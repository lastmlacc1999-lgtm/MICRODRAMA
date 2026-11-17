import asyncio
import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import dotenv_values, load_dotenv, set_key

# Resolve persistent paths relative to this file, not the shell's current
# directory. Tests can point at an isolated, fake environment file.
SERVER_DIR = Path(__file__).resolve().parent
ENV_FILE = Path(os.environ.get("MICRODRAMA_ENV_FILE", str(SERVER_DIR / ".env")))


def load_server_environment(path: Path) -> None:
    load_dotenv(path)
    # UI-saved keys survive a restart even when stale keys were exported in the
    # shell. Other variables retain dotenv's normal environment-first behavior.
    saved = dotenv_values(path, interpolate=False)
    for name in ("A2E_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        token = saved.get(name)
        if token and token.strip():
            os.environ[name] = token.strip()


load_server_environment(ENV_FILE)
_token_write_lock = threading.Lock()

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StrictInt, model_validator

from agents.character_extractor import CharacterExtractor
from interfaces.character import CharacterInScene
from pipelines.idea2video import Idea2VideoPipeline
from pipelines.script2video import Script2VideoPipeline
from tools.a2e_client import A2EClient
from tools.gemini_image import GeminiImageClient
from tools.gemini_text import GeminiTextLLM
from tools.llm import OpenAICompatibleLLM, TextLLM
from tools.openai_text import OpenAITextLLM
from tools.model_catalog import (DEFAULT_GEMINI_TEXT_MODEL, DEFAULT_OPENAI_TEXT_MODEL, IMAGE_BY_KEY, VIDEO_BY_ID,
                                 gemini_text_model, openai_text_model, image_model, public_catalog, video_model)
from tools.project_store import ProjectError, ProjectStore, valid_job_id
from utils.limits import pipeline_limit

app = FastAPI(title="MicroDrama AI · A2E API", version="2.1.0")
# Unlike the old transient outputs/ directory, this name survives workspace
# snapshots. It is private project data and must never be included in downloads.
PROJECTS_DIR = SERVER_DIR / "projects"
LEGACY_OUTPUTS_DIR = SERVER_DIR / "outputs"

# Runtime broadcasts stay in memory; project state and work are stored on disk.
# Use one worker. Jobs interrupted by a restart require explicit manual resume.
jobs: Dict[str, Dict[str, Any]] = {}
_resume_lock = asyncio.Lock()


def _sensitive_keys(*snapshotted: str) -> tuple[str, ...]:
    """Protect both the job's captured credentials and current server keys."""
    candidates = list(snapshotted) + [os.environ.get(name, "").strip() for name in
                                     ("A2E_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")]
    # Redact longer keys first, in case one credential is a prefix of another.
    return tuple(sorted({key for key in candidates if isinstance(key, str) and key}, key=len, reverse=True))


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value


class GenerateRequest(BaseModel):
    idea: str = Field(default="", max_length=3000)
    user_requirement: str = Field(default="", max_length=2000)
    style: str = Field(default="Cinematic", min_length=1, max_length=100)
    mode: Literal["idea2video", "script2video"] = "idea2video"
    script: str = Field(default="", max_length=8000)
    # None retains the original behavior: use the configured external LLM, if any.
    planning_provider: Optional[Literal["simple", "gemini", "openai", "configured"]] = None
    planning_model: Optional[str] = Field(default=None, min_length=1, max_length=100)
    image_provider: Literal["a2e", "gemini"] = "a2e"
    image_model: Optional[str] = Field(default=None, min_length=1, max_length=100)
    video_model: Optional[str] = Field(default=None, min_length=1, max_length=100)
    clip_duration: Optional[StrictInt] = Field(default=None, ge=2, le=30)

    @model_validator(mode="after")
    def require_input(self):
        if self.mode == "idea2video" and not self.idea.strip():
            raise ValueError("An idea is required for idea-to-video")
        if self.mode == "script2video" and not self.script.strip():
            raise ValueError("A script is required for script-to-video")
        return self


class GenerateResponse(BaseModel):
    job_id: str


class JobResult(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    resumable: bool = False


class ResumeRequest(BaseModel):
    acknowledge_costs: bool = False
    retry_uncertain: bool = False


async def _publish(job: dict, event: dict, *, final_status: Optional[str] = None,
                   provider_event: bool = False) -> None:
    async with job["condition"]:
        try:
            if job.get("project"):
                if provider_event:
                    await job["project"].append_provider_event(event)
                else:
                    await job["project"].append_event(event, final_status=final_status)
        except ProjectError:
            # Stop NEW work, but leave the job running until all concurrently
            # submitted tasks settle. Otherwise a resume could race them and
            # accidentally create duplicates. Keep unsaved task IDs visible.
            if job.get("project"):
                job["project"].halt()
            if event.get("type") == "task":
                # One storage-only retry can preserve an already-created task ID;
                # this does not repeat a provider request or spend credits.
                try:
                    await job["project"].append_provider_event(event)
                except ProjectError:
                    pass
                job["events"].append(event)
            message = ("Could not save project checkpoints. Record any task IDs and check provider billing "
                       "before retrying; other in-flight tasks are being settled.")
            job["events"].append({"type": "warning", "message": message})
            if final_status:
                job["status"] = "failed"
                job["error"] = message
                job["events"].append({"type": "error", "message": message, "progress": -1})
            job["condition"].notify_all()
            raise
        job["events"].append(event)
        if final_status:
            job["status"] = final_status
            job["video_url"] = event.get("video_url")
            job["error"] = event.get("message") if final_status in ("failed", "interrupted") else None
        job["condition"].notify_all()


async def run_pipeline(job_id: str, req: GenerateRequest, job_token: str, gemini_key: str,
                       llm: Optional[TextLLM] = None) -> None:
    job = jobs[job_id]
    project: ProjectStore = job["project"]
    # Keys and planner settings were snapshotted when the job was accepted.
    # Saving a replacement affects future jobs only.
    llm_key = getattr(llm, "api_key", "") if llm else ""

    async def progress_callback(stage: str, message: str, progress: int) -> None:
        # A later scene or quote must not make the visible progress move backwards.
        job["progress"] = max(job["progress"], min(99, max(0, progress)))
        await _publish(job, {"type": "progress", "stage": stage, "message": message,
                             "progress": job["progress"]})

    async def a2e_event(event: dict) -> None:
        # Defensive redaction applies to task IDs/messages as well as errors.
        sensitive = _sensitive_keys(job_token, gemini_key, llm_key, *project.secrets)
        event = _redact(event, sensitive)
        await _publish(job, event, provider_event=True)
        if event.get("type") in ("quote", "task"):
            try:
                entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
                with (project.path / "a2e_events.jsonl").open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(entry) + "\n")
            except OSError:
                # A paid task may already exist. Keep its event visible instead
                # of retrying it, even if local persistence failed.
                await _publish(job, {"type": "warning", "message":
                    "Could not save A2E task/quote history locally. Record task IDs from this log."})

    try:
        # Validate all limits before making any potentially billed LLM or A2E call.
        pipeline_limit("MAX_SCENES", 2, 1, 4)
        pipeline_limit("MAX_SHOTS_PER_SCENE", 3, 1, 5)
        pipeline_limit("MAX_CHARACTERS", 2, 0, 5)
        planner = (f"Gemini / {req.planning_model}" if req.planning_provider == "gemini" else
                   f"OpenAI GPT / {req.planning_model}" if req.planning_provider == "openai" else
                   "configured LLM" if req.planning_provider == "configured" else "simple (no text LLM)")
        await progress_callback("frames", f"Planning: {planner}; image: {req.image_provider} / {req.image_model}; video: A2E / {req.video_model}.", 1)
        async with AsyncExitStack() as stack:
            a2e = await stack.enter_async_context(A2EClient(
                api_key=job_token, on_event=a2e_event,
                image_model=req.image_model if req.image_provider == "a2e" else "a2e",
                video_model=req.video_model,
            ))
            # Carry earlier estimates forward across manual resumes. This guard
            # remains conservative and is not a cap on actual provider billing.
            if hasattr(a2e, "budget"):
                a2e.budget.total = Decimal(project.data.get("estimated_total", "0"))
            images = (await stack.enter_async_context(GeminiImageClient(gemini_key, req.image_model, a2e))
                      if req.image_provider == "gemini" else a2e)
            if req.mode == "script2video":
                extractor = CharacterExtractor(llm)
                await progress_callback(
                    "characters", "Extracting or reusing characters..." if llm else
                    "Using your script directly; LLM character extraction is disabled.", 10,
                )
                async def extract(_):
                    return [character.model_dump() for character in await extractor.extract_characters(req.script)]
                character_data = await project.call("characters", "text" if llm else "local",
                                                    "Character descriptions", extract)
                characters = [CharacterInScene.model_validate(value) for value in character_data]
                pipeline = Script2VideoPipeline(a2e, llm, job_id, image_generator=images,
                                                clip_duration=req.clip_duration, project=project,
                                                image_provider=req.image_provider)
                video_path = await pipeline.run(
                    script=req.script, characters=characters, user_requirement=req.user_requirement,
                    style=req.style, working_dir=str(project.path / "scene_00"),
                    progress_callback=progress_callback, scene_idx=0,
                    base_progress=15, progress_range=80,
                )
            else:
                pipeline = Idea2VideoPipeline(a2e, llm, job_id, image_generator=images,
                                              clip_duration=req.clip_duration, project=project,
                                              image_provider=req.image_provider)
                video_path = await pipeline.run(
                    idea=req.idea, user_requirement=req.user_requirement, style=req.style,
                    job_id=job_id, progress_callback=progress_callback,
                )
        rel_path = Path(video_path).relative_to(project.path)
        await _publish(job, {"type": "complete", "video_url": f"/outputs/{job_id}/{rel_path.as_posix()}",
                             "progress": 100}, final_status="completed")
    except Exception as exc:
        # Provider errors must never expose either captured or current keys.
        message = _redact(str(exc), _sensitive_keys(job_token, gemini_key, llm_key, *project.secrets))
        try:
            await _publish(job, {"type": "error", "message": message, "progress": -1},
                           final_status="failed")
        except ProjectError:
            # _publish has already surfaced a safe storage failure in memory.
            pass


def _save_api_token(name: str, token: str) -> None:
    """Persist a key locally without logging it or changing other settings."""
    if name not in ("A2E_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        raise ValueError("Invalid key setting")
    with _token_write_lock:
        if ENV_FILE.is_symlink():
            raise OSError("Refusing to modify a symlinked environment file")
        # set_key writes through a temporary file and preserves the old mode.
        if ENV_FILE.exists():
            os.chmod(ENV_FILE, 0o600)
        set_key(str(ENV_FILE), name, token, quote_mode="always")
        os.chmod(ENV_FILE, 0o600)
        os.environ[name] = token  # update only after persistence succeeds


def _save_a2e_token(token: str) -> None:
    _save_api_token("A2E_API_KEY", token)


def _save_gemini_token(token: str) -> None:
    _save_api_token("GEMINI_API_KEY", token)


def _save_openai_token(token: str) -> None:
    _save_api_token("OPENAI_API_KEY", token)


async def _save_key_request(request: Request, provider: str):
    """Save a key without ever returning it or including it in validation errors."""
    # A custom header and JSON make cross-site browser form submissions fail;
    # this is not authentication. The app must still run on a private server.
    if request.headers.get("x-requested-with") != "MicroDrama":
        raise HTTPException(status_code=403, detail="Request must originate from the MicroDrama app.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="Expected a JSON request.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 8192:
            raise HTTPException(status_code=413, detail="Request is too large.")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON request.") from None
    raw_token = data.get("api_key") if isinstance(data, dict) else None
    invalid_message = {
        "a2e": "Enter an A2E API token beginning with sk_, without spaces.",
        "gemini": "Enter a Gemini API key without spaces.",
        "openai": "Enter an OpenAI API key without spaces.",
    }[provider]
    if not isinstance(raw_token, str):
        raise HTTPException(status_code=422, detail=invalid_message)
    token = raw_token.strip()
    if (not 4 <= len(token) <= 2048 or any(not (33 <= ord(char) <= 126) for char in token) or
            (provider == "a2e" and not token.startswith("sk_"))):
        raise HTTPException(status_code=422, detail=invalid_message)
    try:
        saver = {"a2e": _save_a2e_token, "gemini": _save_gemini_token, "openai": _save_openai_token}[provider]
        await asyncio.to_thread(saver, token)
    except Exception:
        # Never reflect a filesystem or parser exception that might contain a secret.
        raise HTTPException(status_code=500, detail="Could not save the key on the server. Check server file permissions.") from None
    return JSONResponse({"configured": True, "verified": False}, headers={"Cache-Control": "no-store"})


_key_openapi = {"requestBody": {"required": True, "content": {"application/json": {"schema": {
    "type": "object", "required": ["api_key"], "properties": {"api_key": {"type": "string", "writeOnly": True}},
}}}}}


@app.post("/api/settings/a2e-key", openapi_extra=_key_openapi)
async def save_a2e_key(request: Request):
    return await _save_key_request(request, "a2e")


@app.post("/api/settings/gemini-key", openapi_extra=_key_openapi)
async def save_gemini_key(request: Request):
    return await _save_key_request(request, "gemini")


@app.post("/api/settings/openai-key", openapi_extra=_key_openapi)
async def save_openai_key(request: Request):
    return await _save_key_request(request, "openai")


@app.get("/api/health")
async def health():
    return {
        "status": "ok", "service": "microdrama-api", "media_provider": "a2e",
        "a2e_configured": bool(os.environ.get("A2E_API_KEY", "").strip()),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        # Legacy health flag and implicit planner depend only on LLM_API_KEY.
        "planning": "llm" if os.environ.get("LLM_API_KEY", "").strip() else "a2e-only",
    }


@app.get("/api/models")
async def models():
    # A static catalog, not a check that the provider account can use a model.
    default_image = os.environ.get("A2E_IMAGE_MODEL", "a2e").strip()
    default_video = os.environ.get("A2E_VIDEO_MODEL", "a2e").strip()
    if ("a2e", default_image) not in IMAGE_BY_KEY:
        default_image = "a2e"
    if default_video not in VIDEO_BY_ID:
        default_video = "a2e"
    configured_llm = bool(os.environ.get("LLM_API_KEY", "").strip())
    return JSONResponse(public_catalog(default_image, default_video, configured_llm),
                        headers={"Cache-Control": "no-store"})


def _resolve_generation(req: GenerateRequest):
    # Resolve and validate every selection before *any* paid provider call.
    selected_image = req.image_model or (os.environ.get("A2E_IMAGE_MODEL", "a2e").strip()
                                         if req.image_provider == "a2e" else "gemini-nano-banana-2.1")
    selected_video = req.video_model or os.environ.get("A2E_VIDEO_MODEL", "a2e").strip()
    try:
        image_model(req.image_provider, selected_image)
        spec = video_model(selected_video)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    duration = req.clip_duration if req.clip_duration is not None else spec.default_duration
    if (spec.durations and duration not in spec.durations) or (not spec.durations and duration is not None):
        if spec.durations:
            detail = f"{spec.label} supports clip durations: {', '.join(map(str, spec.durations))} seconds."
        else:
            detail = f"{spec.label} does not expose a clip-duration control through the A2E API."
        raise HTTPException(status_code=422, detail=detail)
    # Existing clients that omit the planner keep the previous behavior: use
    # the server-configured LLM if present, otherwise simple planning. Gemini
    # and OpenAI GPT are explicit per-job choices, even when keys are saved.
    planner = req.planning_provider or ("configured" if os.environ.get("LLM_API_KEY", "").strip() else "simple")
    selected_text = None
    if planner in ("gemini", "openai"):
        selected_text = req.planning_model or (DEFAULT_GEMINI_TEXT_MODEL if planner == "gemini" else DEFAULT_OPENAI_TEXT_MODEL)
        try:
            (gemini_text_model if planner == "gemini" else openai_text_model)(selected_text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
    elif req.planning_model is not None:
        raise HTTPException(status_code=422, detail="A planning model may only be specified for Gemini or OpenAI GPT planning.")

    token = os.environ.get("A2E_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="A2E API key is missing. Save it in the website's API key settings.")
    if (req.image_provider == "gemini" or planner == "gemini") and not gemini_key:
        raise HTTPException(status_code=503, detail="Gemini API key is missing. Save it in the website's API key settings.")
    if planner == "openai" and not openai_key:
        raise HTTPException(status_code=503, detail="OpenAI API key is missing. Save it in the website's API key settings.")

    # Snapshot the planner instance (including key and model) when accepting the
    # job. A later key/settings change must not alter an in-flight job. Explicit
    # selection must never silently fall back to a different planner.
    llm: Optional[TextLLM] = None
    if planner == "gemini":
        llm = GeminiTextLLM(gemini_key, selected_text)
    elif planner == "openai":
        llm = OpenAITextLLM(openai_key, selected_text)
    elif planner == "configured":
        try:
            llm = OpenAICompatibleLLM.from_env()
        except ValueError:
            raise HTTPException(status_code=503, detail="Configured LLM settings are incomplete. Check LLM_BASE_URL and LLM_MODEL.") from None
        if llm is None:
            raise HTTPException(status_code=503, detail="No OpenAI-compatible LLM is configured. Set LLM_API_KEY on the server or select another planner.")

    chosen = req.model_copy(update={"planning_provider": planner, "planning_model": selected_text,
                                    "image_model": selected_image, "video_model": selected_video,
                                    "clip_duration": duration})
    return chosen, token, gemini_key, llm


def _project_config(req: GenerateRequest, llm: Optional[TextLLM]) -> dict:
    try:
        limits = {"MAX_SCENES": pipeline_limit("MAX_SCENES", 2, 1, 4),
                  "MAX_SHOTS_PER_SCENE": pipeline_limit("MAX_SHOTS_PER_SCENE", 3, 1, 5),
                  "MAX_CHARACTERS": pipeline_limit("MAX_CHARACTERS", 2, 0, 5)}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    audio = os.environ.get("A2E_AUTO_ADD_AUDIO", "false").strip().lower()
    if audio not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise HTTPException(status_code=503, detail="A2E_AUTO_ADD_AUDIO must be true or false.")
    fingerprint = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    config = {"limits": limits, "native_audio": audio in ("1", "true", "yes", "on"),
              "a2e_origin_hash": fingerprint(os.environ.get("A2E_BASE_URL", "https://api.a2e.ai").rstrip("/"))}
    if req.planning_provider == "configured" and llm:
        config["llm_model"] = getattr(llm, "model", "")
        config["llm_origin_hash"] = fingerprint(getattr(llm, "base_url", ""))
    return config


def _job_from_project(project: ProjectStore) -> dict:
    data = project.data
    return {"status": data.get("status", "failed"), "events": list(data.get("events", [])),
            "video_url": data.get("video_url"), "error": data.get("error"),
            "progress": data.get("progress", 0), "condition": asyncio.Condition(), "project": project}


async def _get_job(job_id: str) -> dict:
    if not valid_job_id(job_id):
        raise HTTPException(status_code=404, detail="Project not found")
    if job_id not in jobs:
        try:
            project = ProjectStore.load(PROJECTS_DIR, job_id)
            if project.data.get("status") == "running":
                await project.mark_interrupted()
        except ProjectError:
            raise HTTPException(status_code=404, detail="Project not found or its saved data is unreadable") from None
        jobs[job_id] = _job_from_project(project)
    return jobs[job_id]


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    chosen, token, gemini_key, llm = _resolve_generation(req)
    llm_key = getattr(llm, "api_key", "") if llm else ""
    secrets = _sensitive_keys(token, gemini_key, llm_key)
    if any(secret in value for secret in secrets for value in
           (chosen.idea, chosen.script, chosen.user_requirement, chosen.style)):
        raise HTTPException(status_code=422, detail="Remove API keys from your project text before saving it.")
    config = _project_config(chosen, llm)
    job_id = str(uuid.uuid4())
    try:
        project = ProjectStore.create(PROJECTS_DIR, job_id, chosen.model_dump(), config, secrets=secrets)
    except (ProjectError, OSError):
        raise HTTPException(status_code=503, detail="Could not create a durable project. No generation was submitted.") from None
    jobs[job_id] = _job_from_project(project)
    background_tasks.add_task(run_pipeline, job_id, chosen, token, gemini_key, llm)
    return GenerateResponse(job_id=job_id)


@app.get("/api/status/{job_id}")
async def status_stream(job_id: str, last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")):
    """Broadcast an event log, replaying from Last-Event-ID on reconnect."""
    job = await _get_job(job_id)
    try:
        start_index = max(0, int(last_event_id or 0))
    except ValueError:
        start_index = 0

    async def event_generator():
        index = min(start_index, len(job["events"]))
        while True:
            async with job["condition"]:
                if index >= len(job["events"]) and job["status"] == "running":
                    try:
                        await asyncio.wait_for(job["condition"].wait(), timeout=15)
                    except asyncio.TimeoutError:
                        pass
                pending = job["events"][index:]
                done = job["status"] in ("completed", "failed", "interrupted")
            if pending:
                for event in pending:
                    index += 1
                    yield f"id: {index}\ndata: {json.dumps(event)}\n\n"
            elif not done:
                yield ": keepalive\n\n"
            if done and index >= len(job["events"]):
                return

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })


@app.get("/api/result/{job_id}", response_model=JobResult)
async def get_result(job_id: str):
    job = await _get_job(job_id)
    return JobResult(job_id=job_id, status=job["status"], video_url=job.get("video_url"),
                     error=job.get("error"), resumable=job["status"] in ("failed", "interrupted"))


@app.get("/api/projects")
async def list_projects():
    summaries = []
    if PROJECTS_DIR.is_dir() and not PROJECTS_DIR.is_symlink():
        for path in PROJECTS_DIR.iterdir():
            if not path.is_dir() or path.is_symlink() or not valid_job_id(path.name):
                continue
            try:
                job = await _get_job(path.name)
                summary = job["project"].summary()
                summary["status"] = job["status"]
                summaries.append(summary)
            except (HTTPException, ProjectError, OSError):
                summaries.append({"job_id": path.name, "title": "Unreadable project", "status": "unavailable",
                                  "updated_at": "", "completed_steps": [], "uncertain_steps": [], "assets": []})
    summaries.sort(key=lambda value: value.get("updated_at") or "", reverse=True)
    return JSONResponse({"projects": summaries}, headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{job_id}")
async def project_details(job_id: str):
    job = await _get_job(job_id)
    summary = job["project"].summary()
    summary["status"] = job["status"]
    summary["error"] = job.get("error")
    return JSONResponse(summary, headers={"Cache-Control": "no-store"})


@app.post("/api/projects/{job_id}/resume", response_model=GenerateResponse)
async def resume_project(job_id: str, body: ResumeRequest, request: Request, background_tasks: BackgroundTasks):
    if request.headers.get("x-requested-with") != "MicroDrama":
        raise HTTPException(status_code=403, detail="Request must originate from the MicroDrama app.")
    if not body.acknowledge_costs:
        raise HTTPException(status_code=422, detail="Confirm that remaining steps may incur charges before resuming.")
    async with _resume_lock:
        job = await _get_job(job_id)
        if job["status"] not in ("failed", "interrupted"):
            raise HTTPException(status_code=409, detail="This project is already running or completed.")
        project: ProjectStore = job["project"]
        uncertain = project.summary()["uncertain_steps"]
        if uncertain and not body.retry_uncertain:
            raise HTTPException(status_code=409, detail=
                "Some requests may already have been billed. Check provider task and billing records and explicitly confirm before retrying them.")
        try:
            saved_request = GenerateRequest.model_validate(project.data["request"])
        except (ValueError, KeyError, TypeError):
            raise HTTPException(status_code=409, detail="The saved project request is invalid; no generation was submitted.") from None
        chosen, token, gemini_key, llm = _resolve_generation(saved_request)
        if _project_config(chosen, llm) != project.data.get("config"):
            raise HTTPException(status_code=409, detail=
                "The server's planning limits, provider URL, configured LLM model or native audio setting changed. Restore the original settings before resuming.")
        llm_key = getattr(llm, "api_key", "") if llm else ""
        # A resume uses the current key, but the original saved model/provider.
        # Never persist credentials in the checkpoint.
        project.secrets = _sensitive_keys(token, gemini_key, llm_key, *project.secrets)
        try:
            # If final status could not be persisted, the live job may have
            # stopped while the last durable snapshot still says "running".
            if project.data.get("status") == "running":
                await project.mark_interrupted()
            await project.begin_attempt(retry_uncertain=body.retry_uncertain)
        except ProjectError:
            raise HTTPException(status_code=503, detail="Could not save the resume state. No generation was submitted.") from None
        jobs[job_id] = _job_from_project(project)
        background_tasks.add_task(run_pipeline, job_id, chosen, token, gemini_key, llm)
    return GenerateResponse(job_id=job_id)


@app.get("/api/projects/{job_id}/assets/{filename}")
async def saved_image(job_id: str, filename: str):
    if not valid_job_id(job_id) or not re.fullmatch(r"[a-f0-9]{24}\.(jpg|png|webp)", filename):
        raise HTTPException(status_code=404, detail="Saved image not found")
    job = await _get_job(job_id)
    directory = job["project"].path / "assets"
    path = directory / filename
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise HTTPException(status_code=404, detail="Saved image not found")
    return FileResponse(path, headers={"Cache-Control": "private, no-store"})


@app.get("/outputs/{job_id}/{relative:path}")
async def saved_video(job_id: str, relative: str):
    if not valid_job_id(job_id):
        raise HTTPException(status_code=404, detail="Video not found")
    parts = Path(relative).parts
    allowed = (relative == "final_video.mp4" or
               (len(parts) == 2 and re.fullmatch(r"scene_\d{2}", parts[0]) and
                re.fullmatch(r"(?:scene_video|shot_\d{3})\.mp4", parts[1])))
    if not allowed:
        raise HTTPException(status_code=404, detail="Video not found")
    for root in (PROJECTS_DIR, LEGACY_OUTPUTS_DIR):
        directory = root / job_id
        path = directory.joinpath(*parts)
        if (not root.is_symlink() and not directory.is_symlink() and not path.is_symlink() and
                all(not parent.is_symlink() for parent in path.parents if parent != root.parent) and path.is_file()):
            return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})
    raise HTTPException(status_code=404, detail="Video not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
