"""Durable, private checkpoints for manually resumed generation jobs.

A paid operation is recorded as pending *before* it starts. If its response is
lost, no new paid request is made on resume unless the user explicitly accepts
the duplicate-charge risk. Known A2E task IDs are polled instead of resubmitted.
"""
import asyncio
import contextvars
import copy
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

CURRENT_STEP: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("microdrama_step", default=None)
PAID_KINDS = {"text", "a2e_image", "gemini_image", "a2e_video", "a2e_upload"}
A2E_TASK_KINDS = {"a2e_image", "a2e_video"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_job_id(job_id: str) -> bool:
    try:
        return str(uuid.UUID(job_id)) == job_id
    except (ValueError, TypeError, AttributeError):
        return False


class ProjectError(RuntimeError):
    pass


class UncertainStepError(ProjectError):
    pass


class ProjectStore:
    VERSION = 1

    def __init__(self, root: Path, job_id: str, data: dict, *, secrets: tuple[str, ...] = (),
                 retry_uncertain: bool = False):
        if not valid_job_id(job_id):
            raise ProjectError("Invalid project ID")
        self.root = Path(root)
        self.job_id = job_id
        self.path = self.root / job_id
        self.data = data
        self.secrets = tuple(s for s in secrets if isinstance(s, str) and s)
        self.retry_uncertain = retry_uncertain
        self._lock = asyncio.Lock()
        self._step_locks: dict[str, asyncio.Lock] = {}
        self._halted = False

    @classmethod
    def create(cls, root: Path, job_id: str, request: dict, config: dict,
               *, secrets: tuple[str, ...] = ()) -> "ProjectStore":
        root = Path(root)
        if root.is_symlink():
            raise ProjectError("Refusing a symlinked projects directory")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(root, 0o700)
        store = cls(root, job_id, {}, secrets=secrets)
        if store.path.exists() or store.path.is_symlink():
            raise ProjectError("Project already exists")
        store.path.mkdir(mode=0o700)
        now = _now()
        store.data = {
            "version": cls.VERSION, "job_id": job_id, "created_at": now, "updated_at": now,
            "status": "running", "attempt": 1, "progress": 0, "error": None, "video_url": None,
            "request": store._clean(request), "config": store._clean(config),
            "steps": {}, "events": [], "estimated_total": "0", "attempt_history": [],
        }
        store._write()
        return store

    @classmethod
    def load(cls, root: Path, job_id: str, *, secrets: tuple[str, ...] = (),
             retry_uncertain: bool = False) -> "ProjectStore":
        if not valid_job_id(job_id):
            raise ProjectError("Invalid project ID")
        root = Path(root)
        path = root / job_id
        if root.is_symlink() or path.is_symlink():
            raise ProjectError("Refusing a symlinked project directory")
        metadata = path / "project.json"
        if metadata.is_symlink():
            raise ProjectError("Refusing symlinked project metadata")
        try:
            data = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ProjectError("Project data is missing or unreadable") from exc
        if not isinstance(data, dict) or data.get("version") != cls.VERSION or data.get("job_id") != job_id:
            raise ProjectError("Project data has an unsupported format")
        if (not isinstance(data.get("steps"), dict) or not isinstance(data.get("events"), list) or
                not isinstance(data.get("request"), dict) or not isinstance(data.get("config"), dict) or
                not isinstance(data.get("attempt_history"), list) or
                data.get("status") not in ("running", "failed", "interrupted", "completed") or
                not isinstance(data.get("attempt"), int) or isinstance(data.get("attempt"), bool) or data["attempt"] < 1 or
                not isinstance(data.get("progress"), int) or isinstance(data.get("progress"), bool) or
                not 0 <= data["progress"] <= 100):
            raise ProjectError("Project checkpoints are invalid")
        if any(not isinstance(event, dict) for event in data["events"]):
            raise ProjectError("Project events are invalid")
        for key, step in data["steps"].items():
            if (not isinstance(key, str) or not isinstance(step, dict) or
                    not isinstance(step.get("kind"), str) or step["kind"] not in (PAID_KINDS | {"local"}) or
                    step.get("state") not in ("pending", "failed", "complete") or
                    not isinstance(step.get("label"), str) or
                    not isinstance(step.get("previous_task_ids", []), list) or
                    any(not isinstance(value, str) for value in step.get("previous_task_ids", [])) or
                    (step.get("task_id") is not None and not isinstance(step["task_id"], str))):
                raise ProjectError("Project step records are invalid")
        try:
            estimate = Decimal(str(data.get("estimated_total", "0")))
            if not estimate.is_finite() or estimate < 0:
                raise InvalidOperation
        except (InvalidOperation, ValueError):
            raise ProjectError("Project estimates are invalid") from None
        return cls(root, job_id, data, secrets=secrets, retry_uncertain=retry_uncertain)

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[redacted]")
            return value
        if isinstance(value, list):
            return [self._clean(item) for item in value]
        if isinstance(value, tuple):
            return [self._clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._clean(item) for key, item in value.items()}
        return value

    def _write(self) -> None:
        """Atomic owner-only write; failure stops the pipeline rather than losing a checkpoint."""
        if self.path.is_symlink() or not self.path.is_dir():
            raise ProjectError("Project directory is not safe or no longer exists")
        self.data["updated_at"] = _now()
        try:
            fd, temporary = tempfile.mkstemp(prefix=".project-", suffix=".tmp", dir=self.path)
            try:
                if os.name == "posix":
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    json.dump(self.data, output, ensure_ascii=False, separators=(",", ":"))
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path / "project.json")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except (OSError, TypeError, ValueError) as exc:
            raise ProjectError("Could not save project checkpoints; no automatic retry was made.") from exc

    def halt(self) -> None:
        # Existing provider tasks may finish, but no new operation may start
        # until a user explicitly begins another attempt.
        self._halted = True

    async def _mutate(self, callback: Callable[[dict], None], *, allow_halted: bool = False) -> None:
        async with self._lock:
            if self._halted and not allow_halted:
                raise ProjectError("Project storage failed; no further operation was started.")
            previous = copy.deepcopy(self.data)
            try:
                callback(self.data)
            except BaseException:
                self.data = previous
                raise
            try:
                self._write()
            except BaseException:
                self.data = previous
                self._halted = True
                raise

    async def append_event(self, event: dict, *, final_status: Optional[str] = None) -> None:
        event = self._clean(event)
        def change(data):
            data["events"].append(event)
            progress = event.get("progress")
            if isinstance(progress, int) and progress >= 0:
                data["progress"] = max(data.get("progress", 0), min(100, progress))
            if final_status:
                data["status"] = final_status
                data["video_url"] = event.get("video_url") if final_status == "completed" else None
                data["error"] = event.get("message") if final_status in ("failed", "interrupted") else None
        await self._mutate(change, allow_halted=bool(final_status))

    async def mark_interrupted(self) -> None:
        if self.data.get("status") != "running":
            return
        await self.append_event({"type": "error", "message":
            "The server stopped before this project finished. Saved work is available; review and resume it manually.",
            "progress": -1}, final_status="interrupted")

    async def begin_attempt(self, *, retry_uncertain: bool) -> None:
        def change(data):
            if data.get("status") not in ("failed", "interrupted"):
                raise ProjectError("Only failed or interrupted projects can be resumed")
            data["attempt_history"].append({"attempt": data["attempt"], "status": data["status"],
                                            "error": data.get("error"), "ended_at": _now()})
            data["attempt"] += 1
            data["status"] = "running"
            data["error"] = None
            data["events"] = []
        await self._mutate(change, allow_halted=True)
        self._halted = False
        self.retry_uncertain = retry_uncertain

    async def append_provider_event(self, event: dict) -> None:
        """Atomically persist a task ID/estimate with its visible event."""
        event = self._clean(event)
        key = CURRENT_STEP.get()
        def change(data):
            if event.get("type") == "task" and key:
                step = data["steps"].get(key)
                if not isinstance(step, dict) or step.get("state") != "pending":
                    raise ProjectError("Cannot associate the A2E task with its checkpoint")
                step["task_id"] = event.get("task_id")
                step["endpoint"] = event.get("endpoint")
            if event.get("type") == "quote" and event.get("estimated_total") is not None:
                try:
                    value = Decimal(str(event.get("estimated_total_exact", event["estimated_total"])))
                    if not value.is_finite() or value < 0:
                        raise InvalidOperation
                    old = Decimal(str(data.get("estimated_total", "0")))
                except (ValueError, InvalidOperation) as exc:
                    raise ProjectError("Could not record the A2E estimate safely") from exc
                data["estimated_total"] = str(max(old, value))
            data["events"].append(event)
        # A task already submitted by another coroutine still needs its ID saved
        # even if a concurrent checkpoint failed. New quotes remain blocked.
        await self._mutate(change, allow_halted=event.get("type") == "task")

    async def call(self, key: str, kind: str, label: str,
                   operation: Callable[[Optional[str]], Awaitable[Any]]) -> Any:
        """Serialize identical steps; reuse work or recover a known task."""
        lock = self._step_locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._call_locked(key, kind, label, operation)

    async def _call_locked(self, key: str, kind: str, label: str,
                           operation: Callable[[Optional[str]], Awaitable[Any]]) -> Any:
        entry = self.data["steps"].get(key)
        if isinstance(entry, dict) and entry.get("state") == "complete":
            return copy.deepcopy(entry.get("value"))
        task_id = None
        prior_ids = []
        if isinstance(entry, dict):
            prior_ids = list(entry.get("previous_task_ids", []))
            existing_id = entry.get("task_id")
            recoverable = kind in A2E_TASK_KINDS and isinstance(existing_id, str) and existing_id and not entry.get("terminal")
            if recoverable:
                task_id = existing_id  # read-only polling; never create another task
            elif kind in PAID_KINDS and (not self.retry_uncertain or
                                            entry.get("attempt") == self.data.get("attempt")):
                # Consent applies to work from a previous attempt, never to a
                # newly ambiguous failure encountered during this attempt.
                raise UncertainStepError(
                    f"{label} may already have been billed. Check the provider's task and billing history, "
                    "then explicitly confirm the duplicate-charge risk before retrying this step."
                )
            elif isinstance(existing_id, str) and existing_id:
                prior_ids.append(existing_id)

        def start(data):
            data["steps"][key] = {"kind": kind, "label": label, "state": "pending",
                                  "task_id": task_id, "previous_task_ids": prior_ids,
                                  "terminal": False, "attempt": data["attempt"]}
        await self._mutate(start)  # written before any potentially billed request
        token = CURRENT_STEP.set(key)
        try:
            value = await operation(task_id)
            safe_value = self._clean(value)
            # Ensure it can be saved before considering the step complete.
            json.dumps(safe_value)
            def complete(data):
                data["steps"][key]["value"] = safe_value
                data["steps"][key]["state"] = "complete"
            # Preserve the results of operations that were already in flight
            # when another checkpoint failed, if the filesystem recovers.
            try:
                await self._mutate(complete, allow_halted=True)
            except ProjectError:
                # Retry only the local write, never the operation itself. Still
                # stop this attempt so no further paid work begins after a fault.
                try:
                    await self._mutate(complete, allow_halted=True)
                except ProjectError:
                    pass
                raise
            return copy.deepcopy(safe_value)
        except BaseException as exc:
            def fail(data):
                step = data["steps"][key]
                if step.get("state") == "complete":
                    return
                step["state"] = "failed"
                step["terminal"] = bool(getattr(exc, "terminal", False))
                step["error"] = self._clean(str(exc))[:800]
            try:
                await self._mutate(fail, allow_halted=True)
            except Exception:
                # The last durable pending record remains conservative.
                pass
            raise
        finally:
            CURRENT_STEP.reset(token)

    def step(self, key: str) -> Optional[dict]:
        value = self.data["steps"].get(key)
        return copy.deepcopy(value) if isinstance(value, dict) else None

    async def invalidate_local(self, key: str) -> None:
        def change(data):
            entry = data["steps"].get(key)
            if entry and entry.get("kind") == "local":
                data["steps"].pop(key)
        await self._mutate(change)

    async def mark_video_expired(self, key: str, task_id: str) -> None:
        """A provider-confirmed lost output requires consent before replacement."""
        def change(data):
            entry = data["steps"].get(key)
            if (not entry or entry.get("kind") != "a2e_video" or entry.get("state") != "complete" or
                    entry.get("task_id") != task_id):
                raise ProjectError("Cannot update the expired video checkpoint.")
            entry["state"] = "failed"
            entry["terminal"] = True
            entry["attempt"] = data["attempt"]
            entry["error"] = "The provider confirmed this video output has expired."
        await self._mutate(change, allow_halted=True)

    async def mark_gemini_image_missing(self, key: str) -> None:
        def change(data):
            entry = data["steps"].get(key)
            if not entry or entry.get("kind") != "gemini_image" or entry.get("state") != "complete":
                raise ProjectError("Cannot update the missing Gemini image checkpoint.")
            entry["state"] = "failed"
            entry["terminal"] = True
            entry["missing_asset"] = True
            entry["attempt"] = data["attempt"]
            entry["error"] = "The local Gemini image is missing or invalid."
        await self._mutate(change, allow_halted=True)

    async def restore_gemini_image_checkpoint(self, key: str) -> None:
        def change(data):
            entry = data["steps"].get(key)
            if not entry or entry.get("kind") != "gemini_image" or not entry.get("missing_asset"):
                raise ProjectError("Cannot restore the Gemini image checkpoint.")
            entry["state"] = "complete"
            entry["terminal"] = False
            entry.pop("missing_asset", None)
            entry.pop("error", None)
        await self._mutate(change)

    async def mark_image_expired(self, key: str, task_id: str) -> None:
        def change(data):
            entry = data["steps"].get(key)
            if (not entry or entry.get("kind") != "a2e_image" or entry.get("state") != "complete" or
                    entry.get("task_id") != task_id):
                raise ProjectError("Cannot update the expired image checkpoint.")
            entry["state"] = "failed"
            entry["terminal"] = True
            entry["attempt"] = data["attempt"]
            entry["error"] = "The provider confirmed this image output has expired."
        await self._mutate(change, allow_halted=True)

    async def refresh_image_url(self, key: str, url: str) -> None:
        """Replace an expired link only with the result of polling the same task."""
        def change(data):
            entry = data["steps"].get(key)
            if not entry or entry.get("kind") != "a2e_image" or entry.get("state") != "complete":
                raise ProjectError("Cannot update the saved image checkpoint.")
            entry["value"] = self._clean(url)
        await self._mutate(change, allow_halted=True)

    def save_asset(self, key: str, content: bytes, mime: str) -> dict:
        extensions = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
        if mime not in extensions or not content or len(content) > 20 * 1024 * 1024:
            raise ProjectError("Cannot save an unsupported or oversized image")
        assets = self.path / "assets"
        if assets.is_symlink():
            raise ProjectError("Refusing a symlinked project assets directory")
        assets.mkdir(mode=0o700, exist_ok=True)
        name = hashlib.sha256(key.encode()).hexdigest()[:24] + extensions[mime]
        destination = assets / name
        try:
            fd, temp = tempfile.mkstemp(prefix=".asset-", dir=assets)
            try:
                if os.name == "posix":
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temp, destination)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        except OSError as exc:
            raise ProjectError("Could not save the generated image in the project.") from exc
        return {"asset": f"assets/{name}", "mime": mime, "sha256": hashlib.sha256(content).hexdigest()}

    def asset_bytes(self, relative: str, sha256: Optional[str] = None) -> bytes:
        if not isinstance(relative, str) or not relative.startswith("assets/"):
            raise ProjectError("Invalid saved image path")
        path = self.path / relative
        if (self.path.is_symlink() or (self.path / "assets").is_symlink() or
                path.parent != self.path / "assets" or path.is_symlink()):
            raise ProjectError("Invalid saved image path")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProjectError("A previously saved image is missing. No replacement was generated automatically.") from exc
        if (not data or len(data) > 20 * 1024 * 1024 or
                (sha256 is not None and hashlib.sha256(data).hexdigest() != sha256)):
            raise ProjectError("A previously saved image is invalid. No replacement was generated automatically.")
        return data

    def summary(self) -> dict:
        request = self.data.get("request", {})
        idea = request.get("idea") if isinstance(request.get("idea"), str) else ""
        script = request.get("script") if isinstance(request.get("script"), str) else ""
        title_text = idea.strip() or script.strip() or "Untitled project"
        title = title_text.splitlines()[0][:100]
        completed = []
        uncertain = []
        assets = []
        tasks = []
        for key, step in self.data["steps"].items():
            if not isinstance(step, dict):
                continue
            state, kind = step.get("state"), step.get("kind")
            if kind in A2E_TASK_KINDS:
                previous = step.get("previous_task_ids")
                for task_id in previous if isinstance(previous, list) else []:
                    if isinstance(task_id, str) and task_id:
                        tasks.append({"label": step.get("label", key), "task_id": task_id, "state": "previous"})
                task_id = step.get("task_id")
                if isinstance(task_id, str) and task_id:
                    tasks.append({"label": step.get("label", key), "task_id": task_id, "state": state})
            if state == "complete":
                completed.append(step.get("label", key))
                value = step.get("value")
                if isinstance(value, dict) and isinstance(value.get("asset"), str):
                    relative = value["asset"]
                    path = self.path / relative
                    if relative.startswith("assets/") and path.parent == self.path / "assets" and path.is_file() and not path.is_symlink():
                        assets.append({"label": step.get("label", "Saved image"),
                                       "url": f"/api/projects/{self.job_id}/assets/{path.name}", "type": "image"})
                    elif (value.get("type") == "video" and
                          (relative == "final_video.mp4" or
                           (len(Path(relative).parts) == 2 and Path(relative).parts[0].startswith("scene_") and
                            path.suffix == ".mp4")) and path.is_file() and not path.is_symlink() and path.stat().st_size):
                        assets.append({"label": step.get("label", "Saved video"),
                                       "url": f"/outputs/{self.job_id}/{relative}", "type": "video"})
            elif state in ("pending", "failed") and kind in PAID_KINDS:
                recoverable = kind in A2E_TASK_KINDS and step.get("task_id") and not step.get("terminal")
                if not recoverable:
                    item = {"label": step.get("label", key), "kind": kind}
                    if isinstance(step.get("task_id"), str) and step["task_id"]:
                        item["task_id"] = step["task_id"]
                    uncertain.append(item)
        return {"job_id": self.job_id, "title": title, "mode": request.get("mode"),
                "status": self.data.get("status"), "error": self.data.get("error"),
                "progress": self.data.get("progress", 0), "attempt": self.data.get("attempt", 1),
                "updated_at": self.data.get("updated_at"), "completed_steps": completed,
                "uncertain_steps": uncertain, "assets": assets, "a2e_tasks": tasks,
                "estimated_a2e_credits": self.data.get("estimated_total", "0"),
                "video_url": self.data.get("video_url")}
