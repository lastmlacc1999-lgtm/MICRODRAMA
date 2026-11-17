import asyncio
import io
import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from tools.a2e_client import A2EClient
from tools.gemini_text import GeminiTextLLM


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    api.jobs.clear()
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "A2E_BASE_URL", "A2E_IMAGE_MODEL",
                 "A2E_VIDEO_MODEL", "A2E_AUTO_ADD_AUDIO", "A2E_MAX_JOB_CREDITS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-secret")
    monkeypatch.setenv("MAX_SCENES", "1")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    monkeypatch.setenv("MAX_CHARACTERS", "1")
    monkeypatch.setattr(api, "_resume_lock", asyncio.Lock())
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(buffer, "JPEG")
    image = buffer.getvalue()

    async def fake_image(url):
        return image, "image/jpeg"

    async def fake_video(url, dest):
        Path(dest).write_bytes(b"saved-video")
        return dest

    async def fake_concat(paths, output):
        assert all(Path(path).is_file() for path in paths)
        Path(output).write_bytes(b"combined-video")
        return output

    async def available(url):
        return True

    monkeypatch.setattr("pipelines.script2video.download_image", fake_image)
    monkeypatch.setattr("pipelines.script2video.image_url_available", available)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_video)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    return tmp_path, image


def fake_a2e_handler(counts, *, processing=None, timeout_first_image=False):
    def handler(request):
        path = request.url.path
        if path == "/v1/generation/quote":
            counts["quote"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if path == "/v1/userText2Image/start":
            counts["image_start"] += 1
            if timeout_first_image and counts["image_start"] == 1:
                raise httpx.ReadTimeout("lost response", request=request)
            return httpx.Response(200, json={"code": 0, "data": [{"_id": f"img-{counts['image_start']}"}]})
        if path.startswith("/v1/userText2Image/img-"):
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed",
                "image_urls": ["https://cdn.test/" + path.rsplit("/", 1)[-1] + ".jpg"]}})
        if path == "/v1/userImage2Video/start":
            counts["video_start"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"_id": f"vid-{counts['video_start']}"}})
        if path.startswith("/v1/userImage2Video/vid-"):
            task = path.rsplit("/", 1)[-1]
            counts.setdefault("polls", []).append(task)
            state = "processing" if processing and processing(task) else "completed"
            return httpx.Response(200, json={"code": 0, "data": {"current_status": state,
                "result_url": "https://cdn.test/" + task + ".mp4"}})
        raise AssertionError(f"Unexpected A2E path: {path}")
    return handler


def install_a2e(monkeypatch, handler):
    class MockA2E(A2EClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handler), poll_interval=.001, poll_timeout=.04)
    monkeypatch.setattr(api, "A2EClient", MockA2E)


def gemini_response(text):
    return {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}


def test_storyboard_503_saves_planning_before_any_media_and_resumes_only_failed_step(monkeypatch, isolated):
    tmp_path, image = isolated
    text_calls = []
    fail_storyboard = {"value": True}
    counts = {"quote": 0, "image_start": 0, "video_start": 0}

    def google(request):
        payload = json.loads(request.content)
        prompt = payload["input"]
        text_calls.append(prompt.split(" ", 3)[0:3])
        if prompt.startswith("Develop this idea"):
            text = "Mira finds a violin and shares music."
        elif prompt.startswith("Extract at most"):
            text = json.dumps({"characters": [{"idx": 0, "name": "Mira", "static_features": "brown fur",
                                               "dynamic_features": "red scarf", "is_visible": True}]})
        elif prompt.startswith("Turn this story"):
            text = json.dumps({"scenes": [{"script": "Mira finds a violin."}]})
        elif prompt.startswith("Break this scene"):
            if fail_storyboard["value"]:
                return httpx.Response(503, json={"error": "temporary fake-gemini-secret"})
            text = json.dumps({"shots": [
                {"idx": 0, "visual_desc": "Mira finds a violin", "motion_desc": "pan", "audio_desc": ""},
                {"idx": 1, "visual_desc": "Mira plays", "motion_desc": "close in", "audio_desc": ""}]})
        else:
            raise AssertionError("Unexpected text request")
        return httpx.Response(200, json=gemini_response(text))

    class MockText(GeminiTextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(google))

    monkeypatch.setattr(api, "GeminiTextLLM", MockText)
    install_a2e(monkeypatch, fake_a2e_handler(counts))
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "An otter discovers a violin.",
            "planning_provider": "gemini", "planning_model": "gemini-3.5-flash-lite"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed" and "HTTP 503" in result["error"] and "temporarily unavailable" in result["error"]
        assert counts == {"quote": 0, "image_start": 0, "video_start": 0}
        assert len(text_calls) == 4
        detail = client.get(f"/api/projects/{job_id}").json()
        assert {"Story outline", "Character descriptions", "Scene scripts"}.issubset(detail["completed_steps"])
        assert detail["uncertain_steps"] == [{"label": "Scene 1 storyboard", "kind": "text"}]
        project_text = (tmp_path / "projects" / job_id / "project.json").read_text()
        assert "fake-gemini-secret" not in project_text and "sk_fake-a2e-secret" not in project_text

        api.jobs.clear()  # simulate a backend process restart
        assert client.get(f"/api/result/{job_id}").json()["status"] == "failed"
        assert any(item["job_id"] == job_id for item in client.get("/api/projects").json()["projects"])
        assert client.post(f"/api/projects/{job_id}/resume", json={"acknowledge_costs": True}).status_code == 403
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers, json={}).status_code == 422
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers,
                           json={"acknowledge_costs": True}).status_code == 409
        assert len(text_calls) == 4 and counts["image_start"] == 0
        fail_storyboard["value"] = False
        response = client.post(f"/api/projects/{job_id}/resume", headers=headers,
                               json={"acknowledge_costs": True, "retry_uncertain": True})
        assert response.status_code == 200 and response.json()["job_id"] == job_id
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert client.get(result["video_url"]).content == b"combined-video"
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["attempt"] == 2 and not detail["uncertain_steps"]
        assert len([asset for asset in detail["assets"] if asset["type"] == "image"]) == 3
        assert client.get(next(a["url"] for a in detail["assets"] if a["type"] == "image")).content == image
        assert len(text_calls) == 5  # successful story, characters and script were not regenerated
        assert counts["image_start"] == 3 and counts["video_start"] == 2
        assert client.get(f"/api/projects/{job_id}/assets/../project.json").status_code == 404


def test_known_paid_task_is_polled_after_restart_without_duplicate_creation(monkeypatch, isolated):
    tmp_path, _image = isolated
    counts = {"quote": 0, "image_start": 0, "video_start": 0}
    running = {"value": True}
    install_a2e(monkeypatch, fake_a2e_handler(counts, processing=lambda task: task == "vid-2" and running["value"]))
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "A small bird returns home.", "planning_provider": "simple"})
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed" and "vid-2" in result["error"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert not detail["uncertain_steps"]  # task ID lets us poll instead of resubmit
        assert len([a for a in detail["assets"] if a["type"] == "video"]) == 1
        assert counts["image_start"] == 2 and counts["video_start"] == 2 and counts["quote"] == 4
        assert Decimal(detail["estimated_a2e_credits"]) == 4
        api.jobs.clear()
        running["value"] = False
        # Even if all input backups disappear, the saved clip and existing
        # video task can still be combined/polled without those inputs.
        for path in (tmp_path / "projects" / job_id / "assets").iterdir():
            path.unlink()

        async def must_not_probe(url):
            raise AssertionError("A completed clip or existing task should not re-upload its frame")
        async def must_not_download_image(url):
            raise AssertionError("A completed clip or existing task does not need an image backup")
        monkeypatch.setattr("pipelines.script2video.image_url_available", must_not_probe)
        monkeypatch.setattr("pipelines.script2video.download_image", must_not_download_image)
        response = client.post(f"/api/projects/{job_id}/resume", headers=headers, json={"acknowledge_costs": True})
        assert response.status_code == 200
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert client.get(result["video_url"]).content == b"combined-video"
        assert counts["image_start"] == 2 and counts["video_start"] == 2 and counts["quote"] == 4
        assert counts["polls"].count("vid-2") >= 2
        assert "no new task was submitted" in client.get(f"/api/status/{job_id}").text
        assert Decimal(client.get(f"/api/projects/{job_id}").json()["estimated_a2e_credits"]) == 4


def test_lost_submission_requires_explicit_confirmation_and_prior_budget_is_carried(monkeypatch, isolated):
    counts = {"quote": 0, "image_start": 0, "video_start": 0}
    monkeypatch.setenv("A2E_MAX_JOB_CREDITS", "1")
    install_a2e(monkeypatch, fake_a2e_handler(counts, timeout_first_image=True))
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A bird.", "planning_provider": "simple"}).json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "failed" and detail["uncertain_steps"]
        assert counts["image_start"] == 1 and counts["quote"] == 1
        api.jobs.clear()
        response = client.post(f"/api/projects/{job_id}/resume", headers=headers, json={"acknowledge_costs": True})
        assert response.status_code == 409 and counts["image_start"] == 1 and counts["quote"] == 1
        # Explicit consent allows an attempt, but the original estimate still
        # counts against the job guard before a second paid POST is submitted.
        response = client.post(f"/api/projects/{job_id}/resume", headers=headers,
                               json={"acknowledge_costs": True, "retry_uncertain": True})
        assert response.status_code == 200
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed" and "budget exceeded" in result["error"]
        assert counts["image_start"] == 1 and counts["quote"] == 2
        assert Decimal(client.get(f"/api/projects/{job_id}").json()["estimated_a2e_credits"]) == 1


def test_interrupted_project_is_listed_after_restart_without_automatic_calls(monkeypatch, isolated):
    tmp_path, _ = isolated
    from tools.project_store import ProjectStore
    request = api.GenerateRequest(idea="A cloud.", planning_provider="simple", image_model="a2e", video_model="a2e", clip_duration=5)
    config = api._project_config(request, None)
    job_id = "092bf4f9-285c-485c-8bce-943048c16405"
    project = ProjectStore.create(Path("projects"), job_id, request.model_dump(), config)
    assert project.data["status"] == "running"
    with TestClient(api.app) as client:
        projects = client.get("/api/projects").json()["projects"]
        assert len(projects) == 1 and projects[0]["status"] == "interrupted"
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "interrupted" and result["resumable"] is True
        assert "server stopped" in result["error"].lower()
        assert (tmp_path / "projects" / job_id / "project.json").is_file()


@pytest.mark.asyncio
async def test_same_checkpoint_is_serialized_and_consent_cannot_retry_new_ambiguity(isolated):
    from tools.project_store import ProjectStore, UncertainStepError
    project = ProjectStore.create(Path("projects"), "4ed2caf3-0d9b-4e08-9dae-87c500018fa1", {"idea": "test"}, {})
    calls = []

    async def succeed(task):
        calls.append("success")
        await asyncio.sleep(.01)
        return "saved"

    values = await asyncio.gather(*(project.call("one", "a2e_image", "one", succeed) for _ in range(3)))
    assert values == ["saved"] * 3 and calls == ["success"]

    async def lost(task):
        calls.append("lost")
        await asyncio.sleep(.01)
        raise RuntimeError("response lost")

    with pytest.raises(RuntimeError):
        await project.call("two", "a2e_image", "two", lost)
    await project.append_event({"type": "error", "message": "failed"}, final_status="failed")
    await project.begin_attempt(retry_uncertain=True)
    before = len(calls)
    # Two simultaneous consumers of an identical key cannot cause two paid
    # submissions, even after the first newly fails during a consented retry.
    results = await asyncio.gather(*(project.call("two", "a2e_image", "two", lost) for _ in range(2)),
                                   return_exceptions=True)
    assert len(calls) == before + 1
    assert any(isinstance(result, UncertainStepError) for result in results)


@pytest.mark.asyncio
async def test_storage_failure_blocks_new_paid_work_and_preserves_atomic_snapshot(monkeypatch, isolated):
    from tools.project_store import ProjectError, ProjectStore
    project = ProjectStore.create(Path("projects"), "6f8f924c-916d-4b9c-9ae8-76e6a6382fe1", {"idea": "test"}, {})
    original = project._write
    before = (project.path / "project.json").read_bytes()
    calls = []
    fail = {"once": True}

    def write_once():
        if fail["once"]:
            fail["once"] = False
            raise ProjectError("simulated disk failure")
        original()

    monkeypatch.setattr(project, "_write", write_once)
    async def paid(task):
        calls.append(1)
        return "created"

    with pytest.raises(ProjectError):
        await project.call("image", "a2e_image", "image", paid)
    assert not calls and (project.path / "project.json").read_bytes() == before
    with pytest.raises(ProjectError):
        await project.call("image", "a2e_image", "image", paid)
    assert not calls  # halted even when the simulated disk is healthy again
    await project.append_event({"type": "error", "message": "disk"}, final_status="failed")
    await project.begin_attempt(retry_uncertain=False)
    assert await project.call("image", "a2e_image", "image", paid) == "created"
    assert calls == [1]


def test_gemini_images_survive_restart_without_regenerating_previous_frames(monkeypatch, isolated):
    import base64
    from tools.gemini_image import GeminiImageClient

    tmp_path, image = isolated
    images, uploads, counts = [], [], {"video_start": 0, "quote": 0, "sign": 0}
    fail = {"second": True}

    def google(request):
        payload = json.loads(request.content)
        images.append(payload)
        if len(images) == 2 and fail["second"]:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output", "content": [{
            "type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()
        }]}]})

    def a2e(request):
        path = request.url.path
        if path == "/v1/feature-pricing":
            return httpx.Response(200, json={"code": 0, "data": {"prices": {"r2": {"credits": 1}}}})
        if path == "/v1/r2/upload-presigned-url":
            counts["sign"] += 1
            return httpx.Response(200, json={"code": 0, "data": {
                "uploadUrl": f"https://upload.test/file{counts['sign']}?signature=private-fake",
                "cdnUrl": f"https://cdn.test/frame{counts['sign']}.jpg"}})
        if path == "/v1/generation/quote":
            counts["quote"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if path == "/v1/userImage2Video/start":
            counts["video_start"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"_id": f"v{counts['video_start']}"}})
        if path.startswith("/v1/userImage2Video/v"):
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.test/video.mp4"}})
        raise AssertionError(path)

    def upload(request):
        assert "authorization" not in request.headers
        uploads.append(request.content)
        return httpx.Response(200)

    class MockA2E(A2EClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(a2e),
                             upload_transport=httpx.MockTransport(upload), poll_interval=.001, poll_timeout=1)

    class MockImage(GeminiImageClient):
        def __init__(self, key, model, client):
            super().__init__(key, model, client, transport=httpx.MockTransport(google))

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr(api, "GeminiImageClient", MockImage)
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        req = {"idea": "A fish finds a lantern.", "planning_provider": "simple", "image_provider": "gemini",
               "image_model": "gemini-3-pro-image"}
        job_id = client.post("/api/generate", json=req).json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "failed" and len(detail["uncertain_steps"]) == 1
        assert len(images) == 2 and not uploads and counts["video_start"] == 0
        assert len([asset for asset in detail["assets"] if asset["type"] == "image"]) == 1
        api.jobs.clear()
        fail["second"] = False
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers,
                           json={"acknowledge_costs": True}).status_code == 409
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers,
                           json={"acknowledge_costs": True, "retry_uncertain": True}).status_code == 200
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert len(images) == 3  # first saved frame never sent to Gemini again
        assert len(images[2]["input"]) == 2
        assert base64.b64decode(images[2]["input"][1]["data"]) == image  # restored reference bytes
        assert counts == {"video_start": 2, "quote": 2, "sign": 2}
        assert uploads == [image, image]
        assert Decimal(client.get(f"/api/projects/{job_id}").json()["estimated_a2e_credits"]) == 4
        saved = (tmp_path / "projects" / job_id / "project.json").read_text()
        assert "private-fake" not in saved and "sk_fake-a2e-secret" not in saved and "fake-gemini-secret" not in saved


def test_expired_frame_reupload_only_after_definitive_expiry_and_consent(monkeypatch, isolated):
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
    counts = {"images": 0, "videos": 0, "sign": 0, "uploads": 0}
    video_quote_fails = {"value": True}
    probe = {"transient": True}

    def a2e(request):
        path = request.url.path
        if path == "/v1/generation/quote":
            body = json.loads(request.content)
            if "userImage2Video" in body["endpoint"] and video_quote_fails["value"]:
                return httpx.Response(200, json={"code": 0, "data": {"available": False, "unavailableReason": "test"}})
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if path == "/v1/userText2Image/start":
            counts["images"] += 1
            return httpx.Response(200, json={"code": 0, "data": [{"_id": "image1"}]})
        if path == "/v1/userText2Image/image1":
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "image_url": "https://cdn.test/expired.jpg"}})
        if path == "/v1/feature-pricing":
            return httpx.Response(200, json={"code": 0, "data": {"prices": {"r2": {"credits": 1}}}})
        if path == "/v1/r2/upload-presigned-url":
            counts["sign"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"uploadUrl": "https://upload.test/file?fake=secret",
                                                                  "cdnUrl": "https://cdn.test/fresh.jpg"}})
        if path == "/v1/userImage2Video/start":
            counts["videos"] += 1
            body = json.loads(request.content)
            assert body["image_url"] == "https://cdn.test/fresh.jpg"
            return httpx.Response(200, json={"code": 0, "data": {"_id": "video1"}})
        if path == "/v1/userImage2Video/video1":
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.test/video.mp4"}})
        raise AssertionError(path)

    def upload(request):
        counts["uploads"] += 1
        assert "authorization" not in request.headers
        return httpx.Response(200)

    async def check(url):
        if probe["transient"]:
            raise RuntimeError("Temporary network failure; no re-upload was made.")
        return False

    class MockA2E(A2EClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(a2e), upload_transport=httpx.MockTransport(upload),
                             poll_interval=.001, poll_timeout=1)

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr("pipelines.script2video.image_url_available", check)
    headers = {"X-Requested-With": "MicroDrama"}
    consent = {"acknowledge_costs": True, "retry_uncertain": True}
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A comet.", "planning_provider": "simple"}).json()["job_id"]
        assert client.get(f"/api/result/{job_id}").json()["status"] == "failed"
        assert counts == {"images": 1, "videos": 0, "sign": 0, "uploads": 0}
        video_quote_fails["value"] = False
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers, json=consent).status_code == 200
        assert client.get(f"/api/result/{job_id}").json()["status"] == "failed"
        assert counts == {"images": 1, "videos": 0, "sign": 0, "uploads": 0}  # 503/network is not proof of expiration
        probe["transient"] = False
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers, json=consent).status_code == 200
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert counts == {"images": 1, "videos": 1, "sign": 1, "uploads": 1}
        assert Decimal(client.get(f"/api/projects/{job_id}").json()["estimated_a2e_credits"]) == 3


def test_expired_video_is_never_replaced_without_duplicate_charge_confirmation(monkeypatch, isolated):
    tmp_path, _ = isolated
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
    counts = {"image_start": 0, "video_start": 0, "quote": 0}
    polls = {}
    base = fake_a2e_handler(counts)

    def a2e(request):
        path = request.url.path
        if path.startswith("/v1/userImage2Video/vid-"):
            task = path.rsplit("/", 1)[-1]
            polls[task] = polls.get(task, 0) + 1
            if task == "vid-1" and polls[task] > 1:
                return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "isExpired": True}})
        return base(request)

    async def download(url, destination):
        if url.endswith("vid-1.mp4"):
            raise RuntimeError("provider output expired")
        Path(destination).write_bytes(b"saved-video")
        return destination

    install_a2e(monkeypatch, a2e)
    monkeypatch.setattr("pipelines.script2video.download_video", download)
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A star.", "planning_provider": "simple"}).json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "failed" and counts["video_start"] == 1
        assert detail["uncertain_steps"] == [{"label": "Scene 1, shot 1 video task", "kind": "a2e_video", "task_id": "vid-1"}]
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers,
                           json={"acknowledge_costs": True}).status_code == 409
        assert counts["video_start"] == 1
        assert client.post(f"/api/projects/{job_id}/resume", headers=headers,
                           json={"acknowledge_costs": True, "retry_uncertain": True}).status_code == 200
        assert client.get(f"/api/result/{job_id}").json()["status"] == "completed"
        assert counts["image_start"] == 1 and counts["video_start"] == 2
        data = json.loads((tmp_path / "projects" / job_id / "project.json").read_text())
        assert data["steps"]["scene/0/clip/0/remote"]["previous_task_ids"] == ["vid-1"]
        tasks = client.get(f"/api/projects/{job_id}").json()["a2e_tasks"]
        assert any(task["task_id"] == "vid-1" and task["state"] == "previous" for task in tasks)
        assert any(task["task_id"] == "vid-2" and task["state"] == "complete" for task in tasks)
        assert Decimal(data["estimated_total"]) == 3


def test_image_backup_can_repoll_same_task_for_fresh_url(monkeypatch, isolated):
    tmp_path, image = isolated
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
    counts = {"image_start": 0, "video_start": 0, "quote": 0}
    polls = {"image": 0}
    base = fake_a2e_handler(counts)

    def a2e(request):
        if request.url.path.startswith("/v1/userText2Image/img-"):
            polls["image"] += 1
            url = "https://cdn.test/old.jpg" if polls["image"] == 1 else "https://cdn.test/fresh.jpg"
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "image_url": url}})
        return base(request)

    async def image_download(url):
        if url.endswith("old.jpg"):
            raise RuntimeError("old link expired")
        assert url.endswith("fresh.jpg")
        return image, "image/jpeg"

    install_a2e(monkeypatch, a2e)
    monkeypatch.setattr("pipelines.script2video.download_image", image_download)
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A cloud.", "planning_provider": "simple"}).json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert counts["image_start"] == 1 and polls["image"] == 2
        data = json.loads((tmp_path / "projects" / job_id / "project.json").read_text())
        assert data["steps"]["scene/0/frame/0"]["value"] == "https://cdn.test/fresh.jpg"


@pytest.mark.asyncio
async def test_storage_error_waits_for_inflight_tasks_before_resume(monkeypatch, isolated):
    from tools.project_store import ProjectError, ProjectStore
    counts = {"image_start": 0, "video_start": 0, "quote": 0}
    base = fake_a2e_handler(counts)
    both_posts = asyncio.Event()
    polling_other = asyncio.Event()
    release = asyncio.Event()
    failure = {"done": False}
    original_write = ProjectStore._write

    async def handler(request):
        path = request.url.path
        if path == "/v1/userImage2Video/start":
            counts["video_start"] += 1
            number = counts["video_start"]
            if number == 1:
                await both_posts.wait()
            else:
                both_posts.set()
            return httpx.Response(200, json={"code": 0, "data": {"_id": f"vid-{number}"}})
        if path == "/v1/userImage2Video/vid-1" and not release.is_set():
            polling_other.set()
            await release.wait()
        return base(request)

    def flaky_write(self):
        events = self.data.get("events", [])
        if (not failure["done"] and events and events[-1].get("type") == "task" and
                events[-1].get("task_id") == "vid-2"):
            failure["done"] = True
            raise ProjectError("simulated transient disk failure")
        return original_write(self)

    monkeypatch.setattr(ProjectStore, "_write", flaky_write)
    install_a2e(monkeypatch, handler)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request = asyncio.create_task(client.post("/api/generate", json={"idea": "A tree.", "planning_provider": "simple"}))
        await asyncio.wait_for(polling_other.wait(), timeout=3)
        assert failure["done"] and counts["video_start"] == 2
        job_id = next(iter(api.jobs))
        assert (await client.get(f"/api/result/{job_id}")).json()["status"] == "running"
        response = await client.post(f"/api/projects/{job_id}/resume", headers={"X-Requested-With": "MicroDrama"},
                                     json={"acknowledge_costs": True, "retry_uncertain": True})
        assert response.status_code == 409 and counts["video_start"] == 2
        release.set()
        response = await asyncio.wait_for(request, timeout=3)
        assert response.status_code == 200
        detail = (await client.get(f"/api/projects/{job_id}")).json()
        assert detail["status"] == "failed" and not detail["uncertain_steps"]
        data = api.jobs[job_id]["project"].data
        # The failed task-ID write was retried locally, and the other task's
        # result was checkpointed while it was already in flight.
        assert data["steps"]["scene/0/clip/0/remote"]["task_id"] == "vid-1"
        assert data["steps"]["scene/0/clip/1/remote"]["task_id"] == "vid-2"
        response = await client.post(f"/api/projects/{job_id}/resume", headers={"X-Requested-With": "MicroDrama"},
                                     json={"acknowledge_costs": True})
        assert response.status_code == 200
        assert (await client.get(f"/api/result/{job_id}")).json()["status"] == "completed"
        assert counts["video_start"] == 2


@pytest.mark.asyncio
async def test_local_result_checkpoint_is_retried_without_repeating_paid_operation(monkeypatch, isolated):
    from tools.project_store import ProjectError, ProjectStore
    project = ProjectStore.create(Path("projects"), "151e047b-4332-4efd-873f-8a0e2a1ef7b0", {"idea": "test"}, {})
    original = project._write
    calls, failed = [], {"once": False}

    def write_once():
        entry = project.data["steps"].get("image", {})
        if entry.get("state") == "complete" and not failed["once"]:
            failed["once"] = True
            raise ProjectError("disk temporarily unavailable")
        original()

    async def paid(task):
        calls.append(1)
        return "the-generated-result"

    monkeypatch.setattr(project, "_write", write_once)
    with pytest.raises(ProjectError):
        await project.call("image", "gemini_image", "image", paid)
    assert calls == [1]
    # The retry was a local metadata write, not a second provider call.
    saved = json.loads((project.path / "project.json").read_text())
    assert saved["steps"]["image"]["state"] == "complete"
    assert saved["steps"]["image"]["value"] == "the-generated-result"
    with pytest.raises(ProjectError):
        await project.call("other", "gemini_image", "other", paid)
    assert calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
@pytest.mark.parametrize("strategy", ["restore", "replace"])
async def test_missing_or_corrupt_gemini_backup_never_silently_regenerates(isolated, damage, strategy):
    from pipelines.script2video import Script2VideoPipeline
    from tools.gemini_image import GeminiImageClient
    from tools.project_store import ProjectError, ProjectStore, UncertainStepError
    _, image = isolated
    project = ProjectStore.create(Path("projects"), "9fc162cc-2c7a-40b0-a37c-dc409d0dd503", {"idea": "test"}, {})
    info = project.save_asset("scene/0/frame/0", image, "image/jpeg")
    info["handle"] = "gemini://0123456789abcdef"
    async def initially(_):
        return info
    await project.call("scene/0/frame/0", "gemini_image", "frame", initially)
    generator = GeminiImageClient("fake-key", "gemini-3-pro-image", object(), transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("must never call Google"))))
    pipeline = Script2VideoPipeline(object(), project=project, image_generator=generator, image_provider="gemini")
    calls = []
    async def replacement(task):
        calls.append(task)
        handle = "gemini://fedcba9876543210"
        generator._images[handle] = (image, "image/jpeg")
        return handle
    async def progress(*args):
        pass
    try:
        path = project.path / info["asset"]
        if damage == "corrupt":
            path.write_bytes(b"corrupt")
        else:
            path.unlink()
        with pytest.raises(ProjectError, match="missing or invalid"):
            await pipeline._image_step("scene/0/frame/0", "frame", replacement, progress)
        assert not calls and project.summary()["uncertain_steps"]
        with pytest.raises(UncertainStepError):
            await pipeline._image_step("scene/0/frame/0", "frame", replacement, progress)
        assert not calls
        await project.append_event({"type": "error", "message": "missing"}, final_status="failed")
        await project.begin_attempt(retry_uncertain=strategy == "replace")
        if strategy == "restore":
            path.write_bytes(image)  # repaired from an external backup
            handle = await pipeline._image_step("scene/0/frame/0", "frame", replacement, progress)
            assert handle == info["handle"] and not calls
        else:
            handle = await pipeline._image_step("scene/0/frame/0", "frame", replacement, progress)
            assert handle == "gemini://fedcba9876543210" and calls == [None]
        assert project.step("scene/0/frame/0")["state"] == "complete"
    finally:
        await generator._client.aclose()


def test_project_files_are_private_and_restricted_routes_reject_symlinks(monkeypatch, isolated):
    import os
    tmp_path, _ = isolated
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
    counts = {"quote": 0, "image_start": 0, "video_start": 0}
    install_a2e(monkeypatch, fake_a2e_handler(counts))
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A fox.", "planning_provider": "simple"}).json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "completed"
        folder = tmp_path / "projects" / job_id
        if os.name == "posix":
            assert (folder.stat().st_mode & 0o077) == 0
            assert ((folder / "project.json").stat().st_mode & 0o077) == 0
        assert client.get(f"/outputs/{job_id}/project.json").status_code == 404
        assert client.get(f"/projects/{job_id}/project.json").status_code == 404
        assert client.get(f"/api/projects/{job_id}/assets/project.json").status_code == 404
        assert client.get("/api/projects/not-a-uuid").status_code == 404
        asset_url = next(item["url"] for item in detail["assets"] if item["type"] == "image")
        image_file = folder / "assets" / asset_url.rsplit("/", 1)[-1]
        secret = tmp_path / "should-not-be-served"
        secret.write_bytes(b"fake-private-content")
        image_file.unlink()
        image_file.symlink_to(secret)
        assert client.get(asset_url).status_code == 404
        final = folder / "final_video.mp4"
        final.unlink()
        final.symlink_to(secret)
        assert client.get(f"/outputs/{job_id}/final_video.mp4").status_code == 404


def test_changed_resume_settings_block_paid_work_until_restored(monkeypatch, isolated):
    counts = {"quote": 0, "image_start": 0, "video_start": 0}
    install_a2e(monkeypatch, fake_a2e_handler(counts, timeout_first_image=True))
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A mouse.", "planning_provider": "simple"}).json()["job_id"]
        assert client.get(f"/api/result/{job_id}").json()["status"] == "failed"
        before = dict(counts)
        monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
        response = client.post(f"/api/projects/{job_id}/resume", headers=headers,
                               json={"acknowledge_costs": True, "retry_uncertain": True})
        assert response.status_code == 409 and "original settings" in response.json()["detail"]
        assert counts == before


def test_storage_defaults_do_not_depend_on_current_working_directory(tmp_path):
    import os
    import subprocess
    import sys
    server = Path(api.__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(server)
    env["MICRODRAMA_ENV_FILE"] = str(tmp_path / "missing-fake-env")
    result = subprocess.run([sys.executable, "-c", "import api; print(api.PROJECTS_DIR); print(api.LEGACY_OUTPUTS_DIR)"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(server / "projects"), str(server / "outputs")]


def test_corrupt_project_is_listed_as_unavailable_without_hiding_other_projects(isolated):
    from tools.project_store import ProjectStore
    broken_id = "b31dc2db-3a9d-4e74-8e91-c33c2fa0296f"
    good_id = "2db07c4c-c0d6-41d3-9925-b88ac2e83331"
    broken = ProjectStore.create(api.PROJECTS_DIR, broken_id, {"idea": "broken"}, {})
    good = ProjectStore.create(api.PROJECTS_DIR, good_id, {"idea": "still here"}, {})
    data = broken.data
    data["steps"]["bad"] = {"kind": ["not", "valid"], "state": "failed"}
    (broken.path / "project.json").write_text(json.dumps(data))
    with TestClient(api.app) as client:
        results = {item["job_id"]: item for item in client.get("/api/projects").json()["projects"]}
        assert results[broken_id]["status"] == "unavailable"
        assert results[good_id]["title"] == "still here" and results[good_id]["status"] == "interrupted"
        assert client.get(f"/api/projects/{broken_id}").status_code == 404


def test_project_history_does_not_hide_older_projects(isolated):
    import uuid
    from tools.project_store import ProjectStore
    ids = []
    for number in range(105):
        job_id = str(uuid.uuid4())
        ids.append(job_id)
        ProjectStore.create(api.PROJECTS_DIR, job_id, {"idea": f"Project {number}"}, {})
    with TestClient(api.app) as client:
        listed = client.get("/api/projects").json()["projects"]
        assert len(listed) == 105
        assert {entry["job_id"] for entry in listed} == set(ids)
        assert all(entry["status"] == "interrupted" for entry in listed)


def test_concurrent_portrait_partial_success_is_saved_and_reused(monkeypatch, isolated):
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "1")
    monkeypatch.setenv("MAX_CHARACTERS", "2")
    text_calls = []
    counts = {"image_start": 0, "video_start": 0, "quote": 0}
    image_bodies = []

    def google(request):
        prompt = json.loads(request.content)["input"]
        text_calls.append(prompt)
        if prompt.startswith("Extract at most"):
            value = {"characters": [
                {"idx": 0, "name": "Mira", "static_features": "brown fur", "dynamic_features": "red scarf", "is_visible": True},
                {"idx": 1, "name": "Sol", "static_features": "black hair", "dynamic_features": "blue coat", "is_visible": True}]}
        elif prompt.startswith("Break this scene"):
            value = {"shots": [{"idx": 0, "visual_desc": "Mira meets Sol at a bridge", "motion_desc": "pan", "audio_desc": ""}]}
        else:
            raise AssertionError("Unexpected planning request")
        return httpx.Response(200, json=gemini_response(json.dumps(value)))

    base = fake_a2e_handler(counts)
    def a2e(request):
        if request.url.path == "/v1/userText2Image/start":
            assert len(text_calls) == 2  # all planning is done before any portraits
            image_bodies.append(json.loads(request.content))
            if counts["image_start"] == 1:  # second portrait response is lost
                counts["image_start"] += 1
                raise httpx.ReadTimeout("lost response", request=request)
        return base(request)

    class MockText(GeminiTextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(google))

    monkeypatch.setattr(api, "GeminiTextLLM", MockText)
    install_a2e(monkeypatch, a2e)
    with TestClient(api.app) as client:
        req = {"mode": "script2video", "script": "Mira and Sol meet at the bridge.", "planning_provider": "gemini",
               "planning_model": "gemini-3.5-flash-lite"}
        job_id = client.post("/api/generate", json=req).json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "failed" and len(detail["uncertain_steps"]) == 1
        assert counts["image_start"] == 2 and counts["video_start"] == 0
        assert len([asset for asset in detail["assets"] if asset["type"] == "image"]) == 1
        assert client.post(f"/api/projects/{job_id}/resume", headers={"X-Requested-With": "MicroDrama"},
                           json={"acknowledge_costs": True, "retry_uncertain": True}).status_code == 200
        assert client.get(f"/api/result/{job_id}").json()["status"] == "completed"
        assert len(text_calls) == 2  # no successful planning repeated
        assert counts["image_start"] == 4 and counts["video_start"] == 1
        # New frame references both the original saved portrait and its replacement.
        assert image_bodies[-1]["input_images"] == ["https://cdn.test/img-1.jpg", "https://cdn.test/img-3.jpg"]
