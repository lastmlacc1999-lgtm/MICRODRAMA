import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import api
from tools.a2e_client import A2EClient as RealA2EClient


def test_health_missing_key_and_request_validation(monkeypatch):
    api.jobs.clear()
    monkeypatch.delenv("A2E_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["planning"] == "a2e-only"
        assert client.get("/api/health").json()["a2e_configured"] is False
        assert client.post("/api/generate", json={"idea": ""}).status_code == 422
        assert client.post("/api/generate", json={"idea": "hello"}).status_code == 503
        assert client.post("/api/generate", json={"mode": "script2video", "idea": "title", "script": ""}).status_code == 422


def test_mocked_end_to_end_job_and_replayable_sse(monkeypatch, tmp_path):
    api.jobs.clear()
    monkeypatch.setenv("A2E_API_KEY", "sk_test")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    monkeypatch.chdir(tmp_path)
    # Pipelines use relative output paths, as they do when the server is started.
    Path("outputs").mkdir()
    counts = {"image": 0, "video": 0}
    quote_bodies = []

    def handler(req):
        path = req.url.path
        if path == "/v1/generation/quote":
            quote_bodies.append(json.loads(req.content))
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if path == "/v1/userText2Image/start":
            counts["image"] += 1
            return httpx.Response(200, json={"code": 0, "data": [{"_id": f"img{counts['image']}"}]})
        if path.startswith("/v1/userText2Image/"):
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "image_urls": ["https://cdn.test/" + path.rsplit("/", 1)[-1] + ".png"]}})
        if path == "/v1/userImage2Video/start":
            counts["video"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"_id": f"vid{counts['video']}"}})
        if path.startswith("/v1/userImage2Video/"):
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.test/" + path.rsplit("/", 1)[-1] + ".mp4"}})
        raise AssertionError(path)

    class MockA2E(RealA2EClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handler), poll_interval=.001, poll_timeout=2)

    async def fake_download(url, dest):
        Path(dest).write_bytes(b"clip")
        return dest

    async def fake_concat(paths, output):
        assert len(paths) == 2
        Path(output).write_bytes(b"fake-mp4")
        return output

    async def fake_image_download(url):
        return b"fake-image", "image/png"

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr("pipelines.script2video.download_image", fake_image_download)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "A small robot finds a flower."})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed"
        assert result["video_url"] == f"/outputs/{job_id}/final_video.mp4"
        assert client.get(result["video_url"]).content == b"fake-mp4"
        manifest_text = Path(f"projects/{job_id}/a2e_events.jsonl").read_text()
        manifest = [json.loads(line) for line in manifest_text.splitlines()]
        assert len(manifest) == 8 and "sk_test" not in manifest_text
        assert len([entry for entry in manifest if entry["type"] == "task"]) == 4
        stream1 = client.get(f"/api/status/{job_id}").text
        stream2 = client.get(f"/api/status/{job_id}").text
        assert stream1 == stream2
        assert stream1.count('"type": "quote"') == 4
        assert stream1.count('"type": "task"') == 4
        assert stream1.count('"type": "complete"') == 1
        last_id = int([line for line in stream1.splitlines() if line.startswith("id:")][-1].split(":")[1])
        replay = client.get(f"/api/status/{job_id}", headers={"Last-Event-ID": str(last_id - 1)}).text
        assert replay.count("data:") == 1 and '"type": "complete"' in replay
    assert counts == {"image": 2, "video": 2}
    assert quote_bodies[1]["requestBody"]["input_images"] == ["https://cdn.test/img1.png"]


def test_provider_error_does_not_expose_token(monkeypatch):
    api.jobs.clear()
    monkeypatch.setenv("A2E_API_KEY", "sk_test-secret")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    class BrokenA2E:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            raise RuntimeError("upstream echoed sk_test-secret")

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(api, "A2EClient", BrokenA2E)
    with TestClient(api.app) as client:
        job_id = client.post("/api/generate", json={"idea": "A bird flies home"}).json()["job_id"]
        error = client.get(f"/api/result/{job_id}").json()["error"]
        assert "[redacted]" in error and "sk_test-secret" not in error


def test_save_a2e_key_is_persistent_private_and_not_echoed(monkeypatch, tmp_path):
    import os
    import stat
    from dotenv import dotenv_values

    env_file = tmp_path / ".env"
    env_file.write_text("# keep settings\nOTHER_SETTING=keep-me\nA2E_API_KEY=old-value\n", encoding="utf-8")
    env_file.chmod(0o644)
    monkeypatch.setattr(api, "ENV_FILE", env_file)
    monkeypatch.delenv("A2E_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_SETTING", "from-shell")
    token = "sk_test-quote'backslash\\and#symbol"
    with TestClient(api.app) as client:
        response = client.post("/api/settings/a2e-key", json={"api_key": token},
                               headers={"X-Requested-With": "MicroDrama"})
        assert response.status_code == 200
        assert response.json() == {"configured": True, "verified": False}
        assert token not in response.text
        assert response.headers["cache-control"] == "no-store"
        health = client.get("/api/health")
        assert health.json()["a2e_configured"] is True
        assert token not in health.text
        assert os.environ["A2E_API_KEY"] == token
    values = dotenv_values(env_file, interpolate=False)
    assert values["A2E_API_KEY"] == token
    assert values["OTHER_SETTING"] == "keep-me"
    assert "# keep settings" in env_file.read_text()
    if os.name == "posix":
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    # An explicitly saved key remains active after restart even if another
    # A2E key was exported; unrelated variables keep environment precedence.
    monkeypatch.setenv("A2E_API_KEY", "sk_stale-exported")
    api.load_server_environment(env_file)
    assert os.environ["A2E_API_KEY"] == token
    assert os.environ["OTHER_SETTING"] == "from-shell"


def test_save_key_rejects_bad_input_without_echoing_it(monkeypatch, tmp_path):
    import os

    env_file = tmp_path / ".env"
    monkeypatch.setattr(api, "ENV_FILE", env_file)
    monkeypatch.setenv("A2E_API_KEY", "sk_original")
    with TestClient(api.app) as client:
        response = client.post("/api/settings/a2e-key", json={"api_key": "sk_no-header"})
        assert response.status_code == 403
        headers = {"X-Requested-With": "MicroDrama"}
        for bad in ("not-a-token", "sk_bad secret", "sk_", "sk_bad\nINJECTED=value", "sk_" + "x" * 3000):
            response = client.post("/api/settings/a2e-key", json={"api_key": bad}, headers=headers)
            assert response.status_code == 422
            if bad != "sk_":
                assert bad not in response.text
        response = client.post("/api/settings/a2e-key", content="{broken", headers={**headers, "Content-Type": "application/json"})
        assert response.status_code == 400 and "broken" not in response.text
        response = client.post("/api/settings/a2e-key", content="x" * 9000, headers={**headers, "Content-Type": "application/json"})
        assert response.status_code == 413
        assert os.environ["A2E_API_KEY"] == "sk_original"
        assert not env_file.exists()


def test_save_failure_never_changes_active_key_or_exposes_new_one(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("A2E_API_KEY", "sk_original")

    def fail(*args, **kwargs):
        raise OSError("cannot save sk_private-new")

    monkeypatch.setattr(api, "set_key", fail)
    with TestClient(api.app) as client:
        response = client.post("/api/settings/a2e-key", json={"api_key": "sk_private-new"},
                               headers={"X-Requested-With": "MicroDrama"})
        assert response.status_code == 500
        assert "sk_private-new" not in response.text
        assert os.environ["A2E_API_KEY"] == "sk_original"


def test_save_refuses_symlinked_env(monkeypatch, tmp_path):
    import os

    target = tmp_path / "target.env"
    target.write_text("A2E_API_KEY=unchanged\n")
    symlink = tmp_path / ".env"
    symlink.symlink_to(target)
    monkeypatch.setattr(api, "ENV_FILE", symlink)
    monkeypatch.setenv("A2E_API_KEY", "sk_original")
    with TestClient(api.app) as client:
        response = client.post("/api/settings/a2e-key", json={"api_key": "sk_new"},
                               headers={"X-Requested-With": "MicroDrama"})
        assert response.status_code == 500
    assert target.read_text() == "A2E_API_KEY=unchanged\n"
    assert os.environ["A2E_API_KEY"] == "sk_original"


def test_saved_key_is_used_immediately_by_new_jobs(monkeypatch, tmp_path):
    api.jobs.clear()
    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("A2E_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    captured = []

    class StubA2E:
        def __init__(self, *, api_key, on_event, **kwargs):
            captured.append(api_key)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class StubPipeline:
        def __init__(self, *args, project, **kwargs):
            self.project = project

        async def run(self, *, job_id, **kwargs):
            path = self.project.path / "final_video.mp4"
            path.write_bytes(b"test-video")
            return str(path)

    monkeypatch.setattr(api, "A2EClient", StubA2E)
    monkeypatch.setattr(api, "Idea2VideoPipeline", StubPipeline)
    with TestClient(api.app) as client:
        saved = client.post("/api/settings/a2e-key", json={"api_key": "sk_saved-for-job"},
                            headers={"X-Requested-With": "MicroDrama"})
        assert saved.status_code == 200
        response = client.post("/api/generate", json={"idea": "A fox in a forest"})
        assert response.status_code == 200
        result = client.get(f"/api/result/{response.json()['job_id']}").json()
        assert result["status"] == "completed"
        assert "sk_saved-for-job" not in response.text + json.dumps(result)
    assert captured == ["sk_saved-for-job"]


def test_gemini_key_save_is_private_persistent_and_does_not_overwrite_a2e(monkeypatch, tmp_path):
    import os
    import stat
    from dotenv import dotenv_values

    env_file = tmp_path / ".env"
    env_file.write_text("# preserve me\nA2E_API_KEY=sk_previous\nOTHER=unchanged\n", encoding="utf-8")
    env_file.chmod(0o644)
    monkeypatch.setattr(api, "ENV_FILE", env_file)
    monkeypatch.setenv("A2E_API_KEY", "sk_previous")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    key = "fake-gemini-key-with'quote\\and#symbol"
    headers = {"X-Requested-With": "MicroDrama"}
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["gemini_configured"] is False
        assert client.post("/api/settings/gemini-key", json={"api_key": key}).status_code == 403
        for bad in ("abc", "bad key", "line1\nline2", "x" * 3000):
            response = client.post("/api/settings/gemini-key", json={"api_key": bad}, headers=headers)
            assert response.status_code == 422 and bad not in response.text
        response = client.post("/api/settings/gemini-key", json={"api_key": key}, headers=headers)
        assert response.status_code == 200 and response.json() == {"configured": True, "verified": False}
        assert response.headers["cache-control"] == "no-store" and key not in response.text
        health = client.get("/api/health")
        assert health.json()["gemini_configured"] is True and key not in health.text
    saved = dotenv_values(env_file, interpolate=False)
    assert saved["GEMINI_API_KEY"] == key and saved["A2E_API_KEY"] == "sk_previous" and saved["OTHER"] == "unchanged"
    if os.name == "posix":
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    monkeypatch.setenv("GEMINI_API_KEY", "stale-export")
    api.load_server_environment(env_file)
    assert os.environ["GEMINI_API_KEY"] == key
    assert os.environ["A2E_API_KEY"] == "sk_previous"


def test_model_catalog_and_invalid_selections_fail_before_creating_jobs(monkeypatch):
    api.jobs.clear()
    monkeypatch.setenv("A2E_API_KEY", "sk_fake")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("A2E_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("A2E_VIDEO_MODEL", raising=False)
    with TestClient(api.app) as client:
        models = client.get("/api/models")
        catalog = models.json()
        assert models.status_code == 200 and models.headers["cache-control"] == "no-store"
        assert len(catalog["video_models"]) >= 30
        assert any(m["id"] == "gemini-nano-banana-2.1" and m["provider"] == "gemini" for m in catalog["image_models"])
        assert any(m["id"] == "wan2.6-r2v" and m["input_mode"] == "reference" for m in catalog["video_models"])
        for body in (
            {"image_provider": "other"},
            {"image_provider": "gemini", "image_model": "a2e"},
            {"video_model": "not-a-video-model"},
            {"video_model": "grok-video-1.5", "clip_duration": 5},
            {"video_model": "veo3", "clip_duration": 8},
            {"video_model": "a2e", "clip_duration": "5"},
        ):
            response = client.post("/api/generate", json={"idea": "a small bird", **body})
            assert response.status_code == 422, (body, response.text)
        response = client.post("/api/generate", json={"idea": "bird", "image_provider": "gemini"})
        assert response.status_code == 503 and "Gemini API key" in response.text
        assert not api.jobs


def test_mocked_gemini_to_selected_a2e_video_full_job(monkeypatch, tmp_path):
    import base64
    import io
    from PIL import Image
    from tools.gemini_image import GeminiImageClient as RealGeminiClient

    api.jobs.clear()
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("A2E_BASE_URL", raising=False)
    monkeypatch.delenv("A2E_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("A2E_VIDEO_MODEL", raising=False)
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    monkeypatch.chdir(tmp_path)
    Path("outputs").mkdir()
    image_buffer = io.BytesIO()
    Image.new("RGB", (30, 20), (25, 50, 100)).save(image_buffer, "JPEG")
    image = image_buffer.getvalue()
    gemini_requests, storage_requests, video_bodies, quote_bodies = [], [], [], []
    signed = "https://storage.test/upload?signature=private-signature"

    def google(req):
        assert req.headers["x-goog-api-key"] == "fake-google-key"
        assert "authorization" not in req.headers
        gemini_requests.append(json.loads(req.content))
        return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output", "content": [
            {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}
        ]}]})

    def a2e(req):
        assert req.headers["authorization"] == "Bearer sk_fake-a2e"
        body = json.loads(req.content) if req.content else None
        if req.url.path == "/v1/feature-pricing":
            return httpx.Response(200, json={"code": 0, "data": {"prices": {"r2": {"credits": 1}}}})
        if req.url.path == "/v1/r2/upload-presigned-url":
            assert body["fileSize"] == len(image)
            return httpx.Response(200, json={"code": 0, "data": {"uploadUrl": signed, "cdnUrl": "https://cdn.test/frame.jpg"}})
        if req.url.path == "/v1/generation/quote":
            quote_bodies.append(body)
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if req.url.path == "/v1/seedance2Video/start":
            video_bodies.append(body)
            return httpx.Response(200, json={"code": 0, "data": {"_id": f"vid-{len(video_bodies)}"}})
        if req.url.path.startswith("/v1/seedance2Video/vid-"):
            return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.test/movie.mp4"}})
        raise AssertionError(f"Unexpected A2E path {req.url.path}")

    def storage(req):
        assert "authorization" not in req.headers and "x-goog-api-key" not in req.headers
        assert req.content == image and str(req.url) == signed
        storage_requests.append(req)
        return httpx.Response(200)

    class MockA2E(RealA2EClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(a2e),
                             upload_transport=httpx.MockTransport(storage), poll_interval=.001, poll_timeout=2)

    class MockGemini(RealGeminiClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(google))

    async def fake_download(url, dest):
        Path(dest).write_bytes(b"clip")
        return dest

    async def fake_concat(paths, output):
        assert len(paths) == 2
        Path(output).write_bytes(b"fake-mp4")
        return output

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr(api, "GeminiImageClient", MockGemini)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "An otter finds a violin.",
            "image_provider": "gemini", "image_model": "gemini-3-pro-image",
            "video_model": "seedance-2.5-flash", "clip_duration": 4})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert client.get(result["video_url"]).content == b"fake-mp4"
        stream = client.get(f"/api/status/{job_id}").text
        manifest = Path(f"projects/{job_id}/a2e_events.jsonl").read_text()
        assert "sk_fake-a2e" not in stream + manifest
        assert "fake-google-key" not in stream + manifest and "private-signature" not in stream + manifest
    assert len(gemini_requests) == len(storage_requests) == len(video_bodies) == 2
    assert all(r["model"] == "gemini-3-pro-image" and r["store"] is False for r in gemini_requests)
    assert gemini_requests[1]["input"][1]["type"] == "image"  # previous frame used for continuity
    assert all(b["model_version"] == "2.5-flash" and b["duration"] == 4 and
               b["image_url"] == "https://cdn.test/frame.jpg" for b in video_bodies)
    assert all(q["requestBody"] in video_bodies for q in quote_bodies)
