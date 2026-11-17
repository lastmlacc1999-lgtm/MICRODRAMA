import base64
import io
import json
import os
import stat
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from PIL import Image

import api
from tools.gemini_image import GeminiImageClient
from tools.gemini_text import GeminiTextLLM
from tools.llm import OpenAICompatibleLLM
from tools.model_catalog import DEFAULT_OPENAI_TEXT_MODEL, OPENAI_TEXT_MODELS
from tools.openai_text import OpenAITextLLM


HEADERS = {"X-Requested-With": "MicroDrama"}


def gpt_response(text):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    ]}


def test_openai_key_save_is_private_persistent_separate_and_not_echoed(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# preserve me\nA2E_API_KEY=sk_previous\nGEMINI_API_KEY=old-google\nOTHER=unchanged\n")
    env_file.chmod(0o644)
    monkeypatch.setattr(api, "ENV_FILE", env_file)
    monkeypatch.setenv("A2E_API_KEY", "sk_previous")
    monkeypatch.setenv("GEMINI_API_KEY", "old-google")
    monkeypatch.setenv("LLM_API_KEY", "separate-compatible-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    key = "sk-proj-fake-openai'with\\and#symbols"
    with TestClient(api.app) as client:
        assert client.get("/api/health").json()["openai_configured"] is False
        assert client.post("/api/settings/openai-key", json={"api_key": key}).status_code == 403
        for bad in ("abc", "bad key", "line1\nline2", "x" * 3000, "non-ascii-💡", None):
            response = client.post("/api/settings/openai-key", json={"api_key": bad}, headers=HEADERS)
            assert response.status_code == 422
            if isinstance(bad, str):
                assert bad not in response.text
        response = client.post("/api/settings/openai-key", json={"api_key": key}, headers=HEADERS)
        assert response.status_code == 200 and response.json() == {"configured": True, "verified": False}
        assert response.headers["cache-control"] == "no-store" and key not in response.text
        health = client.get("/api/health")
        assert health.json()["openai_configured"] is True and key not in health.text
        assert health.json()["a2e_configured"] and health.json()["gemini_configured"]
    saved = dotenv_values(env_file, interpolate=False)
    assert saved["OPENAI_API_KEY"] == key and saved["A2E_API_KEY"] == "sk_previous"
    assert saved["GEMINI_API_KEY"] == "old-google" and saved["OTHER"] == "unchanged"
    assert os.environ["LLM_API_KEY"] == "separate-compatible-key"
    if os.name == "posix":
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    monkeypatch.setenv("OPENAI_API_KEY", "stale-export")
    api.load_server_environment(env_file)
    assert os.environ["OPENAI_API_KEY"] == key
    assert os.environ["LLM_API_KEY"] == "separate-compatible-key"


def test_openai_key_save_failure_keeps_old_key_and_hides_new_key(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("OPENAI_API_KEY", "old-openai-key")
    def fail(*args, **kwargs):
        raise OSError("write failed for new-private-openai")
    monkeypatch.setattr(api, "set_key", fail)
    with TestClient(api.app) as client:
        response = client.post("/api/settings/openai-key", json={"api_key": "new-private-openai"}, headers=HEADERS)
        assert response.status_code == 500 and "new-private-openai" not in response.text
        assert os.environ["OPENAI_API_KEY"] == "old-openai-key"


def test_gpt_provider_catalog_selection_independence_and_legacy_defaults(monkeypatch):
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    seen = []
    async def capture(job_id, req, token, gemini_key, llm):
        seen.append((req, token, gemini_key, llm))
    monkeypatch.setattr(api, "run_pipeline", capture)

    with TestClient(api.app) as client:
        catalog_response = client.get("/api/models")
        catalog = catalog_response.json()
        assert catalog_response.headers["cache-control"] == "no-store"
        assert catalog["defaults"]["openai_text"] == DEFAULT_OPENAI_TEXT_MODEL
        assert {item["id"] for item in catalog["openai_text_models"]} == {model.id for model in OPENAI_TEXT_MODELS}
        assert catalog["defaults"]["planning_provider"] == "simple"  # saving an OpenAI key does not select GPT
        assert client.post("/api/generate", json={"idea": "A bird"}).status_code == 200
        assert seen[-1][0].planning_provider == "simple" and seen[-1][3] is None

        response = client.post("/api/generate", json={"idea": "A bird", "planning_provider": "openai"})
        assert response.status_code == 200
        req, token, gemini_key, llm = seen[-1]
        assert req.planning_model == DEFAULT_OPENAI_TEXT_MODEL and req.image_provider == "a2e"
        assert isinstance(llm, OpenAITextLLM) and llm.api_key == "sk-fake-openai" and llm.model == DEFAULT_OPENAI_TEXT_MODEL
        assert token == "sk_fake-a2e" and not gemini_key
        saved = (api.PROJECTS_DIR / response.json()["job_id"] / "project.json").read_text()
        assert json.loads(saved)["request"]["planning_provider"] == "openai"
        assert json.loads(saved)["request"]["planning_model"] == DEFAULT_OPENAI_TEXT_MODEL
        assert "sk-fake-openai" not in saved and "sk_fake-a2e" not in saved

        before = len(seen)
        for body, expected in (
            ({"planning_provider": "openai", "planning_model": "not-a-model"}, 422),
            ({"planning_provider": "openai", "planning_model": "gemini-3.8-flash"}, 422),
            ({"planning_provider": "gemini", "planning_model": "gpt-6-luna"}, 422),
            ({"planning_provider": "simple", "planning_model": "gpt-6-luna"}, 422),
            ({"planning_provider": "configured", "planning_model": "gpt-6-luna"}, 422),
            ({"planning_provider": "openai", "image_provider": "gemini"}, 503),
        ):
            result = client.post("/api/generate", json={"idea": "A bird", **body})
            assert result.status_code == expected, (body, result.text)
        assert len(seen) == before

        monkeypatch.setenv("GEMINI_API_KEY", "fake-google")
        result = client.post("/api/generate", json={"mode": "script2video", "script": "A bird flies.",
            "planning_provider": "openai", "planning_model": "gpt-6.1-sol", "image_provider": "gemini"})
        assert result.status_code == 200
        req, _, google_key, llm = seen[-1]
        assert req.image_provider == "gemini" and google_key == "fake-google"
        assert isinstance(llm, OpenAITextLLM) and llm.model == "gpt-6.1-sol" and llm.api_key == "sk-fake-openai"

        monkeypatch.setenv("LLM_API_KEY", "fake-compat-key")
        monkeypatch.setenv("LLM_MODEL", "compat-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://provider.test/v1")
        assert client.post("/api/generate", json={"idea": "A bird"}).status_code == 200
        req, _, _, llm = seen[-1]
        assert req.planning_provider == "configured" and isinstance(llm, OpenAICompatibleLLM)
        assert llm.api_key == "fake-compat-key" and llm.model == "compat-model"
        assert client.get("/api/models").json()["defaults"]["planning_provider"] == "configured"

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        before = len(seen)
        response = client.post("/api/generate", json={"idea": "A bird", "planning_provider": "openai"})
        assert response.status_code == 503 and "OpenAI API key is missing" in response.text
        assert len(seen) == before  # configured and Gemini keys cannot be substituted
        result = client.post("/api/generate", json={"idea": "A bird", "planning_provider": "gemini"})
        assert result.status_code == 200 and isinstance(seen[-1][3], GeminiTextLLM)
        assert client.post("/api/generate", json={"idea": "A bird", "planning_provider": "simple"}).status_code == 200
        assert seen[-1][3] is None


def test_project_input_cannot_accidentally_save_an_unused_openai_key(monkeypatch):
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-private-openai-key")
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "please use unused-private-openai-key", "planning_provider": "simple"})
        assert response.status_code == 422 and "unused-private-openai-key" not in response.text
        assert not api.jobs and not api.PROJECTS_DIR.exists()


@pytest.mark.asyncio
async def test_gpt_job_snapshots_keys_and_redacts_old_and_rotated_credentials(monkeypatch):
    old = ["sk_a2e-old-private", "google-old-private", "openai-old-private", "compat-old-private"]
    new = ["sk_a2e-new-private", "google-new-private", "openai-new-private", "compat-new-private"]
    for name, value in zip(("A2E_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"), old):
        monkeypatch.setenv(name, value)
    observed = []

    class MockA2E:
        def __init__(self, *, api_key, on_event, **kwargs):
            self.api_key, self.on_event = api_key, on_event
        async def __aenter__(self):
            await self.on_event({"type": "quote", "message": " ".join(old + new),
                                 "nested": {"detail": " ".join(old + new)}})
            return self
        async def __aexit__(self, *args):
            pass

    class FailingPipeline:
        def __init__(self, a2e, llm, *args, **kwargs):
            observed.append((a2e.api_key, llm.api_key, llm.model))
        async def run(self, **kwargs):
            raise RuntimeError("provider echoed " + " ".join(old + new))

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr(api, "Idea2VideoPipeline", FailingPipeline)
    req = api.GenerateRequest(idea="private scene", planning_provider="openai", planning_model="gpt-6.1-terra")
    tasks = BackgroundTasks()
    accepted = await api.generate(req, tasks)
    for name, value in zip(("A2E_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"), new):
        monkeypatch.setenv(name, value)
    await tasks()
    job = api.jobs[accepted.job_id]
    assert observed == [(old[0], old[2], "gpt-6.1-terra")]
    assert job["status"] == "failed" and "[redacted]" in job["error"]
    manifest = (api.PROJECTS_DIR / accepted.job_id / "a2e_events.jsonl").read_text()
    saved = (api.PROJECTS_DIR / accepted.job_id / "project.json").read_text()
    exposed = json.dumps(job["events"]) + job["error"] + manifest + saved
    assert "[redacted]" in manifest
    for key in old + new:
        assert key not in exposed


@pytest.mark.parametrize("mode", ["idea2video", "script2video"])
@pytest.mark.parametrize("image_provider", ["a2e", "gemini"])
def test_gpt_full_mocked_job_with_both_modes_and_image_paths(monkeypatch, mode, image_provider):
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAX_SCENES", "1")
    monkeypatch.setenv("MAX_CHARACTERS", "1")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    text_requests, image_requests, a2e_images, uploads, videos = [], [], [], [], []
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    def openai_handler(request):
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer sk-fake-openai"
        payload = json.loads(request.content)
        text_requests.append(payload)
        prompt = payload["input"]
        if prompt.startswith("Develop this idea"):
            text = "Mira the otter discovers a violin and shares music."
        elif prompt.startswith("Turn this story"):
            text = json.dumps({"scenes": [{"script": "Mira discovers and plays a violin."}]})
        elif prompt.startswith("Extract at most"):
            text = json.dumps({"characters": [{"idx": 0, "name": "Mira", "static_features": "brown fur",
                                               "dynamic_features": "red scarf", "is_visible": True}]})
        elif prompt.startswith("Break this scene"):
            assert "Mira" in prompt and "7 seconds" in prompt
            text = json.dumps({"shots": [
                {"idx": 0, "visual_desc": "Mira finds the violin.", "motion_desc": "Pan gently.", "audio_desc": ""},
                {"idx": 1, "visual_desc": "Mira plays the violin.", "motion_desc": "Move closer.", "audio_desc": ""},
            ]})
        else:
            raise AssertionError("Unexpected GPT prompt")
        return httpx.Response(200, json=gpt_response(text))

    def image_handler(request):
        assert request.headers["x-goog-api-key"] == "fake-google"
        image_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output", "content": [{
            "type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()
        }]}]})

    class MockText(OpenAITextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(openai_handler))

    class MockImage(GeminiImageClient):
        def __init__(self, key, model, a2e):
            super().__init__(key, model, a2e, transport=httpx.MockTransport(image_handler))

    class MockA2E:
        supports_references = True
        def __init__(self, *, api_key, on_event, image_model, video_model):
            assert api_key == "sk_fake-a2e" and video_model == "wan2.6-i2v-flash"
            assert image_model == ("a2e-v2" if image_provider == "a2e" else "a2e")
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def generate_image(self, prompt, *, aspect_ratio, name, reference_urls=None):
            a2e_images.append((prompt, aspect_ratio, reference_urls))
            return f"https://cdn.test/a2e-frame-{len(a2e_images)}.jpg"
        async def upload_image(self, content, mime):
            assert content == image_bytes and mime == "image/jpeg"
            uploads.append(content)
            return f"https://cdn.test/upload-{len(uploads)}.jpg"
        async def generate_video(self, prompt, image_url, *, duration, name):
            videos.append((prompt, image_url, duration))
            return f"https://cdn.test/video-{len(videos)}.mp4"

    async def fake_download(url, dest):
        Path(dest).write_bytes(b"clip")
        return dest
    async def fake_concat(paths, output):
        assert len(paths) == 2
        Path(output).write_bytes(b"fake-mp4")
        return output
    async def fake_image_download(url):
        return image_bytes, "image/jpeg"

    monkeypatch.setattr(api, "OpenAITextLLM", MockText)
    monkeypatch.setattr(api, "GeminiImageClient", MockImage)
    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr("pipelines.script2video.download_image", fake_image_download)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    body = {"idea": "An otter finds a violin", "mode": mode, "script": "Mira follows music through the woods.",
            "planning_provider": "openai", "planning_model": "gpt-6.1-terra",
            "image_provider": image_provider, "image_model": "a2e-v2" if image_provider == "a2e" else "gemini-3-pro-image",
            "video_model": "wan2.6-i2v-flash", "clip_duration": 7}
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json=body)
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert client.get(result["video_url"]).content == b"fake-mp4"
        stream = client.get(f"/api/status/{job_id}").text
        assert "OpenAI GPT / gpt-6.1-terra" in stream
        persisted = (api.PROJECTS_DIR / job_id / "project.json").read_text()
        assert json.loads(persisted)["request"]["planning_model"] == "gpt-6.1-terra"
        for secret in ("sk-fake-openai", "fake-google", "sk_fake-a2e"):
            assert secret not in stream + persisted + json.dumps(result)

    assert len(text_requests) == (4 if mode == "idea2video" else 2)
    assert all(item["model"] == "gpt-6.1-terra" and item["store"] is False for item in text_requests)
    assert sum("text" in item for item in text_requests) == (3 if mode == "idea2video" else 2)
    if mode == "script2video":
        assert all("Develop this idea" not in item["input"] for item in text_requests)
        assert "Mira follows music through the woods." in text_requests[0]["input"]
    assert len(videos) == 2 and all(item[2] == 7 for item in videos)
    if image_provider == "a2e":
        assert len(a2e_images) == 3 and not image_requests and not uploads
    else:
        assert len(image_requests) == 3 and len(uploads) == 2 and not a2e_images
        assert all(item["model"] == "gemini-3-pro-image" and item["store"] is False for item in image_requests)
        assert len(image_requests[1]["input"]) == 2


@pytest.mark.parametrize("mode", ["idea2video", "script2video"])
def test_gpt_failure_stops_without_media_or_fallback(monkeypatch, mode):
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-secret")
    monkeypatch.setenv("LLM_API_KEY", "fake-compatible-secret")
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"message": "sk-fake-openai-secret and private prompt"}})
    class MockText(OpenAITextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(handler))
    class NoMedia:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def generate_image(self, *args, **kwargs):
            raise AssertionError("No media or fallback may be requested")
        async def generate_video(self, *args, **kwargs):
            raise AssertionError("No media or fallback may be requested")
    monkeypatch.setattr(api, "OpenAITextLLM", MockText)
    monkeypatch.setattr(api, "A2EClient", NoMedia)
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "private prompt", "script": "private script",
            "mode": mode, "planning_provider": "openai", "planning_model": "gpt-6-luna"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed" and "HTTP 403" in result["error"]
        stream = client.get(f"/api/status/{job_id}").text
        for value in ("sk-fake-openai-secret", "fake-google-secret", "fake-compatible-secret", "sk_fake-a2e", "private prompt"):
            assert value not in result["error"] + stream
        assert client.get(f"/api/projects/{job_id}").json()["uncertain_steps"]
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["idea2video", "script2video"])
def test_gpt_failed_storyboard_resumes_original_model_using_current_key_and_saved_work(monkeypatch, mode):
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old-openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAX_SCENES", "1")
    monkeypatch.setenv("MAX_CHARACTERS", "1")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    text_calls, images, videos = [], [], []
    fail = {"storyboard": True}
    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(image_buffer, "JPEG")
    image = image_buffer.getvalue()

    def handler(request):
        body = json.loads(request.content)
        text_calls.append((body["input"], body["model"], request.headers["authorization"]))
        prompt = body["input"]
        if prompt.startswith("Develop this idea"):
            text = "Mira finds a violin and shares music."
        elif prompt.startswith("Extract at most"):
            text = json.dumps({"characters": [{"idx": 0, "name": "Mira", "static_features": "brown fur",
                                               "dynamic_features": "red scarf", "is_visible": True}]})
        elif prompt.startswith("Turn this story"):
            text = json.dumps({"scenes": [{"script": "Mira finds a violin."}]})
        elif prompt.startswith("Break this scene"):
            if fail["storyboard"]:
                return httpx.Response(503, json={"error": "sk-old-openai private text"})
            text = json.dumps({"shots": [
                {"idx": 0, "visual_desc": "Mira finds a violin", "motion_desc": "pan", "audio_desc": ""},
                {"idx": 1, "visual_desc": "Mira plays", "motion_desc": "close in", "audio_desc": ""}]})
        else:
            raise AssertionError("Unexpected GPT planning prompt")
        return httpx.Response(200, json=gpt_response(text))

    class MockText(OpenAITextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(handler))
    class MockA2E:
        supports_references = True
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def generate_image(self, prompt, **kwargs):
            images.append(prompt)
            return f"https://cdn.test/img{len(images)}.jpg"
        async def generate_video(self, prompt, image_url, **kwargs):
            videos.append((prompt, image_url))
            return f"https://cdn.test/video{len(videos)}.mp4"
    async def fake_image(url):
        return image, "image/jpeg"
    async def fake_video(url, destination):
        Path(destination).write_bytes(b"clip")
        return destination
    async def fake_concat(paths, output):
        assert len(paths) == 2
        Path(output).write_bytes(b"combined")
        return output
    monkeypatch.setattr(api, "OpenAITextLLM", MockText)
    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr("pipelines.script2video.download_image", fake_image)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_video)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)

    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "An otter finds a violin.", "mode": mode,
            "script": "Mira finds a violin.", "planning_provider": "openai", "planning_model": "gpt-6.1-sol"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["status"] == "failed" and "HTTP 503" in detail["error"]
        expected_completed = ({"Story outline", "Character descriptions", "Scene scripts"}
                              if mode == "idea2video" else {"Character descriptions"})
        assert expected_completed.issubset(detail["completed_steps"])
        assert detail["uncertain_steps"] == [{"label": "Scene 1 storyboard", "kind": "text"}]
        initial_calls = 4 if mode == "idea2video" else 2
        assert len(text_calls) == initial_calls and not images and not videos
        api.jobs.clear()  # simulate a new backend process reading durable checkpoints
        monkeypatch.delenv("OPENAI_API_KEY")
        consent = {"acknowledge_costs": True, "retry_uncertain": True}
        assert client.post(f"/api/projects/{job_id}/resume", headers=HEADERS, json=consent).status_code == 503
        assert len(text_calls) == initial_calls and not images
        monkeypatch.setenv("OPENAI_API_KEY", "sk-new-openai")
        monkeypatch.setattr(api, "DEFAULT_OPENAI_TEXT_MODEL", "gpt-5.4-mini")
        assert client.post(f"/api/projects/{job_id}/resume", headers=HEADERS,
                           json={"acknowledge_costs": True}).status_code == 409
        assert len(text_calls) == initial_calls
        fail["storyboard"] = False
        response = client.post(f"/api/projects/{job_id}/resume", headers=HEADERS, json=consent)
        assert response.status_code == 200 and response.json()["job_id"] == job_id
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "completed", result
        assert client.get(result["video_url"]).content == b"combined"
        assert len(text_calls) == initial_calls + 1  # completed GPT planning calls were reused
        assert all(model == "gpt-6.1-sol" for _, model, _ in text_calls)
        assert all(key == "Bearer sk-old-openai" for _, _, key in text_calls[:initial_calls])
        assert text_calls[-1][2] == "Bearer sk-new-openai"
        assert len(images) == 3 and len(videos) == 2
        detail = client.get(f"/api/projects/{job_id}").json()
        assert detail["attempt"] == 2 and not detail["uncertain_steps"]
        saved = (api.PROJECTS_DIR / job_id / "project.json").read_text()
        assert json.loads(saved)["request"]["planning_model"] == "gpt-6.1-sol"
        assert "sk-old-openai" not in saved and "sk-new-openai" not in saved and "sk_fake-a2e" not in saved
