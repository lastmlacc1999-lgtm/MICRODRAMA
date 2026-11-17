import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from PIL import Image

import api
from tools.gemini_image import GeminiImageClient
from tools.gemini_text import GeminiTextLLM
from tools.llm import OpenAICompatibleLLM
from tools.model_catalog import DEFAULT_GEMINI_TEXT_MODEL, GEMINI_TEXT_MODELS


def test_planner_catalog_validation_and_backward_compatible_defaults(monkeypatch):
    api.jobs.clear()
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("A2E_IMAGE_MODEL", raising=False)
    monkeypatch.delenv("A2E_VIDEO_MODEL", raising=False)
    captured = []

    async def capture(job_id, req, a2e_key, gemini_key, llm):
        captured.append((req, a2e_key, gemini_key, llm))

    monkeypatch.setattr(api, "run_pipeline", capture)
    with TestClient(api.app) as client:
        catalog = client.get("/api/models").json()
        assert catalog["configured_llm_available"] is False
        assert catalog["defaults"]["planning_provider"] == "simple"
        assert catalog["defaults"]["gemini_text"] == DEFAULT_GEMINI_TEXT_MODEL
        assert {x["id"] for x in catalog["gemini_text_models"]} == {x.id for x in GEMINI_TEXT_MODELS}

        for body, expected in (
            ({"planning_provider": "gemini", "image_provider": "a2e"}, 503),
            ({"planning_provider": "gemini", "planning_model": "no-such-model"}, 422),
            ({"planning_provider": "configured"}, 503),
            ({"planning_provider": "other"}, 422),
            ({"planning_provider": "simple", "planning_model": "gemini-3.8-flash"}, 422),
            ({"planning_provider": "configured", "planning_model": "gemini-3.8-flash"}, 422),
        ):
            response = client.post("/api/generate", json={"idea": "A small bird", **body})
            assert response.status_code == expected, (body, response.text)
        assert not api.jobs and not captured

        # Legacy requests with no LLM still use the simple planner.
        assert client.post("/api/generate", json={"idea": "A small bird"}).status_code == 200
        req, token, gemini_key, llm = captured[-1]
        assert req.planning_provider == "simple" and req.planning_model is None and llm is None
        assert token == "sk_fake-a2e" and gemini_key == ""

        monkeypatch.setenv("GEMINI_API_KEY", "fake-google-key")
        response = client.post("/api/generate", json={"idea": "A small bird", "planning_provider": "gemini"})
        assert response.status_code == 200
        req, _, gemini_key, llm = captured[-1]
        assert req.planning_provider == "gemini" and req.planning_model == DEFAULT_GEMINI_TEXT_MODEL
        assert isinstance(llm, GeminiTextLLM) and llm.model == DEFAULT_GEMINI_TEXT_MODEL
        assert llm.api_key == gemini_key == "fake-google-key"

        monkeypatch.setenv("LLM_API_KEY", "fake-openai-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
        monkeypatch.setenv("LLM_MODEL", "configured-text-model")
        catalog = client.get("/api/models").json()
        assert catalog["configured_llm_available"] is True and catalog["defaults"]["planning_provider"] == "configured"
        assert client.post("/api/generate", json={"idea": "bird", "planning_provider": "simple"}).status_code == 200
        assert captured[-1][0].planning_provider == "simple" and captured[-1][3] is None
        assert client.post("/api/generate", json={"idea": "bird"}).status_code == 200
        req, _, _, llm = captured[-1]
        assert req.planning_provider == "configured" and isinstance(llm, OpenAICompatibleLLM)
        assert llm.api_key == "fake-openai-key" and llm.model == "configured-text-model"
        assert client.post("/api/generate", json={"mode": "script2video", "script": "A bird flies.",
            "planning_provider": "gemini", "planning_model": "gemini-4-hydrogen"}).status_code == 200
        req, _, _, llm = captured[-1]
        assert req.planning_provider == "gemini" and llm.model == "gemini-4-hydrogen"
        assert isinstance(llm, GeminiTextLLM) and llm.api_key == "fake-google-key"
        monkeypatch.setenv("LLM_MODEL", "")
        response = client.post("/api/generate", json={"idea": "bird", "planning_provider": "configured"})
        assert response.status_code == 503 and "settings are incomplete" in response.text
        # Explicit simple still works if the unrelated configured provider is malformed.
        assert client.post("/api/generate", json={"idea": "bird", "planning_provider": "simple"}).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("planner", ["gemini", "configured"])
async def test_accepted_job_snapshots_keys_and_model_and_redacts_old_keys(monkeypatch, tmp_path, planner):
    api.jobs.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("A2E_API_KEY", "sk_old-a2e-private")
    monkeypatch.setenv("GEMINI_API_KEY", "old-google-private")
    monkeypatch.setenv("LLM_API_KEY", "old-llm-private")
    monkeypatch.setenv("LLM_MODEL", "old-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
    old = ["sk_old-a2e-private", "old-google-private"]
    if planner == "configured":
        old.append("old-llm-private")
    seen = []

    class MockA2E:
        def __init__(self, *, api_key, on_event, **kwargs):
            self.api_key = api_key
            self.on_event = on_event

        async def __aenter__(self):
            await self.on_event({"type": "quote", "message": " ".join(old)})
            return self

        async def __aexit__(self, *args):
            pass

    class FailingPipeline:
        def __init__(self, a2e, llm, *args, **kwargs):
            seen.append((a2e.api_key, llm.api_key, llm.model))

        async def run(self, **kwargs):
            raise RuntimeError("provider echoed " + " ".join(old))

    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr(api, "Idea2VideoPipeline", FailingPipeline)
    req = api.GenerateRequest(idea="test", planning_provider=planner,
                              planning_model="gemini-4-hydrogen" if planner == "gemini" else None)
    tasks = BackgroundTasks()
    accepted = await api.generate(req, tasks)
    monkeypatch.setenv("A2E_API_KEY", "sk_new-a2e")
    monkeypatch.setenv("GEMINI_API_KEY", "new-google")
    monkeypatch.setenv("LLM_API_KEY", "new-llm")
    monkeypatch.setenv("LLM_MODEL", "new-model")
    await tasks()
    job = api.jobs[accepted.job_id]
    assert job["status"] == "failed" and "[redacted]" in job["error"]
    assert seen == [("sk_old-a2e-private", "old-google-private" if planner == "gemini" else "old-llm-private",
                     "gemini-4-hydrogen" if planner == "gemini" else "old-model")]
    manifest = (tmp_path / "projects" / accepted.job_id / "a2e_events.jsonl").read_text()
    exposed = json.dumps(job["events"]) + job["error"] + manifest
    for key in old:
        assert key not in exposed
    assert "[redacted]" in manifest


@pytest.mark.parametrize("mode", ["idea2video", "script2video"])
@pytest.mark.parametrize("image_provider", ["a2e", "gemini"])
def test_gemini_planner_full_mocked_job_in_both_modes_and_image_paths(monkeypatch, tmp_path, mode, image_provider):
    api.jobs.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("MAX_SCENES", "1")
    monkeypatch.setenv("MAX_CHARACTERS", "1")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    Path("outputs").mkdir()
    text_requests, image_requests, a2e_images, uploads, videos = [], [], [], [], []
    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(image_buffer, format="JPEG")
    image_bytes = image_buffer.getvalue()

    def text_handler(request):
        assert request.headers["x-goog-api-key"] == "fake-google-key"
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
            raise AssertionError("Unexpected Gemini planning prompt")
        return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output",
            "content": [{"type": "text", "text": text}]}]})

    def image_handler(request):
        assert request.headers["x-goog-api-key"] == "fake-google-key"
        payload = json.loads(request.content)
        image_requests.append(payload)
        return httpx.Response(200, json={"status": "completed", "steps": [{"type": "model_output", "content": [{
            "type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()
        }]}]})

    class MockText(GeminiTextLLM):
        def __init__(self, key, model):
            super().__init__(key, model, transport=httpx.MockTransport(text_handler))

    class MockImage(GeminiImageClient):
        def __init__(self, key, model, a2e):
            super().__init__(key, model, a2e, transport=httpx.MockTransport(image_handler))

    class MockA2E:
        supports_references = True

        def __init__(self, *, api_key, on_event, image_model, video_model):
            assert api_key == "sk_fake-a2e"
            assert video_model == "wan2.6-i2v-flash"
            assert image_model == ("a2e-v2" if image_provider == "a2e" else "a2e")
            self.on_event = on_event

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

    monkeypatch.setattr(api, "GeminiTextLLM", MockText)
    monkeypatch.setattr(api, "GeminiImageClient", MockImage)
    monkeypatch.setattr(api, "A2EClient", MockA2E)
    monkeypatch.setattr("pipelines.script2video.download_image", fake_image_download)
    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    body = {"idea": "An otter finds a violin", "mode": mode, "script": "Mira follows music through the woods.",
            "planning_provider": "gemini", "planning_model": "gemini-4-hydrogen",
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
        assert "Gemini / gemini-4-hydrogen" in stream
        assert "fake-google-key" not in stream and "sk_fake-a2e" not in stream

    assert len(text_requests) == (4 if mode == "idea2video" else 2)
    assert all(item["model"] == "gemini-4-hydrogen" and item["store"] is False for item in text_requests)
    assert sum("response_format" in item for item in text_requests) == (3 if mode == "idea2video" else 2)
    if mode == "script2video":
        assert all("Develop this idea" not in item["input"] for item in text_requests)
        assert "Mira follows music through the woods." in text_requests[0]["input"]
    assert len(videos) == 2 and all(item[2] == 7 for item in videos)
    if image_provider == "a2e":
        assert len(a2e_images) == 3 and not image_requests and not uploads
        assert all("a2e-frame" in item[1] for item in videos)
    else:
        assert len(image_requests) == 3 and len(uploads) == 2 and not a2e_images
        assert all(item["model"] == "gemini-3-pro-image" and item["store"] is False for item in image_requests)
        assert all("upload-" in item[1] for item in videos)
        assert len(image_requests[1]["input"]) == 2  # the portrait is reused as a frame reference


@pytest.mark.parametrize("mode", ["idea2video", "script2video"])
def test_gemini_failure_stops_without_media_or_fallback(monkeypatch, tmp_path, mode):
    api.jobs.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("A2E_API_KEY", "sk_fake-a2e")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-google-secret")
    monkeypatch.setenv("LLM_API_KEY", "fake-configured-secret")  # must not be used as a fallback
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"message": "fake-google-secret private prompt"}})

    class MockText(GeminiTextLLM):
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
            raise AssertionError("No image or fallback may be requested")

        async def generate_video(self, *args, **kwargs):
            raise AssertionError("No video or fallback may be requested")

    monkeypatch.setattr(api, "GeminiTextLLM", MockText)
    monkeypatch.setattr(api, "A2EClient", NoMedia)
    with TestClient(api.app) as client:
        response = client.post("/api/generate", json={"idea": "private prompt", "script": "private script",
            "mode": mode, "planning_provider": "gemini", "planning_model": "gemini-4-argon"})
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        result = client.get(f"/api/result/{job_id}").json()
        assert result["status"] == "failed" and "HTTP 403" in result["error"]
        stream = client.get(f"/api/status/{job_id}").text
        for value in ("fake-google-secret", "fake-configured-secret", "sk_fake-a2e", "private prompt"):
            assert value not in result["error"] + stream
    assert len(calls) == 1
