import httpx
import pytest

from tools.a2e_client import A2EClient
from tools.a2e_requests import build_image_request, build_video_request
from tools.model_catalog import IMAGE_MODELS, VIDEO_MODELS


@pytest.mark.asyncio
@pytest.mark.parametrize("model", VIDEO_MODELS, ids=lambda model: model.id)
async def test_every_video_model_can_poll_saved_task_without_input_or_new_post(model, monkeypatch):
    monkeypatch.delenv("A2E_BASE_URL", raising=False)
    requests = []
    _, detail, _ = build_video_request(model, "unused", "https://cdn.test/frame", model.default_duration, "test")
    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.test/out.mp4"}})
    async with A2EClient(api_key="sk_fake", video_model=model.id, transport=httpx.MockTransport(handler),
                         poll_interval=.001, poll_timeout=1) as client:
        assert await client.generate_video("", "", resume_task_id="saved-task") == "https://cdn.test/out.mp4"
    assert requests == [("GET", detail + "/saved-task")]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [model for model in IMAGE_MODELS if model.provider == "a2e"], ids=lambda model: model.id)
async def test_every_a2e_image_model_can_poll_saved_task_without_new_post(model, monkeypatch):
    monkeypatch.delenv("A2E_BASE_URL", raising=False)
    requests = []
    _, detail, _ = build_image_request(model, "unused", "16:9", [], "test")
    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "image_url": "https://cdn.test/out.jpg"}})
    async with A2EClient(api_key="sk_fake", image_model=model.id, transport=httpx.MockTransport(handler),
                         poll_interval=.001, poll_timeout=1) as client:
        assert await client.generate_image("", resume_task_id="saved-task") == "https://cdn.test/out.jpg"
    assert requests == [("GET", detail + "/saved-task")]
