import json

import httpx
import pytest

from tools.a2e_client import A2EClient
from tools.model_catalog import IMAGE_MODELS, VIDEO_MODELS


VIDEO_ROOTS = {
    "native": "/v1/userImage2Video", "wan25": "/v1/userWan25", "wanspicy": "/v1/userWanSpicy",
    "wanr2v": "/v1/userWan26R2V", "wan30": "/v1/userWan30", "happyhorse": "/v1/userHappyhorseVideo",
    "veo": "/v1/veoVideo", "kling": "/v1/klingVideo", "kling4": "/v1/klingVideo",
    "omni": "/v1/klingOmni", "grok": "/v1/grokVideo", "hailuo": "/v1/hailuoVideo",
    "minimax": "/v1/minimaxH3Video", "seedance1": "/v1/seedanceVideo", "seedance2": "/v1/seedance2Video",
}
IMAGE_ROOTS = {
    "native": "/v1/userText2Image", "seedream": "/v1/userText2Image", "nano": "/v1/userNanoBanana",
    "gpt": "/v1/userGptImage", "wan26image": "/v1/userWan26Image", "wan27image": "/v1/userWan27Image",
    "qwen": "/v1/userQwen2Image", "flux": "/v1/userFlux2", "grokimage": "/v1/userGrokImage",
    "klingimage": "/v1/userKlingImage",
}


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for key in ("A2E_BASE_URL", "A2E_IMAGE_MODEL", "A2E_VIDEO_MODEL", "A2E_AUTO_ADD_AUDIO",
                "A2E_MAX_CONCURRENCY", "A2E_MAX_JOB_CREDITS"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", VIDEO_MODELS, ids=lambda m: m.id)
async def test_every_video_model_quotes_exact_payload_and_polls_its_own_endpoint(model):
    root = VIDEO_ROOTS[model.family]
    frame = "https://cdn.example.com/our-frame.jpg"
    calls = []

    def handler(req):
        assert req.headers["authorization"] == "Bearer sk_test"
        body = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, body))
        if req.url.path == "/v1/generation/quote":
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 2}})
        if req.url.path == root + "/start":
            return httpx.Response(200, json={"code": 0, "data": {"_id": "vid-task"}})
        assert req.url.path == root + "/vid-task"
        return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "result_url": "https://cdn.example.com/movie.mp4"}})

    async with A2EClient("sk_test", video_model=model.id, transport=httpx.MockTransport(handler),
                         poll_interval=.001, poll_timeout=2) as client:
        result = await client.generate_video("gentle motion", frame, duration=model.default_duration)
    assert result == "https://cdn.example.com/movie.mp4"
    assert [item[1] for item in calls] == ["/v1/generation/quote", root + "/start", root + "/vid-task"]
    post = calls[1][2]
    assert calls[0][2] == {"endpoint": "/api" + root + "/start", "requestBody": post}
    if model.family == "wan30":
        assert post["input"]["media"] == [{"type": "first_frame", "url": frame}]
        assert post["parameters"]["duration"] == model.default_duration
        assert post["model"] == model.variant and post["parameters"]["audio"] is False
    elif model.family == "veo":
        assert post["imageUrls"] == [frame] and post["generationType"] == "FIRST_AND_LAST_FRAMES_2_VIDEO"
        assert post["model"] == model.variant and "duration" not in post
    elif model.family == "wanr2v":
        assert post["reference_urls"] == [frame] and post["model"] == model.variant
        assert post["audio"] is (model.variant == "wan2.6-r2v")
    elif model.family == "omni":
        assert post["image_list"] == [frame] and "<<<image_1>>>" in post["prompt"]
        assert post["mode"] == model.quality and post["sound"] is False
    elif model.family == "kling4":
        assert post["keyframes"] == [{"type": "first_frame", "url": frame}]
        assert post["version"] == model.variant and post["resolution"] == model.quality
        assert "sound" not in post and "quality_mode" not in post
    elif model.family == "minimax":
        assert post["first_frame_url"] == frame and post["mode"] == "image-to-video"
    else:
        assert post["image_url"] == frame
    if model.family == "native":
        assert post["model_version"] == model.variant and post["video_time"] == model.default_duration
        assert post["number_of_images"] == 1 and post["auto_add_audio"] is False
    if model.family == "kling":
        assert post["sound"] is False
        if model.variant == "3.0-fast":
            assert post["version"] == "3.0" and post["model_version"] == "fast" and post["resolution"] == model.quality
        else:
            assert post["version"] == model.variant and post["quality_mode"] == model.quality
    if model.family == "seedance2":
        assert post["model_version"] == model.variant and post["duration"] == model.default_duration
        assert post["generate_audio"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("model", VIDEO_MODELS, ids=lambda m: m.id)
async def test_invalid_video_duration_never_contacts_provider(model):
    def handler(req):
        raise AssertionError("Invalid duration must not trigger any HTTP request")

    async with A2EClient("sk_test", video_model=model.id, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="duration"):
            await client.generate_video("motion", "https://cdn.example.com/frame.jpg", duration=31 if model.durations else 5)


A2E_IMAGES = [m for m in IMAGE_MODELS if m.provider == "a2e"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", A2E_IMAGES, ids=lambda m: m.id)
@pytest.mark.parametrize("ratio", ["16:9", "2:3"])
async def test_every_a2e_image_model_uses_own_payload_and_detail_route(model, ratio):
    root = IMAGE_ROOTS[model.family]
    ref = "https://cdn.example.com/reference.jpg"
    seen = []

    def handler(req):
        body = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.path, body))
        if req.url.path == "/v1/generation/quote":
            return httpx.Response(200, json={"code": 0, "data": {"available": True, "coins": 1}})
        if req.url.path == root + "/start":
            data = [{"_id": "image-id"}] if model.family in ("native", "seedream", "klingimage") else {"_id": "image-id"}
            return httpx.Response(200, json={"code": 0, "data": data})
        expected_detail = root + ("/image-id" if model.family in ("native", "seedream") else "/detail/image-id")
        assert req.url.path == expected_detail
        return httpx.Response(200, json={"code": 0, "data": {"current_status": "completed", "image_url": "https://cdn.example.com/output.jpg"}})

    async with A2EClient("sk_test", image_model=model.id, transport=httpx.MockTransport(handler),
                         poll_interval=.001, poll_timeout=2) as client:
        result = await client.generate_image("a lovely landscape", aspect_ratio=ratio, reference_urls=[ref])
    assert result == "https://cdn.example.com/output.jpg"
    assert len(seen) == 3
    body = seen[1][2]
    assert seen[0][2] == {"endpoint": "/api" + root + "/start", "requestBody": body}
    assert (body.get("input_images") == [ref]) is model.supports_references
    if model.family == "seedream":
        assert body["model_type"] == "seedream" and body["model_version"] == model.variant
        assert body["aspect_ratio"] == ratio and body["max_images"] == 1
    if model.family == "native":
        assert body["model_type"] == model.variant and body["max_images"] == 1
    if model.family in ("nano", "gpt", "wan27image", "qwen", "flux"):
        assert body["model"] == model.variant
    if model.family == "gpt" and model.variant == "gpt-image-1.5" and ratio == "16:9":
        assert body["aspect_ratio"] == "3:2"
    if model.family == "klingimage":
        assert body["n"] == 1
