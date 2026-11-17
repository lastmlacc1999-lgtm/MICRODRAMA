import asyncio
import json
import time
from decimal import Decimal

import httpx
import pytest

from tools.a2e_client import A2EClient, A2EError


def envelope(data, code=0, **extra):
    return {"code": code, "data": data, **extra}


@pytest.fixture(autouse=True)
def clean_a2e_settings(monkeypatch):
    for key in ("A2E_BASE_URL", "A2E_IMAGE_MODEL", "A2E_VIDEO_MODEL", "A2E_AUTO_ADD_AUDIO",
                "A2E_MAX_CONCURRENCY", "A2E_MAX_JOB_CREDITS"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
async def test_image_and_video_quote_exact_payload_then_submit_and_poll():
    calls = []
    events = []

    async def handler(req):
        assert req.headers["Authorization"] == "Bearer sk_test"
        body = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, body))
        if req.url.path == "/v1/generation/quote":
            return httpx.Response(200, json=envelope({"available": True, "coins": 3}))
        if req.url.path == "/v1/userText2Image/start":
            return httpx.Response(200, json=envelope([{"_id": "img-1", "current_status": "initialized"}]))
        if req.url.path == "/v1/userText2Image/img-1":
            return httpx.Response(200, json=envelope({"_id": "img-1", "current_status": "completed", "image_urls": ["https://cdn.test/frame.png"]}))
        if req.url.path == "/v1/userImage2Video/start":
            return httpx.Response(200, json=envelope({"_id": "vid-1"}))
        if req.url.path == "/v1/userImage2Video/vid-1":
            return httpx.Response(200, json=envelope({"_id": "vid-1", "current_status": "completed", "result_url": "https://cdn.test/clip.mp4"}))
        raise AssertionError(req.url)

    async def event(event):
        events.append(event)

    async with A2EClient("sk_test", on_event=event, transport=httpx.MockTransport(handler),
                         poll_interval=0.001, poll_timeout=2) as client:
        image = await client.generate_image("dramatic moon", reference_urls=["https://cdn.test/ref.png"])
        video = await client.generate_video("camera pans", image)
        assert client.budget.total == Decimal("6")
    assert image == "https://cdn.test/frame.png"
    assert video == "https://cdn.test/clip.mp4"
    quotes = [body for _, path, body in calls if path == "/v1/generation/quote"]
    image_post = next(body for _, path, body in calls if path == "/v1/userText2Image/start")
    video_post = next(body for _, path, body in calls if path == "/v1/userImage2Video/start")
    assert quotes[0] == {"endpoint": "/api/v1/userText2Image/start", "requestBody": image_post}
    assert quotes[1] == {"endpoint": "/api/v1/userImage2Video/start", "requestBody": video_post}
    assert image_post["input_images"] == ["https://cdn.test/ref.png"]
    assert (image_post["width"], image_post["height"], image_post["max_images"]) == (1280, 720, 1)
    assert video_post["image_url"] == image and video_post["video_time"] == 5
    assert video_post["model_version"] == "a2e" and video_post["auto_add_audio"] is False
    assert [e["type"] for e in events] == ["quote", "task", "quote", "task"]
    assert [e["task_id"] for e in events if e["type"] == "task"] == ["img-1", "vid-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("quote_data", [
    {"available": False, "coins": None, "unavailableReason": "pricing_input_required"},
    {"available": True, "coins": None},
    {"available": True, "coins": -1},
    {"available": True, "coins": True},
])
async def test_invalid_or_unavailable_quote_never_submits(quote_data):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json=envelope(quote_data))

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001) as client:
        with pytest.raises(A2EError, match="quote"):
            await client.generate_image("anything")
    assert calls == ["/v1/generation/quote"]


@pytest.mark.asyncio
async def test_business_error_in_http_200_is_not_success():
    def handler(req):
        return httpx.Response(200, json={"code": 100004, "msg": "not allowed", "trace_id": "trace-123"})
    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001) as client:
        with pytest.raises(A2EError, match="100004.*trace-123.*not allowed"):
            await client.generate_image("anything")


@pytest.mark.asyncio
async def test_moderation_without_task_id_is_not_polled():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if "quote" in req.url.path:
            return httpx.Response(200, json=envelope({"available": True, "coins": 1}))
        return httpx.Response(200, json=envelope({"nsfw_detected": True, "message": "Content blocked"}, trace_id="trace-1"))

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001) as client:
        with pytest.raises(A2EError, match="Content blocked.*trace-1"):
            await client.generate_video("anything", "https://cdn.test/image.png")
    assert calls == ["/v1/generation/quote", "/v1/userImage2Video/start"]


@pytest.mark.asyncio
async def test_creation_timeout_never_retries_paid_post():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if "quote" in req.url.path:
            return httpx.Response(200, json=envelope({"available": True, "coins": 1}))
        raise httpx.ReadTimeout("response lost", request=req)

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001) as client:
        with pytest.raises(A2EError, match="may have been created and charged"):
            await client.generate_image("anything")
    assert calls.count("/v1/userText2Image/start") == 1


@pytest.mark.asyncio
async def test_poll_retries_429_and_honors_retry_after():
    read_times = []
    starts = []

    def handler(req):
        if "quote" in req.url.path:
            return httpx.Response(200, json=envelope({"available": True, "coins": 1}))
        if req.url.path.endswith("/start"):
            starts.append(req.url.path)
            return httpx.Response(200, json=envelope({"_id": "vid"}))
        read_times.append(time.monotonic())
        if len(read_times) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.12"}, json={"code": 429, "msg": "busy"})
        return httpx.Response(200, json=envelope({"current_status": "completed", "result_url": "https://cdn.test/x.mp4"}))

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001, poll_timeout=2) as client:
        assert await client.generate_video("motion", "https://cdn.test/image.png") == "https://cdn.test/x.mp4"
    assert len(starts) == 1
    assert read_times[1] - read_times[0] >= 0.115


@pytest.mark.asyncio
async def test_failed_task_surfaces_id_reason_and_pending_refund():
    def handler(req):
        if "quote" in req.url.path:
            return httpx.Response(200, json=envelope({"available": True, "coins": 1}))
        if req.url.path.endswith("/start"):
            return httpx.Response(200, json=envelope({"_id": "v-123"}))
        return httpx.Response(200, json=envelope({"current_status": "failed", "failed_code": "X1",
                                                  "failed_message": "provider failure", "hasRefundCoin": True,
                                                  "refund_pending": True}, trace_id="detail-trace"))
    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001) as client:
        with pytest.raises(A2EError, match="v-123 failed.*X1.*provider failure.*refund pending.*detail-trace"):
            await client.generate_video("motion", "https://cdn.test/image.png")


@pytest.mark.asyncio
async def test_concurrent_estimates_cannot_exceed_job_budget():
    starts = []

    async def handler(req):
        if "quote" in req.url.path:
            await asyncio.sleep(.001)
            return httpx.Response(200, json=envelope({"available": True, "coins": 5}))
        if req.url.path.endswith("/start"):
            starts.append(req.url.path)
            return httpx.Response(200, json=envelope([{"_id": "only-one"}]))
        return httpx.Response(200, json=envelope({"current_status": "completed", "image_urls": ["https://cdn.test/a.png"]}))

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001,
                         max_job_credits=Decimal("5")) as client:
        results = await asyncio.gather(client.generate_image("one"), client.generate_image("two"),
                                       return_exceptions=True)
    assert len(starts) == 1
    assert sum(isinstance(r, A2EError) for r in results) == 1
    assert "budget exceeded" in str(next(r for r in results if isinstance(r, Exception)))


@pytest.mark.asyncio
async def test_poll_timeout_keeps_task_id_and_does_not_resubmit():
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if "quote" in req.url.path:
            return httpx.Response(200, json=envelope({"available": True, "coins": 1}))
        if req.url.path.endswith("/start"):
            return httpx.Response(200, json=envelope({"_id": "still-running"}))
        return httpx.Response(200, json=envelope({"current_status": "processing"}))

    async with A2EClient("sk_test", transport=httpx.MockTransport(handler), poll_interval=.001, poll_timeout=.03) as client:
        with pytest.raises(A2EError, match="still-running.*still be running"):
            await client.generate_video("motion", "https://cdn.test/image.png")
    assert calls.count("/v1/userImage2Video/start") == 1
