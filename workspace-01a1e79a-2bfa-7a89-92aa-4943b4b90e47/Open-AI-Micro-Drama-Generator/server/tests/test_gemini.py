import base64
import io
import json
from decimal import Decimal

import httpx
import pytest
from PIL import Image

from tools.a2e_client import A2EClient, A2EError
from tools.gemini_image import GeminiImageClient, GeminiImageError
from tools.model_catalog import IMAGE_MODELS


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for key in ("A2E_BASE_URL", "A2E_IMAGE_MODEL", "A2E_VIDEO_MODEL", "A2E_AUTO_ADD_AUDIO",
                "A2E_MAX_CONCURRENCY", "A2E_MAX_JOB_CREDITS"):
        monkeypatch.delenv(key, raising=False)


def jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB", (40, 30), (90, 140, 210)).save(output, format="JPEG")
    return output.getvalue()


def image_response(image):
    return {"id": "interaction-1", "status": "completed", "steps": [
        {"type": "model_output", "content": [{"type": "text", "text": "Here it is"},
          {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}]}]}


@pytest.mark.asyncio
async def test_gemini_generates_with_references_in_memory_and_uploads_frames_lazily():
    image = jpeg_bytes()
    requests = []
    uploads = []

    class FakeA2E:
        async def upload_image(self, content, content_type):
            uploads.append((content, content_type))
            return "https://cdn.example.com/frame.jpg"

    def gemini(req):
        assert req.url == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert req.headers["x-goog-api-key"] == "fake-google-key"
        assert "authorization" not in req.headers
        assert req.headers["api-revision"] == "2026-05-20"
        requests.append(json.loads(req.content))
        return httpx.Response(200, json=image_response(image))

    async with GeminiImageClient("fake-google-key", "gemini-nano-banana-2.1", FakeA2E(),
                                 transport=httpx.MockTransport(gemini)) as client:
        first = await client.generate_image("A quiet hillside", aspect_ratio="2:3", name="portrait")
        second = await client.generate_image("The same hillside at dawn", aspect_ratio="16:9", reference_urls=[first])
        assert first.startswith("gemini://") and second != first
        assert not uploads  # portraits/frames do not pay A2E upload fees until a video needs them
        assert await client.public_url(second) == "https://cdn.example.com/frame.jpg"
        assert await client.public_url(second) == "https://cdn.example.com/frame.jpg"
        assert len(uploads) == 1 and uploads[0] == (image, "image/jpeg")
    assert requests[0]["model"] == "gemini-nano-banana-2.1"
    assert requests[0]["store"] is False
    assert requests[0]["response_format"]["aspect_ratio"] == "2:3"
    assert requests[1]["response_format"] == {"type": "image", "aspect_ratio": "16:9", "mime_type": "image/jpeg", "delivery": "inline"}
    assert requests[1]["input"][1] == {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [m for m in IMAGE_MODELS if m.provider == "gemini"], ids=lambda m: m.id)
async def test_each_gemini_catalog_choice_is_sent_to_google(model):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=image_response(jpeg_bytes()))

    async with GeminiImageClient("fake-google-key", model.id, object(),
                                 transport=httpx.MockTransport(handler)) as client:
        assert (await client.generate_image("A quiet hillside")).startswith("gemini://")
    assert len(seen) == 1 and seen[0]["model"] == model.id
    assert seen[0]["store"] is False


@pytest.mark.asyncio
async def test_gemini_provider_error_is_safe_and_never_retried():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(403, json={"error": {"message": "fake-google-secret has no access"}})

    async with GeminiImageClient("fake-google-secret", "gemini-3-pro-image", object(),
                                 transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GeminiImageError) as raised:
            await client.generate_image("prompt")
    assert len(calls) == 1 and "HTTP 403" in str(raised.value)
    assert "fake-google-secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_gemini_timeout_warns_of_possible_google_charge_and_does_not_retry():
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("lost", request=req)

    async with GeminiImageClient("fake-google-key", "gemini-2.5-flash-image", object(),
                                 transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GeminiImageError, match="may have been billed.*no automatic retry"):
            await client.generate_image("prompt")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"status": "completed", "steps": []},
    {"status": "failed", "steps": []},
    {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "image", "mime_type": "image/jpeg", "data": "not-base64"}]}]},
    {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "image", "mime_type": "image/png", "data": base64.b64encode(jpeg_bytes()).decode()}]}]},
])
async def test_gemini_invalid_response_or_filtered_result_does_not_upload(payload):
    class NeverUpload:
        async def upload_image(self, *args):
            raise AssertionError("must not upload invalid content")
    async with GeminiImageClient("fake-google-key", "gemini-nano-banana-2.1", NeverUpload(),
                                 transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))) as client:
        with pytest.raises(GeminiImageError):
            await client.generate_image("prompt")


@pytest.mark.asyncio
async def test_presigned_upload_uses_exact_bytes_no_bearer_and_reserves_current_signing_price():
    image = jpeg_bytes()
    signed_url = "https://storage.example.com/objects/a%2Fb.jpg?X-Amz-Signature=a%2Fb%2Bc&part=1"
    a2e_calls, storage_calls, events = [], [], []

    def api_handler(req):
        a2e_calls.append(req)
        assert req.headers["authorization"] == "Bearer sk_test-a2e"
        if req.url.path == "/v1/feature-pricing":
            return httpx.Response(200, json={"code": 0, "data": {"prices": {"r2": {"credits": 1, "unit": "per_use"}}}})
        assert req.url.path == "/v1/r2/upload-presigned-url"
        payload = json.loads(req.content)
        assert payload["purpose"] == "STAGING" and payload["contentType"] == "image/jpeg"
        assert payload["fileSize"] == len(image) and payload["key"].endswith(".jpg")
        return httpx.Response(200, json={"code": 0, "data": {"uploadUrl": signed_url, "cdnUrl": "https://cdn.example.com/frame.jpg"}})

    def storage_handler(req):
        storage_calls.append(req)
        assert str(req.url) == signed_url
        assert "authorization" not in req.headers and "x-goog-api-key" not in req.headers
        assert req.method == "PUT" and req.content == image
        assert req.headers["content-type"] == "image/jpeg" and int(req.headers["content-length"]) == len(image)
        return httpx.Response(200)

    async def on_event(event):
        events.append(event)

    async with A2EClient("sk_test-a2e", on_event=on_event, transport=httpx.MockTransport(api_handler),
                         upload_transport=httpx.MockTransport(storage_handler), max_job_credits=Decimal("5")) as client:
        assert await client.upload_image(image, "image/jpeg") == "https://cdn.example.com/frame.jpg"
        assert client.budget.total == Decimal("1")
    assert len(a2e_calls) == 2 and len(storage_calls) == 1
    assert [e["type"] for e in events] == ["quote", "upload"]
    assert signed_url not in json.dumps(events) and "sk_test-a2e" not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["upload", "signing-timeout", "budget", "bad-price"])
async def test_upload_failures_do_not_retry_and_never_expose_signed_url(failure):
    image = jpeg_bytes()
    requests = []
    signed = "https://storage.example.com/frame.jpg?sig=private-signature"

    def api_handler(req):
        requests.append(req.url.path)
        if req.url.path == "/v1/feature-pricing":
            price = None if failure == "bad-price" else 3
            return httpx.Response(200, json={"code": 0, "data": {"prices": {"r2": {"credits": price}}}})
        if failure == "signing-timeout":
            raise httpx.ReadTimeout("ambiguous", request=req)
        return httpx.Response(200, json={"code": 0, "data": {"uploadUrl": signed, "cdnUrl": "https://cdn.example.com/frame.jpg"}})

    def storage_handler(req):
        requests.append("storage-put")
        assert "authorization" not in req.headers
        return httpx.Response(503)

    async with A2EClient("sk_test", transport=httpx.MockTransport(api_handler),
                         upload_transport=httpx.MockTransport(storage_handler), max_job_credits=Decimal("1") if failure == "budget" else Decimal("5")) as client:
        with pytest.raises(A2EError) as raised:
            await client.upload_image(image, "image/jpeg")
    assert requests.count("/v1/r2/upload-presigned-url") <= 1 and requests.count("storage-put") <= 1
    assert "private-signature" not in str(raised.value)
    if failure in ("upload", "signing-timeout"):
        assert "fee may" in str(raised.value)
    if failure in ("budget", "bad-price"):
        assert "/v1/r2/upload-presigned-url" not in requests
