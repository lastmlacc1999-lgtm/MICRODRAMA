import io

import httpx
import pytest
from PIL import Image

from utils.images import download_image, image_url_available


@pytest.mark.asyncio
async def test_image_backup_validates_bytes_and_never_sends_authorization():
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, "PNG")
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, content=image.getvalue())
    data, mime = await download_image("https://cdn.test/a.png?signature=fake-secret", transport=httpx.MockTransport(handler))
    assert data == image.getvalue() and mime == "image/png"

    with pytest.raises(RuntimeError, match="verified"):
        await download_image("https://cdn.test/a.png", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not an image")))
    with pytest.raises(RuntimeError) as error:
        await download_image("https://cdn.test/a.png?signature=fake-secret", transport=httpx.MockTransport(lambda request: httpx.Response(403)))
    assert "fake-secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(200, True), (206, True), (403, False), (404, False), (410, False), (416, True)])
async def test_expiration_probe_only_treats_clear_unavailability_as_missing(status, expected):
    def handler(request):
        assert request.headers["range"] == "bytes=0-0" and "authorization" not in request.headers
        return httpx.Response(status)
    assert await image_url_available("https://cdn.test/a.png", transport=httpx.MockTransport(handler)) is expected


@pytest.mark.asyncio
async def test_expiration_probe_will_not_trigger_reupload_on_transient_http_error():
    with pytest.raises(RuntimeError, match="no re-upload"):
        await image_url_available("https://cdn.test/a.png", transport=httpx.MockTransport(lambda request: httpx.Response(503)))
