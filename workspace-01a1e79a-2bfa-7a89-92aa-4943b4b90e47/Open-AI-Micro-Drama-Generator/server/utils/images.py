"""Save and check provider image URLs without sending API credentials."""
import io
from typing import Optional
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("The provider returned an invalid image URL.")


async def download_image(url: str, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> tuple[bytes, str]:
    """Download a bounded, verified local backup before a provider link expires."""
    _validate_url(url)
    chunks = []
    size = 0
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), follow_redirects=True,
                                     transport=transport) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise RuntimeError("The provider image exceeds the 20 MB backup limit.")
                    chunks.append(chunk)
    except httpx.HTTPError as exc:
        # Do not expose a potentially signed URL in errors or event logs.
        raise RuntimeError("Could not download the generated image for safe keeping. The existing task was not resubmitted.") from exc
    data = b"".join(chunks)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.verify()
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(image_format)
        if not mime:
            raise ValueError("unsupported image")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise RuntimeError("The provider image could not be verified for local backup. The existing task was not resubmitted.") from exc
    return data, mime


async def image_url_available(url: str, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> bool:
    """Return False for a clearly inaccessible link; transient failures never cause re-upload."""
    _validate_url(url)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), follow_redirects=True,
                                     transport=transport) as client:
            async with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
                if response.status_code in (403, 404, 410):
                    return False
                if 200 <= response.status_code < 300 or response.status_code == 416:
                    return True
                raise RuntimeError(f"Could not check the saved image URL (HTTP {response.status_code}); no re-upload was made.")
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not check the saved image URL; no re-upload was made.") from exc
