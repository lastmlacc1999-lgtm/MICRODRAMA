"""Generate frames with the Gemini Interactions API; upload only frames used for video.

Every request is made once. A lost response can still be billed by Google; there
is no automatic retry, fallback, or generation through A2E on failure.
"""
import asyncio
import base64
import binascii
import io
import uuid
from typing import Any, Optional

import httpx
from PIL import Image, UnidentifiedImageError

from tools.a2e_client import A2EClient
from tools.model_catalog import image_model


class GeminiImageError(RuntimeError):
    pass


class GeminiImageClient:
    supports_references = True
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
    MAX_BYTES = 20 * 1024 * 1024

    def __init__(self, api_key: str, model: str, a2e: A2EClient, *,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        if not api_key.strip():
            raise GeminiImageError("Gemini API key is missing. Save it in the website's API key settings.")
        image_model("gemini", model)
        self.model = model
        self.a2e = a2e
        self._client = httpx.AsyncClient(
            headers={"x-goog-api-key": api_key.strip(), "Content-Type": "application/json",
                     "Api-Revision": "2026-05-20"},
            transport=transport, follow_redirects=False,
            timeout=httpx.Timeout(240.0, connect=10.0),
        )
        self._images: dict[str, tuple[bytes, str]] = {}
        self._urls: dict[str, str] = {}
        self._upload_locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(2)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self._images.clear()
        self._urls.clear()
        await self._client.aclose()

    @classmethod
    def _image_from_response(cls, response: dict[str, Any]) -> tuple[bytes, str]:
        steps = response.get("steps")
        if not isinstance(steps, list):
            raise GeminiImageError("Gemini returned no image output. The request may have been billed; check Google billing.")
        blocks = [block for step in steps if isinstance(step, dict) and step.get("type") == "model_output"
                  for block in (step.get("content") if isinstance(step.get("content"), list) else [])
                  if isinstance(block, dict) and block.get("type") == "image"]
        if not blocks:
            raise GeminiImageError("Gemini returned no image (possibly due to a safety filter). The request may have been billed.")
        encoded = blocks[-1].get("data")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((cls.MAX_BYTES + 2) // 3):
            raise GeminiImageError("Gemini returned an invalid or oversized image; no A2E upload was requested.")
        try:
            content = base64.b64decode(encoded, validate=True)
            if not content or len(content) > cls.MAX_BYTES:
                raise ValueError("invalid size")
            with Image.open(io.BytesIO(content)) as image:
                kind = image.format
                image.verify()
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(kind)
            supplied_mime = blocks[-1].get("mime_type")
            if mime is None or (supplied_mime and supplied_mime != mime):
                raise ValueError("unsupported or mismatched type")
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise GeminiImageError("Gemini returned invalid image bytes or an unsupported image type; no A2E upload was requested.") from exc
        return content, mime

    async def generate_image(self, prompt: str, *, aspect_ratio: str = "16:9",
                             reference_urls: Optional[list[str]] = None,
                             name: str = "MicroDrama frame") -> str:
        if aspect_ratio not in ("16:9", "2:3", "1:1"):
            raise ValueError("Unsupported Gemini image aspect ratio")
        refs = reference_urls or []
        if len(refs) > 2:
            raise ValueError("This pipeline accepts at most two image references")
        parts: list[dict[str, str]] = [{"type": "text", "text": prompt}]
        for ref in refs:
            if ref not in self._images:
                # References must be images generated in this job, not arbitrary
                # URLs fetched by the server or submitted to Google.
                raise GeminiImageError("A Gemini image reference is no longer available in this job.")
            data, mime = self._images[ref]
            parts.append({"type": "image", "data": base64.b64encode(data).decode("ascii"), "mime_type": mime})
        body = {"model": self.model, "input": parts, "store": False,
                "response_format": {"type": "image", "aspect_ratio": aspect_ratio,
                                    "mime_type": "image/jpeg", "delivery": "inline"}}
        async with self._semaphore:
            try:
                response = await self._client.post(self.ENDPOINT, json=body)
            except httpx.RequestError as exc:
                raise GeminiImageError("Gemini image request lost its response and may have been billed. Check Google billing before starting again; no automatic retry was made.") from exc
        if not response.is_success:
            # Provider messages can contain prompts or credentials; do not echo them.
            raise GeminiImageError(f"Gemini image request failed (HTTP {response.status_code}). Check the Gemini key, model access and Google billing; no automatic retry was made.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiImageError("Gemini returned invalid JSON; the request may have been billed.") from exc
        if not isinstance(payload, dict):
            raise GeminiImageError("Gemini returned an invalid response; the request may have been billed.")
        if payload.get("status") not in (None, "completed"):
            raise GeminiImageError("Gemini did not complete the image request. It may have been billed; no automatic retry was made.")
        image, mime = self._image_from_response(payload)
        handle = f"gemini://{uuid.uuid4().hex}"
        self._images[handle] = (image, mime)
        self._upload_locks[handle] = asyncio.Lock()
        return handle

    def export_image(self, handle: str) -> tuple[bytes, str]:
        if handle not in self._images:
            raise GeminiImageError("A Gemini image is no longer available in this job.")
        return self._images[handle]

    def restore_image(self, handle: str, content: bytes, mime: str) -> None:
        """Restore a previously verified local image without calling Google again."""
        if (not isinstance(handle, str) or not handle.startswith("gemini://") or
                mime not in ("image/jpeg", "image/png", "image/webp") or
                not content or len(content) > self.MAX_BYTES):
            raise GeminiImageError("The saved Gemini image is invalid; no replacement was requested.")
        try:
            with Image.open(io.BytesIO(content)) as image:
                kind = image.format
                image.verify()
            if {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(kind) != mime:
                raise ValueError("mismatched image type")
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise GeminiImageError("The saved Gemini image is invalid; no replacement was requested.") from exc
        self._images[handle] = (content, mime)
        self._upload_locks.setdefault(handle, asyncio.Lock())

    async def public_url(self, handle: str) -> str:
        if handle not in self._images:
            raise GeminiImageError("A Gemini frame is no longer available in this job.")
        async with self._upload_locks[handle]:
            if handle not in self._urls:
                content, mime = self._images[handle]
                self._urls[handle] = await self.a2e.upload_image(content, mime)
            return self._urls[handle]
