"""A2E generation client, based on the public A2E OpenAPI specification.

Only read-only task polling is retried. A paid creation POST is never replayed:
if its response is lost, the caller must reconcile A2E task/billing records.
"""
import asyncio
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, urlsplit

import httpx

from tools.a2e_requests import build_image_request, build_video_request
from tools.model_catalog import image_model as find_image_model, video_model as find_video_model

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class A2EError(RuntimeError):
    """A safe-to-display A2E error, with retry hints for read requests only."""

    def __init__(self, message: str, *, retryable: bool = False,
                 retry_after: Optional[float] = None, terminal: bool = False,
                 output_expired: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.terminal = terminal
        self.output_expired = output_expired


def _positive_number(name: str, default: str) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as exc:
        raise A2EError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise A2EError(f"{name} must be a positive number")
    return value


def _positive_integer(name: str, default: str) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as exc:
        raise A2EError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise A2EError(f"{name} must be a positive integer")
    return value


def _retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class CreditBudget:
    """Per-job guard on quoted estimates, not an A2E spending limit."""

    def __init__(self, maximum: Optional[Decimal]):
        self.maximum = maximum
        self.total = Decimal("0")
        self._lock = asyncio.Lock()

    async def reserve(self, amount: Decimal) -> Decimal:
        async with self._lock:
            new_total = self.total + amount
            if self.maximum is not None and new_total > self.maximum:
                raise A2EError(
                    f"A2E estimated job budget exceeded: {new_total} credits would exceed "
                    f"the configured limit of {self.maximum}. Earlier tasks may already "
                    "have been charged. Adjust A2E_MAX_JOB_CREDITS if appropriate."
                )
            self.total = new_total
            return new_total


class A2EClient:
    IMAGE_PATH = "/v1/userText2Image/start"
    VIDEO_PATH = "/v1/userImage2Video/start"

    def __init__(self, api_key: Optional[str] = None, *,
                 on_event: Optional[EventCallback] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None,
                 upload_transport: Optional[httpx.AsyncBaseTransport] = None,
                 image_model: Optional[str] = None,
                 video_model: Optional[str] = None,
                 poll_interval: Optional[float] = None,
                 poll_timeout: Optional[float] = None,
                 max_job_credits: Optional[Decimal] = None):
        token = (api_key or os.environ.get("A2E_API_KEY", "")).strip()
        if not token:
            raise A2EError("A2E_API_KEY is missing. Save it in the website's API key settings.")
        base_url = os.environ.get("A2E_BASE_URL", "https://api.a2e.ai").rstrip("/")
        self.image_model = image_model or os.environ.get("A2E_IMAGE_MODEL", "a2e").strip()
        self.video_model = video_model or os.environ.get("A2E_VIDEO_MODEL", "a2e").strip()
        try:
            self.image_spec = find_image_model("a2e", self.image_model)
            self.video_spec = find_video_model(self.video_model)
        except ValueError as exc:
            raise A2EError(str(exc)) from exc
        self.supports_references = self.image_spec.supports_references
        self.upload_transport = upload_transport
        audio_setting = os.environ.get("A2E_AUTO_ADD_AUDIO", "false").strip().lower()
        if audio_setting not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            raise A2EError("A2E_AUTO_ADD_AUDIO must be true or false")
        self.auto_add_audio = audio_setting in ("1", "true", "yes", "on")
        self.on_event = on_event
        self.interval = poll_interval if poll_interval is not None else _positive_number("A2E_POLL_INTERVAL_SECONDS", "5")
        self.poll_timeout = poll_timeout if poll_timeout is not None else _positive_number("A2E_POLL_TIMEOUT_SECONDS", "900")
        if self.interval <= 0 or self.poll_timeout <= 0:
            raise A2EError("Poll interval and timeout must be positive")
        self.semaphore = asyncio.Semaphore(_positive_integer("A2E_MAX_CONCURRENCY", "2"))
        if max_job_credits is None:
            try:
                max_job_credits = Decimal(os.environ.get("A2E_MAX_JOB_CREDITS", "300"))
            except InvalidOperation as exc:
                raise A2EError("A2E_MAX_JOB_CREDITS must be a non-negative number") from exc
        if not max_job_credits.is_finite() or max_job_credits < 0:
            raise A2EError("A2E_MAX_JOB_CREDITS must be a non-negative number")
        self.budget = CreditBudget(max_job_credits)
        # Construct the HTTP client only after all local settings are validated.
        self.http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(320.0, connect=10.0),
            follow_redirects=False,
            transport=transport,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.http.aclose()

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event:
            await self.on_event(event)

    async def _request(self, method: str, path: str, *, body: Optional[dict] = None) -> dict:
        paid_creation = method == "POST" and path.endswith("/start")
        paid_signing = method == "POST" and path == "/v1/r2/upload-presigned-url"
        uncertain = (" The task may have been created and charged; check task and billing records before retrying."
                     if paid_creation else
                     " A signing fee may have been charged; check A2E billing before trying again."
                     if paid_signing else "")
        try:
            response = await self.http.request(
                method, path, json=body,
                timeout=httpx.Timeout(30.0, connect=10.0) if method == "GET" else httpx.Timeout(320.0, connect=10.0),
            )
        except httpx.RequestError as exc:
            if paid_creation or paid_signing:
                raise A2EError(f"A2E submission to {path} lost its response.{uncertain}") from exc
            raise A2EError(f"A2E {method} {path} network error: {type(exc).__name__}",
                           retryable=(method == "GET")) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise A2EError(
                f"A2E {method} {path} returned non-JSON or invalid JSON (HTTP {response.status_code})."
                + uncertain,
                retryable=method == "GET" and (response.status_code == 429 or response.status_code >= 500),
                retry_after=_retry_after(response.headers.get("Retry-After")),
            )
        code = payload.get("code")
        if not 200 <= response.status_code < 300 or code != 0:
            msg = str(payload.get("msg") or payload.get("message") or "request failed")[:500]
            trace = payload.get("trace_id")
            extra = f", trace_id={trace}" if trace else ""
            ambiguous = uncertain if response.status_code >= 500 else ""
            raise A2EError(
                f"A2E {method} {path}: HTTP {response.status_code}, code={code}{extra}: {msg}.{ambiguous}",
                retryable=method == "GET" and (response.status_code == 429 or response.status_code >= 500 or str(code) == "30101"),
                retry_after=_retry_after(response.headers.get("Retry-After")),
            )
        return payload

    async def _quote_and_reserve(self, path: str, body: dict) -> None:
        # The quote API examples use the /api/v1 alias, while requests use /v1.
        envelope = await self._request("POST", "/v1/generation/quote", body={
            "endpoint": "/api" + path,
            "requestBody": body,
        })
        result = envelope.get("data")
        if not isinstance(result, dict) or result.get("available") is not True:
            reason = result.get("unavailableReason", "unknown reason") if isinstance(result, dict) else "invalid response"
            raise A2EError(f"A2E quote unavailable for {path}: {reason}. No generation was submitted.")
        raw_coins = result.get("coins")
        try:
            if isinstance(raw_coins, bool) or raw_coins is None:
                raise InvalidOperation
            coins = Decimal(str(raw_coins))
            if not coins.is_finite() or coins < 0:
                raise InvalidOperation
        except (InvalidOperation, ValueError) as exc:
            raise A2EError(f"A2E quote for {path} did not contain a valid credit estimate; no generation was submitted.") from exc
        total = await self.budget.reserve(coins)
        await self._emit({
            "type": "quote",
            "message": f"A2E estimate: {coins} credits; job estimate so far: {total} credits.",
            "endpoint": path,
            "estimated_credits": float(coins),
            "estimated_total": float(total),
            "estimated_total_exact": str(total),
        })

    async def _submit_and_poll(self, path: str, detail_root: str, body: dict,
                               *, result_kind: str) -> str:
        async with self.semaphore:
            await self._quote_and_reserve(path, body)
            submitted = await self._request("POST", path, body=body)
            data = submitted.get("data")
            # Text-to-image returns an array of task records; I2V returns an object.
            if isinstance(data, list):
                record = data[0] if len(data) == 1 else None
            else:
                record = data
            task_id = record.get("_id") if isinstance(record, dict) else None
            if not isinstance(task_id, str) or not task_id.strip():
                msg = record.get("message") if isinstance(record, dict) else None
                trace = submitted.get("trace_id")
                raise A2EError(
                    f"A2E did not return a task ID for {path}"
                    + (f": {msg}" if msg else " (the request may have been moderated or rejected)")
                    + (f"; trace_id={trace}" if trace else "")
                    + ". No polling or automatic resubmission was performed. Check A2E task/billing records before retrying."
                )
            await self._emit({"type": "task", "task_id": task_id, "endpoint": path,
                              "message": f"A2E {result_kind} task submitted: {task_id}"})
            return await self._poll_result(detail_root, task_id, result_kind)

    async def _poll_result(self, detail_root: str, task_id: str, result_kind: str) -> str:
        record = await self._poll(f"{detail_root}/{quote(task_id, safe='')}", task_id)
        if record.get("isExpired"):
            raise A2EError(f"A2E {result_kind} task {task_id} finished but its output has expired. "
                           "Check A2E history before deciding whether to pay for a replacement.",
                           terminal=True, output_expired=True)
        if result_kind == "image":
            for key in ("result_image_urls", "result_image_url", "image_urls", "image_url"):
                value = record.get(key)
                if isinstance(value, list):
                    value = next((url for url in value if isinstance(url, str) and url.startswith(("https://", "http://"))), None)
                if isinstance(value, str) and value.startswith(("https://", "http://")):
                    return value
        else:
            for key in ("result_url", "result_video_url", "video_url"):
                value = record.get(key)
                if isinstance(value, str) and value.startswith(("https://", "http://")):
                    return value
        raise A2EError(f"A2E {result_kind} task {task_id} completed without a usable output URL.")

    async def _resume_task(self, detail_root: str, task_id: str, result_kind: str) -> str:
        if not isinstance(task_id, str) or not task_id.strip():
            raise A2EError("Cannot recover an A2E task without its task ID.")
        async with self.semaphore:
            await self._emit({"type": "recovery", "task_id": task_id,
                              "message": f"Checking saved A2E {result_kind} task {task_id}; no new task was submitted."})
            return await self._poll_result(detail_root, task_id, result_kind)

    async def _poll(self, path: str, task_id: str) -> dict:
        deadline = time.monotonic() + self.poll_timeout
        delay = self.interval
        retry_not_before = 0.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise A2EError(
                    f"Timed out waiting for A2E task {task_id}; it may still be running or billed. "
                    "Check its detail and billing records before resubmitting."
                )
            # Jitter prevents simultaneous tasks from polling in lockstep, while
            # an explicit Retry-After is always a minimum, never jittered shorter.
            wait = max(delay * random.uniform(0.85, 1.15), retry_not_before - time.monotonic())
            await asyncio.sleep(min(remaining, wait))
            retry_not_before = 0.0
            if time.monotonic() >= deadline:
                continue
            try:
                envelope = await self._request("GET", path)
            except A2EError as exc:
                if not exc.retryable:
                    raise
                # Honor Retry-After when present; otherwise use bounded exponential backoff.
                delay = min(30.0, max(self.interval, delay * 1.5, 0.1))
                if exc.retry_after is not None:
                    retry_not_before = time.monotonic() + exc.retry_after
                continue
            record = envelope.get("data")
            if not isinstance(record, dict):
                raise A2EError(f"Invalid detail response for A2E task {task_id}.")
            status_value = record.get("current_status")
            if not isinstance(status_value, str) or not status_value:
                raise A2EError(f"A2E detail response for task {task_id} did not include current_status.")
            status = status_value.lower()
            if status == "completed":
                return record
            if status in ("failed", "blocked", "block", "canceled", "cancelled"):
                reason = record.get("failed_message") or record.get("failed_reason") or "no reason supplied"
                code = record.get("failed_code")
                refund = ("refund pending" if record.get("refund_pending") is True else
                          "refund flag true" if record.get("hasRefundCoin") is True else
                          "refund not confirmed")
                trace = envelope.get("trace_id")
                raise A2EError(
                    f"A2E task {task_id} {status} (code={code or 'unknown'}): {reason}; {refund}"
                    + (f"; trace_id={trace}" if trace else ""), terminal=True,
                )
            delay = min(30.0, max(self.interval, delay * 1.5))

    async def generate_image(self, prompt: str, *, aspect_ratio: str = "16:9",
                             reference_urls: Optional[list[str]] = None,
                             name: str = "MicroDrama frame", resume_task_id: Optional[str] = None) -> str:
        path, detail_root, body = build_image_request(
            self.image_spec, prompt, aspect_ratio, reference_urls or [], name,
        )
        if resume_task_id:
            return await self._resume_task(detail_root, resume_task_id, "image")
        return await self._submit_and_poll(path, detail_root, body, result_kind="image")

    async def generate_video(self, prompt: str, image_url: str, *, duration: Optional[int] = None,
                             name: str = "MicroDrama shot", resume_task_id: Optional[str] = None) -> str:
        if resume_task_id:
            # Derive the documented detail route without requiring the old
            # input image. This dummy URL is used only to build the local route;
            # polling never sends the body or creates a new task.
            _, detail_root, _ = build_video_request(
                self.video_spec, "", "https://recovery.invalid/unused", self.video_spec.default_duration,
                "Recovery", native_audio=self.auto_add_audio,
            )
            return await self._resume_task(detail_root, resume_task_id, "video")
        if duration is None:
            duration = self.video_spec.default_duration
        path, detail_root, body = build_video_request(
            self.video_spec, prompt, image_url, duration, name, native_audio=self.auto_add_audio,
        )
        return await self._submit_and_poll(path, detail_root, body, result_kind="video")

    async def upload_image(self, content: bytes, content_type: str) -> str:
        """Upload a Gemini frame; presigning itself is billed for API-token users.

        Never send the bearer token to the storage host and never log a signed URL.
        The non-idempotent presign and the PUT are not automatically retried.
        """
        types = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
        if content_type not in types or not content or len(content) > 20 * 1024 * 1024:
            raise A2EError("Gemini returned an unsupported or oversized image for A2E upload.")
        async with self.semaphore:
            # The public pricing endpoint is the documented price source for the
            # presign fee; this is not a generation/quote endpoint.
            prices = await self._request("GET", "/v1/feature-pricing")
            data = prices.get("data")
            price_list = data.get("prices") if isinstance(data, dict) else None
            price_entry = price_list.get("r2") if isinstance(price_list, dict) else None
            raw = price_entry.get("credits") if isinstance(price_entry, dict) else None
            try:
                if isinstance(raw, bool) or raw is None:
                    raise InvalidOperation
                cost = Decimal(str(raw))
                if not cost.is_finite() or cost < 0:
                    raise InvalidOperation
            except (InvalidOperation, ValueError) as exc:
                raise A2EError("A2E did not provide a valid R2 upload-signing price; no upload was requested.") from exc
            total = await self.budget.reserve(cost)
            await self._emit({"type": "quote", "endpoint": "/v1/r2/upload-presigned-url",
                              "estimated_credits": float(cost), "estimated_total": float(total),
                              "estimated_total_exact": str(total),
                              "message": f"A2E upload-signing estimate: {cost} credits; job estimate so far: {total} credits."})
            signed = await self._request("POST", "/v1/r2/upload-presigned-url", body={
                "key": f"microdrama-{uuid.uuid4().hex}.{types[content_type]}",
                "purpose": "STAGING", "contentType": content_type, "fileSize": len(content),
            })
            result = signed.get("data")
            upload_url = result.get("uploadUrl") if isinstance(result, dict) else None
            cdn_url = result.get("cdnUrl") if isinstance(result, dict) else None
            for value in (upload_url, cdn_url):
                if not isinstance(value, str):
                    raise A2EError("A2E returned an invalid upload destination. A signing fee may already have been charged.")
                parsed = urlsplit(value)
                if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                    raise A2EError("A2E returned an invalid upload destination. A signing fee may already have been charged.")
            await self._emit({"type": "upload", "message": "Uploading a Gemini frame to A2E staging storage."})
            try:
                # This separate client has no A2E Authorization header. The
                # signature in uploadUrl is the only upload credential.
                async with httpx.AsyncClient(transport=self.upload_transport, follow_redirects=False,
                                             timeout=httpx.Timeout(120.0, connect=10.0)) as uploader:
                    response = await uploader.put(upload_url, content=content,
                                                  headers={"Content-Type": content_type})
            except httpx.RequestError as exc:
                raise A2EError("Uploading the Gemini frame failed. The A2E signing fee may already have been charged; no automatic retry was made.") from exc
            if not 200 <= response.status_code < 300:
                raise A2EError(f"Uploading the Gemini frame failed (HTTP {response.status_code}). The A2E signing fee may already have been charged; no automatic retry was made.")
            return cdn_url
