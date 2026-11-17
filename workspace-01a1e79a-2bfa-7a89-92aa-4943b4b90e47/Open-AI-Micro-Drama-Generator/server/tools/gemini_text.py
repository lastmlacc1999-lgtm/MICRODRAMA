"""Google Gemini Interactions text client for story and shot planning.

Each generation request is sent once. A timeout can still be billed; the app
never silently retries or switches to another provider after an error.
"""
from typing import Any, Optional

import httpx

from tools.model_catalog import gemini_text_model


class GeminiTextError(RuntimeError):
    """Safe-to-display planning error that does not expose provider content."""


class GeminiTextLLM:
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
    MAX_OUTPUT_TOKENS = 8192

    def __init__(self, api_key: str, model: str, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        if not api_key.strip():
            raise GeminiTextError("Gemini API key is missing. Save it in the website's API key settings.")
        gemini_text_model(model)
        self.api_key = api_key.strip()
        self.model = model
        self.transport = transport

    async def complete(self, prompt: str, *, system_prompt: Optional[str] = None,
                       response_schema: Optional[dict[str, Any]] = None, timeout: int = 120) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "generation_config": {"max_output_tokens": self.MAX_OUTPUT_TOKENS},
        }
        if system_prompt:
            body["system_instruction"] = system_prompt
        if response_schema is not None:
            body["response_format"] = {"type": "text", "mime_type": "application/json", "schema": response_schema}
        try:
            async with httpx.AsyncClient(
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json",
                         "Api-Revision": "2026-05-20"},
                timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.post(self.ENDPOINT, json=body)
        except httpx.RequestError as exc:
            raise GeminiTextError(
                "Gemini story-planning request lost its response and may have been billed. "
                "Check Google billing before starting again; no automatic retry was made."
            ) from exc
        if not response.is_success:
            # Provider response bodies may contain prompts or credentials.
            if response.status_code >= 500:
                detail = "Gemini's service is temporarily unavailable; it may have billed this request."
            elif response.status_code == 429:
                detail = "Gemini's rate or quota limit was reached; check the project's limits and billing."
            else:
                detail = "Check the Gemini key, selected text model access and Google billing."
            raise GeminiTextError(
                f"Gemini story-planning request failed (HTTP {response.status_code}). "
                f"{detail} There was no automatic retry."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiTextError("Gemini returned invalid planning JSON; the request may have been billed.") from exc
        if not isinstance(payload, dict) or payload.get("status") not in (None, "completed"):
            raise GeminiTextError("Gemini did not complete the planning request; it may have been billed. No automatic retry was made.")
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise GeminiTextError("Gemini returned no planning output; the request may have been billed.")
        output = next((step for step in reversed(steps)
                       if isinstance(step, dict) and step.get("type") == "model_output"), None)
        content = output.get("content") if isinstance(output, dict) else None
        blocks = content if isinstance(content, list) else []
        text = "".join(block.get("text", "") for block in blocks
                       if isinstance(block, dict) and block.get("type") == "text"
                       and isinstance(block.get("text"), str)).strip()
        if not text:
            raise GeminiTextError("Gemini returned no planning text (possibly due to a safety filter); the request may have been billed.")
        return text
