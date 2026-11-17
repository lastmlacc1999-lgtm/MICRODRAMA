"""Text-planning interface and optional OpenAI-compatible implementation.

The supplied A2E spec does not provide a general chat-completions endpoint.
"""
import json
import os
import re
from typing import Any, Optional, Protocol

import httpx


class TextLLM(Protocol):
    async def complete(self, prompt: str, *, system_prompt: Optional[str] = None,
                       response_schema: Optional[dict[str, Any]] = None, timeout: int = 120) -> str:
        ...


class OpenAICompatibleLLM:
    def __init__(self, api_key: str, base_url: str, model: str,
                 *, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.transport = transport

    @classmethod
    def from_env(cls) -> Optional["OpenAICompatibleLLM"]:
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
        if not base_url or not model:
            raise ValueError("LLM_BASE_URL and LLM_MODEL must not be empty when LLM_API_KEY is set")
        return cls(api_key, base_url, model)

    async def complete(self, prompt: str, *, system_prompt: Optional[str] = None,
                       response_schema: Optional[dict[str, Any]] = None, timeout: int = 120) -> str:
        # Compatibility providers vary in their structured-output support. The
        # agents still request JSON in the prompt and validate it themselves.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport,
                                     follow_redirects=False) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages},
                )
            except httpx.RequestError as exc:
                raise RuntimeError(f"LLM request failed: {type(exc).__name__}") from exc
        if response.is_error:
            # Do not reflect provider response bodies, which can contain sensitive data.
            raise RuntimeError(f"LLM request failed with HTTP {response.status_code}; check LLM configuration and access.")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM returned an invalid chat-completions response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty completion")
        return content.strip()


def parse_json_object(raw: str, kind: str) -> dict:
    """Accept JSON or one fenced JSON block, and fail rather than invent content."""
    text = raw.strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"LLM returned invalid {kind} JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"LLM returned invalid {kind} JSON: expected an object")
    return value
