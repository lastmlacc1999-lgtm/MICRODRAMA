"""OpenAI Responses API client for optional story and shot planning.

The GPT API has its own key and fixed endpoint; it never changes the legacy
OpenAI-compatible chat-completions client. Each request is sent only once.
"""
import copy
import json
from typing import Any, Optional

import httpx

from tools.model_catalog import openai_text_model


class OpenAITextError(RuntimeError):
    """Safe-to-display GPT planning error with no upstream content or key."""


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make the agents' all-required schemas compatible with strict output.

    This is a copy: Gemini and other clients must keep their original schemas.
    Refuse unsupported optional properties rather than silently changing their
    meaning. All current agent schemas have every object property required.
    """
    result = copy.deepcopy(schema)
    if not isinstance(result, dict) or result.get("type") != "object":
        raise ValueError("OpenAI planning schemas must have an object root; no request was sent.")

    def close_objects(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                props = value.get("properties")
                required = value.get("required")
                if (not isinstance(props, dict) or not isinstance(required, list) or
                        set(required) != set(props)):
                    raise ValueError("OpenAI planning schemas must require every property; no request was sent.")
                value["additionalProperties"] = False
            for child in value.values():
                close_objects(child)
        elif isinstance(value, list):
            for child in value:
                close_objects(child)

    close_objects(result)
    return result


class OpenAITextLLM:
    ENDPOINT = "https://api.openai.com/v1/responses"
    MAX_OUTPUT_TOKENS = 8192

    def __init__(self, api_key: str, model: str, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        if not api_key.strip():
            raise OpenAITextError("OpenAI API key is missing. Save it in the website's API key settings.")
        openai_text_model(model)
        self.api_key = api_key.strip()
        self.model = model
        self.transport = transport

    async def complete(self, prompt: str, *, system_prompt: Optional[str] = None,
                       response_schema: Optional[dict[str, Any]] = None, timeout: int = 120) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
            "max_output_tokens": self.MAX_OUTPUT_TOKENS,
        }
        if system_prompt:
            body["instructions"] = system_prompt
        if response_schema is not None:
            body["text"] = {"format": {
                "type": "json_schema", "name": "microdrama_plan", "strict": True,
                "schema": _strict_schema(response_schema),
            }}
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = await client.post(self.ENDPOINT, json=body)
        except httpx.RequestError as exc:
            raise OpenAITextError(
                "OpenAI story-planning request lost its response and may have been billed. "
                "Check OpenAI billing before starting again; no automatic retry was made."
            ) from exc
        if not response.is_success:
            # Never echo an upstream body: it can contain a key or prompt.
            if response.status_code >= 500:
                detail = "OpenAI's service is temporarily unavailable; it may have billed this request."
            elif response.status_code == 429:
                detail = "OpenAI's rate, quota or billing limit was reached."
            else:
                detail = "Check the OpenAI API key, selected model access and API billing."
            raise OpenAITextError(
                f"OpenAI story-planning request failed (HTTP {response.status_code}). "
                f"{detail} There was no automatic retry."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenAITextError("OpenAI returned an invalid planning response; the request may have been billed. No automatic retry was made.") from exc
        if not isinstance(payload, dict) or payload.get("status") != "completed":
            raise OpenAITextError("OpenAI did not complete the planning request; it may have been billed. No automatic retry was made.")
        output = payload.get("output")
        if not isinstance(output, list):
            raise OpenAITextError("OpenAI returned no planning output; the request may have been billed. No automatic retry was made.")
        blocks = []
        for item in output:
            if (isinstance(item, dict) and item.get("type") == "message" and
                    item.get("role") == "assistant" and isinstance(item.get("content"), list)):
                blocks.extend(item["content"])
        if any(isinstance(block, dict) and block.get("type") == "refusal" for block in blocks):
            raise OpenAITextError("OpenAI declined the planning request; it may have been billed. No automatic retry was made.")
        text = "".join(block.get("text", "") for block in blocks
                       if isinstance(block, dict) and block.get("type") == "output_text"
                       and isinstance(block.get("text"), str)).strip()
        if not text:
            raise OpenAITextError("OpenAI returned no planning text; the request may have been billed. No automatic retry was made.")
        if response_schema is not None:
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError) as exc:
                raise OpenAITextError("OpenAI returned invalid structured planning JSON; the request may have been billed. No automatic retry was made.") from exc
            if not isinstance(parsed, dict):
                raise OpenAITextError("OpenAI returned invalid structured planning JSON; the request may have been billed. No automatic retry was made.")
        return text
