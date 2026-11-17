import copy
import json

import httpx
import pytest

from agents.character_extractor import CHARACTERS_SCHEMA, CharacterExtractor
from agents.screenwriter import SCENES_SCHEMA, Screenwriter
from agents.storyboard_artist import SHOTS_SCHEMA, StoryboardArtist
from tools.model_catalog import OPENAI_TEXT_MODELS
from tools.openai_text import OpenAITextError, OpenAITextLLM, _strict_schema


def reply(text):
    return {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
    ]}


def assert_all_objects_are_closed(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["required"]) == set(schema["properties"])
        for value in schema.values():
            assert_all_objects_are_closed(value)
    elif isinstance(schema, list):
        for value in schema:
            assert_all_objects_are_closed(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", OPENAI_TEXT_MODELS, ids=lambda item: item.id)
async def test_each_gpt_model_uses_stateless_responses_with_strict_schema(model):
    calls = []
    schema = {"type": "object", "properties": {"answers": {"type": "array", "items": {
        "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]
    }}}, "required": ["answers"]}
    original = copy.deepcopy(schema)

    def handler(request):
        calls.append(request)
        assert request.method == "POST" and str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer fake-openai-private"
        assert "x-goog-api-key" not in request.headers
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "reasoning", "summary": [{"text": "not part of the answer"}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": '{"answers":'},
                {"type": "output_text", "text": '[{"answer":"yes"}]}'},
            ]},
        ]})

    llm = OpenAITextLLM("fake-openai-private", model.id, transport=httpx.MockTransport(handler))
    result = await llm.complete("Question", system_prompt="System", response_schema=schema)
    assert json.loads(result) == {"answers": [{"answer": "yes"}]}
    assert len(calls) == 1 and schema == original
    body = json.loads(calls[0].content)
    assert body["model"] == model.id and body["input"] == "Question" and body["instructions"] == "System"
    assert body["store"] is False and body["max_output_tokens"] == 8192
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["name"] == "microdrama_plan" and body["text"]["format"]["strict"] is True
    assert_all_objects_are_closed(body["text"]["format"]["schema"])
    assert "additionalProperties" not in schema
    assert "fake-openai-private" not in calls[0].content.decode() + str(calls[0].url)


@pytest.mark.asyncio
async def test_gpt_drives_all_agents_without_modifying_gemini_schemas(monkeypatch):
    monkeypatch.setenv("MAX_SCENES", "2")
    monkeypatch.setenv("MAX_CHARACTERS", "2")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    original_schemas = copy.deepcopy([SCENES_SCHEMA, CHARACTERS_SCHEMA, SHOTS_SCHEMA])
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        prompt = body["input"]
        if prompt.startswith("Develop this idea"):
            assert "text" not in body
            text = "A lonely otter discovers a violin and shares music."
        elif prompt.startswith("Turn this story"):
            assert body["text"]["format"]["schema"] == _strict_schema(SCENES_SCHEMA)
            text = json.dumps({"scenes": [{"script": "Mira the otter finds a violin."}]})
        elif prompt.startswith("Extract at most"):
            assert body["text"]["format"]["schema"] == _strict_schema(CHARACTERS_SCHEMA)
            text = json.dumps({"characters": [{"idx": 0, "name": "Mira", "static_features": "brown fur",
                                               "dynamic_features": "red scarf", "is_visible": True}]})
        elif prompt.startswith("Break this scene"):
            assert body["text"]["format"]["schema"] == _strict_schema(SHOTS_SCHEMA)
            assert "Mira: brown fur" in prompt and "7 seconds" in prompt
            text = json.dumps({"shots": [{"idx": 0, "visual_desc": "Mira finds a violin.",
                                         "motion_desc": "Camera moves closer.", "audio_desc": ""}]})
        else:
            raise AssertionError("Unexpected planning prompt")
        return httpx.Response(200, json=reply(text))

    llm = OpenAITextLLM("fake-key", "gpt-6.1-terra", transport=httpx.MockTransport(handler))
    writer = Screenwriter(llm)
    story = await writer.develop_story("An otter finds a violin", "hopeful")
    scenes = await writer.write_script_based_on_story(story, "hopeful")
    characters = await CharacterExtractor(llm).extract_characters(story)
    shots = await StoryboardArtist(llm, 7).design_storyboard(scenes[0], characters, "hopeful")
    assert "music" in story and scenes == ["Mira the otter finds a violin."]
    assert characters[0].name == "Mira" and shots[0].visual_desc == "Mira finds a violin."
    assert len(calls) == 4 and all(item["model"] == "gpt-6.1-terra" and not item["store"] for item in calls)
    assert all(item.get("instructions") for item in calls)
    assert [SCENES_SCHEMA, CHARACTERS_SCHEMA, SHOTS_SCHEMA] == original_schemas


@pytest.mark.asyncio
async def test_plain_completion_needs_no_schema_or_instructions():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=reply("  A short story.  "))
    llm = OpenAITextLLM("fake-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    assert await llm.complete("Make a story") == "A short story."
    assert "text" not in calls[0] and "instructions" not in calls[0]
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 400, 401, 403, 404, 429, 500, 503])
async def test_gpt_http_error_is_safe_and_never_retried(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "fake-private-key and secret prompt"}},
                              headers={"Location": "https://should-not-be-followed.test/"})
    llm = OpenAITextLLM("fake-private-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAITextError) as raised:
        await llm.complete("secret prompt")
    error = str(raised.value)
    assert len(calls) == 1 and f"HTTP {status}" in error and "no automatic retry" in error
    assert "fake-private-key" not in error and "secret prompt" not in error


@pytest.mark.asyncio
async def test_gpt_timeout_warns_of_possible_charge_without_retry():
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("fake-private-key, secret prompt", request=request)
    llm = OpenAITextLLM("fake-private-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAITextError, match="may have been billed.*no automatic retry") as raised:
        await llm.complete("secret prompt")
    assert len(calls) == 1 and "fake-private-key" not in str(raised.value) and "secret prompt" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    [],
    {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
    {"status": "failed", "error": {"message": "fake-private-key"}, "output": []},
    {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial"}]}]},
    {"status": "completed", "output": []},
    {"status": "completed", "output": [{"type": "reasoning"}]},
    {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": "not a list"}]},
    {"status": "completed", "output": [{"type": "message", "role": "user", "content": [{"type": "output_text", "text": "not assistant output"}]}]},
    {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}]},
    {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": "partial"}, {"type": "refusal", "refusal": "fake-private-key"}]}]},
])
async def test_gpt_incomplete_refused_or_malformed_response_stops_without_fallback(payload):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)
    llm = OpenAITextLLM("fake-private-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAITextError) as raised:
        await llm.complete("prompt")
    assert len(calls) == 1 and "fake-private-key" not in str(raised.value)
    assert "no automatic retry" in str(raised.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["not json with fake-private-key", "[]", "null", "123"])
async def test_gpt_malformed_structured_output_fails_before_agents_and_is_safe(text):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=reply(text))
    llm = OpenAITextLLM("fake-private-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAITextError, match="invalid structured planning JSON") as raised:
        await Screenwriter(llm).write_script_based_on_story("story", "")
    assert len(calls) == 1 and "fake-private-key" not in str(raised.value)


@pytest.mark.asyncio
async def test_gpt_valid_json_without_required_business_content_does_not_fallback():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=reply('{"scenes":[]}'))
    llm = OpenAITextLLM("fake-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="no scene scripts"):
        await Screenwriter(llm).write_script_based_on_story("story", "")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_invalid_http_json_and_invalid_schema_never_retry():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="not JSON fake-private-key")
    llm = OpenAITextLLM("fake-private-key", "gpt-6-luna", transport=httpx.MockTransport(handler))
    with pytest.raises(OpenAITextError, match="invalid planning response") as raised:
        await llm.complete("prompt")
    assert "fake-private-key" not in str(raised.value) and len(calls) == 1
    invalid = {"type": "object", "properties": {"optional": {"type": "string"}}, "required": []}
    with pytest.raises(ValueError, match="must require every property"):
        await llm.complete("prompt", response_schema=invalid)
    assert len(calls) == 1 and "additionalProperties" not in invalid


def test_gpt_requires_key_and_known_model():
    with pytest.raises(OpenAITextError, match="key is missing"):
        OpenAITextLLM("", "gpt-6-luna")
    with pytest.raises(ValueError, match="Unknown OpenAI story-planning model"):
        OpenAITextLLM("fake-key", "unknown-model")
