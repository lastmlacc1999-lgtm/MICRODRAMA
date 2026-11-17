import json

import httpx
import pytest

from agents.character_extractor import CHARACTERS_SCHEMA, CharacterExtractor
from agents.screenwriter import SCENES_SCHEMA, Screenwriter
from agents.storyboard_artist import SHOTS_SCHEMA, StoryboardArtist
from tools.gemini_text import GeminiTextError, GeminiTextLLM
from tools.model_catalog import GEMINI_TEXT_MODELS


def reply(text):
    return {"status": "completed", "steps": [
        {"type": "thought", "signature": "ignored"},
        {"type": "model_output", "content": [{"type": "text", "text": text}]},
    ]}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", GEMINI_TEXT_MODELS, ids=lambda item: item.id)
async def test_gemini_text_catalog_models_use_native_stateless_interactions(model):
    calls = []
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

    def handler(request):
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert request.headers["x-goog-api-key"] == "fake-gemini-key"
        assert request.headers["api-revision"] == "2026-05-20"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"status": "completed", "steps": [
            {"type": "model_output", "content": [{"type": "text", "text": "old"}]},
            {"type": "thought", "signature": "never included in text"},
            {"type": "model_output", "content": [
                {"type": "text", "text": '{"answer":'}, {"type": "text", "text": '"yes"}'},
            ]},
        ]})

    client = GeminiTextLLM("fake-gemini-key", model.id, transport=httpx.MockTransport(handler))
    result = await client.complete("Question", system_prompt="System", response_schema=schema)
    assert json.loads(result) == {"answer": "yes"}
    assert len(calls) == 1
    payload = json.loads(calls[0].content)
    assert payload == {
        "model": model.id,
        "input": "Question",
        "system_instruction": "System",
        "store": False,
        "generation_config": {"max_output_tokens": 8192},
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
    }
    assert "fake-gemini-key" not in str(calls[0].url) and "fake-gemini-key" not in calls[0].content.decode()


@pytest.mark.asyncio
async def test_gemini_drives_story_scene_character_and_shot_agents_with_schemas(monkeypatch):
    monkeypatch.setenv("MAX_SCENES", "2")
    monkeypatch.setenv("MAX_CHARACTERS", "2")
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "2")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        prompt = payload["input"]
        if prompt.startswith("Develop this idea"):
            assert "response_format" not in payload
            text = "A lonely otter finds music and shares it with friends."
        elif prompt.startswith("Turn this story"):
            assert payload["response_format"]["schema"] == SCENES_SCHEMA
            text = json.dumps({"scenes": [{"script": "Mira the otter finds a violin."}]})
        elif prompt.startswith("Extract at most"):
            assert payload["response_format"]["schema"] == CHARACTERS_SCHEMA
            text = json.dumps({"characters": [{"idx": 0, "name": "Mira", "static_features": "brown fur",
                                               "dynamic_features": "red scarf", "is_visible": True}]})
        elif prompt.startswith("Break this scene"):
            assert payload["response_format"]["schema"] == SHOTS_SCHEMA
            assert "Mira: brown fur" in prompt and "7 seconds" in prompt
            text = json.dumps({"shots": [{"idx": 0, "visual_desc": "Mira finds a violin.",
                                         "motion_desc": "Camera moves closer.", "audio_desc": ""}]})
        else:
            raise AssertionError("Unexpected Gemini planning prompt")
        return httpx.Response(200, json=reply(text))

    llm = GeminiTextLLM("fake-gemini-key", "gemini-3.8-flash", transport=httpx.MockTransport(handler))
    writer = Screenwriter(llm)
    story = await writer.develop_story("An otter finds a violin", "hopeful")
    assert "music" in story
    scenes = await writer.write_script_based_on_story(story, "hopeful")
    characters = await CharacterExtractor(llm).extract_characters(story)
    shots = await StoryboardArtist(llm, 7).design_storyboard(scenes[0], characters, "hopeful")
    assert scenes == ["Mira the otter finds a violin."]
    assert len(characters) == 1 and characters[0].name == "Mira"
    assert len(shots) == 1 and shots[0].visual_desc == "Mira finds a violin."
    assert len(calls) == 4 and all(item["store"] is False for item in calls)
    assert all(item["model"] == "gemini-3.8-flash" for item in calls)
    assert "An otter finds a violin" in calls[0]["input"] and "hopeful" in calls[1]["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403, 429, 503])
async def test_gemini_text_http_failure_is_safe_and_never_retried(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"error": {"message": "fake-private-key and prompt"}})

    llm = GeminiTextLLM("fake-private-key", "gemini-3.8-flash", transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiTextError) as raised:
        await llm.complete("sensitive prompt")
    error = str(raised.value)
    assert len(seen) == 1 and f"HTTP {status}" in error and "no automatic retry" in error
    assert "fake-private-key" not in error and "sensitive prompt" not in error


@pytest.mark.asyncio
async def test_gemini_text_timeout_warns_of_possible_charge_without_retry():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private provider details", request=request)

    llm = GeminiTextLLM("fake-key", "gemini-3.8-flash", transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiTextError, match="may have been billed.*no automatic retry"):
        await llm.complete("prompt")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    [],
    {"status": "failed", "steps": []},
    {"status": "completed", "steps": []},
    {"status": "completed", "steps": [{"type": "thought"}]},
    {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": ""}]}]},
    {"status": "completed", "steps": [{"type": "model_output", "content": "not-a-list"}]},
])
async def test_gemini_text_invalid_responses_stop_without_fallback(payload):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    llm = GeminiTextLLM("fake-key", "gemini-3.8-flash", transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiTextError):
        await llm.complete("prompt")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gemini_malformed_structured_output_does_not_get_fallback():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=reply("this is not JSON"))

    llm = GeminiTextLLM("fake-key", "gemini-3.8-flash", transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="invalid scene JSON"):
        await Screenwriter(llm).write_script_based_on_story("story", "")
    assert len(calls) == 1


def test_gemini_requires_key_and_known_model():
    with pytest.raises(GeminiTextError, match="key is missing"):
        GeminiTextLLM("", "gemini-3.8-flash")
    with pytest.raises(ValueError, match="Unknown Gemini story-planning model"):
        GeminiTextLLM("fake-key", "unknown-model")
