import json

import httpx
import pytest

from agents.character_extractor import CharacterExtractor
from agents.screenwriter import Screenwriter
from agents.storyboard_artist import StoryboardArtist
from tools.llm import OpenAICompatibleLLM, parse_json_object


@pytest.mark.asyncio
async def test_a2e_only_planner_uses_user_input_without_canned_fallback(monkeypatch):
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "3")
    idea = "An otter finds a forgotten violin in a rainy train station."
    writer = Screenwriter()
    assert await writer.develop_story(idea, "") == idea
    assert await writer.write_script_based_on_story(idea, "") == [idea]
    assert await CharacterExtractor().extract_characters(idea) == []
    shots = await StoryboardArtist().design_storyboard(idea, [], "watercolor")
    assert len(shots) == 3
    assert all(idea in s.visual_desc and "watercolor" in s.visual_desc for s in shots)
    assert all("arctic" not in s.visual_desc.lower() for s in shots)


@pytest.mark.asyncio
async def test_optional_llm_uses_chat_completions_and_agents():
    seen = []

    def handler(req):
        seen.append(req)
        assert req.url.path == "/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer llm-key"
        content = json.dumps({"scenes": [{"script": "An otter plays violin."}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = OpenAICompatibleLLM("llm-key", "https://llm.test/v1", "model-x", transport=httpx.MockTransport(handler))
    result = await Screenwriter(llm).write_script_based_on_story("An otter", "")
    assert result == ["An otter plays violin."]
    payload = json.loads(seen[0].content)
    assert payload["model"] == "model-x" and len(payload["messages"]) == 2


def test_llm_json_parser_and_no_silent_fallback():
    assert parse_json_object('```json\n{"shots": []}\n```', "storyboard") == {"shots": []}
    with pytest.raises(RuntimeError, match="invalid storyboard JSON"):
        parse_json_object("not JSON", "storyboard")


@pytest.mark.asyncio
async def test_openai_compatible_planner_still_drives_all_agents():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.headers["authorization"] == "Bearer fake-key"
        assert "response_format" not in payload  # remain compatible with older endpoints
        prompt = payload["messages"][-1]["content"]
        if prompt.startswith("Develop this idea"):
            text = "A heron finds its way home."
        elif prompt.startswith("Turn this story"):
            text = json.dumps({"scenes": [{"script": "A heron crosses a river."}]})
        elif prompt.startswith("Extract at most"):
            text = json.dumps({"characters": [{"idx": 0, "name": "Heron", "static_features": "gray feathers",
                                              "dynamic_features": "none", "is_visible": True}]})
        elif prompt.startswith("Break this scene"):
            text = json.dumps({"shots": [{"idx": 0, "visual_desc": "Heron crosses a river",
                                         "motion_desc": "slow pan", "audio_desc": ""}]})
        else:
            raise AssertionError("Unexpected prompt")
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    llm = OpenAICompatibleLLM("fake-key", "https://llm.test/v1", "model-x", transport=httpx.MockTransport(handler))
    writer = Screenwriter(llm)
    story = await writer.develop_story("a heron", "")
    characters = await CharacterExtractor(llm).extract_characters(story)
    scenes = await writer.write_script_based_on_story(story, "")
    shots = await StoryboardArtist(llm).design_storyboard(scenes[0], characters, "")
    assert story == "A heron finds its way home."
    assert len(characters) == len(scenes) == len(shots) == 1
    assert len(calls) == 4 and all(item["model"] == "model-x" for item in calls)
