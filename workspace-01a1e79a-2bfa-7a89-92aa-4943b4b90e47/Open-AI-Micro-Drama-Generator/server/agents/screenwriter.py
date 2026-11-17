from typing import List, Optional

from tools.llm import TextLLM, parse_json_object
from utils.limits import pipeline_limit


SCENES_SCHEMA = {
    "type": "object", "properties": {"scenes": {"type": "array", "items": {
        "type": "object", "properties": {"script": {"type": "string"}}, "required": ["script"],
    }}}, "required": ["scenes"],
}


class Screenwriter:
    def __init__(self, llm: Optional[TextLLM] = None):
        self.llm = llm

    async def develop_story(self, idea: str, user_requirement: str) -> str:
        # A2E does not expose a chat model in the supplied API spec. Without an
        # external LLM, use the actual idea; never substitute a canned story.
        if self.llm is None:
            return idea.strip()
        return await self.llm.complete(
            f"Develop this idea into a short, visually filmable story outline. "
            f"Include a protagonist, conflict, climax and resolution.\n\n"
            f"Idea: {idea}\n\nAdditional requirements: {user_requirement or 'None'}",
            system_prompt="You are a professional screenwriter. Write concise, visually descriptive prose.",
        )

    async def write_script_based_on_story(self, story: str, user_requirement: str) -> List[str]:
        if self.llm is None:
            return [story]
        limit = pipeline_limit("MAX_SCENES", 2, 1, 4)
        raw = await self.llm.complete(
            f"Turn this story into at most {limit} filmable scene scripts. Include actions "
            f"and setting, and obey the additional requirements.\n\nStory: {story}\n\n"
            f"Additional requirements: {user_requirement or 'None'}\n\n"
            'Return only JSON: {"scenes": [{"script": "full scene script"}]}',
            system_prompt="You are a professional screenwriter. Return valid JSON only.",
            response_schema=SCENES_SCHEMA,
        )
        data = parse_json_object(raw, "scene")
        scenes = data.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise RuntimeError("LLM returned no scene scripts")
        result = []
        for scene in scenes[:limit]:
            script = scene.get("script") if isinstance(scene, dict) else None
            if not isinstance(script, str) or not script.strip():
                raise RuntimeError("LLM returned an invalid scene script")
            result.append(script.strip())
        return result
