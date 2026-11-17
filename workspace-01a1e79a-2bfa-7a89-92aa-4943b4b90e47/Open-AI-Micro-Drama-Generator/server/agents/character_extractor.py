from typing import List, Optional

from interfaces.character import CharacterInScene
from tools.llm import TextLLM, parse_json_object
from utils.limits import pipeline_limit


CHARACTERS_SCHEMA = {
    "type": "object", "properties": {"characters": {"type": "array", "items": {
        "type": "object", "properties": {
            "idx": {"type": "integer"}, "name": {"type": "string"},
            "static_features": {"type": "string"}, "dynamic_features": {"type": "string"},
            "is_visible": {"type": "boolean"},
        }, "required": ["idx", "name", "static_features", "dynamic_features", "is_visible"],
    }}}, "required": ["characters"],
}


class CharacterExtractor:
    def __init__(self, llm: Optional[TextLLM] = None):
        self.llm = llm

    async def extract_characters(self, script: str) -> List[CharacterInScene]:
        # No speculative character extraction in simple-planning mode. The story text
        # and prior frames still serve as visual context for generated shots.
        if self.llm is None:
            return []
        limit = pipeline_limit("MAX_CHARACTERS", 2, 0, 5)
        if limit == 0:
            return []
        raw = await self.llm.complete(
            f"Extract at most {limit} visible characters from the following story or script. "
            "Give each a detailed appearance and an outfit for reference images.\n\n"
            f"Script: {script}\n\n"
            'Return only JSON: {"characters": [{"idx": 0, "name": "name", '
            '"static_features": "appearance", "dynamic_features": "clothing", "is_visible": true}]}',
            system_prompt="You are a casting director and character designer. Return valid JSON only.",
            response_schema=CHARACTERS_SCHEMA,
        )
        data = parse_json_object(raw, "character")
        characters = data.get("characters")
        if not isinstance(characters, list):
            raise RuntimeError("LLM returned an invalid characters list")
        try:
            return [CharacterInScene(**character) for character in characters[:limit]]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("LLM returned invalid character details") from exc
