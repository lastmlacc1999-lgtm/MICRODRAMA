import re
from typing import List, Optional

from interfaces.character import CharacterInScene
from interfaces.shot import ShotBriefDescription
from tools.llm import TextLLM, parse_json_object
from utils.limits import pipeline_limit


SHOTS_SCHEMA = {
    "type": "object", "properties": {"shots": {"type": "array", "items": {
        "type": "object", "properties": {
            "idx": {"type": "integer"}, "visual_desc": {"type": "string"},
            "motion_desc": {"type": "string"}, "audio_desc": {"type": "string"},
        }, "required": ["idx", "visual_desc", "motion_desc", "audio_desc"],
    }}}, "required": ["shots"],
}


class StoryboardArtist:
    def __init__(self, llm: Optional[TextLLM] = None, clip_duration: Optional[int] = 5):
        self.llm = llm
        self.clip_duration = clip_duration

    async def design_storyboard(
        self, script: str, characters: List[CharacterInScene], user_requirement: str,
    ) -> List[ShotBriefDescription]:
        limit = pipeline_limit("MAX_SHOTS_PER_SCENE", 3, 1, 5)
        if self.llm is None:
            return self._simple_storyboard(script, user_requirement, limit)

        character_descriptions = "\n".join(
            f"- {c.name}: {c.static_features}. Wearing: {c.dynamic_features}"
            for c in characters if c.is_visible
        ) or "No named characters"
        length_hint = f"each about {self.clip_duration} seconds long" if self.clip_duration else "each a short clip"
        raw = await self.llm.complete(
            f"Break this scene into at most {limit} shots, {length_hint}. "
            "Make visual descriptions self-contained and explicitly name visible characters. "
            "Describe camera motion and optionally sound, without assuming dialogue will be generated.\n\n"
            f"Script: {script}\n\nCharacters:\n{character_descriptions}\n\n"
            f"Requirements: {user_requirement or 'Cinematic'}\n\n"
            'Return only JSON: {"shots": [{"idx": 0, "visual_desc": "scene and composition", '
            '"motion_desc": "motion and camera", "audio_desc": "optional audio"}]}',
            system_prompt="You are a professional storyboard artist. Return valid JSON only.",
            response_schema=SHOTS_SCHEMA,
        )
        data = parse_json_object(raw, "storyboard")
        shots = data.get("shots")
        if not isinstance(shots, list) or not shots:
            raise RuntimeError("LLM returned no storyboard shots")
        try:
            return [ShotBriefDescription(**shot) for shot in shots[:limit]]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("LLM returned invalid storyboard shots") from exc

    @staticmethod
    def _simple_storyboard(script: str, requirements: str, limit: int) -> List[ShotBriefDescription]:
        """Use the submitted text, not an invented plot, for a simple visual plan."""
        script = script.strip()
        if not script:
            raise ValueError("A non-empty idea or script is required")
        pieces = [s.strip() for s in re.split(r"(?:\n\s*\n|(?<=[.!?])\s+)", script) if s.strip()]
        count = min(limit, 3)
        if len(pieces) >= count:
            # Partition all text among shots; do not silently discard the end.
            beats = [" ".join(pieces[i * len(pieces) // count:(i + 1) * len(pieces) // count])
                     for i in range(count)]
        else:
            beats = [script] * count
        directions = [
            ("opening wide establishing composition", "A slow, steady establishing camera move."),
            ("closer view of the central action", "Natural character and environmental movement; gently move the camera closer."),
            ("closing composition emphasizing the outcome", "A measured camera move that settles on the scene's final moment."),
        ]
        shots = []
        for i, beat in enumerate(beats):
            framing, motion = directions[i]
            shots.append(ShotBriefDescription(
                idx=i,
                visual_desc=(f"Story context: {script}\nThis shot focuses on: {beat}\n"
                             f"Composition: {framing}. Requirements: {requirements or 'Cinematic'}."),
                motion_desc=motion,
                audio_desc="",
            ))
        return shots
