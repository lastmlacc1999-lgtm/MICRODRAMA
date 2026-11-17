import asyncio
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Optional

from agents.screenwriter import Screenwriter
from agents.character_extractor import CharacterExtractor
from interfaces.character import CharacterInScene
from pipelines.script2video import Script2VideoPipeline
from tools.a2e_client import A2EClient
from tools.llm import TextLLM
from tools.project_store import ProjectStore
from utils.video import concatenate_videos

ProgressCallback = Callable[[str, str, int], Awaitable[None]]


class Idea2VideoPipeline:
    def __init__(self, a2e: A2EClient, llm: Optional[TextLLM] = None,
                 job_id: str = "job", *, image_generator=None, clip_duration: Optional[int] = 5,
                 project: Optional[ProjectStore] = None, image_provider: str = "a2e"):
        self.llm = llm
        self.project = project
        self.screenwriter = Screenwriter(llm)
        self.character_extractor = CharacterExtractor(llm)
        self.script2video = Script2VideoPipeline(a2e, llm, job_id, image_generator=image_generator,
                                                clip_duration=clip_duration, project=project,
                                                image_provider=image_provider)

    async def _step(self, key: str, label: str, operation):
        if self.project:
            return await self.project.call(key, "text" if self.llm else "local", label, operation)
        return await operation(None)

    async def run(
        self, idea: str, user_requirement: str, style: str, job_id: str,
        progress_callback: ProgressCallback,
    ) -> str:
        outputs_dir = self.project.path if self.project else Path("outputs") / job_id
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / "final_video.mp4"
        final = self.project.step("final") if self.project else None
        if final and final.get("state") == "complete" and output_path.is_file() and output_path.stat().st_size:
            await progress_callback("concat", "Reusing the saved final video.", 90)
            return str(output_path)
        await progress_callback(
            "screenwriter", "Developing or reusing the story..." if self.llm
            else "Using your idea directly (simple planning; no text LLM).", 10,
        )
        async def develop(_):
            return await self.screenwriter.develop_story(idea, user_requirement)
        story = await self._step("story", "Story outline", develop)

        await progress_callback(
            "characters", "Extracting or reusing characters..." if self.llm
            else "Skipping LLM character extraction in simple-planning mode.", 20,
        )
        async def extract(_):
            return [character.model_dump() for character in await self.character_extractor.extract_characters(story)]
        character_data = await self._step("characters", "Character descriptions", extract)
        characters = [CharacterInScene.model_validate(item) for item in character_data]

        await progress_callback("screenwriter", "Writing or reusing scene scripts..." if self.llm
                                else "Planning one scene from your idea.", 25)
        async def write(_):
            return await self.screenwriter.write_script_based_on_story(story, user_requirement)
        scene_scripts = await self._step("scenes", "Scene scripts", write)
        if not scene_scripts:
            raise RuntimeError("No scene scripts were produced")

        # Finish planning every scene before any paid image/video generation.
        storyboards = []
        for i, scene_script in enumerate(scene_scripts):
            await progress_callback("storyboard", f"Planning or reusing shots for scene {i + 1}...", 28)
            storyboards.append(await self.script2video.plan_storyboard(
                scene_script, characters, user_requirement, style, i,
            ))

        paths = []
        progress_per_scene = int(58 / len(scene_scripts))
        for i, scene_script in enumerate(scene_scripts):
            path = await self.script2video.run(
                script=scene_script, characters=characters, user_requirement=user_requirement,
                style=style, working_dir=str(outputs_dir / f"scene_{i:02d}"),
                progress_callback=progress_callback, scene_idx=i,
                base_progress=30 + i * progress_per_scene, progress_range=progress_per_scene,
                briefs=storyboards[i],
            )
            paths.append(path)
        await progress_callback("concat", "Combining scenes into the final video...", 90)
        async def combine():
            if len(paths) == 1:
                await asyncio.to_thread(shutil.copyfile, paths[0], output_path)
            else:
                await concatenate_videos(paths, str(output_path))
        await self.script2video._local_video_step("final", "Final video", output_path, combine)
        return str(output_path)
