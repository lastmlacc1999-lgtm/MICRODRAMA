import asyncio
import hashlib
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from agents.storyboard_artist import StoryboardArtist
from interfaces.character import CharacterInScene
from interfaces.shot import ShotBriefDescription, ShotDescription
from tools.a2e_client import A2EClient
from tools.gemini_image import GeminiImageError
from tools.llm import TextLLM
from tools.project_store import ProjectError, ProjectStore
from utils.images import download_image, image_url_available
from utils.video import concatenate_videos, download_video

ProgressCallback = Callable[[str, str, int], Awaitable[None]]


class Script2VideoPipeline:
    def __init__(self, a2e: A2EClient, llm: Optional[TextLLM] = None,
                 job_id: str = "job", *, image_generator=None, clip_duration: Optional[int] = 5,
                 project: Optional[ProjectStore] = None, image_provider: str = "a2e"):
        self.a2e = a2e
        self.image_generator = image_generator or a2e
        self.image_provider = image_provider
        self.clip_duration = clip_duration
        self.storyboard_artist = StoryboardArtist(llm, clip_duration)
        self.project = project
        self.job_id = job_id[:8]
        self.portraits: dict[str, str] = {}
        self.image_assets: dict[str, dict] = {}
        self.resolved_urls: dict[str, str] = {}

    async def _step(self, key: str, kind: str, label: str, operation):
        if self.project:
            return await self.project.call(key, kind, label, operation)
        return await operation(None)

    async def plan_storyboard(self, script: str, characters: List[CharacterInScene],
                              user_requirement: str, style: str, scene_idx: int) -> List[ShotBriefDescription]:
        async def plan(_):
            briefs = await self.storyboard_artist.design_storyboard(
                script, characters, f"{user_requirement or 'No additional requirements'}. Style: {style}"
            )
            if not briefs:
                raise RuntimeError("Storyboard contains no shots")
            return [brief.model_dump() for brief in briefs]
        values = await self._step(f"scene/{scene_idx}/storyboard",
                                  "text" if self.storyboard_artist.llm else "local",
                                  f"Scene {scene_idx + 1} storyboard", plan)
        return [ShotBriefDescription.model_validate(value) for value in values]

    async def run(
        self, script: str, characters: List[CharacterInScene], user_requirement: str,
        style: str, working_dir: str, progress_callback: ProgressCallback,
        scene_idx: int = 0, base_progress: int = 30, progress_range: int = 60,
        briefs: Optional[List[ShotBriefDescription]] = None,
    ) -> str:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        scene_path = Path(working_dir) / "scene_video.mp4"
        saved_scene = self.project.step(f"scene/{scene_idx}/combined") if self.project else None
        if saved_scene and saved_scene.get("state") == "complete" and scene_path.is_file() and scene_path.stat().st_size:
            await progress_callback("concat", f"Reusing the saved video for scene {scene_idx + 1}.", base_progress + progress_range)
            return str(scene_path)

        # Complete all text planning before any paid portrait or frame creation.
        if briefs is None:
            await progress_callback("storyboard", f"Planning shots for scene {scene_idx + 1}...",
                                    base_progress + int(progress_range * 0.05))
            briefs = await self.plan_storyboard(script, characters, user_requirement, style, scene_idx)
        if not briefs:
            raise RuntimeError("Storyboard contains no shots")
        shots = [ShotDescription(**brief.model_dump()) for brief in briefs]

        # A saved clip or known remote task does not need its input image again.
        # Only load frames (and their backups) for work that still needs them.
        supports_refs = getattr(self.image_generator, "supports_references", True)
        need_frame = [not self._clip_reusable(scene_idx, i, working_dir) for i in range(len(shots))]
        will_submit_frame = []
        for i, needed in enumerate(need_frame):
            frame_key = f"scene/{scene_idx}/frame/{i}"
            entry = self.project.step(frame_key) if self.project else None
            if needed and self.image_provider == "gemini" and await self._restore_gemini_checkpoint(frame_key, entry):
                entry = self.project.step(frame_key)
            ready_or_pollable = entry and (entry.get("state") == "complete" or
                                           (entry.get("task_id") and not entry.get("terminal")))
            will_submit_frame.append(bool(needed and not ready_or_pollable))
        if supports_refs and not characters:
            for i in range(1, len(shots)):
                if will_submit_frame[i]:
                    need_frame[i - 1] = True  # continuity reference for a new frame
        visible = [c for c in characters if c.is_visible and c.name.strip()]
        needed_characters = [c for c in visible if any(
            will_submit_frame[i] and c.name.casefold() in shot.visual_desc.casefold()
            for i, shot in enumerate(shots))]
        missing = ({c.name.casefold(): c for c in needed_characters if c.name.casefold() not in self.portraits}
                   if supports_refs else {})
        await progress_callback(
            "portraits",
            ("Selected image model cannot use references; skipping character portraits." if not supports_refs else
             f"Generating or reusing {len(missing)} character reference portrait(s)..." if missing else
             "No character portraits needed for the remaining work."),
            base_progress + int(progress_range * 0.15),
        )
        if missing:
            async def portrait(key, character):
                ident = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                return await self._image_step(
                    f"portrait/{ident}", f"Character portrait: {character.name}",
                    lambda task: self._generate_portrait(character, style, task), progress_callback,
                )
            results = await asyncio.gather(*(portrait(key, character) for key, character in missing.items()),
                                           return_exceptions=True)
            errors = []
            for (key, character), result in zip(missing.items(), results):
                if isinstance(result, BaseException):
                    errors.append(f"{character.name}: {result}")
                else:
                    self.portraits[key] = result
            if errors:
                raise RuntimeError("Character portrait generation failed: " + "; ".join(errors))

        previous_frame = None
        for i, shot in enumerate(shots):
            if not need_frame[i]:
                # The clip or its task is already saved. Do not require a
                # potentially expired input URL or missing image backup.
                continue
            await progress_callback(
                "frames", f"Generating or reusing frame {i + 1}/{len(shots)} for scene {scene_idx + 1}...",
                base_progress + int(progress_range * (0.2 + 0.35 * i / len(shots))),
            )
            shot.first_frame_url = await self._image_step(
                f"scene/{scene_idx}/frame/{i}", f"Scene {scene_idx + 1}, shot {i + 1} frame",
                lambda task, shot=shot, i=i, previous_frame=previous_frame: self._generate_first_frame(
                    shot, characters, style, scene_idx, i, previous_frame, task, progress_callback
                ), progress_callback,
            )
            previous_frame = shot.first_frame_url

        await progress_callback(
            "video", f"Generating or reusing {len(shots)} video clip(s) for scene {scene_idx + 1}...",
            base_progress + int(progress_range * 0.60),
        )
        results = await asyncio.gather(
            *(self._generate_shot_video(shot, i, scene_idx, working_dir, progress_callback)
              for i, shot in enumerate(shots)), return_exceptions=True,
        )
        errors = [f"shot {i + 1}: {result}" for i, result in enumerate(results)
                  if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("Video generation failed: " + "; ".join(errors))
        paths = [result for result in results if isinstance(result, str)]
        if len(paths) != len(shots):
            raise RuntimeError("One or more video clips were not returned")

        await progress_callback("concat", f"Combining shots for scene {scene_idx + 1}...",
                                base_progress + int(progress_range * 0.90))
        await self._local_video_step(f"scene/{scene_idx}/combined", f"Scene {scene_idx + 1} video",
                                     scene_path, lambda: concatenate_videos(paths, str(scene_path)))
        return str(scene_path)

    def _clip_reusable(self, scene_idx: int, shot_idx: int, working_dir: str) -> bool:
        if not self.project:
            return False
        local = self.project.step(f"scene/{scene_idx}/clip/{shot_idx}/local")
        path = Path(working_dir) / f"shot_{shot_idx:03d}.mp4"
        if local and local.get("state") == "complete" and path.is_file() and path.stat().st_size:
            return True
        remote = self.project.step(f"scene/{scene_idx}/clip/{shot_idx}/remote")
        return bool(remote and (remote.get("state") == "complete" or
                                (remote.get("task_id") and not remote.get("terminal"))))

    async def _restore_gemini_checkpoint(self, key: str, entry: Optional[dict]) -> bool:
        # Prefer an externally restored backup to any paid replacement.
        if not self.project or not entry or not entry.get("missing_asset") or not isinstance(entry.get("value"), dict):
            return False
        info = entry["value"]
        try:
            content = self.project.asset_bytes(info["asset"], info.get("sha256"))
            self.image_generator.restore_image(info["handle"], content, info["mime"])
        except (ProjectError, GeminiImageError, KeyError, TypeError):
            return False
        await self.project.restore_gemini_image_checkpoint(key)
        self.image_assets[info["handle"]] = info
        return True

    async def _image_step(self, key: str, label: str, operation, progress: ProgressCallback) -> str:
        if self.project is None:
            return await operation(None)
        previous = self.project.step(key)
        reused = bool(previous and previous.get("state") == "complete")
        if self.image_provider == "gemini":
            if await self._restore_gemini_checkpoint(key, previous):
                return previous["value"]["handle"]
            async def create(task_id):
                if task_id:
                    raise ProjectError("A Gemini image cannot be recovered using an A2E task ID.")
                handle = await operation(None)
                content, mime = self.image_generator.export_image(handle)
                info = self.project.save_asset(key, content, mime)
                return {**info, "handle": handle}
            info = await self.project.call(key, "gemini_image", label, create)
            if not isinstance(info, dict) or not all(isinstance(info.get(k), str) for k in ("asset", "mime", "handle")):
                raise ProjectError("The saved Gemini image checkpoint is invalid; no replacement was requested.")
            try:
                content = self.project.asset_bytes(info["asset"], info.get("sha256"))
                self.image_generator.restore_image(info["handle"], content, info["mime"])
            except (ProjectError, GeminiImageError) as exc:
                await self.project.mark_gemini_image_missing(key)
                raise ProjectError("The saved Gemini image is missing or invalid. No replacement was requested; "
                                   "restore the backup or review provider billing before explicitly retrying.") from exc
            self.image_assets[info["handle"]] = info
            return info["handle"]

        url = await self.project.call(key, "a2e_image", label, operation)
        if not isinstance(url, str):
            raise ProjectError("The saved A2E image checkpoint is invalid; no replacement was requested.")
        cache_key = key + "/local"
        saved = self.project.step(cache_key)
        if saved and saved.get("state") == "complete":
            try:
                self.project.asset_bytes(saved["value"]["asset"], saved["value"].get("sha256"))
            except (ProjectError, KeyError, TypeError, AttributeError):
                await self.project.invalidate_local(cache_key)
        async def cache(_):
            nonlocal url
            try:
                content, mime = await download_image(url)
            except Exception:
                # If a result link expired or a backup was lost, ask the SAME
                # task for its output again. Never create a replacement image.
                existing = self.project.step(key)
                task_id = existing.get("task_id") if existing else None
                if not task_id:
                    raise
                try:
                    refreshed = await operation(task_id)
                except Exception as exc:
                    if getattr(exc, "output_expired", False):
                        await self.project.mark_image_expired(key, task_id)
                    raise
                if not isinstance(refreshed, str):
                    raise ProjectError("The saved A2E image task returned an invalid URL.")
                content, mime = await download_image(refreshed)
                await self.project.refresh_image_url(key, refreshed)
                url = refreshed
            return self.project.save_asset(cache_key, content, mime)
        info = await self.project.call(cache_key, "local", label + " · local backup", cache)
        self.image_assets[url] = {**info, "key": key, "reused": reused}
        return url

    async def _ready_image_url(self, url: str, progress: ProgressCallback) -> str:
        if url in self.resolved_urls:
            return self.resolved_urls[url]
        info = self.image_assets.get(url)
        if self.project and info and info.get("reused") and self.project.data.get("attempt", 1) > 1:
            live = await self._ensure_live_url(url, info, info["key"], progress)
            self.resolved_urls[url] = live
            return live
        return url

    async def _ensure_live_url(self, url: str, info: dict, key: str, progress: ProgressCallback) -> str:
        """Only re-upload a saved image if its URL is clearly inaccessible."""
        if not self.project:
            return url
        for _ in range(4):
            if await image_url_available(url):
                return url
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
            refresh_key = f"refresh/{key}/{digest}"
            previous = self.project.step(refresh_key)
            await progress("frames", "A saved image link is inaccessible; re-uploading its local copy may incur an A2E signing fee.", 30)
            content = self.project.asset_bytes(info["asset"], info.get("sha256"))
            async def upload(_):
                return await self.a2e.upload_image(content, info["mime"])
            url = await self.project.call(refresh_key, "a2e_upload", "Re-upload saved image", upload)
            if not isinstance(url, str):
                raise ProjectError("The saved image upload checkpoint is invalid.")
            if not previous or previous.get("state") != "complete":
                return url  # a newly uploaded destination has not expired yet
        raise ProjectError("The saved image URL repeatedly expired. No further re-upload was made.")

    async def _local_video_step(self, key: str, label: str, path: Path, operation) -> None:
        if self.project:
            saved = self.project.step(key)
            if saved and saved.get("state") == "complete" and (not path.is_file() or not path.stat().st_size):
                await self.project.invalidate_local(key)
        async def produce(_):
            await operation()
            if not path.is_file() or not path.stat().st_size:
                raise ProjectError("The local video file was not saved successfully.")
            if self.project:
                return {"asset": path.relative_to(self.project.path).as_posix(), "type": "video"}
            return str(path)
        await self._step(key, "local", label, produce)

    async def _generate_portrait(self, character: CharacterInScene, style: str, task_id: Optional[str] = None) -> str:
        prompt = (f"Character reference portrait of {character.name}. {character.static_features}. "
                  f"Wearing {character.dynamic_features}. Style: {style}. "
                  "Detailed face, neutral background, consistent character reference.")
        kwargs = {"aspect_ratio": "2:3", "name": f"MicroDrama {self.job_id} character {character.idx}"}
        if task_id:
            kwargs["resume_task_id"] = task_id
        return await self.image_generator.generate_image(prompt, **kwargs)

    async def _generate_first_frame(
        self, shot: ShotDescription, characters: List[CharacterInScene], style: str,
        scene_idx: int, shot_idx: int, previous_frame: Optional[str], task_id: Optional[str],
        progress_callback: ProgressCallback,
    ) -> str:
        mentioned = [c for c in characters if c.is_visible and c.name.casefold() in shot.visual_desc.casefold()]
        supports_refs = getattr(self.image_generator, "supports_references", True)
        references = ([self.portraits[c.name.casefold()] for c in mentioned
                       if c.name.casefold() in self.portraits][:2] if supports_refs else [])
        if supports_refs and not characters and previous_frame:
            references = [previous_frame]
        # Polling a known task does not require live input references. Re-upload
        # an expired saved reference only when actually submitting a new task.
        if not task_id and self.image_provider == "a2e":
            references = [await self._ready_image_url(ref, progress_callback) for ref in references]
        character_context = " ".join(f"{c.name}: {c.static_features}; wearing {c.dynamic_features}." for c in mentioned)
        prompt = (f"{shot.visual_desc}\n{character_context}\n"
                  f"Visual style: {style}. Cinematic 16:9 composition. "
                  "If references are supplied, preserve the visual identity while composing this shot.")
        kwargs = {"aspect_ratio": "16:9", "reference_urls": references,
                  "name": f"MicroDrama {self.job_id} scene {scene_idx + 1} frame {shot_idx + 1}"}
        if task_id:
            kwargs["resume_task_id"] = task_id
        return await self.image_generator.generate_image(prompt, **kwargs)

    async def _generate_shot_video(
        self, shot: ShotDescription, shot_idx: int, scene_idx: int, working_dir: str,
        progress_callback: ProgressCallback,
    ) -> str:
        local_path = Path(working_dir) / f"shot_{shot_idx:03d}.mp4"
        local_key = f"scene/{scene_idx}/clip/{shot_idx}/local"
        saved_local = self.project.step(local_key) if self.project else None
        if saved_local and saved_local.get("state") == "complete" and local_path.is_file() and local_path.stat().st_size:
            return str(local_path)
        video_key = f"scene/{scene_idx}/clip/{shot_idx}/remote"
        saved_video = self.project.step(video_key) if self.project else None
        recovering = bool(saved_video and (saved_video.get("state") == "complete" or
                           (saved_video.get("task_id") and not saved_video.get("terminal"))))
        if not shot.first_frame_url and not recovering:
            raise ValueError(f"Shot {shot_idx + 1} has no first frame")
        prompt = f"{shot.motion_desc[:500]}\nVisual context: {shot.visual_desc[:1200]}"
        if shot.audio_desc:
            prompt += f"\nRequested sound, if audio is enabled: {shot.audio_desc[:250]}"
        image_url = shot.first_frame_url or ""
        if hasattr(self.image_generator, "public_url") and not recovering:
            upload_key = f"scene/{scene_idx}/frame/{shot_idx}/upload"
            previous = self.project.step(upload_key) if self.project else None
            async def upload(_):
                return await self.image_generator.public_url(shot.first_frame_url)
            image_url = await self._step(upload_key, "a2e_upload", f"Scene {scene_idx + 1}, shot {shot_idx + 1} upload", upload)
            if self.project and previous and previous.get("state") == "complete":
                info = self.image_assets.get(shot.first_frame_url)
                if not info:
                    raise ProjectError("The saved Gemini frame is missing; no new image was requested.")
                image_url = await self._ensure_live_url(image_url, info, upload_key, progress_callback)
        elif not recovering:
            image_url = await self._ready_image_url(image_url, progress_callback)

        name = f"MicroDrama {self.job_id} scene {scene_idx + 1} shot {shot_idx + 1}"
        async def create_video(task_id):
            kwargs = {"duration": self.clip_duration, "name": name}
            if task_id:
                kwargs["resume_task_id"] = task_id
            return await self.a2e.generate_video(prompt, image_url, **kwargs)
        video_url = await self._step(video_key, "a2e_video", f"Scene {scene_idx + 1}, shot {shot_idx + 1} video task", create_video)
        async def download():
            try:
                await download_video(video_url, str(local_path))
            except Exception:
                # A lost/expired download URL must never create another paid task.
                # If a task ID was saved, poll it again for a fresh result URL.
                saved = self.project.step(video_key) if self.project else None
                task_id = saved.get("task_id") if saved else None
                if not task_id:
                    raise
                try:
                    refreshed = await self.a2e.generate_video(prompt, image_url, duration=self.clip_duration,
                                                              name=name, resume_task_id=task_id)
                except Exception as exc:
                    if getattr(exc, "output_expired", False):
                        await self.project.mark_video_expired(video_key, task_id)
                    raise
                await download_video(refreshed, str(local_path))
        await self._local_video_step(local_key, f"Scene {scene_idx + 1}, shot {shot_idx + 1} clip",
                                     local_path, download)
        return str(local_path)
