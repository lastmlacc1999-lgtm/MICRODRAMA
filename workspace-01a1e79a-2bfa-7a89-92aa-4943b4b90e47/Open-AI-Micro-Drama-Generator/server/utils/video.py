import asyncio
import tempfile
from pathlib import Path
from typing import List

import httpx


async def download_video(url: str, dest_path: str) -> str:
    """Download promptly before A2E output URLs expire; never send API credentials."""
    dest = Path(dest_path)
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with dest.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        output.write(chunk)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        # HTTP exceptions can contain a signed provider URL; never echo it to
        # the progress log or saved error message.
        raise RuntimeError("Could not download the saved A2E video. No new video task was submitted.") from exc
    return str(dest)


async def concatenate_videos(video_paths: List[str], output_path: str) -> str:
    """Concatenate local files (or download URLs first) in a worker thread."""
    local_paths = []
    temp_files = []
    try:
        for video in video_paths:
            if video.startswith(("http://", "https://")):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    path = tmp.name
                temp_files.append(path)
                local_paths.append(await download_video(video, path))
            else:
                local_paths.append(video)
        await asyncio.to_thread(_concat_sync, local_paths, output_path)
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
    return output_path


def _concat_sync(video_paths: List[str], output_path: str) -> None:
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy import VideoFileClip, concatenate_videoclips
    if not video_paths:
        raise ValueError("No video clips to concatenate")
    clips = []
    final = None
    try:
        for path in video_paths:
            clips.append(VideoFileClip(path))
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    finally:
        if final is not None:
            final.close()
        for clip in clips:
            clip.close()
