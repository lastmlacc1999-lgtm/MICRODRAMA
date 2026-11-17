from pathlib import Path

import pytest
from moviepy import ColorClip, VideoFileClip

from utils.video import concatenate_videos


@pytest.mark.asyncio
async def test_real_moviepy_concatenation(tmp_path):
    paths = []
    for i, color in enumerate(((0, 0, 255), (255, 0, 0))):
        path = tmp_path / f"clip{i}.mp4"
        clip = ColorClip(size=(160, 90), color=color, duration=0.6).with_fps(10)
        try:
            clip.write_videofile(str(path), codec="libx264", logger=None)
        finally:
            clip.close()
        paths.append(str(path))
    output = str(tmp_path / "combined.mp4")
    assert await concatenate_videos(paths, output) == output
    clip = VideoFileClip(output)
    try:
        assert 1.1 <= clip.duration <= 1.3
        assert Path(output).stat().st_size > 0
    finally:
        clip.close()
