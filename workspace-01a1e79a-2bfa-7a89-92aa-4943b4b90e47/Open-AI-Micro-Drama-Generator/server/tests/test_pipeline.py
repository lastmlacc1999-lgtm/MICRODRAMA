from pathlib import Path

import pytest

from pipelines.script2video import Script2VideoPipeline


class FakeA2E:
    def __init__(self, fail_video=None):
        self.images = []
        self.videos = []
        self.fail_video = fail_video

    async def generate_image(self, prompt, *, aspect_ratio, reference_urls=None, name):
        self.images.append({"prompt": prompt, "refs": reference_urls, "name": name})
        return f"https://cdn.test/frame{len(self.images)}.png"

    async def generate_video(self, prompt, image_url, *, duration, name):
        self.videos.append({"prompt": prompt, "image": image_url, "name": name})
        if self.fail_video and self.fail_video in name:
            raise RuntimeError("test generation failure")
        return f"https://cdn.test/clip{len(self.videos)}.mp4"


@pytest.mark.asyncio
async def test_a2e_only_pipeline_links_frames_and_keeps_clip_order(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "3")
    captured = []

    async def fake_download(url, dest):
        Path(dest).write_text(url)
        return dest

    async def fake_concat(paths, output):
        captured.extend(paths)
        Path(output).write_text("video")
        return output

    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", fake_concat)
    a2e = FakeA2E()
    pipeline = Script2VideoPipeline(a2e, job_id="abcdef123456")
    events = []

    async def progress(stage, message, percentage):
        events.append((stage, percentage))

    path = await pipeline.run("An owl sees the moon.", [], "", "Anime", str(tmp_path), progress)
    assert Path(path).exists()
    assert len(a2e.images) == len(a2e.videos) == 3
    assert not a2e.images[0]["refs"]
    assert a2e.images[1]["refs"] == ["https://cdn.test/frame1.png"]
    assert a2e.images[2]["refs"] == ["https://cdn.test/frame2.png"]
    assert [v["image"] for v in a2e.videos] == [f"https://cdn.test/frame{i}.png" for i in (1, 2, 3)]
    assert [Path(p).name for p in captured] == ["shot_000.mp4", "shot_001.mp4", "shot_002.mp4"]


@pytest.mark.asyncio
async def test_failed_shot_does_not_silently_generate_partial_video(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_SHOTS_PER_SCENE", "3")

    async def fake_download(url, dest):
        Path(dest).write_text(url)
        return dest

    async def never_concat(*args):
        raise AssertionError("Should not concatenate an incomplete scene")

    monkeypatch.setattr("pipelines.script2video.download_video", fake_download)
    monkeypatch.setattr("pipelines.script2video.concatenate_videos", never_concat)
    a2e = FakeA2E(fail_video="shot 2")
    pipeline = Script2VideoPipeline(a2e)

    async def progress(*_):
        pass

    with pytest.raises(RuntimeError, match="shot 2.*test generation failure"):
        await pipeline.run("An owl sees the moon.", [], "", "Anime", str(tmp_path), progress)
    assert len(a2e.videos) == 3  # all in-flight tasks settled, no fallback submitted
