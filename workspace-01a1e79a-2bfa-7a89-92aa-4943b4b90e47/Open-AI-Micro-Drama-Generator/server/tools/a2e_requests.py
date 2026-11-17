"""Build endpoint-specific A2E requests for one generated frame per shot."""
from typing import Any, Optional

from tools.model_catalog import ImageModel, VideoModel


def build_image_request(model: ImageModel, prompt: str, aspect_ratio: str,
                        refs: list[str], name: str) -> tuple[str, str, dict[str, Any]]:
    if aspect_ratio not in ("16:9", "2:3", "1:1"):
        raise ValueError(f"Unsupported image aspect ratio: {aspect_ratio}")
    if len(refs) > 2:
        raise ValueError("This pipeline accepts at most two image references")
    family = model.family
    body: dict[str, Any] = {"name": name, "prompt": prompt}
    if family in ("native", "seedream"):
        root = "/v1/userText2Image"
        if family == "native":
            width, height = {"16:9": (1280, 720), "2:3": (768, 1152), "1:1": (1024, 1024)}[aspect_ratio]
            body.update(model_type=model.variant, width=width, height=height, resolution="1080P", max_images=1)
        else:
            tier = "2K" if model.variant in ("4.5", "5.0") else "1K"
            sizes = ({"16:9": (2048, 1152), "2:3": (1365, 2048), "1:1": (2048, 2048)}
                     if tier == "2K" else
                     {"16:9": (1280, 720), "2:3": (768, 1152), "1:1": (1024, 1024)})
            width, height = sizes[aspect_ratio]
            body.update(model_type="seedream", model_version=model.variant, width=width, height=height,
                        aspect_ratio=aspect_ratio, resolution=tier, max_images=1)
        detail_root = root
    elif family == "nano":
        root = "/v1/userNanoBanana"
        body.update(model=model.variant, aspect_ratio=aspect_ratio)
        if model.variant != "nano-banana":
            body["image_size"] = "1K"
        detail_root = root + "/detail"
    elif family == "gpt":
        root = "/v1/userGptImage"
        # GPT Image 1.5 does not support 16:9; its closest landscape size is 3:2.
        ratio = "3:2" if model.variant == "gpt-image-1.5" and aspect_ratio == "16:9" else aspect_ratio
        body.update(model=model.variant, aspect_ratio=ratio)
        if model.variant == "gpt-image-1.5":
            body["quality"] = "medium"
        else:
            body["resolution"] = "1K"
        detail_root = root + "/detail"
    elif family == "wan26image":
        root = "/v1/userWan26Image"
        body.update(prompt=prompt[:2000], aspect_ratio=aspect_ratio, prompt_extend=False)
        detail_root = root + "/detail"
    elif family == "wan27image":
        root = "/v1/userWan27Image"
        body.update(prompt=prompt[:5000], model=model.variant, aspect_ratio=aspect_ratio)
        detail_root = root + "/detail"
    elif family == "qwen":
        root = "/v1/userQwen2Image"
        size = {"16:9": "1280*720", "2:3": "880*1328", "1:1": "1024*1024"}[aspect_ratio]
        body.update(model=model.variant, size=size)
        detail_root = root + "/detail"
    elif family == "flux":
        root = "/v1/userFlux2"
        body.update(model=model.variant, aspect_ratio=aspect_ratio, resolution="1K")
        detail_root = root + "/detail"
    elif family == "grokimage":
        root = "/v1/userGrokImage"
        body.update(prompt=prompt[:8000], aspect_ratio=aspect_ratio)
        detail_root = root + "/detail"
    elif family == "klingimage":
        root = "/v1/userKlingImage"
        body.update(aspect_ratio=aspect_ratio, resolution="1k", n=1)
        detail_root = root + "/detail"
    else:
        raise ValueError(f"Unsupported A2E image family: {family}")
    if refs and model.supports_references:
        body["input_images"] = refs
    return root + "/start", detail_root, body


def build_video_request(model: VideoModel, prompt: str, image_url: str,
                        duration: Optional[int], name: str, *, native_audio: bool = False
                        ) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(image_url, str) or not image_url.startswith(("http://", "https://")):
        raise ValueError("A public image URL is required for A2E video generation")
    if model.durations:
        if isinstance(duration, bool) or not isinstance(duration, int) or duration not in model.durations:
            raise ValueError(f"Unsupported clip duration for {model.label}; choose one of {list(model.durations)} seconds")
    elif duration is not None:
        raise ValueError(f"{model.label} does not expose clip duration through the A2E API")
    family = model.family
    d = str(duration) if duration is not None else None
    body: dict[str, Any] = {"name": name, "prompt": prompt}
    if family == "native":
        root = "/v1/userImage2Video"
        body.update(image_url=image_url, model_version=model.variant, video_time=duration,
                    number_of_images=1, auto_add_audio=native_audio)
    elif family == "wan25":
        root = "/v1/userWan25"
        body.update(model=model.variant, image_url=image_url, duration=d, resolution="720p")
        if model.variant == "wan2.7-i2v":
            body["task_type"] = "first_frame"  # upstream may always produce audio
        else:
            body["audio"] = False
    elif family == "wanspicy":
        if not image_url.startswith("https://"):
            raise ValueError("Wan Spicy requires an HTTPS image URL")
        root = "/v1/userWanSpicy"
        body.update(model=model.variant, image_url=image_url, duration=duration, resolution="720p")
    elif family == "wanr2v":
        root = "/v1/userWan26R2V"
        # Standard R2V generates audio; only Flash supports disabling it.
        ref_prompt = "Use character1 as the visual reference where applicable. " + prompt
        body.update(model=model.variant, prompt=ref_prompt[:1500], reference_urls=[image_url],
                    duration=d, resolution="720p", aspect_ratio="16:9", shot_type="single",
                    audio=model.variant == "wan2.6-r2v")
    elif family == "wan30":
        root = "/v1/userWan30"
        body = {"name": name, "model": model.variant, "mode": "first-frame",
                "input": {"prompt": prompt[:5000], "media": [{"type": "first_frame", "url": image_url}]},
                "parameters": {"duration": duration, "resolution": "720P", "ratio": "adaptive",
                               "audio": False, "prompt_extend": False}}
    elif family == "happyhorse":
        root = "/v1/userHappyhorseVideo"
        body.update(mode="i2v", model_version=model.variant, image_url=image_url, duration=d, resolution="720P")
    elif family == "veo":
        root = "/v1/veoVideo"
        body.update(model=model.variant, generationType="FIRST_AND_LAST_FRAMES_2_VIDEO",
                    imageUrls=[image_url], aspectRatio="16:9")
    elif family == "kling":
        root = "/v1/klingVideo"
        body.update(mode="image-to-video", image_url=image_url, duration=d, sound=False)
        if model.variant == "3.0-fast":
            body.update(version="3.0", model_version="fast", resolution=model.quality)
        else:
            body.update(version=model.variant, quality_mode=model.quality)
    elif family == "kling4":
        root = "/v1/klingVideo"
        # Kling 4 rejects the earlier-version sound and quality_mode fields.
        body.update(mode="image-to-video", version=model.variant, duration=d, resolution=model.quality,
                    keyframes=[{"type": "first_frame", "url": image_url}])
    elif family == "omni":
        root = "/v1/klingOmni"
        body.update(prompt=f"Use <<<image_1>>> as the visual reference. {prompt}", image_list=[image_url],
                    mode=model.quality, duration=d, aspect_ratio="16:9", sound=False, multi_shot=False)
    elif family == "grok":
        root = "/v1/grokVideo"
        body.update(model_type="image-to-video", model_version=model.variant, image_url=image_url,
                    duration=d, resolution="720p", mode="normal", aspect_ratio="16:9")
    elif family == "hailuo":
        root = "/v1/hailuoVideo"
        body.update(model=model.variant, image_url=image_url, duration=d, resolution="768P")
    elif family == "minimax":
        root = "/v1/minimaxH3Video"
        body.update(mode="image-to-video", first_frame_url=image_url, duration=d, resolution="768P")
    elif family == "seedance1":
        root = "/v1/seedanceVideo"
        body.update(mode="image-to-video", image_url=image_url, duration=d, resolution="720p",
                    aspect_ratio="16:9", generate_audio=False)
    elif family == "seedance2":
        root = "/v1/seedance2Video"
        body.update(mode="image-to-video", model_version=model.variant, image_url=image_url,
                    duration=duration, resolution="720p", aspect_ratio="16:9", generate_audio=False)
    else:
        raise ValueError(f"Unsupported A2E video family: {family}")
    return root + "/start", root, body
