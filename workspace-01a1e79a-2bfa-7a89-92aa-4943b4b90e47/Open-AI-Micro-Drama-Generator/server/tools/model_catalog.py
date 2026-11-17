"""Supported one-image workflows, based on the supplied A2E OpenAPI spec.

This is a curated integration catalog, not a live entitlement or pricing list.
Model availability, prices, and upstream API contracts may change.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ImageModel:
    id: str
    label: str
    provider: str
    group: str
    family: str
    variant: str = ""
    supports_references: bool = True
    note: str = ""


@dataclass(frozen=True)
class TextModel:
    id: str
    label: str
    note: str = ""


@dataclass(frozen=True)
class VideoModel:
    id: str
    label: str
    group: str
    family: str
    variant: str
    durations: tuple[int, ...]
    default_duration: Optional[int] = 5
    quality: str = ""
    input_mode: str = "first_frame"
    note: str = ""


IMAGE_MODELS = [
    ImageModel("a2e", "A2E · default", "a2e", "A2E", "native", "a2e"),
    ImageModel("a2e-v2", "A2E · V2", "a2e", "A2E", "native", "a2e-v2"),
    ImageModel("zimage", "Z-Image Turbo", "a2e", "A2E", "native", "zimage", False,
               "Text-to-image only: reference portraits and previous frames cannot be used for continuity."),
    *[ImageModel(f"seedream-{v}", f"Seedream {v}", "a2e", "Seedream", "seedream", v)
      for v in ("4.5", "5.0", "5.0-pro", "5.5")],
    *[ImageModel(v, label, "a2e", "Nano Banana via A2E", "nano", v) for v, label in (
        ("nano-banana", "Nano Banana"), ("nano-banana-pro", "Nano Banana Pro"),
        ("nano-banana-2", "Nano Banana 2"), ("nano-banana-2-1", "Nano Banana 2.1"),
        ("nano-banana-2-lite", "Nano Banana 2 Lite"))],
    *[ImageModel(v, f"GPT Image {label}", "a2e", "GPT Image via A2E", "gpt", v,
                 note="GPT Image 1.5 produces 3:2 landscape frames, not 16:9." if v == "gpt-image-1.5" else "")
      for v, label in (("gpt-image-1.5", "1.5"), ("gpt-image-2", "2"),
                       ("gpt-image-2.5-flare", "2.5 Flare"), ("gpt-image-2.5-sunburst", "2.5 Sunburst"),
                       ("gpt-image-2.6-flare", "2.6 Flare"), ("gpt-image-2.6-sunburst", "2.6 Sunburst"),
                       ("gpt-image-2.6-economy", "2.6 Economy"))],
    ImageModel("wan2.6-image", "Wan 2.6 Image", "a2e", "Wan via A2E", "wan26image", note=
               "Prompts longer than 2,000 characters are shortened for this model."),
    ImageModel("wan2.7-image", "Wan 2.7 Image", "a2e", "Wan via A2E", "wan27image", "wan2.7-image", note=
               "Prompts longer than 5,000 characters are shortened for this model."),
    ImageModel("wan2.7-image-pro", "Wan 2.7 Image Pro", "a2e", "Wan via A2E", "wan27image", "wan2.7-image-pro", note=
               "Prompts longer than 5,000 characters are shortened for this model."),
    *[ImageModel(v, v.replace("qwen-image-", "Qwen Image "), "a2e", "Qwen via A2E", "qwen", v)
      for v in ("qwen-image-2.0", "qwen-image-2.0-pro", "qwen-image-3.0", "qwen-image-3.0-pro")],
    ImageModel("flux-2-pro", "Flux 2 Pro", "a2e", "Flux via A2E", "flux", "flux-2-pro"),
    ImageModel("flux-3-klein", "Flux 3 Klein", "a2e", "Flux via A2E", "flux", "flux-3-klein"),
    ImageModel("grok-imagine-image-2.0", "Grok Imagine Image 2.0", "a2e", "Other via A2E", "grokimage"),
    ImageModel("kling-image-3.0", "Kling Image 3.0", "a2e", "Other via A2E", "klingimage"),
    ImageModel("gemini-nano-banana-2.1", "Nano Banana 2.1", "gemini", "Gemini API", "gemini", note=
               "Google Gemini API; 1K images with reference inputs. Availability and billing depend on your Google account."),
    ImageModel("gemini-3.1-flash-lite-image", "Nano Banana 2 Lite", "gemini", "Gemini API", "gemini", note=
               "Fast 1K model. Google says it is not optimized for multiple references or sequential editing."),
    ImageModel("gemini-3.1-flash-image", "Nano Banana 2 · deprecated", "gemini", "Gemini API", "gemini", note=
               "Deprecated by Google; Nano Banana 2.1 is recommended. Availability and billing may change."),
    ImageModel("gemini-3-pro-image", "Nano Banana Pro", "gemini", "Gemini API", "gemini", note=
               "Premium Gemini image model; Google pricing and access apply."),
    ImageModel("gemini-2.5-flash-image", "Nano Banana (2.5 Flash)", "gemini", "Gemini API", "gemini", note=
               "Legacy Gemini image model; availability may change."),
]

# Google text models are independent of the image model; this static list does
# not imply access on the user's project. All support text + structured output.
GEMINI_TEXT_MODELS = [
    TextModel("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "Lower-cost, faster option; Google pricing and access apply."),
    TextModel("gemini-3.6-flash", "Gemini 3.6 Flash", "Previous-generation Flash model; Google pricing and access apply."),
    TextModel("gemini-3.8-flash", "Gemini 3.8 Flash", "General-purpose Flash model for story writing and planning."),
    TextModel("gemini-4-hydrogen", "Gemini 4 Hydrogen", "Gemini 4 text model; Google pricing and access apply."),
    TextModel("gemini-4-argon", "Gemini 4 Argon", "Advanced reasoning model; Google pricing and access apply."),
    TextModel("gemini-4-carbon", "Gemini 4 Carbon", "Premium reasoning model; Google pricing and access apply."),
]
DEFAULT_GEMINI_TEXT_MODEL = "gemini-3.8-flash"

# These OpenAI Responses API models support text generation and structured
# outputs. Availability and pricing depend on the user's OpenAI project.
OPENAI_TEXT_MODELS = [
    TextModel("gpt-6-luna", "GPT-6 Luna", "Efficient, lower-cost option for focused tasks."),
    TextModel("gpt-6.1-terra", "GPT-6.1 Terra", "Fast model for clear, natural writing."),
    TextModel("gpt-6.1-sol", "GPT-6.1 Sol", "Higher-capability model with higher token costs."),
    TextModel("gpt-6-astra", "GPT-6 Astra", "Flagship model with higher token costs."),
    TextModel("gpt-5.4-mini", "GPT-5.4 mini", "Earlier efficient GPT model; account access may vary."),
    TextModel("gpt-5.5", "GPT-5.5", "Earlier flagship model; account access and prices vary."),
]
DEFAULT_OPENAI_TEXT_MODEL = "gpt-6-luna"

_range = lambda lo, hi: tuple(range(lo, hi + 1))
VIDEO_MODELS = [
    *[VideoModel(v, label, "A2E", "native", v, _range(3, 20), note=(
        "Clips over 5 seconds are rounded up to 5-second segments: a 6-second request can produce about 10 seconds."
        if v == "a2e" else "Frame rounding can make the output up to about 0.7 seconds longer than requested."))
      for v, label in (("a2e", "A2E · original"), ("a2e-v2", "A2E · V2"), ("a2e-v2-flash", "A2E · V2 Flash"))],
    VideoModel("wan2.5-i2v-preview", "Wan 2.5 I2V", "Wan", "wan25", "wan2.5-i2v-preview", (5, 10)),
    VideoModel("wan2.6-i2v", "Wan 2.6 I2V", "Wan", "wan25", "wan2.6-i2v", _range(2, 15),
               note="A2E documents VIP/Max access for this model."),
    VideoModel("wan2.6-i2v-flash", "Wan 2.6 I2V Flash", "Wan", "wan25", "wan2.6-i2v-flash", _range(2, 15)),
    VideoModel("wan2.7-i2v", "Wan 2.7 I2V", "Wan", "wan25", "wan2.7-i2v", _range(2, 15),
               note="A2E documents VIP/Max access; this model may generate audio."),
    VideoModel("wan2.2-i2v-spicy", "Wan 2.2 I2V Spicy", "Wan", "wanspicy", "wan2.2-i2v-spicy", (5, 8)),
    VideoModel("wan2.7-i2v-spicy", "Wan 2.7 I2V Spicy", "Wan", "wanspicy", "wan2.7-i2v-spicy", _range(2, 15),
               note="Audio behavior is controlled by the provider."),
    VideoModel("wan2.6-r2v-flash", "Wan 2.6 R2V Flash · reference", "Wan", "wanr2v", "wan2.6-r2v-flash", _range(2, 10), input_mode="reference",
               note="Uses the image as a reference, not a guaranteed first frame."),
    VideoModel("wan2.6-r2v", "Wan 2.6 R2V · reference", "Wan", "wanr2v", "wan2.6-r2v", _range(2, 10), input_mode="reference",
               note="Uses a reference image, not a guaranteed first frame; audio is included."),
    *[VideoModel(v, label, "Wan", "wan30", v, _range(2, 30), note=(
        "Standard can take much longer than Prime; increase A2E_POLL_TIMEOUT_SECONDS if needed."
        if v == "wan3.0-video" else "Prime trades a higher per-second price for faster inference.")) for v, label in (
        ("wan3.0-video", "Wan 3.0 Video"), ("wan3.0-video-prime", "Wan 3.0 Video Prime"))],
    *[VideoModel(f"happyhorse-{v}", f"HappyHorse {v}", "HappyHorse", "happyhorse", v, _range(3, 15),
                 note="Audio behavior is controlled by the provider.") for v in ("1.0", "1.1")],
    *[VideoModel(v, label, "Veo", "veo", v, (), None, note=
                 "A2E does not expose a duration or audio-off control for this endpoint; the provider chooses the clip length and may include audio.")
      for v, label in (("veo3_fast", "Veo 3.1 Fast"), ("veo3", "Veo 3.1 Quality"))],
    VideoModel("kling-2.6-std", "Kling 2.6 Standard", "Kling", "kling", "2.6", (5, 10), quality="std"),
    VideoModel("kling-2.6-pro", "Kling 2.6 Pro", "Kling", "kling", "2.6", (5, 10), quality="pro"),
    VideoModel("kling-3.0-std", "Kling 3.0 Standard", "Kling", "kling", "3.0", _range(3, 15), quality="std"),
    VideoModel("kling-3.0-pro", "Kling 3.0 Pro", "Kling", "kling", "3.0", _range(3, 15), quality="pro"),
    VideoModel("kling-3.0-4k", "Kling 3.0 4K", "Kling", "kling", "3.0", _range(3, 15), quality="4k"),
    VideoModel("kling-3.0-fast-720p", "Kling 3.0 Fast · 720p", "Kling", "kling", "3.0-fast", _range(3, 15), quality="720p"),
    VideoModel("kling-3.0-fast-1080p", "Kling 3.0 Fast · 1080p", "Kling", "kling", "3.0-fast", _range(3, 15), quality="1080p"),
    *[VideoModel(f"kling-{v}-{q}", f"Kling {v} · {q}", "Kling", "kling4", v, ds, quality=q,
                 note="Uses a first keyframe; native audio may be included.")
      for v, q, ds in (("4.0", "720p", _range(3, 30)), ("4.0", "1080p", _range(3, 30)),
                       ("4.0-flash", "720p", _range(3, 20)))],
    *[VideoModel(f"kling-omni-{q}", f"Kling Omni · {label} · reference", "Kling", "omni", "kling-v3-omni",
                 _range(3, 15), quality=q, input_mode="reference",
                 note="Uses the image as a reference, not a guaranteed first frame.")
      for q, label in (("std", "Standard"), ("pro", "Pro"), ("4k", "4K"))],
    *[VideoModel(f"grok-video-{v}", f"Grok Imagine Video {label}", "Grok", "grok", v, (6, 10, 15), 6,
                 note="A2E does not expose an audio-off control for this endpoint.")
      for v, label in (("legacy", "Legacy"), ("1.5", "1.5"))],
    *[VideoModel(f"hailuo-{v}", f"Hailuo {v.title()}", "Hailuo & MiniMax", "hailuo", v, (6, 10), 6,
                 note="This integration uses 768P; audio behavior is controlled by the provider.") for v in ("standard", "pro")],
    VideoModel("minimax-h3", "MiniMax H3", "Hailuo & MiniMax", "minimax", "h3", _range(4, 15),
               note="This integration uses 768P; audio behavior is controlled by the provider."),
    VideoModel("seedance-1.5-pro", "Seedance 1.5 Pro", "Seedance", "seedance1", "1.5-pro", (5, 10)),
    *[VideoModel(f"seedance-2.0-{v}", f"Seedance 2.0 {v.title()}", "Seedance", "seedance2", v, _range(4, 15))
      for v in ("standard", "mini", "fast")],
    *[VideoModel(f"seedance-{v}", f"Seedance {v.title()}", "Seedance", "seedance2", v, _range(4, 30))
      for v in ("2.5", "2.5-flash")],
]

IMAGE_BY_KEY = {(m.provider, m.id): m for m in IMAGE_MODELS}
VIDEO_BY_ID = {m.id: m for m in VIDEO_MODELS}
GEMINI_TEXT_BY_ID = {m.id: m for m in GEMINI_TEXT_MODELS}
OPENAI_TEXT_BY_ID = {m.id: m for m in OPENAI_TEXT_MODELS}


def openai_text_model(model_id: str) -> TextModel:
    try:
        return OPENAI_TEXT_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown OpenAI story-planning model: {model_id}") from exc


def gemini_text_model(model_id: str) -> TextModel:
    try:
        return GEMINI_TEXT_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Gemini story-planning model: {model_id}") from exc


def image_model(provider: str, model_id: str) -> ImageModel:
    try:
        return IMAGE_BY_KEY[(provider, model_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown {provider} image model: {model_id}") from exc


def video_model(model_id: str) -> VideoModel:
    try:
        return VIDEO_BY_ID[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown A2E video model: {model_id}") from exc


def public_catalog(default_a2e_image: str, default_video: str, configured_llm: bool = False) -> dict:
    # Never claim these models are enabled for the caller; this is a static catalog.
    return {
        "gemini_text_models": [{"id": m.id, "label": m.label, "note": m.note} for m in GEMINI_TEXT_MODELS],
        "openai_text_models": [{"id": m.id, "label": m.label, "note": m.note} for m in OPENAI_TEXT_MODELS],
        "configured_llm_available": configured_llm,
        "image_models": [{"id": m.id, "label": m.label, "provider": m.provider,
                          "group": m.group, "supports_references": m.supports_references,
                          "note": m.note} for m in IMAGE_MODELS],
        "video_models": [{"id": m.id, "label": m.label, "group": m.group,
                          "durations": m.durations, "default_duration": m.default_duration,
                          "input_mode": m.input_mode, "note": m.note} for m in VIDEO_MODELS],
        "defaults": {"a2e_image": default_a2e_image, "gemini_image": "gemini-nano-banana-2.1",
                     "video": default_video, "planning_provider": "configured" if configured_llm else "simple",
                     "gemini_text": DEFAULT_GEMINI_TEXT_MODEL,
                     "openai_text": DEFAULT_OPENAI_TEXT_MODEL},
    }
