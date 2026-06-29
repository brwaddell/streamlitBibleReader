"""Leonardo storybook generation defaults (reference library lives in Supabase per story+grade)."""

import os
from typing import Any, Dict

# Phoenix 1.0 — https://docs.leonardo.ai/docs/phoenix
PHOENIX_1_MODEL_ID = "de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3"

_VALID_CONTROLNET_STRENGTHS = frozenset({"Low", "Mid", "High"})


def _env_controlnet_strength(name: str, default: str) -> str:
    v = (os.getenv(name) or "").strip()
    return v if v in _VALID_CONTROLNET_STRENGTHS else default

LEONARDO_GENERATION_DEFAULTS: Dict[str, Any] = {
    "modelId": PHOENIX_1_MODEL_ID,
    "alchemy": True,
    "presetStyle": "ILLUSTRATION",
    "width": 1024,
    "height": 768,
    "num_images": 1,
    "guidance_scale": 7,
    "num_inference_steps": 30,
    "negative_prompt": (
        "photorealistic, 3d render, anime, comic book, cinematic, text, watermark, logo, "
        "distorted face, extra limbs"
    ),
}

# Controlnet settings for scene generation (when optional refs are selected).
STYLE_CONTROLNET_PREPROCESSOR_ID = 67
# Reference preview generation (style/character refs that should match the series look).
STYLE_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_STYLE_CONTROLNET_STRENGTH", "High")
# Story scene images — Low keeps palette similar without cloning one reference composition.
STYLE_CONTROLNET_SCENE_STRENGTH = _env_controlnet_strength(
    "LEONARDO_STYLE_CONTROLNET_SCENE_STRENGTH", "Low"
)
CHARACTER_CONTROLNET_PREPROCESSOR_ID = 133
CHARACTER_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_CHARACTER_CONTROLNET_STRENGTH", "Low")
LOCATION_CONTROLNET_PREPROCESSOR_ID = 100
LOCATION_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_LOCATION_CONTROLNET_STRENGTH", "Low")
