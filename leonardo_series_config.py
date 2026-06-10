"""Leonardo storybook generation defaults (reference library lives in Supabase per story+grade)."""

import os
from typing import Any, Dict

_VALID_CONTROLNET_STRENGTHS = frozenset({"Low", "Mid", "High"})


def _env_controlnet_strength(name: str, default: str) -> str:
    v = (os.getenv(name) or "").strip()
    return v if v in _VALID_CONTROLNET_STRENGTHS else default

LEONARDO_GENERATION_DEFAULTS: Dict[str, Any] = {
    "modelId": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
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
STYLE_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_STYLE_CONTROLNET_STRENGTH", "High")
CHARACTER_CONTROLNET_PREPROCESSOR_ID = 133
CHARACTER_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_CHARACTER_CONTROLNET_STRENGTH", "Low")
LOCATION_CONTROLNET_PREPROCESSOR_ID = 100
LOCATION_CONTROLNET_STRENGTH = _env_controlnet_strength("LEONARDO_LOCATION_CONTROLNET_STRENGTH", "Low")
