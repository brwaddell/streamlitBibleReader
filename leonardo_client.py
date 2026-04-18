"""Leonardo.ai REST API client (no Streamlit). See https://docs.leonardo.ai/"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

LEONARDO_API_BASE = "https://cloud.leonardo.ai/api/rest/v1"

# https://docs.leonardo.ai/docs/phoenix — Phoenix 1.0 / 0.9 modelIds and Character Reference preprocessor 397
PHOENIX_1_MODEL_ID = "de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3"
PHOENIX_09_MODEL_ID = "6b645e3a-d64f-4341-a6d8-7a3690fbf042"
PHOENIX_MODEL_IDS = frozenset({PHOENIX_1_MODEL_ID.lower(), PHOENIX_09_MODEL_ID.lower()})

# Legacy SD-style models — Character Reference often uses 133 (see Image Guidance docs per model).
DEFAULT_CHARACTER_PREPROCESSOR_ID = int(os.getenv("LEONARDO_CHARACTER_PREPROCESSOR_ID", "133"))
DEFAULT_PHOENIX_CHARACTER_PREPROCESSOR_ID = int(os.getenv("LEONARDO_PHOENIX_CHARACTER_PREPROCESSOR_ID", "397"))
# Phoenix requires contrast in {3, 3.5, 4} when using Quality/Alchemy path. Default 3 = softer, less harsh than 3.5.
DEFAULT_PHOENIX_CONTRAST = float(os.getenv("LEONARDO_PHOENIX_CONTRAST", "3"))

# Output size: Leonardo requires 32–1536 and multiples of 8 (OpenAPI). Default 800 matches mobile export cap in lib.
_LEO_DIM_MIN = 32
_LEO_DIM_MAX = 1536


def snap_leonardo_dimension(x: int) -> int:
    """Clamp to API range and round down to a multiple of 8."""
    v = int(round(float(x)))
    v = max(_LEO_DIM_MIN, min(_LEO_DIM_MAX, v))
    return (v // 8) * 8


def _env_dimension(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return snap_leonardo_dimension(default)
    try:
        return snap_leonardo_dimension(int(raw))
    except ValueError:
        return snap_leonardo_dimension(default)


DEFAULT_LEONARDO_WIDTH = _env_dimension("LEONARDO_WIDTH", 800)
DEFAULT_LEONARDO_HEIGHT = _env_dimension("LEONARDO_HEIGHT", 800)

# Square presets offered in UI (all multiples of 8)
LEONARDO_SQUARE_PRESETS: Tuple[int, ...] = (512, 640, 768, 800, 1024)


def nearest_square_preset(side: int, presets: Tuple[int, ...] = LEONARDO_SQUARE_PRESETS) -> int:
    s = snap_leonardo_dimension(side)
    return min(presets, key=lambda p: abs(p - s))


def is_phoenix_model(model_id: str) -> bool:
    return (model_id or "").strip().lower() in PHOENIX_MODEL_IDS


def character_preprocessor_for_model(model_id: str) -> int:
    if is_phoenix_model(model_id):
        return DEFAULT_PHOENIX_CHARACTER_PREPROCESSOR_ID
    return DEFAULT_CHARACTER_PREPROCESSOR_ID


def _snap_phoenix_contrast(value: float) -> float:
    allowed = (3.0, 3.5, 4.0)
    return min(allowed, key=lambda x: abs(x - float(value)))


def effective_contrast_for_model(model_id: str, contrast: Optional[float]) -> Optional[float]:
    """Phoenix: always send contrast (default 3.5). Other models: only if caller passes contrast."""
    if is_phoenix_model(model_id):
        base = DEFAULT_PHOENIX_CONTRAST if contrast is None else float(contrast)
        return _snap_phoenix_contrast(base)
    if contrast is not None:
        return float(contrast)
    return None


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _extension_from_bytes(image_bytes: bytes) -> str:
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"\xff\xd8":
        return "jpeg"
    return "png"


def request_init_image_slot(api_key: str, extension: str) -> Dict[str, Any]:
    r = requests.post(
        f"{LEONARDO_API_BASE}/init-image",
        headers=_headers(api_key),
        json={"extension": extension},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    slot = data.get("uploadInitImage") or data.get("upload_init_image")
    if not slot:
        raise ValueError(f"Unexpected init-image response: {data}")
    return slot


def upload_init_image_bytes(api_key: str, image_bytes: bytes) -> str:
    """Register init image with Leonardo and upload bytes. Returns init image id for controlnets / init_image_id."""
    ext = _extension_from_bytes(image_bytes)
    slot = request_init_image_slot(api_key, ext)
    fields_raw = slot.get("fields")
    url = slot.get("url")
    img_id = slot.get("id")
    if not fields_raw or not url or not img_id:
        raise ValueError(f"Incomplete uploadInitImage: {slot}")
    fields = json.loads(fields_raw) if isinstance(fields_raw, str) else fields_raw
    fname = f"ref.{ext}" if ext != "jpeg" else "ref.jpg"
    up = requests.post(url, data=fields, files={"file": (fname, image_bytes)}, timeout=120)
    up.raise_for_status()
    return str(img_id)


def create_generation(
    api_key: str,
    prompt: str,
    model_id: str,
    *,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_images: int = 1,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
    preset_style: str = "ILLUSTRATION",
    alchemy: bool = True,
    init_image_id: Optional[str] = None,
    init_strength: Optional[float] = None,
    controlnets: Optional[List[Dict[str, Any]]] = None,
    contrast: Optional[float] = None,
) -> str:
    """POST /generations. Returns generationId (UUID string)."""
    w = snap_leonardo_dimension(width if width is not None else DEFAULT_LEONARDO_WIDTH)
    h = snap_leonardo_dimension(height if height is not None else DEFAULT_LEONARDO_HEIGHT)
    body: Dict[str, Any] = {
        "prompt": prompt,
        "modelId": model_id,
        "width": w,
        "height": h,
        "num_images": num_images,
        "guidance_scale": guidance_scale,
        "alchemy": alchemy,
        "presetStyle": preset_style,
    }
    ec = effective_contrast_for_model(model_id, contrast)
    if ec is not None:
        body["contrast"] = ec
    if negative_prompt:
        body["negative_prompt"] = negative_prompt
    if seed is not None:
        body["seed"] = int(seed)
    if init_image_id and init_strength is not None:
        body["init_image_id"] = init_image_id
        body["init_strength"] = float(init_strength)
    if controlnets:
        body["controlnets"] = controlnets

    r = requests.post(
        f"{LEONARDO_API_BASE}/generations",
        headers=_headers(api_key),
        json=body,
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Leonardo create generation failed: {r.status_code} {r.text}")
    data = r.json()
    job = data.get("sdGenerationJob") or data.get("sd_generation_job") or {}
    gen_id = job.get("generationId") or job.get("generation_id")
    if not gen_id:
        for _k, v in (data or {}).items():
            if isinstance(v, dict):
                gen_id = v.get("generationId") or v.get("generation_id")
                if gen_id:
                    break
    if not gen_id:
        raise ValueError(f"No generationId in response: {data}")
    return str(gen_id)


def get_generation(api_key: str, generation_id: str) -> Optional[Dict[str, Any]]:
    """GET /generations/{id} -> generations_by_pk object or None."""
    r = requests.get(
        f"{LEONARDO_API_BASE}/generations/{generation_id}",
        headers=_headers(api_key),
        timeout=60,
    )
    if not r.ok:
        return None
    data = r.json()
    return data.get("generations_by_pk") or data.get("generationsByPk")


def download_image(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def format_generation_cost(gen: Optional[Dict[str, Any]]) -> Optional[str]:
    """Human-readable cost from GET /generations/{id} (apiCreditCost and/or PAYG cost object)."""
    if not gen or not isinstance(gen, dict):
        return None
    bits: List[str] = []
    acc = gen.get("apiCreditCost")
    if acc is None:
        acc = gen.get("api_credit_cost")
    if acc is not None:
        try:
            bits.append(f"{int(acc)} API credits")
        except (TypeError, ValueError):
            bits.append(f"{acc} API credits")
    cost = gen.get("cost")
    if isinstance(cost, dict):
        amt = cost.get("amount") or cost.get("value") or cost.get("total")
        unit = (cost.get("unit") or cost.get("currency") or "").strip()
        if amt is not None:
            bits.append(f"{amt} {unit}".strip() if unit else str(amt))
    if bits:
        return "Leonardo: " + " · ".join(bits)
    return None


def poll_generation_until_done(
    api_key: str,
    generation_id: str,
    *,
    interval_sec: float = 2.0,
    max_wait_sec: float = 300.0,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Poll GET /generations/{id} until COMPLETE, FAILED, or timeout.
    Returns (image_bytes, error_message, cost_caption). On success error is None; cost_caption may be None if API omits it.
    """
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        gen = get_generation(api_key, generation_id)
        if not gen:
            time.sleep(interval_sec)
            continue
        status = (gen.get("status") or "").upper()
        if status == "COMPLETE":
            images = gen.get("generated_images") or gen.get("generatedImages") or []
            if not images:
                return None, "COMPLETE but no generated_images", None
            url = (images[0] or {}).get("url")
            if not url:
                return None, "COMPLETE but no image url", None
            try:
                cost_line = format_generation_cost(gen)
                return download_image(url), None, cost_line
            except Exception as e:
                return None, str(e), None
        if status == "FAILED":
            return None, "Generation FAILED", None
        time.sleep(interval_sec)
    return None, "Timeout waiting for Leonardo generation", None


def character_reference_controlnets(
    init_image_id: str,
    *,
    model_id: Optional[str] = None,
    preprocessor_id: Optional[int] = None,
    strength_type: str = "Mid",
    weight: float = 1.0,
) -> List[Dict[str, Any]]:
    if preprocessor_id is not None:
        pid = preprocessor_id
    elif model_id:
        pid = character_preprocessor_for_model(model_id)
    else:
        pid = DEFAULT_CHARACTER_PREPROCESSOR_ID
    # Phoenix image guidance: strengthType only — weight returns 400 (see Leonardo API error).
    use_phoenix_style_controlnet = is_phoenix_model(model_id or "") or pid == DEFAULT_PHOENIX_CHARACTER_PREPROCESSOR_ID
    entry: Dict[str, Any] = {
        "initImageId": init_image_id,
        "initImageType": "UPLOADED",
        "preprocessorId": pid,
        "strengthType": strength_type,
    }
    if not use_phoenix_style_controlnet:
        entry["weight"] = weight
    return [entry]
