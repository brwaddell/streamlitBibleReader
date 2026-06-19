"""Shared utilities for Storybook Image Processor."""
import base64
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from auth import get_secret
from scene_prompts import (
    LEONARDO_ILLUSTRATOR_LOCK,
    LEONARDO_NEGATIVE_IMAGE_HARD_RULES,
    LEONARDO_NEGATIVE_KID_SOFT,
    LEONARDO_POSITIVE_IMAGE_RULES,
    SCENE_HARD_RULES,
)

READING_LEVELS = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
IMAGE_PROCESSOR_GRADES = ("grade_1", "grade_2", "grade_3", "grade_4")
PAGES_PER_IMAGE = 3
# Languages offered in Story Text / Audio / Book Pages dropdowns (English + Spanish only).
LANGUAGE_CODES = ["en", "es"]


def ui_language_select_options(versions: List[dict]) -> List[str]:
    """Intersect Supabase version codes with LANGUAGE_CODES; default to full UI list if none match."""
    from_db = {v.get("language_code") for v in (versions or []) if v.get("language_code")}
    filtered = sorted(from_db & set(LANGUAGE_CODES))
    return filtered if filtered else list(LANGUAGE_CODES)

# ElevenLabs voice options (name, voice_id, description) — shared by Audio Generator and Book Pages
VOICES_MALE = [
    ("Hale", "nzFihrBIvB34imQBuxub", "Warm, friendly"),
    ("Johnny Kid", "8JVbfL6oEdmuxKn5DK2C", "Serious, calm narrator"),
    ("David", "FF7KdobWPaiR0vkcALHF", "Deep, engaging"),
    ("Father Christmas", "1wg2wOjdEWKA7yQD8Kca", "Magical storyteller"),
]
VOICES_FEMALE = [
    ("Zara", "jqcCZkN6Knx8BJ5TBdYR", "Clear, natural"),
    ("Amelia", "ZF6FPAbjXT4488VcRRnw", "Enthusiastic, expressive"),
    ("Emma Taylor", "S9EGwlCtMF7VXtENq79v", "Gentle, thoughtful"),
    ("Rachel", "21m00Tcm4TlvDq8ikWAM", "Professional, warm"),
]

# Audio Generator batch: ElevenLabs quality model, fixed narrators, per-grade playback speed.
ELEVENLABS_TTS_MODEL_ID = "eleven_multilingual_v2"
ELEVENLABS_TTS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_VOICE_MALE_DEFAULT = "8JVbfL6oEdmuxKn5DK2C"  # Johnny Kid
ELEVENLABS_VOICE_FEMALE_DEFAULT = "jqcCZkN6Knx8BJ5TBdYR"  # Zara
AUDIO_TTS_SPEED_BY_GRADE = {
    "grade_1": 0.75,
    "grade_2": 0.8,
    "grade_3": 0.8,
    "grade_4": 0.8,
    "grade_5": 0.9,
}


def audio_tts_speed_for_grade(reading_level: str) -> float:
    return AUDIO_TTS_SPEED_BY_GRADE.get(reading_level or "", 0.8)


STORAGE_BUCKET = "storybook-images"
AUDIO_BUCKET = "storybook-audio"
MAX_IMAGE_SIZE = 800
TARGET_BYTES = 100_000

# One reference image for Gemini and Leonardo (Leonardo rejects multiple character-reference controlnets).
MAX_REFERENCE_IMAGES = 1
LEONARDO_MAX_CHARACTER_REFERENCE_REFS = 1


def parse_reference_images_json(raw: Any) -> List[Dict[str, str]]:
    """Normalize DB / JSON value to a list of {label, url} with non-empty URLs."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        lab = str(item.get("label") or "").strip() or "Reference"
        out.append({"label": lab, "url": url})
    return out[:MAX_REFERENCE_IMAGES]


def parse_storybook_references(raw: Any) -> Dict[str, Any]:
    """Normalize storybook_references JSON from story_grade_styles."""
    empty: Dict[str, Any] = {"style": None, "characters": [], "locations": []}
    if raw is None:
        return dict(empty)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return dict(empty)
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return dict(empty)
    if not isinstance(raw, dict):
        return dict(empty)
    style = raw.get("style")
    if style is not None and not isinstance(style, dict):
        style = None
    chars = raw.get("characters") if isinstance(raw.get("characters"), list) else []
    locs = raw.get("locations") if isinstance(raw.get("locations"), list) else []
    return {
        "style": style,
        "characters": [c for c in chars if isinstance(c, dict)],
        "locations": [loc for loc in locs if isinstance(loc, dict)],
    }


def _saved_ref_entry(entry: Optional[dict]) -> bool:
    """Approved reference with a stored URL — enough to show in the UI."""
    if not entry or not isinstance(entry, dict):
        return False
    if not entry.get("approved"):
        return False
    return bool((entry.get("url") or "").strip())


def _approved_ref_entry(entry: Optional[dict]) -> bool:
    """Approved reference usable for Leonardo generation (needs init image id)."""
    if not _saved_ref_entry(entry):
        return False
    return bool((entry.get("leonardo_init_image_id") or "").strip())


def get_saved_style_ref(refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    style = refs.get("style")
    return style if _saved_ref_entry(style) else None


def get_saved_character_refs(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in (refs.get("characters") or []) if _saved_ref_entry(c)]


def get_saved_location_refs(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [loc for loc in (refs.get("locations") or []) if _saved_ref_entry(loc)]


def get_approved_style_ref(refs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    style = refs.get("style")
    return style if _approved_ref_entry(style) else None


def get_approved_character_refs(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in (refs.get("characters") or []) if _approved_ref_entry(c)]


def get_approved_location_refs(refs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [loc for loc in (refs.get("locations") or []) if _approved_ref_entry(loc)]


def find_character_ref_by_id(refs: Dict[str, Any], ref_id: str) -> Optional[Dict[str, Any]]:
    rid = (ref_id or "").strip()
    if not rid:
        return None
    for entry in get_approved_character_refs(refs):
        if entry.get("id") == rid:
            return entry
    return None


def find_location_ref_by_id(refs: Dict[str, Any], ref_id: str) -> Optional[Dict[str, Any]]:
    rid = (ref_id or "").strip()
    if not rid:
        return None
    for entry in get_approved_location_refs(refs):
        if entry.get("id") == rid:
            return entry
    return None


def storybook_references_from_grade_style(style: Optional[dict]) -> Dict[str, Any]:
    if not style:
        return parse_storybook_references(None)
    return parse_storybook_references(style.get("storybook_references"))


def save_storybook_references(
    supabase, story_id: int, reading_level: str, refs: Dict[str, Any]
) -> bool:
    existing = get_story_grade_style(supabase, story_id, reading_level) or {}
    data = {k: existing[k] for k in STORY_STYLE_COLUMNS if k in existing}
    data["storybook_references"] = refs
    return upsert_story_grade_style(supabase, story_id, reading_level, data)


def seed_reference_text_slots(
    story_id: int,
    reading_level: str,
    max_slots: int,
    key_prefix: str,
    entries: List[Dict[str, str]],
) -> None:
    """Initialize label/URL text_input session keys from saved reference list.

    Must run before Streamlit creates widgets for those keys on the same run, or Streamlit raises.
    """
    for i in range(max_slots):
        lk = f"{key_prefix}_l_{story_id}_{reading_level}_{i}"
        uk = f"{key_prefix}_u_{story_id}_{reading_level}_{i}"
        if i < len(entries):
            st.session_state[lk] = str(entries[i].get("label") or "")
            st.session_state[uk] = str(entries[i].get("url") or "")
        else:
            st.session_state[lk] = ""
            st.session_state[uk] = ""


def reference_entries_from_text_slots(
    story_id: int,
    reading_level: str,
    max_slots: int,
    key_prefix: str,
) -> List[Dict[str, str]]:
    """Read label/URL from fixed Streamlit text_input keys; skip rows with empty URL."""
    out: List[Dict[str, str]] = []
    for i in range(max_slots):
        lk = f"{key_prefix}_l_{story_id}_{reading_level}_{i}"
        uk = f"{key_prefix}_u_{story_id}_{reading_level}_{i}"
        url = (st.session_state.get(uk) or "").strip()
        if not url:
            continue
        lab = (st.session_state.get(lk) or "").strip() or "Reference"
        out.append({"label": lab, "url": url})
    return out


def format_visual_reference_instructions(labels: List[str]) -> str:
    """Prompt text so the model maps each supplied image index to a named design."""
    if not labels:
        return ""
    bits = [
        f"Image {i} ({lab}): match this design whenever it appears; keep it consistent across the series."
        for i, lab in enumerate(labels, start=1)
    ]
    return "VISUAL REFERENCE IMAGES are provided in order. " + " ".join(bits)


def _normalize_ref_url(url: str) -> str:
    return (url or "").strip().lower().rstrip("/")


def collect_reference_images(
    ref_file=None,
    selected_page_index: Optional[int] = None,
    pages=None,
    ref_image_url: Optional[str] = None,
    additional_ref_urls: Optional[List[str]] = None,
    saved_labeled: Optional[List[Dict[str, str]]] = None,
    legacy_canonical_url: Optional[str] = None,
) -> Tuple[List[bytes], str]:
    """
    Build ordered reference image bytes (max MAX_REFERENCE_IMAGES) and matching prompt instructions.
    Order: file upload → saved labeled URLs → legacy canonical URL → additional URLs → in-session page
    image → published page URL. De-duplicates by URL.
    """
    seen_urls: set = set()
    refs: List[bytes] = []
    labels: List[str] = []

    def add_bytes(data: bytes, label: str) -> None:
        if len(refs) >= MAX_REFERENCE_IMAGES or not data:
            return
        refs.append(data)
        labels.append(label)

    if ref_file is not None:
        if hasattr(ref_file, "seek"):
            ref_file.seek(0)
        data = ref_file.read()
        if data:
            add_bytes(data, "Uploaded reference")

    for entry in saved_labeled or []:
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        nu = _normalize_ref_url(url)
        if nu in seen_urls:
            continue
        seen_urls.add(nu)
        try:
            r = requests.get(url, timeout=15)
            if r.ok and r.content:
                add_bytes(r.content, str(entry.get("label") or "").strip() or "Reference")
        except Exception:
            pass

    leg = (legacy_canonical_url or "").strip()
    if leg:
        nu = _normalize_ref_url(leg)
        if nu not in seen_urls:
            seen_urls.add(nu)
            try:
                r = requests.get(leg, timeout=15)
                if r.ok and r.content:
                    add_bytes(r.content, "Character (canonical)")
            except Exception:
                pass

    for u in additional_ref_urls or []:
        u = str(u or "").strip()
        if not u:
            continue
        nu = _normalize_ref_url(u)
        if nu in seen_urls:
            continue
        seen_urls.add(nu)
        try:
            r = requests.get(u, timeout=15)
            if r.ok and r.content:
                add_bytes(r.content, "Reference")
        except Exception:
            pass

    if selected_page_index is not None and pages and 0 <= selected_page_index < len(pages):
        p = pages[selected_page_index]
        img = p.get("image")
        if img:
            add_bytes(img, "Page reference")

    pub = (ref_image_url or "").strip()
    if pub:
        nu = _normalize_ref_url(pub)
        if nu not in seen_urls:
            seen_urls.add(nu)
            try:
                r = requests.get(pub, timeout=15)
                if r.ok and r.content:
                    add_bytes(r.content, "Published page reference")
            except Exception:
                pass

    note = format_visual_reference_instructions(labels)
    return refs, note

# Flattened story content table (replaces book_pages, localized_story_versions, story_assets)
TABLE_STORY_CONTENT_FLAT = "story_content_flat"
PAGE_NUMBER_COLUMN = "page_number"  # Always order by this ascending for correct story sequence

# --- Image generation (Gemini); shared by Image Processor and Book Pages ---
SYSTEM_INSTRUCTIONS = (
    "You are a consistent storybook illustrator. CRITICAL: The main character must have the EXACT SAME face, hair, beard, and clothing in every image. "
    "Use the provided character description as a fixed design—do not vary it. No creative reinterpretation of the character. "
    "Quality: Standard. Aspect Ratio: 1024x1024. Never include text, words, or letters in the image.\n\n"
    + SCENE_HARD_RULES
    + "\n"
)


def build_prompt(
    extra_details: str,
    story_text: str,
    age_appropriateness: str,
    global_style: str,
    character_ref: str,
    lighting: str,
    palette: str,
    framing: str = "",
    visual_reference_note: str = "",
) -> str:
    """Build a structured prompt with explicit sections for all style controls, main text, and extra details."""
    extra = (extra_details or "").strip()
    story = (story_text or "").strip()
    age = (age_appropriateness or "").strip()
    style = (global_style or "").strip()
    char = (character_ref or "").strip()
    light = (lighting or "").strip()
    pal = (palette or "").strip()
    frame = (framing or "").strip()
    vref = (visual_reference_note or "").strip()

    sections = []
    sections.append(f"SCENE TO ILLUSTRATE: {story}" if story else "SCENE TO ILLUSTRATE: (illustrate the story moment)")
    if vref:
        sections.append(vref)
    if extra:
        sections.append(f"EXTRA VISUAL DETAILS: {extra}")
    if age:
        sections.append(f"AGE APPROPRIATENESS: {age}")
    if style:
        sections.append(f"GLOBAL STYLE: {style}")
    if char:
        sections.append(f"CHARACTER REFERENCE (use exactly, keep consistent): {char}")
    if light:
        sections.append(f"LIGHTING: {light}")
    if pal:
        sections.append(f"COLOR PALETTE: {pal}")
    if frame:
        sections.append(f"FRAMING: {frame}")

    body = ". ".join(sections)
    return f"{SYSTEM_INSTRUCTIONS}{body}"


LEONARDO_NEGATIVE_BASE = (
    "text, words, letters, typography, watermark, signature, logo, blurry, low quality, "
    "deformed hands, extra fingers, cropped, worst quality, "
    + LEONARDO_NEGATIVE_IMAGE_HARD_RULES
    + ", "
    + LEONARDO_NEGATIVE_KID_SOFT
)


def _leonardo_style_signal_mode(reading_level: Optional[str]) -> str:
    """Fewer stacked style cues for young grades = less conflict with simple art direction."""
    r = (reading_level or "").strip().lower()
    if r in ("grade_1", "grade_2"):
        return "minimal"
    if r == "grade_3":
        return "standard"
    return "full"


def _trim_leonardo_positive(text: str, max_len: int = 1480) -> str:
    """Stay under Leonardo body limits without hard mid-word API truncation."""
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    cut = t[:max_len]
    if " " in cut:
        return cut.rsplit(" ", 1)[0].rstrip(",; ")
    return cut


def build_leonardo_prompt(
    extra_details: str,
    story_text: str,
    age_appropriateness: str,
    global_style: str,
    character_ref: str,
    lighting: str,
    palette: str,
    framing: str = "",
    user_negative_suffix: str = "",
    visual_reference_note: str = "",
    reading_level: Optional[str] = None,
) -> Tuple[str, str]:
    """Build (positive_prompt, negative_prompt) tuned for Leonardo — Gemini-aligned lead, then scene and style."""
    extra = (extra_details or "").strip()
    story = (story_text or "").strip()
    vref = (visual_reference_note or "").strip()
    mode = _leonardo_style_signal_mode(reading_level)

    parts: List[str] = [LEONARDO_ILLUSTRATOR_LOCK.strip(), LEONARDO_POSITIVE_IMAGE_RULES]
    if (age_appropriateness or "").strip():
        parts.append(f"Audience: {(age_appropriateness or '').strip()[:320]}")
    if story:
        parts.append(f"Scene: {story[:1200]}")
    if vref:
        parts.append(vref[:600])
    if extra:
        parts.append(extra[:800])
    if (character_ref or "").strip():
        parts.append(f"Characters: {(character_ref or '').strip()[:600]}")

    style_bits: List[str] = []
    if (global_style or "").strip():
        style_bits.append((global_style or "").strip()[:400])
    if (palette or "").strip():
        style_bits.append(f"Colors: {(palette or '').strip()[:120]}")
    if mode in ("standard", "full") and (lighting or "").strip():
        style_bits.append(f"Lighting: {(lighting or '').strip()[:120]}")
    if mode == "full" and (framing or "").strip():
        style_bits.append(f"Framing: {(framing or '').strip()[:120]}")
    if style_bits:
        parts.append("Style: " + "; ".join(style_bits))
    if mode == "minimal":
        parts.append(
            "Composition: one clear friendly focal subject, gentle non-scary mood, uncluttered background."
        )
    parts.append("Children's storybook illustration, cohesive series, no text in image.")
    positive = _trim_leonardo_positive(" ".join(p for p in parts if p).strip())
    if not positive:
        positive = "Storybook illustration, friendly scene, no text in image."
    neg = LEONARDO_NEGATIVE_BASE
    if user_negative_suffix and user_negative_suffix.strip():
        neg = f"{neg}, {user_negative_suffix.strip()}"
    return positive, neg


def build_simple_leonardo_prompt(
    scene_description: str,
    reading_level: Optional[str] = None,
    *,
    style_scene: Optional[str] = None,
    character_ref: Optional[Dict[str, Any]] = None,
    location_ref: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Grade-aware storybook prompt for scene generation (Image Processor)."""
    from grade_style_defaults import grade_style_defaults_for
    from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS
    from scene_prompts import LEONARDO_ILLUSTRATOR_LOCK, LEONARDO_POSITIVE_IMAGE_RULES

    g = grade_style_defaults_for(reading_level or "grade_1")
    scene = (scene_description or "").strip()
    parts = [
        LEONARDO_ILLUSTRATOR_LOCK.strip(),
        LEONARDO_POSITIVE_IMAGE_RULES,
    ]
    style = (style_scene or "").strip()
    if style:
        parts.append(f"STYLE SCENE (world, setting, and visual tone for this book): {style}")
    parts.append(f"SCENE (this illustration — composition and action must follow this): {scene}")
    if character_ref:
        label = (character_ref.get("label") or "the character").strip()
        parts.append(
            f"Character reference: match {label}'s face, hair, and clothing only; "
            f"pose, action, and scene staging follow the SCENE description."
        )
    if location_ref:
        label = (location_ref.get("label") or "the location").strip()
        parts.append(
            f"Location reference: borrow background architecture and mood from {label} only; "
            f"do not copy the empty reference composition; characters and action follow the SCENE."
        )
    age = (g.get("age_appropriateness") or "").strip()
    if age:
        parts.append(f"Audience: {age}")
    palette = (g.get("color_palette") or "").strip()
    if palette:
        parts.append(f"Colors: {palette}")
    lighting = (g.get("lighting") or "").strip()
    if lighting:
        parts.append(f"Lighting: {lighting}")
    framing = (g.get("framing") or "").strip()
    if framing:
        parts.append(f"Framing: {framing}")
    parts.append("Children's storybook illustration, cohesive series, no text in image.")
    positive = _trim_leonardo_positive("\n".join(p for p in parts if p))
    negative = LEONARDO_GENERATION_DEFAULTS["negative_prompt"]
    return positive, negative


def build_leonardo_reference_prompt(
    description: str, reading_level: Optional[str] = None
) -> Tuple[str, str]:
    """Prompt for generating a reference image (style / character / location), not a story scene."""
    from grade_style_defaults import grade_style_defaults_for
    from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS

    g = grade_style_defaults_for(reading_level or "grade_1")
    desc = (description or "").strip()
    parts = [
        "A children's storybook illustration reference image.",
        desc,
        (g.get("age_appropriateness") or "").strip(),
        "Iconic, reusable visual design, clean composition, no text in image.",
    ]
    positive = _trim_leonardo_positive("\n".join(p for p in parts if p))
    negative = LEONARDO_GENERATION_DEFAULTS["negative_prompt"]
    return positive, negative


def get_reference_images(
    ref_file=None,
    selected_page_index: Optional[int] = None,
    pages=None,
    ref_image_url: Optional[str] = None,
    additional_ref_urls: Optional[List[str]] = None,
    saved_labeled: Optional[List[Dict[str, str]]] = None,
    legacy_canonical_url: Optional[str] = None,
) -> List[bytes]:
    """Collect reference image bytes (see collect_reference_images for order and de-duplication)."""
    refs, _ = collect_reference_images(
        ref_file=ref_file,
        selected_page_index=selected_page_index,
        pages=pages,
        ref_image_url=ref_image_url,
        additional_ref_urls=additional_ref_urls,
        saved_labeled=saved_labeled,
        legacy_canonical_url=legacy_canonical_url,
    )
    return refs


def reference_entries_from_grade_style(style: Optional[dict]) -> List[Dict[str, str]]:
    """Labeled URL rows from saved style, migrating legacy character_reference_image_url when list is empty."""
    if not style:
        return []
    entries = parse_reference_images_json(style.get("reference_images"))
    if entries:
        return entries[:MAX_REFERENCE_IMAGES]
    leg = (style.get("character_reference_image_url") or "").strip()
    if leg:
        return [{"label": "Character", "url": leg}]
    return []


def get_gemini():
    """Get Google Gemini client (cached in session state)."""
    if st.session_state.gemini_client is None:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            st.error("Set GEMINI_API_KEY in .env for Gemini image generation.")
            return None
        try:
            from google import genai
            st.session_state.gemini_client = genai.Client(api_key=api_key)
        except ImportError:
            st.error("Install google-genai: pip install google-genai")
            return None
    return st.session_state.gemini_client


def _gemini_usage_line(response) -> Optional[str]:
    """Short caption from generate_content usage_metadata, if present."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None

    def _g(obj, snake: str, camel: str):
        v = getattr(obj, snake, None)
        if v is not None:
            return v
        return getattr(obj, camel, None)

    pt = _g(um, "prompt_token_count", "promptTokenCount")
    ct = _g(um, "candidates_token_count", "candidatesTokenCount")
    tt = _g(um, "total_token_count", "totalTokenCount")
    bits = []
    if pt is not None:
        bits.append(f"prompt {pt} tok")
    if ct is not None:
        bits.append(f"candidates {ct} tok")
    if tt is not None:
        bits.append(f"total {tt} tok")
    if bits:
        return "Gemini: " + ", ".join(bits)
    return None


def _prepare_reference_images(reference_images: Optional[List[bytes]]) -> List[Image.Image]:
    """Validate and normalize reference images to RGB PIL Images. Skips any that fail."""
    prepared = []
    if not reference_images:
        return prepared
    for img_bytes in reference_images[:MAX_REFERENCE_IMAGES]:
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            prepared.append(img)
        except Exception:
            pass
    return prepared


def _generate_image_with_client(
    client, prompt: str, reference_images: Optional[List[bytes]] = None
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Generate image using an existing Gemini client. Returns (image_bytes, error_message, usage_caption)."""
    try:
        from google.genai import types
        ref_imgs = _prepare_reference_images(reference_images)
        contents: list = [prompt] + ref_imgs
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        if not response.candidates:
            reason = "No response candidates"
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                pf = response.prompt_feedback
                if getattr(pf, "block_reason", None):
                    reason = f"Blocked: {pf.block_reason}"
                if getattr(pf, "block_reason_message", None):
                    reason = f"{reason} — {pf.block_reason_message}"
            return (None, reason, None)
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            fr = getattr(finish_reason, "name", str(finish_reason)).upper()
            if fr not in ("STOP", "END_TURN", "1", ""):
                return (None, f"Response finish reason: {finish_reason}", None)
        usage = _gemini_usage_line(response)
        parts = candidate.content.parts
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                data = getattr(part.inline_data, "data", None)
                if data is not None:
                    if isinstance(data, bytes):
                        return (data, None, usage)
                    try:
                        return (base64.b64decode(data), None, usage)
                    except Exception:
                        return (data if isinstance(data, bytes) else bytes(data), None, usage)
            img = getattr(part, "as_image", lambda: None)()
            if img is not None:
                buf = BytesIO()
                try:
                    img.save(buf, "PNG")
                except TypeError:
                    img.save(buf)
                return (buf.getvalue(), None, usage)
        return (None, "No image in response", usage)
    except Exception as e:
        return (None, str(e), None)


_GEMINI_MAX_RETRIES = 2


def generate_image_gemini(prompt: str, reference_images: Optional[List[bytes]] = None) -> Tuple[Optional[bytes], Optional[str]]:
    """Generate image via Gemini (gemini-3-pro-image-preview). Retries on transient errors.

    Returns (image_bytes, usage_caption). usage_caption is set on success when the API returns usage metadata.
    """
    client = get_gemini()
    if not client:
        return None, None
    last_error = None
    last_usage: Optional[str] = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        result, error, usage = _generate_image_with_client(client, prompt, reference_images)
        if result is not None:
            return result, usage
        last_usage = usage
        last_error = error or "Unknown error"
        if attempt < _GEMINI_MAX_RETRIES - 1:
            if any(code in last_error for code in ("400", "500", "503", "UNAVAILABLE")):
                time.sleep(2)
                continue
            break
        break
    ref_count = len(reference_images) if reference_images else 0
    diag = f"Prompt length: {len(prompt)} chars, reference images: {ref_count}"
    st.error(f"Gemini image generation failed: {last_error}")
    st.caption(f"Debug: {diag}")
    return None, last_usage


def generate_image_leonardo(
    positive: str,
    negative: str,
    reference_images: Optional[List[bytes]] = None,
    *,
    api_key: str,
    model_id: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    guidance_scale: float = 6.0,
    seed: Optional[int] = None,
    controlnet_strength: str = "Low",
    preprocessor_id: Optional[int] = None,
    contrast: Optional[float] = None,
    controlnets: Optional[List[Dict[str, Any]]] = None,
    preset_style: str = "ILLUSTRATION",
    alchemy: bool = True,
) -> Tuple[Optional[bytes], Optional[str]]:
    """Generate one image via Leonardo; poll until complete.

    Pass pre-built controlnets for the storybook flow, or reference_images for legacy single character ref.
    Returns (image_bytes, cost_caption).
    """
    import leonardo_client as leo

    if not api_key or not model_id:
        st.error("Set LEONARDO_API_KEY and LEONARDO_MODEL_ID.")
        return None, None
    cn = controlnets
    try:
        if cn is None and reference_images:
            cn = []
            for chunk in reference_images[:LEONARDO_MAX_CHARACTER_REFERENCE_REFS]:
                init_id = leo.upload_init_image_bytes(api_key, chunk)
                cn.extend(
                    leo.character_reference_controlnets(
                        init_id,
                        model_id=model_id,
                        strength_type=controlnet_strength,
                        preprocessor_id=preprocessor_id,
                    )
                )
        gen_id = leo.create_generation(
            api_key,
            positive[:1500],
            model_id,
            negative_prompt=negative[:1000],
            width=width,
            height=height,
            num_images=1,
            guidance_scale=guidance_scale,
            seed=seed,
            preset_style=preset_style,
            alchemy=alchemy,
            controlnets=cn,
            contrast=contrast,
        )
        img, err, cost_line, _gen_img_id = leo.poll_generation_until_done(
            api_key, gen_id, interval_sec=2.0, max_wait_sec=360.0
        )
        if img is not None:
            return img, cost_line
        st.error(f"Leonardo generation failed: {err or 'unknown'}")
        return None, None
    except Exception as e:
        st.error(f"Leonardo error: {e}")
        return None, None


def generate_leonardo_reference_preview(
    prompt: str,
    *,
    api_key: str,
    model_id: str,
    reading_level: Optional[str] = None,
    controlnets: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Generate a reference image. Returns (bytes, cost_caption, generated_image_id)."""
    import leonardo_client as leo
    from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS

    if not api_key or not model_id:
        st.error("Set LEONARDO_API_KEY.")
        return None, None, None
    text = (prompt or "").strip()
    if not text:
        st.warning("Enter a prompt first.")
        return None, None, None
    pos, neg = build_leonardo_reference_prompt(text, reading_level)
    defaults = LEONARDO_GENERATION_DEFAULTS
    try:
        gen_id = leo.create_generation(
            api_key,
            pos[:1500],
            model_id,
            negative_prompt=neg[:1000],
            width=int(defaults["width"]),
            height=int(defaults["height"]),
            num_images=1,
            guidance_scale=float(defaults["guidance_scale"]),
            preset_style=str(defaults["presetStyle"]),
            alchemy=bool(defaults["alchemy"]),
            controlnets=controlnets,
        )
        img, err, cost_line, gen_img_id = leo.poll_generation_until_done(
            api_key, gen_id, interval_sec=2.0, max_wait_sec=360.0
        )
        if img is not None:
            return img, cost_line, gen_img_id
        st.error(f"Leonardo generation failed: {err or 'unknown'}")
        return None, None, None
    except Exception as e:
        st.error(f"Leonardo error: {e}")
        return None, None, None


def _build_batch_request_parts(prompt: str, ref_images: Optional[List[bytes]] = None) -> list:
    """Build contents parts for a Batch API request: text + optional base64 ref images."""
    parts = [{"text": prompt}]
    if ref_images:
        for img_bytes in ref_images[:MAX_REFERENCE_IMAGES]:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    return parts


def _extract_image_from_batch_response(response) -> Optional[bytes]:
    """Extract image bytes from a batch response part. Returns None if no image."""
    if not response or not hasattr(response, "candidates"):
        return None
    candidates = getattr(response, "candidates", []) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    if not content or not hasattr(content, "parts"):
        return None
    for part in content.parts or []:
        if hasattr(part, "inline_data") and part.inline_data:
            data = getattr(part.inline_data, "data", None)
            if data:
                try:
                    return base64.b64decode(data) if isinstance(data, str) else data
                except Exception:
                    pass
        img = getattr(part, "as_image", lambda: None)()
        if img is not None:
            buf = BytesIO()
            try:
                img.save(buf, "PNG")
            except TypeError:
                img.save(buf)
            return buf.getvalue()
    return None


def fetch_image_for_display(url: str) -> Optional[bytes]:
    """Fetch image from URL server-side (avoids CORS with R2/Supabase). Returns bytes or None."""
    if not url or not url.strip():
        return None
    try:
        r = requests.get(url.strip(), timeout=10)
        if r.ok and r.content:
            return r.content
    except Exception:
        pass
    return None


def get_supabase():
    """Get Supabase client (cached in session state)."""
    if getattr(st.session_state, "supabase", None) is None:
        url = get_secret("SUPABASE_URL")
        key = get_secret("SUPABASE_SERVICE_KEY")
        if not url or not key:
            st.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
            return None
        from supabase import create_client
        st.session_state.supabase = create_client(url, key)
    return st.session_state.supabase


def fetch_stories():
    """Fetch all stories from Supabase for dropdown. Include character_ref and other style fields if present."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        r = (
            sb.table("stories")
            .select('id, title, description, "Free", publish, created_at')
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        data = r.data or []
        # Normalize case-sensitive DB columns to lowercase for app code
        for s in data:
            if "Free" in s:
                s["free"] = s.pop("Free")
            if "publish" in s:
                s["published"] = s.pop("publish")
        return data
    except Exception as e:
        st.error(f"Failed to fetch stories: {e}")
        return []


# Non-linked columns on stories table that can be edited in CRUD (excludes id, created_at, updated_at, etc.)
STORY_EDITABLE_COLUMNS = ["title", "description", "free", "published"]

# Style columns (Image Style Controls) - persisted per (story, grade) in story_grade_styles
STORY_STYLE_COLUMNS = [
    "age_appropriateness",
    "global_style",
    "character_ref",
    "color_palette",
    "lighting",
    "framing",
    "reference_page_index",
    "reference_images",
    "character_reference_image_url",
    "leonardo_seed",
    "default_image_provider",
    "storybook_references",
]


def get_story_grade_style(supabase, story_id: int, reading_level: str):
    """Fetch saved style for (story_id, reading_level) from story_grade_styles. Returns dict or None."""
    try:
        r = (
            supabase.table("story_grade_styles")
            .select("*")
            .eq("story_id", story_id)
            .eq("reading_level", reading_level)
            .limit(1)
            .execute()
        )
        rows = r.data or []
        return rows[0] if rows else None
    except Exception as e:
        st.error(f"Failed to load style: {e}")
        return None


def upsert_story_grade_style(supabase, story_id: int, reading_level: str, data: dict):
    """Insert or update style for (story_id, reading_level). Persists per story and grade."""
    try:
        row = {"story_id": story_id, "reading_level": reading_level}
        for k in STORY_STYLE_COLUMNS:
            if k not in data:
                continue
            if k == "reference_page_index":
                row[k] = data[k]  # int or None
            elif k == "leonardo_seed":
                row[k] = data[k]  # int or None
            elif k in ("reference_images", "storybook_references"):
                v = data[k]
                if k == "reference_images":
                    if v is None:
                        row[k] = []
                    elif isinstance(v, str):
                        row[k] = json.loads(v) if v.strip() else []
                    else:
                        row[k] = list(v)
                else:
                    if v is None:
                        row[k] = {}
                    elif isinstance(v, str):
                        row[k] = json.loads(v) if v.strip() else {}
                    elif isinstance(v, dict):
                        row[k] = v
                    else:
                        row[k] = {}
            else:
                row[k] = data[k] if data[k] is not None else ""
        supabase.table("story_grade_styles").upsert(row, on_conflict="story_id,reading_level").execute()
        return True
    except Exception as e:
        st.error(f"Failed to save style: {e}")
        return False


def insert_story(supabase, data: dict):
    """Insert a row into stories. Sends title, description, Free, publish (id is auto-generated)."""
    try:
        row = {}
        for k in STORY_EDITABLE_COLUMNS:
            if k in data:
                row[k] = data[k]
            elif k in ("free", "published"):
                row[k] = False
        row.pop("id", None)
        # Map to case-sensitive DB column names
        if "free" in row:
            row["Free"] = row.pop("free")
        if "published" in row:
            row["publish"] = row.pop("published")
        r = supabase.table("stories").insert(row).execute()
        return r.data[0] if r.data else None
    except Exception as e:
        if "23505" in str(e) and "stories_pkey" in str(e):
            st.error(
                "Create failed: the stories id sequence is out of sync. In Supabase SQL Editor run: "
                "SELECT setval(pg_get_serial_sequence('stories', 'id'), COALESCE((SELECT MAX(id) FROM stories), 0) + 1);"
            )
        else:
            st.error(f"Failed to create story: {e}")
        return None


def update_story(supabase, story_id: int, data: dict):
    """Update a story by id. Only updates keys that are in STORY_EDITABLE_COLUMNS and present in data."""
    try:
        row = {k: data[k] for k in STORY_EDITABLE_COLUMNS if k in data}
        if not row:
            return True
        # Map to case-sensitive DB column names
        if "free" in row:
            row["Free"] = row.pop("free")
        if "published" in row:
            row["publish"] = row.pop("published")
        supabase.table("stories").update(row).eq("id", story_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to update story: {e}")
        return False


def delete_story(supabase, story_id: int):
    """Delete a story by id. May fail if referenced by localized_story_versions or elsewhere."""
    try:
        supabase.table("stories").delete().eq("id", story_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to delete story: {e}")
        return False


def get_story_page_data(
    supabase,
    story_id: int,
    language_code: str,
    reading_level: str,
) -> List[dict]:
    """Fetch story content via get_story_page_data RPC. Returns pages with text, image_url, audio_male_url, audio_female_url, timing_json.
    RPC param names must match your Supabase function (e.g. p_story_id if your function uses that)."""
    try:
        r = supabase.rpc(
            "get_story_page_data",
            {"p_story_id": story_id, "p_lang": language_code, "p_gender": "male"},
        ).execute()
        return r.data or []
    except Exception as e:
        st.error(f"Failed to fetch story pages via RPC: {e}")
        return []


def check_subscription_active(supabase, profile_id: str) -> bool:
    """Verify profile_id has 'active' status in user_subscriptions before allowing story previews."""
    if not profile_id:
        return False
    try:
        r = (
            supabase.table("user_subscriptions")
            .select("status")
            .eq("profile_id", profile_id)
            .limit(1)
            .execute()
        )
        rows = r.data or []
        return bool(rows and rows[0].get("status") == "active")
    except Exception as e:
        st.error(f"Failed to check subscription: {e}")
        return False


def get_or_create_localized_version(
    supabase, story_id: int, language_code: str, reading_level: str
) -> Optional[int]:
    """No-op for simplified schema: no localized_story_versions table. Returns story_id for API compatibility."""
    return story_id


def insert_story_asset(supabase, image_url: str) -> Optional[str]:
    """No-op for simplified schema: no story_assets table. Returns image_url for story_content_flat.image_url."""
    return image_url


def find_book_page(
    supabase, story_id: int, reading_level: str, page_index: int, language_code: str = "en"
) -> Optional[dict]:
    """Find a row in story_content_flat by story_id, language_code, reading_level, page_number. Returns row with id, image_url or None."""
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select("*")
            .eq("story_id", story_id)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level)
            .eq(PAGE_NUMBER_COLUMN, page_index)
            .limit(1)
            .execute()
        )
        rows = r.data or []
        row = rows[0] if rows else None
        if row is not None and "page_index" not in row:
            row["page_index"] = row.get(PAGE_NUMBER_COLUMN)
        return row
    except Exception as e:
        st.error(f"Failed to find book page: {e}")
        return None


def fetch_book_pages(
    supabase, story_id: int, reading_level: str, language_code: str = "en", search: str = ""
):
    """Fetch story_content_flat by story_id, reading_level, language_code. Ordered by page_number ascending."""
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select("*")
            .eq("story_id", story_id)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level)
            .order(PAGE_NUMBER_COLUMN)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            row["_display_text"] = _get_page_text(row)
            if "page_index" not in row:
                row["page_index"] = row.get(PAGE_NUMBER_COLUMN)
        if search and search.strip():
            s = search.strip().lower()
            rows = [row for row in rows if s in (row.get("_display_text", "") or "").lower()]
        return rows
    except Exception as e:
        st.error(f"Failed to fetch book pages: {e}")
        return []


def delete_book_page(supabase, row_id):
    """Delete a story_content_flat row by id."""
    try:
        supabase.table(TABLE_STORY_CONTENT_FLAT).delete().eq("id", row_id).execute()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


# Column name for page text in story_content_flat - use "text" if your schema uses that
PAGE_TEXT_COLUMN = "page_text"


def _get_page_text(row: dict) -> str:
    """Get page text from row, supports both page_text and text column names."""
    return (row.get("page_text") or row.get("text") or "").strip()


def page_number_for_row(row: dict) -> int:
    v = row.get("page_index")
    if v is None:
        v = row.get(PAGE_NUMBER_COLUMN)
    return int(v or 0)


def is_image_anchor_page(page_number: int) -> bool:
    return page_number % PAGES_PER_IMAGE == 0


def image_block_start(page_number: int) -> int:
    return (page_number // PAGES_PER_IMAGE) * PAGES_PER_IMAGE


def uses_triple_page_images(reading_level: str) -> bool:
    return reading_level in IMAGE_PROCESSOR_GRADES


def format_page_range(block_start: int, block_end: int) -> str:
    if block_start == block_end:
        return str(block_start)
    return f"{block_start}–{block_end}"


def combined_text_for_pages(pages: List[dict]) -> str:
    parts = [_get_page_text(p) for p in pages]
    return "\n\n".join(p for p in parts if p)


def group_pages_into_image_blocks(pages: List[dict]) -> List[dict]:
    """Group ordered story pages into 3-page image blocks (anchor = first page of each block)."""
    if not pages:
        return []
    sorted_pages = sorted(pages, key=page_number_for_row)
    groups: Dict[int, List[dict]] = {}
    for row in sorted_pages:
        bs = image_block_start(page_number_for_row(row))
        groups.setdefault(bs, []).append(row)
    blocks = []
    for bs in sorted(groups.keys()):
        members = sorted(groups[bs], key=page_number_for_row)
        anchor = members[0]
        for m in members:
            if is_image_anchor_page(page_number_for_row(m)):
                anchor = m
                break
        end_pn = page_number_for_row(members[-1])
        blocks.append(
            {
                "anchor_row": anchor,
                "member_rows": members,
                "combined_text": combined_text_for_pages(members),
                "page_range_label": format_page_range(bs, end_pn),
                "block_start": bs,
            }
        )
    return blocks


def block_needs_image(block: dict) -> bool:
    urls = [(m.get("image_url") or "").strip() for m in block["member_rows"]]
    if not any(urls):
        return True
    return not all(urls)


def find_image_block_for_row_id(blocks: List[dict], row_id) -> Optional[dict]:
    if row_id is None:
        return None
    for block in blocks:
        for m in block["member_rows"]:
            if m.get("id") == row_id:
                return block
    return None


def apply_image_url_to_block(supabase, block: dict, image_url: str) -> Tuple[int, List[str]]:
    """Write the same image_url to every row in the block. Returns (success_count, error labels)."""
    url = (image_url or "").strip()
    ok = 0
    errs: List[str] = []
    for member in block["member_rows"]:
        rid = member.get("id")
        pn = page_number_for_row(member)
        if rid and update_book_page(supabase, rid, image_url=url):
            ok += 1
        else:
            errs.append(f"page {pn}")
    return ok, errs

def insert_book_page(
    supabase,
    story_id: int,
    language_code: str,
    reading_level: str,
    page_index: int,
    page_text: str,
    image_url: Optional[str] = None,
    audio_male_url: Optional[str] = None,
    audio_female_url: Optional[str] = None,
    timing_male_json: Optional[dict] = None,
    timing_female_json: Optional[dict] = None,
):
    """Insert a story_content_flat row. Sends: story_id, language_code, reading_level, page_number, page_text, image_url, audio/timing."""
    try:
        text_val = (page_text or "").strip() if page_text is not None else ""
        row = {
            "story_id": story_id,
            "language_code": language_code,
            "reading_level": reading_level,
            PAGE_NUMBER_COLUMN: page_index,
            PAGE_TEXT_COLUMN: text_val,
        }
        if image_url is not None and (image_url or "").strip():
            row["image_url"] = (image_url or "").strip()
        if audio_male_url is not None:
            row["audio_male_url"] = audio_male_url
        if audio_female_url is not None:
            row["audio_female_url"] = audio_female_url
        if timing_male_json is not None:
            row["timing_male_json"] = timing_male_json
        if timing_female_json is not None:
            row["timing_female_json"] = timing_female_json
        supabase.table(TABLE_STORY_CONTENT_FLAT).insert(row).execute()
        return True
    except Exception as e:
        st.error(f"Insert book page failed: {e}")
        return False


def update_book_page(
    supabase,
    row_id: int,
    page_text: Optional[str] = None,
    image_url: Optional[str] = None,
    page_index: Optional[int] = None,
    audio_male_url: Optional[str] = None,
    audio_female_url: Optional[str] = None,
    timing_male_json: Optional[dict] = None,
    timing_female_json: Optional[dict] = None,
):
    """Update a story_content_flat row by id. Uses image_url and page_number."""
    try:
        updates = {}
        if page_text is not None:
            updates[PAGE_TEXT_COLUMN] = (page_text or "").strip()
        if image_url is not None:
            updates["image_url"] = (image_url or "").strip() or None
        if page_index is not None:
            updates[PAGE_NUMBER_COLUMN] = page_index
        if audio_male_url is not None:
            updates["audio_male_url"] = (audio_male_url or "").strip() or None
        if audio_female_url is not None:
            updates["audio_female_url"] = (audio_female_url or "").strip() or None
        if timing_male_json is not None:
            updates["timing_male_json"] = timing_male_json
        if timing_female_json is not None:
            updates["timing_female_json"] = timing_female_json
        if not updates:
            return True
        supabase.table(TABLE_STORY_CONTENT_FLAT).update(updates).eq("id", row_id).execute()
        return True
    except Exception as e:
        st.error(f"Update failed: {e}")
        return False


def optimize_image_for_mobile(image_bytes: bytes) -> bytes:
    """Resize to max 800x800 and convert to WebP."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        ratio = min(MAX_IMAGE_SIZE / w, MAX_IMAGE_SIZE / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    for quality in [85, 80, 75, 70, 65, 60]:
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, "WEBP", quality=quality)
        if buf.tell() <= TARGET_BYTES:
            break
    buf.seek(0)
    return buf.read()


def upload_image_to_storage(supabase, story_id: int, reading_level: str, page_index: int, image_bytes: bytes) -> Optional[str]:
    """Upload image to R2. Returns public URL or None."""
    from storage_r2 import upload_image

    path = f"{story_id}/{reading_level}/page_{page_index}.webp"
    return upload_image(path, image_bytes)


def upload_reference_image_to_storage(
    supabase, story_id: int, reading_level: str, slot_index: int, image_bytes: bytes
) -> Optional[str]:
    """Upload a style reference image to R2 under refs/. Returns public URL or None."""
    return upload_typed_reference_image(
        supabase, story_id, reading_level, "ref", str(int(slot_index)), image_bytes
    )


def upload_typed_reference_image(
    supabase,
    story_id: int,
    reading_level: str,
    ref_type: str,
    ref_id: str,
    image_bytes: bytes,
) -> Optional[str]:
    """Upload reference image to R2 under refs/{type}_{id}.webp. Returns public URL or None."""
    from storage_r2 import upload_image

    opt = optimize_image_for_mobile(image_bytes)
    safe_type = (ref_type or "ref").strip().replace("/", "_")
    safe_id = (ref_id or "0").strip().replace("/", "_")
    path = f"{story_id}/{reading_level}/refs/{safe_type}_{safe_id}.webp"
    return upload_image(path, opt)


def upload_audio_to_storage(
    supabase, story_id: int, language_code: str, reading_level: str, page_index: int, audio_bytes: bytes, voice: str
) -> Optional[str]:
    """Upload audio to R2. voice is 'male' or 'female'. Returns public URL or None."""
    return upload_audio_to_stories_path(
        supabase, story_id, language_code, reading_level, voice, page_index, audio_bytes
    )


def fetch_localized_versions(supabase, story_id: int) -> List[dict]:
    """Fetch distinct (story_id, language_code, reading_level) from story_content_flat for this story."""
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select("story_id, language_code, reading_level")
            .eq("story_id", story_id)
            .execute()
        )
        rows = r.data or []
        seen = set()
        out = []
        for row in rows:
            key = (row.get("story_id"), row.get("language_code"), row.get("reading_level"))
            if key not in seen:
                seen.add(key)
                out.append({"id": None, "story_id": key[0], "language_code": key[1], "reading_level": key[2]})
        return sorted(out, key=lambda x: (x.get("language_code") or "", x.get("reading_level") or ""))
    except Exception as e:
        st.error(f"Failed to fetch localized versions: {e}")
        return []


def _row_has_male_audio(row: dict) -> bool:
    """Check if row has a non-empty audio_male_url."""
    val = (row.get("audio_male_url") or "")
    return bool(val.strip()) if isinstance(val, str) else bool(val)


def _row_has_female_audio(row: dict) -> bool:
    """Check if row has a non-empty audio_female_url."""
    val = (row.get("audio_female_url") or "")
    return bool(val.strip()) if isinstance(val, str) else bool(val)


def fetch_pages_missing_audio(
    supabase, story_id: int, language_code: str, reading_level: str
) -> List[dict]:
    """Fetch story_content_flat rows missing male or female audio. Ordered by page_number."""
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select("*")
            .eq("story_id", story_id)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level)
            .order(PAGE_NUMBER_COLUMN)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            if "page_index" not in row:
                row["page_index"] = row.get(PAGE_NUMBER_COLUMN)
        return [
            row for row in rows
            if not _row_has_male_audio(row) or not _row_has_female_audio(row)
        ]
    except Exception as e:
        st.error(f"Failed to fetch pages missing audio: {e}")
        return []


def fetch_pages_missing_images(
    supabase, story_id: int, language_code: str, reading_level: str
) -> List[dict]:
    """Fetch rows needing images. Grades 1–4: anchor page per 3-page block; grade 5: none."""
    if reading_level == "grade_5":
        return []
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select("*")
            .eq("story_id", story_id)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level)
            .order(PAGE_NUMBER_COLUMN)
            .execute()
        )
        rows = r.data or []
        for row in rows:
            row["_display_text"] = _get_page_text(row)
            if "page_index" not in row:
                row["page_index"] = row.get(PAGE_NUMBER_COLUMN)
        if uses_triple_page_images(reading_level):
            missing = []
            for block in group_pages_into_image_blocks(rows):
                if block_needs_image(block):
                    missing.append(block["anchor_row"])
            return missing
        return [row for row in rows if not (row.get("image_url") or "").strip()]
    except Exception as e:
        st.error(f"Failed to fetch pages missing images: {e}")
        return []


def fetch_pages_missing_translation(
    supabase, story_id: int, reading_level: str, target_language_code: str
) -> List[dict]:
    """Return English pages (story_id, en, reading_level) that don't have a row for (story_id, target_language_code, reading_level) with the same page_number. Used to translate only missing pages."""
    english_pages = fetch_book_pages(supabase, story_id, reading_level, "en")
    if not english_pages:
        return []
    try:
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select(PAGE_NUMBER_COLUMN)
            .eq("story_id", story_id)
            .eq("language_code", target_language_code)
            .eq("reading_level", reading_level)
            .execute()
        )
        existing_page_numbers = {row.get(PAGE_NUMBER_COLUMN) for row in (r.data or [])}
        return [
            p
            for p in english_pages
            if p.get("page_index", p.get(PAGE_NUMBER_COLUMN)) not in existing_page_numbers
        ]
    except Exception as e:
        st.error(f"Failed to fetch pages missing translation: {e}")
        return []


def delete_book_pages_for_version(
    supabase, story_id: int, language_code: str, reading_level: str
) -> bool:
    """Delete all story_content_flat rows for (story_id, language_code, reading_level). Returns True on success."""
    try:
        (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .delete()
            .eq("story_id", story_id)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level)
            .execute()
        )
        return True
    except Exception as e:
        st.error(f"Failed to delete pages for version: {e}")
        return False


# ---------------------------------------------------------------------------
# Image batch jobs (Gemini Batch API)
# ---------------------------------------------------------------------------

TABLE_IMAGE_BATCH_JOBS = "image_batch_jobs"


def insert_batch_job(supabase, batch_name: str, story_id: int, reading_level: str, page_count: int):
    """Insert an image batch job row. Returns the inserted row id or None."""
    try:
        r = (
            supabase.table(TABLE_IMAGE_BATCH_JOBS)
            .insert({
                "batch_name": batch_name,
                "story_id": story_id,
                "reading_level": reading_level,
                "page_count": page_count,
            })
            .execute()
        )
        rows = r.data or []
        return rows[0]["id"] if rows else None
    except Exception as e:
        st.error(f"Failed to insert batch job: {e}")
        return None


def fetch_batch_jobs_for_version(supabase, story_id: int, reading_level: str) -> List[dict]:
    """Fetch image batch jobs for (story_id, reading_level), ordered by created_at desc."""
    try:
        r = (
            supabase.table(TABLE_IMAGE_BATCH_JOBS)
            .select("*")
            .eq("story_id", story_id)
            .eq("reading_level", reading_level)
            .order("created_at", desc=True)
            .execute()
        )
        return r.data or []
    except Exception as e:
        st.error(f"Failed to fetch batch jobs: {e}")
        return []


def fetch_batch_jobs_for_story(supabase, story_id: int) -> List[dict]:
    """Fetch image batch jobs for story_id across all reading levels, ordered by reading_level, created_at desc."""
    try:
        r = (
            supabase.table(TABLE_IMAGE_BATCH_JOBS)
            .select("*")
            .eq("story_id", story_id)
            .order("reading_level")
            .order("created_at", desc=True)
            .execute()
        )
        return r.data or []
    except Exception as e:
        st.error(f"Failed to fetch batch jobs for story: {e}")
        return []


def update_batch_job_status(
    supabase, batch_name: str, status: str, error_message: Optional[str] = None
) -> bool:
    """Update batch job status and completed_at when done. Returns True on success."""
    try:
        updates = {"status": status}
        if status in ("succeeded", "failed", "cancelled", "expired"):
            from datetime import datetime, timezone
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        if error_message is not None:
            updates["error_message"] = error_message
        (
            supabase.table(TABLE_IMAGE_BATCH_JOBS)
            .update(updates)
            .eq("batch_name", batch_name)
            .execute()
        )
        return True
    except Exception as e:
        st.error(f"Failed to update batch job status: {e}")
        return False


# ---------------------------------------------------------------------------
# Per-page image jobs (Leonardo async queue)
# ---------------------------------------------------------------------------

TABLE_IMAGE_GENERATION_JOBS = "image_generation_jobs"


def insert_image_generation_job(
    supabase,
    story_id: int,
    reading_level: str,
    story_content_flat_id: str,
    external_generation_id: str,
    provider: str = "leonardo",
) -> Optional[str]:
    """Insert a pending job row. Returns job UUID string or None."""
    try:
        r = (
            supabase.table(TABLE_IMAGE_GENERATION_JOBS)
            .insert({
                "story_id": story_id,
                "reading_level": reading_level,
                "story_content_flat_id": str(story_content_flat_id),
                "provider": provider,
                "external_generation_id": external_generation_id,
                "status": "pending",
            })
            .execute()
        )
        rows = r.data or []
        return str(rows[0]["id"]) if rows else None
    except Exception as e:
        st.error(f"Failed to insert image generation job: {e}")
        return None


def fetch_pending_image_generation_jobs(
    supabase, story_id: int, reading_level: str
) -> List[dict]:
    try:
        r = (
            supabase.table(TABLE_IMAGE_GENERATION_JOBS)
            .select("*")
            .eq("story_id", story_id)
            .eq("reading_level", reading_level)
            .eq("status", "pending")
            .order("created_at")
            .execute()
        )
        return r.data or []
    except Exception as e:
        st.error(f"Failed to fetch image generation jobs: {e}")
        return []


def update_image_generation_job(
    supabase,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    try:
        from datetime import datetime, timezone

        updates = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error_message is not None:
            updates["error_message"] = error_message
        supabase.table(TABLE_IMAGE_GENERATION_JOBS).update(updates).eq("id", job_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to update image generation job: {e}")
        return False


def generate_elevenlabs_audio(
    api_key: str,
    voice_id: str,
    text: str,
    language_code: str = "en",
    stability: Optional[float] = None,
    similarity_boost: Optional[float] = None,
    use_speaker_boost: Optional[bool] = None,
    speed: Optional[float] = None,
    model_id: Optional[str] = None,
    output_format: Optional[str] = None,
    optimize_streaming_latency: Optional[int] = None,
    apply_text_normalization: Optional[str] = None,
) -> Tuple[Optional[bytes], Optional[dict]]:
    """Generate audio via ElevenLabs with-timestamps endpoint. Returns (audio_bytes, timing_json) or (None, None)."""
    if not text or not text.strip():
        return None, None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    params = {}
    if output_format:
        params["output_format"] = output_format
    if optimize_streaming_latency is not None:
        params["optimize_streaming_latency"] = optimize_streaming_latency
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {"text": text.strip()}
    if language_code:
        payload["language_code"] = language_code
    if model_id:
        payload["model_id"] = model_id
    if apply_text_normalization:
        payload["apply_text_normalization"] = apply_text_normalization
    voice_settings = {}
    if stability is not None:
        voice_settings["stability"] = max(0, min(1, float(stability)))
    if similarity_boost is not None:
        voice_settings["similarity_boost"] = max(0, min(1, float(similarity_boost)))
    if use_speaker_boost is not None:
        voice_settings["use_speaker_boost"] = bool(use_speaker_boost)
    if speed is not None:
        voice_settings["speed"] = max(0.5, min(2.0, float(speed)))
    if voice_settings:
        payload["voice_settings"] = voice_settings
    try:
        resp = requests.post(url, json=payload, headers=headers, params=params or None, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        audio_b64 = data.get("audio_base64")
        alignment = data.get("alignment") or data.get("normalized_alignment")
        timing_json = alignment if isinstance(alignment, dict) else None
        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            return audio_bytes, timing_json
    except Exception as e:
        st.error(f"ElevenLabs generation failed: {e}")
    return None, None


def upload_audio_to_stories_path(
    supabase,
    story_id: int,
    language_code: str,
    reading_level: str,
    gender: str,
    page_index: int,
    audio_bytes: bytes,
) -> Optional[str]:
    """Upload audio to R2 at stories/{story_id}/{language}/{reading_level}/{gender}/page_{page_index}.mp3.
    reading_level is required so grade_1 and grade_2 (and other levels) do not overwrite the same file."""
    from storage_r2 import upload_audio

    path = f"stories/{story_id}/{language_code}/{reading_level}/{gender}/page_{page_index}.mp3"
    return upload_audio(path, audio_bytes)


def remove_page_audio(supabase, row_id: int, gender: str, old_url: Optional[str] = None) -> bool:
    """Delete R2 audio object and clear URL + timing fields for male or female voice."""
    from storage_r2 import delete_audio_by_url

    if old_url:
        delete_audio_by_url(old_url)
    try:
        updates = (
            {"audio_male_url": None, "timing_male_json": None}
            if gender == "male"
            else {"audio_female_url": None, "timing_female_json": None}
        )
        supabase.table(TABLE_STORY_CONTENT_FLAT).update(updates).eq("id", row_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to remove {gender} audio: {e}")
        return False


def approve_audio_for_page(
    supabase,
    *,
    row_id: int,
    story_id: int,
    language_code: str,
    reading_level: str,
    gender: str,
    page_index: int,
    audio_bytes: bytes,
    timing_json: Optional[dict],
    old_url: Optional[str] = None,
) -> Optional[str]:
    """Delete old R2 audio, upload replacement, cache-bust URL, and update story_content_flat."""
    from storage_r2 import delete_audio_by_url

    if old_url:
        delete_audio_by_url(old_url)
    url = upload_audio_to_stories_path(
        supabase, story_id, language_code, reading_level, gender, page_index, audio_bytes
    )
    if not url:
        return None
    url = f"{url}?v={int(time.time())}"
    kwargs = (
        {"audio_male_url": url, "timing_male_json": timing_json}
        if gender == "male"
        else {"audio_female_url": url, "timing_female_json": timing_json}
    )
    if update_book_page(supabase, row_id, **kwargs):
        return url
    return None


@st.dialog("Replace audio", width="medium")
def replace_audio_modal(row: dict, story_id: int, language_code: str, reading_level: str, voice: str):
    """Modal to upload male or female audio for a page."""
    st.caption(f"Upload {voice} voice audio for page {row.get('page_index', row.get(PAGE_NUMBER_COLUMN, 0))}. Will update story_content_flat.")
    audio_file = st.file_uploader("Choose audio", type=["mp3", "wav", "m4a"], key="bp_audio_modal")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload & update", type="primary"):
            if not audio_file:
                st.error("Select an audio file first.")
            else:
                data = audio_file.read()
                pn = row.get("page_index", row.get(PAGE_NUMBER_COLUMN, 0))
                old_url = (row.get("audio_male_url") if voice == "male" else row.get("audio_female_url")) or None
                url = approve_audio_for_page(
                    get_supabase(),
                    row_id=row["id"],
                    story_id=story_id,
                    language_code=language_code,
                    reading_level=reading_level,
                    gender=voice,
                    page_index=pn,
                    audio_bytes=data,
                    timing_json=None,
                    old_url=old_url,
                )
                if url:
                    st.success(f"{voice.capitalize()} audio uploaded.")
                    if "bp_editing_audio" in st.session_state:
                        del st.session_state.bp_editing_audio
                    st.rerun()
    with col2:
        if st.button("Cancel"):
            if "bp_editing_audio" in st.session_state:
                del st.session_state.bp_editing_audio
            st.rerun()


@st.dialog("Replace image", width="medium")
def replace_image_modal(row: dict, story_id: int, reading_level: str):
    """Modal to replace image: upload to Storage, update story_content_flat.image_url."""
    st.caption("The new image will be uploaded to Storage and story_content_flat will be updated.")
    new_img = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp"], key="bp_modal_upload")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload & update", type="primary"):
            if not new_img:
                st.error("Select an image first.")
            else:
                data = new_img.read()
                opt = optimize_image_for_mobile(data)
                pn = row.get("page_index", row.get(PAGE_NUMBER_COLUMN, 0))
                url = upload_image_to_storage(
                    get_supabase(), story_id, reading_level, pn, opt
                )
                if url:
                    if update_book_page(get_supabase(), row["id"], image_url=url):
                        st.success("Image uploaded. story_content_flat updated.")
                        if "bp_editing_image" in st.session_state:
                            del st.session_state.bp_editing_image
                        st.rerun()
    with col2:
        if st.button("Remove image", type="secondary"):
            if update_book_page(get_supabase(), row["id"], image_url=""):
                st.success("Image removed.")
                if "bp_editing_image" in st.session_state:
                    del st.session_state.bp_editing_image
                st.rerun()


def run_audio_generator_view(*, as_wizard_step: bool = False):
    """Audio Generator: batch create male/female audio via ElevenLabs for pages missing audio."""
    if as_wizard_step:
        st.caption(
            "Generate male and female voice audio for pages missing audio. Uses ElevenLabs with timestamps."
        )
    else:
        st.title("Audio Generator")
        st.caption("Generate male and female voice audio for pages missing audio. Uses ElevenLabs with timestamps.")

    sb = get_supabase()
    if not sb:
        return

    api_key = get_secret("ELEVENLABS_API_KEY")

    if not api_key:
        st.warning("Set ELEVENLABS_API_KEY in .env or Streamlit secrets.")
        return

    stories = fetch_stories()
    if not stories:
        st.warning("No stories found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
        story_label = st.selectbox("Story", options=list(story_options.keys()), key="ag_story")
        story_id = story_options.get(story_label) if story_label else None

    versions = []
    if story_id:
        versions = fetch_localized_versions(sb, story_id)

    lang_options = ui_language_select_options(versions)
    with col2:
        ag_language = st.selectbox("Language", options=lang_options, key="ag_language")

    ag_all_levels = st.radio(
        "Mode",
        options=["Single level", "All levels"],
        key="ag_mode",
        horizontal=True,
        help="Single level: one grade at a time. All levels: generate audio for all 5 grades at once (for this language).",
    )
    ag_all_levels_mode = ag_all_levels == "All levels"

    if not ag_all_levels_mode:
        ag_grade = st.selectbox("Grade", options=READING_LEVELS, key="ag_grade")
    else:
        ag_grade = None

    # ElevenLabs: multilingual model, fixed Johnny Kid + Zara, per-grade speed (not editable in UI).
    stability_default = 0.5
    similarity_boost_default = 0.75
    use_speaker_boost = False
    model_id = ELEVENLABS_TTS_MODEL_ID
    output_format = ELEVENLABS_TTS_OUTPUT_FORMAT
    optimize_streaming_latency = 0
    apply_text_normalization = "auto"

    def _get_ag_settings_for_level(lev: str) -> dict:
        return {
            "speed": audio_tts_speed_for_grade(lev),
            "voice_male": ELEVENLABS_VOICE_MALE_DEFAULT,
            "voice_female": ELEVENLABS_VOICE_FEMALE_DEFAULT,
        }

    st.caption(
        f"Narrators: **Johnny Kid** (male), **Zara** (female) · Model `{model_id}` · {output_format} · "
        "Speed by grade: 1 → 0.75, 2–4 → 0.8, 5 → 0.9"
    )

    language_code = ag_language
    story_id_ver = story_id
    if not story_id_ver:
        st.info("Select a story.")
        return

    if ag_all_levels_mode:
        levels_missing = {
            level: fetch_pages_missing_audio(sb, story_id_ver, language_code, level)
            for level in READING_LEVELS
        }
        total_missing = sum(len(p) for p in levels_missing.values())
        if total_missing == 0:
            st.info(
                f"No pages missing audio for Story {story_id_ver}, **{language_code}** across any grade. "
                "Add translated pages or pick another story/language."
            )
        else:
            parts = [f"**{lev}**: {len(levels_missing[lev])} missing" for lev in READING_LEVELS]
            st.success(" · ".join(parts) + f" → **Total: {total_missing} pages**")
            with st.expander("Incomplete pages (by level)", expanded=False):
                for lev in READING_LEVELS:
                    pm = levels_missing.get(lev, [])
                    if pm:
                        st.caption(f"{lev}: {len(pm)} pages")
        reading_level_ver = next((lev for lev in READING_LEVELS if levels_missing.get(lev)), READING_LEVELS[0])
        pages_missing = levels_missing.get(reading_level_ver, [])
    else:
        version = next(
            (v for v in versions if v.get("language_code") == ag_language and v.get("reading_level") == ag_grade),
            None,
        )
        if not version and story_id:
            version = {"story_id": story_id, "language_code": ag_language, "reading_level": ag_grade}
        reading_level_ver = version.get("reading_level", ag_grade)
        pages_missing = fetch_pages_missing_audio(sb, story_id_ver, language_code, reading_level_ver)

    if not ag_all_levels_mode:
        if pages_missing:
            st.success(
                f"**{len(pages_missing)}** pages missing audio for **Story {story_id_ver}**, "
                f"**{language_code}**, **{reading_level_ver}** (these are the story_content_flat rows that will get URLs)."
            )
            with st.expander("Incomplete pages (rows that need audio)", expanded=True):
                for p in pages_missing:
                    txt = _get_page_text(p)
                    male_ok = "✓" if _row_has_male_audio(p) else "—"
                    female_ok = "✓" if _row_has_female_audio(p) else "—"
                    row_id = p.get("id")
                    pn = p.get("page_index", p.get(PAGE_NUMBER_COLUMN, "?"))
                    st.caption(f"Row id={row_id} · Page {pn} · male {male_ok} female {female_ok} — {txt[:60]}...")
        else:
            st.info(
                f"No pages missing audio for Story {story_id_ver}, {language_code}, {reading_level_ver}. "
                "Add translated pages (e.g. via Translator) or pick another story/language/grade."
            )

    stability = stability_default
    similarity_boost = similarity_boost_default

    st.subheader("Generate")
    levels_missing_agg = levels_missing if ag_all_levels_mode else {}
    total_pages_display = (sum(len(p) for p in levels_missing_agg.values()) if ag_all_levels_mode else len(pages_missing))
    st.write(f"**{total_pages_display}** pages missing male and/or female audio.")
    if st.button("Generate All Missing Audio", type="secondary", key="ag_generate"):
        if ag_all_levels_mode and levels_missing and sum(len(p) for p in levels_missing.values()) > 0:
            # Generate for all levels
            prog = st.progress(0)
            batches_list = []
            total_to_gen = sum(len(p) for p in levels_missing.values())
            total_done = 0
            for level in READING_LEVELS:
                pm = levels_missing.get(level, [])
                if not pm:
                    continue
                level_settings = _get_ag_settings_for_level(level)
                batch = {"story_id": story_id_ver, "language_code": language_code, "reading_level": level, "pages": []}
                for idx, page in enumerate(pm):
                    page_text = _get_page_text(page)
                    page_index = page.get("page_index", page.get(PAGE_NUMBER_COLUMN, idx))
                    row_id = page.get("id")
                    if row_id is None:
                        continue
                    entry = {"row_id": row_id, "page_index": page_index, "page_text": page_text[:60], "male": None, "female": None}
                    if page_text:
                        if not _row_has_male_audio(page):
                            audio_bytes, timing = generate_elevenlabs_audio(
                                api_key, level_settings["voice_male"], page_text, language_code,
                                stability=stability, similarity_boost=similarity_boost,
                                use_speaker_boost=use_speaker_boost, speed=level_settings["speed"],
                                model_id=model_id, output_format=output_format,
                                optimize_streaming_latency=optimize_streaming_latency,
                                apply_text_normalization=apply_text_normalization,
                            )
                            if audio_bytes:
                                entry["male"] = (audio_bytes, timing)
                        total_done += 1
                        prog.progress(min(1.0, total_done / (total_to_gen * 2)))
                        if not _row_has_female_audio(page):
                            audio_bytes, timing = generate_elevenlabs_audio(
                                api_key, level_settings["voice_female"], page_text, language_code,
                                stability=stability, similarity_boost=similarity_boost,
                                use_speaker_boost=use_speaker_boost, speed=level_settings["speed"],
                                model_id=model_id, output_format=output_format,
                                optimize_streaming_latency=optimize_streaming_latency,
                                apply_text_normalization=apply_text_normalization,
                            )
                            if audio_bytes:
                                entry["female"] = (audio_bytes, timing)
                    total_done += 1
                    prog.progress(min(1.0, total_done / (total_to_gen * 2)))
                    if entry["male"] or entry["female"]:
                        batch["pages"].append(entry)
                if batch["pages"]:
                    batches_list.append(batch)
            prog.progress(1.0)
            st.session_state["ag_generated_batches"] = batches_list
            if "ag_generated_batch" in st.session_state:
                del st.session_state["ag_generated_batch"]
            st.success(f"Generated {sum(len(b['pages']) for b in batches_list)} pages across {len(batches_list)} levels. Preview below, then Approve and save.")
            st.rerun()
        elif not pages_missing:
            st.info("No pages missing audio.")
        else:
            prog = st.progress(0)
            rl = reading_level_ver or ""
            speed = audio_tts_speed_for_grade(rl)
            batch = {
                "story_id": story_id_ver,
                "language_code": language_code,
                "reading_level": reading_level_ver,
                "pages": [],
            }
            total = len(pages_missing) * 2
            done = 0
            for idx, page in enumerate(pages_missing):
                page_text = _get_page_text(page)
                page_index = page.get("page_index", page.get(PAGE_NUMBER_COLUMN, idx))
                # Use story_content_flat row id so we update the correct row (same story_id, language_code, reading_level)
                row_id = page.get("id")
                if row_id is None:
                    continue  # skip rows without id (shouldn’t happen for select *)
                entry = {"row_id": row_id, "page_index": page_index, "page_text": page_text[:60], "male": None, "female": None}

                if page_text:
                    if not _row_has_male_audio(page):
                        audio_bytes, timing = generate_elevenlabs_audio(
                            api_key, ELEVENLABS_VOICE_MALE_DEFAULT, page_text, language_code,
                            stability=stability, similarity_boost=similarity_boost,
                            use_speaker_boost=use_speaker_boost, speed=speed,
                            model_id=model_id, output_format=output_format,
                            optimize_streaming_latency=optimize_streaming_latency,
                            apply_text_normalization=apply_text_normalization,
                        )
                        if audio_bytes:
                            entry["male"] = (audio_bytes, timing)
                    done += 1
                    prog.progress(done / total)

                    if not _row_has_female_audio(page):
                        audio_bytes, timing = generate_elevenlabs_audio(
                            api_key, ELEVENLABS_VOICE_FEMALE_DEFAULT, page_text, language_code,
                            stability=stability, similarity_boost=similarity_boost,
                            use_speaker_boost=use_speaker_boost, speed=speed,
                            model_id=model_id, output_format=output_format,
                            optimize_streaming_latency=optimize_streaming_latency,
                            apply_text_normalization=apply_text_normalization,
                        )
                        if audio_bytes:
                            entry["female"] = (audio_bytes, timing)
                done += 1
                prog.progress(done / total)
                if entry["male"] or entry["female"]:
                    batch["pages"].append(entry)

            st.session_state["ag_generated_batch"] = batch
            if "ag_generated_batches" in st.session_state:
                del st.session_state["ag_generated_batches"]
            st.success(f"Generated {len(batch['pages'])} pages. Preview below, then Approve and save.")
            st.rerun()

    batches_list = st.session_state.get("ag_generated_batches") or []
    if batches_list and (batches_list[0].get("story_id") != story_id_ver or batches_list[0].get("language_code") != language_code):
        if "ag_generated_batches" in st.session_state:
            del st.session_state["ag_generated_batches"]
        batches_list = []
    batch = st.session_state.get("ag_generated_batch")
    if batch and batch.get("pages"):
        if (
            batch.get("story_id") != story_id_ver
            or batch.get("language_code") != language_code
            or batch.get("reading_level") != reading_level_ver
        ):
            if "ag_generated_batch" in st.session_state:
                del st.session_state["ag_generated_batch"]
            batch = None
    if batches_list:
        st.subheader("Review generated batch (all levels)")
        st.caption("Listen by level. When ready, click Approve and save to Supabase to save all levels.")
        ag_preview_level = st.selectbox(
            "Preview level",
            options=[b["reading_level"] for b in batches_list],
            key="ag_preview_level",
            format_func=lambda x: x.replace("_", " ").title(),
        )
        selected_batch = next((b for b in batches_list if b["reading_level"] == ag_preview_level), batches_list[0])
        for entry in selected_batch["pages"]:
            st.write(f"**Page {entry['page_index']}:** {entry.get('page_text', '')}...")
            c1, c2 = st.columns(2)
            with c1:
                if entry.get("male"):
                    st.audio(entry["male"][0], format="audio/mp3")
                    st.caption("Male")
            with c2:
                if entry.get("female"):
                    st.audio(entry["female"][0], format="audio/mp3")
                    st.caption("Female")
            st.divider()
        if st.button("Approve and save to Supabase", type="primary", key="ag_approve_save_batches"):
            sb_save = get_supabase()
            ok = 0
            for b in batches_list:
                reading_level_b = b.get("reading_level") or ""
                for entry in b["pages"]:
                    row_id = entry.get("row_id")
                    page_index = entry.get("page_index")
                    if row_id is None:
                        continue
                    if entry.get("male"):
                        b_bytes, t = entry["male"]
                        if approve_audio_for_page(
                            sb_save,
                            row_id=row_id,
                            story_id=b["story_id"],
                            language_code=b["language_code"],
                            reading_level=reading_level_b,
                            gender="male",
                            page_index=page_index,
                            audio_bytes=b_bytes,
                            timing_json=t,
                        ):
                            ok += 1
                    if entry.get("female"):
                        b_bytes, t = entry["female"]
                        if approve_audio_for_page(
                            sb_save,
                            row_id=row_id,
                            story_id=b["story_id"],
                            language_code=b["language_code"],
                            reading_level=reading_level_b,
                            gender="female",
                            page_index=page_index,
                            audio_bytes=b_bytes,
                            timing_json=t,
                        ):
                            ok += 1
            del st.session_state["ag_generated_batches"]
            st.success(f"Saved {ok} audio URLs to story_content_flat (all levels).")
            st.rerun()
    elif batch and batch.get("pages"):
        st.subheader("Review generated batch")
        st.caption("Listen to the generated audio. When ready, click Approve and save to Supabase.")
        for entry in batch["pages"]:
            st.write(f"**Page {entry['page_index']}:** {entry.get('page_text', '')}...")
            c1, c2 = st.columns(2)
            with c1:
                if entry.get("male"):
                    st.audio(entry["male"][0], format="audio/mp3")
                    st.caption("Male")
            with c2:
                if entry.get("female"):
                    st.audio(entry["female"][0], format="audio/mp3")
                    st.caption("Female")
            st.divider()

        if st.button("Approve and save to Supabase", type="primary", key="ag_approve_save"):
            sb_save = get_supabase()
            ok = 0
            for entry in batch["pages"]:
                # Each entry is one story_content_flat row; row_id is that row’s primary key (correct for Spanish etc.)
                row_id = entry.get("row_id")
                page_index = entry.get("page_index")
                if row_id is None:
                    st.warning(f"Page {page_index}: missing row id, skip saving.")
                    continue
                reading_level = batch.get("reading_level") or ""
                if entry.get("male"):
                    b, t = entry["male"]
                    if approve_audio_for_page(
                        sb_save,
                        row_id=row_id,
                        story_id=batch["story_id"],
                        language_code=batch["language_code"],
                        reading_level=reading_level,
                        gender="male",
                        page_index=page_index,
                        audio_bytes=b,
                        timing_json=t,
                    ):
                        ok += 1
                if entry.get("female"):
                    b, t = entry["female"]
                    if approve_audio_for_page(
                        sb_save,
                        row_id=row_id,
                        story_id=batch["story_id"],
                        language_code=batch["language_code"],
                        reading_level=reading_level,
                        gender="female",
                        page_index=page_index,
                        audio_bytes=b,
                        timing_json=t,
                    ):
                        ok += 1
            del st.session_state["ag_generated_batch"]
            st.success(f"Saved {ok} audio URLs to story_content_flat (Story {batch['story_id']}, {batch['language_code']}, {batch.get('reading_level', '')}).")
            st.rerun()

    st.divider()
    st.subheader("Preview generated audio")
    all_pages = []
    try:
        r = (
            sb.table(TABLE_STORY_CONTENT_FLAT)
            .select("*")
            .eq("story_id", story_id_ver)
            .eq("language_code", language_code)
            .eq("reading_level", reading_level_ver)
            .order(PAGE_NUMBER_COLUMN)
            .execute()
        )
        all_pages = r.data or []
        for p in all_pages:
            if "page_index" not in p:
                p["page_index"] = p.get(PAGE_NUMBER_COLUMN)
    except Exception:
        pass

    pages_with_audio = [p for p in all_pages if _row_has_male_audio(p) or _row_has_female_audio(p)]
    st.caption(f"Story {story_id_ver}, {language_code}, {reading_level_ver} — **{len(pages_with_audio)}** of {len(all_pages)} pages with audio.")
    for p in pages_with_audio:
        male_url = (p.get("audio_male_url") or "").strip() or None
        female_url = (p.get("audio_female_url") or "").strip() or None
        st.caption(f"Page {p.get('page_index', p.get(PAGE_NUMBER_COLUMN, '?'))}: {_get_page_text(p)[:50]}...")
        c1, c2 = st.columns(2)
        with c1:
            if male_url:
                st.audio(male_url, format="audio/mp3")
                st.caption("Male")
            else:
                st.caption("—")
        with c2:
            if female_url:
                st.audio(female_url, format="audio/mp3")
                st.caption("Female")
            else:
                st.caption("—")
        st.divider()


def _book_page_audio_voice_section(
    sb,
    *,
    voice: str,
    row_id: int,
    story_id: int,
    language_code: str,
    reading_level: str,
    page_idx: int,
    page_text: str,
    new_text: str,
    selected_row: dict,
    voices: list,
    voice_select_index: int,
    speed: float,
):
    """Render regenerate / remove / upload / preview / approve UI for one voice."""
    label = voice.capitalize()
    url = (selected_row.get(f"audio_{voice}_url") or "").strip() or None
    key_prefix = voice[0]

    st.markdown(f"**{label} audio**")
    if url:
        st.audio(url, format="audio/mpeg")
    else:
        st.caption("—")

    pending_all = st.session_state.setdefault("bp_pending_audio", {})
    pending_row = pending_all.setdefault(row_id, {})
    pending_entry = pending_row.get(voice)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button(f"Regenerate {voice}", key=f"bp_regen_{key_prefix}_{row_id}"):
            api_key = get_secret("ELEVENLABS_API_KEY")
            if not api_key:
                st.error("Set ELEVENLABS_API_KEY in .env for Regenerate audio.")
            else:
                text_for_audio = (st.session_state.get(f"bp_text_{row_id}", new_text) or "").strip() or page_text
                if not text_for_audio:
                    st.warning("Page text is empty.")
                else:
                    voice_id = voices[voice_select_index][1]
                    with st.spinner(f"Generating {voice} audio..."):
                        audio_bytes, timing = generate_elevenlabs_audio(
                            api_key,
                            voice_id,
                            text_for_audio,
                            language_code,
                            stability=0.5,
                            similarity_boost=0.75,
                            speed=speed,
                            model_id=ELEVENLABS_TTS_MODEL_ID,
                            output_format=ELEVENLABS_TTS_OUTPUT_FORMAT,
                            optimize_streaming_latency=0,
                            apply_text_normalization="auto",
                        )
                    if audio_bytes:
                        pending_row[voice] = (audio_bytes, timing)
                        st.success(f"{label} audio generated. Preview below, then Approve and save.")
                        st.rerun()
                    else:
                        st.error("Audio generation failed.")
    with btn_col2:
        if url and st.button(f"Remove {voice}", key=f"bp_remove_{key_prefix}_{row_id}", type="secondary"):
            if remove_page_audio(sb, row_id, voice, old_url=url):
                pending_row.pop(voice, None)
                if not pending_row:
                    pending_all.pop(row_id, None)
                st.success(f"{label} audio removed.")
                st.rerun()
    with btn_col3:
        if st.button(f"Upload {voice} audio", key=f"bp_upload_{key_prefix}_{row_id}"):
            st.session_state.bp_editing_audio = (selected_row, voice)
            st.rerun()

    if pending_entry:
        audio_bytes, timing = pending_entry
        st.caption("Preview (not saved until you approve)")
        st.audio(audio_bytes, format="audio/mpeg")
        ap_col1, ap_col2 = st.columns(2)
        with ap_col1:
            if st.button(f"Approve and save {voice}", key=f"bp_approve_{key_prefix}_{row_id}", type="primary"):
                result = approve_audio_for_page(
                    sb,
                    row_id=row_id,
                    story_id=story_id,
                    language_code=language_code,
                    reading_level=reading_level,
                    gender=voice,
                    page_index=page_idx,
                    audio_bytes=audio_bytes,
                    timing_json=timing,
                    old_url=url,
                )
                if result:
                    pending_row.pop(voice, None)
                    if not pending_row:
                        pending_all.pop(row_id, None)
                    st.success(f"{label} audio saved.")
                    st.rerun()
                else:
                    st.error("Upload or save failed.")
        with ap_col2:
            if st.button(f"Discard {voice}", key=f"bp_discard_{key_prefix}_{row_id}"):
                pending_row.pop(voice, None)
                if not pending_row:
                    pending_all.pop(row_id, None)
                st.rerun()


def run_book_pages_view():
    """Render Story Content Manager: page selector + focus panel with edit text, regenerate image/audio, upload, delete."""
    st.title("Story Content Manager")
    st.caption("View, search, filter, sort, and manage each page (edit text, regenerate or upload image/audio, delete).")
    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories()
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    top1, top2, top3 = st.columns([1.2, 1, 1])
    with top1:
        story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
        story_label = st.selectbox("Story", options=list(story_options.keys()), key="bp_story")
        story_id = story_options.get(story_label) if story_label else None
    versions = fetch_localized_versions(sb, story_id) if story_id else []
    bp_lang_options = ui_language_select_options(versions)
    with top2:
        language_code = st.selectbox("Language", options=bp_lang_options, key="bp_language")
    with top3:
        reading_level = st.selectbox("Reading Level", READING_LEVELS, key="bp_reading_level")

    if not story_id:
        st.info("Select a story to load book pages.")
        return

    rows = fetch_book_pages(sb, story_id, reading_level, language_code, "")
    for r in rows:
        r["_display_text"] = _get_page_text(r)
    if not rows:
        st.info("No book pages found for this story, language, and reading level.")
        return

    # Search / filter / sort controls
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        search = st.text_input(
            "Search",
            value=st.session_state.get("bp_search", ""),
            placeholder="Search page text...",
            key="bp_search",
        ).strip()
    with c2:
        media_filter = st.selectbox(
            "Filter",
            options=[
                "All",
                "Has image",
                "No image",
                "Has male audio",
                "Has female audio",
                "Missing male audio",
                "Missing female audio",
            ],
            key="bp_filter",
        )
    with c3:
        sort_col = st.selectbox("Sort by", options=["page_number", "id", "page_text"], key="bp_sort_col")
    with c4:
        sort_dir = st.selectbox("Order", options=["Ascending", "Descending"], key="bp_sort_dir")

    filtered_rows = rows
    if search:
        q = search.lower()
        filtered_rows = [r for r in filtered_rows if q in (r.get("_display_text", "") or "").lower()]

    if media_filter == "Has image":
        filtered_rows = [r for r in filtered_rows if (r.get("image_url") or "").strip()]
    elif media_filter == "No image":
        filtered_rows = [r for r in filtered_rows if not (r.get("image_url") or "").strip()]
    elif media_filter == "Has male audio":
        filtered_rows = [r for r in filtered_rows if (r.get("audio_male_url") or "").strip()]
    elif media_filter == "Has female audio":
        filtered_rows = [r for r in filtered_rows if (r.get("audio_female_url") or "").strip()]
    elif media_filter == "Missing male audio":
        filtered_rows = [r for r in filtered_rows if not (r.get("audio_male_url") or "").strip()]
    elif media_filter == "Missing female audio":
        filtered_rows = [r for r in filtered_rows if not (r.get("audio_female_url") or "").strip()]

    ascending = sort_dir == "Ascending"
    if sort_col == "page_text":
        filtered_rows = sorted(filtered_rows, key=lambda r: (r.get("_display_text", "") or "").lower(), reverse=not ascending)
    elif sort_col == "id":
        filtered_rows = sorted(filtered_rows, key=lambda r: int(r.get("id", 0) or 0), reverse=not ascending)
    else:
        filtered_rows = sorted(
            filtered_rows,
            key=lambda r: int(r.get("page_index", r.get(PAGE_NUMBER_COLUMN, 0)) or 0),
            reverse=not ascending,
        )

    st.caption(f"Showing {len(filtered_rows)} of {len(rows)} page(s).")
    if not filtered_rows:
        st.info("No pages match current search/filter.")
        return

    # Page selector
    def _pn(r):
        return r.get("page_index", r.get(PAGE_NUMBER_COLUMN, 0))

    action_rows = [r for r in filtered_rows if r.get("id") is not None]
    page_options = []
    page_map = {}
    for r in action_rows:
        snippet = (r.get("_display_text", "") or "")[:50]
        if len((r.get("_display_text") or "")) > 50:
            snippet += "..."
        label = f"Page {_pn(r)} — {snippet or '(no text)'}"
        page_options.append(label)
        page_map[label] = r

    current_label = st.session_state.get("bp_page_sel")
    if current_label not in page_options:
        current_idx = 0
        if page_options:
            st.session_state["bp_page_sel"] = page_options[0]
    else:
        current_idx = page_options.index(current_label)

    nav_prev, nav_pos, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("← Previous", key="bp_page_prev", disabled=current_idx <= 0):
            st.session_state["bp_page_sel"] = page_options[current_idx - 1]
            st.rerun()
    with nav_pos:
        st.caption(f"Page {current_idx + 1} of {len(page_options)}")
    with nav_next:
        if st.button("Next →", key="bp_page_next", disabled=current_idx >= len(page_options) - 1):
            st.session_state["bp_page_sel"] = page_options[current_idx + 1]
            st.rerun()

    selected_label = st.selectbox("Select a page", options=page_options, key="bp_page_sel")
    selected_row = page_map.get(selected_label) if selected_label else None

    # Audio settings (for Regenerate audio)
    with st.expander("Audio settings (for Regenerate)", expanded=False):
        bp_voice_male = st.selectbox(
            "Male voice",
            options=range(len(VOICES_MALE)),
            format_func=lambda i: f"{VOICES_MALE[i][0]} — {VOICES_MALE[i][2]}",
            key="bp_voice_male",
        )
        bp_voice_female = st.selectbox(
            "Female voice",
            options=range(len(VOICES_FEMALE)),
            format_func=lambda i: f"{VOICES_FEMALE[i][0]} — {VOICES_FEMALE[i][2]}",
            key="bp_voice_female",
        )
        bp_speed = st.slider("Speed", min_value=0.7, max_value=1.2, value=1.0, step=0.05, key="bp_speed")

    # Focus panel for selected page
    if selected_row:
        row_id = selected_row.get("id")
        page_text = selected_row.get("_display_text", "")
        page_idx = _pn(selected_row)

        st.divider()
        st.subheader(f"Page {page_idx}")

        # Text edit
        st.markdown("**Page text**")
        new_text = st.text_area(
            "Edit text",
            value=page_text,
            height=100,
            key=f"bp_text_{row_id}",
        )
        if st.button("Save text", key=f"bp_save_text_{row_id}"):
            current = st.session_state.get(f"bp_text_{row_id}", new_text)
            if update_book_page(sb, row_id, page_text=(current or "").strip()):
                st.success("Text saved.")
                st.rerun()
            else:
                st.error("Failed to save text.")

        # Image: thumbnail + Regenerate / Replace / Remove
        st.markdown("**Image**")
        img_col, img_actions = st.columns([1, 2])
        with img_col:
            img_url = selected_row.get("image_url") or ""
            if img_url:
                img_bytes = fetch_image_for_display(img_url)
                if img_bytes:
                    try:
                        st.image(img_bytes, width=180)
                    except Exception:
                        try:
                            st.image(img_url, width=180)
                        except Exception:
                            st.caption("(image)")
                else:
                    try:
                        st.image(img_url, width=180)
                    except Exception:
                        st.caption("(image)")
            else:
                st.caption("No image")

        with img_actions:
            # Reference image for regeneration (same story+level pages with image_url)
            published_with_images = []
            for r in rows:
                url = (r.get("image_url") or "").strip()
                if url and r.get("id") != row_id:
                    pi = r.get("page_index", r.get(PAGE_NUMBER_COLUMN, 0))
                    published_with_images.append((pi, url))
            published_with_images.sort(key=lambda x: x[0])
            ref_page_options = ["None"] + [f"Page {pi}" for pi, _ in published_with_images]
            ref_page_labels = {f"Page {pi}": url for pi, url in published_with_images}
            grade_style = get_story_grade_style(sb, story_id, reading_level)
            ref_default = "None"
            if grade_style and grade_style.get("reference_page_index") is not None:
                ref_label = f"Page {grade_style['reference_page_index']}"
                if ref_label in ref_page_options:
                    ref_default = ref_label
            ref_idx = ref_page_options.index(ref_default) if ref_default in ref_page_options else 0
            ref_selected = st.selectbox("Reference image (for Regenerate)", options=ref_page_options, index=ref_idx, key=f"bp_ref_{row_id}")
            selected_ref_url = ref_page_labels.get(ref_selected) if ref_selected != "None" else None

            correction = st.text_input("Correction (optional)", key=f"bp_corr_{row_id}", placeholder="e.g. Add a dragon in the background")
            if st.button("Regenerate image", key=f"bp_regen_img_{row_id}"):
                client = get_gemini()
                if not client:
                    st.error("Set GEMINI_API_KEY in .env.")
                else:
                    style = grade_style or {}
                    ap = (style.get("age_appropriateness") or "").strip()
                    sp = (style.get("global_style") or "").strip()
                    cr = (style.get("character_ref") or "").strip()
                    li = (style.get("lighting") or "").strip()
                    cp = (style.get("color_palette") or "").strip()
                    fr = (style.get("framing") or "").strip()
                    text_for_prompt = (st.session_state.get(f"bp_text_{row_id}", new_text) or "").strip() or page_text
                    labeled = reference_entries_from_grade_style(style)
                    refs, ref_note = collect_reference_images(
                        ref_image_url=selected_ref_url,
                        saved_labeled=labeled if labeled else None,
                    )
                    prompt = build_prompt(
                        (correction or "").strip(),
                        text_for_prompt,
                        ap, sp, cr, li, cp, fr,
                        visual_reference_note=ref_note,
                    )
                    with st.spinner("Generating image..."):
                        img, _usage = generate_image_gemini(prompt, refs if refs else None)
                    if img:
                        opt = optimize_image_for_mobile(img)
                        url_new = upload_image_to_storage(sb, story_id, reading_level, page_idx, opt)
                        if url_new and update_book_page(sb, row_id, image_url=url_new):
                            st.success("Image regenerated and saved.")
                            st.rerun()
                        else:
                            st.error("Upload or save failed.")
                    else:
                        st.error("Image generation failed.")

            if st.button("Replace image", key=f"bp_replace_img_{row_id}"):
                st.session_state.bp_editing_image = selected_row
                st.rerun()
            if img_url and st.button("Remove image", key=f"bp_remove_img_{row_id}", type="secondary"):
                if update_book_page(sb, row_id, image_url=""):
                    st.success("Image removed.")
                    st.rerun()

        _book_page_audio_voice_section(
            sb,
            voice="male",
            row_id=row_id,
            story_id=story_id,
            language_code=language_code,
            reading_level=reading_level,
            page_idx=page_idx,
            page_text=page_text,
            new_text=new_text,
            selected_row=selected_row,
            voices=VOICES_MALE,
            voice_select_index=bp_voice_male,
            speed=bp_speed,
        )
        _book_page_audio_voice_section(
            sb,
            voice="female",
            row_id=row_id,
            story_id=story_id,
            language_code=language_code,
            reading_level=reading_level,
            page_idx=page_idx,
            page_text=page_text,
            new_text=new_text,
            selected_row=selected_row,
            voices=VOICES_FEMALE,
            voice_select_index=bp_voice_female,
            speed=bp_speed,
        )

        # Delete
        with st.expander("Danger zone", expanded=False):
            if st.button("Delete this page", key=f"bp_del_{row_id}", type="secondary"):
                st.session_state.bp_pending_delete = [row_id]
                st.rerun()

    # Replace image modal
    if st.session_state.get("bp_editing_image"):
        replace_image_modal(st.session_state.bp_editing_image, story_id, reading_level)

    # Replace audio modal
    if st.session_state.get("bp_editing_audio"):
        row, voice = st.session_state.bp_editing_audio
        replace_audio_modal(row, story_id, language_code, reading_level, voice)

    # Delete confirmation
    pending = st.session_state.get("bp_pending_delete", [])
    if pending:
        st.warning(f"Delete {len(pending)} row(s)? This cannot be undone.")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("Yes, delete", key="bp_confirm_delete"):
                for rid in pending:
                    delete_book_page(sb, rid)
                del st.session_state.bp_pending_delete
                st.success(f"Deleted {len(pending)} row(s).")
                st.rerun()
        with d2:
            if st.button("Cancel", key="bp_cancel_delete"):
                del st.session_state.bp_pending_delete
                st.rerun()

    st.caption("Use the page selector above. Edit text and click Save text. Regenerate or upload image/audio, or delete in Danger zone.")
