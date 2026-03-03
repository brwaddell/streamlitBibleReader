"""Shared utilities for Storybook Image Processor."""
import base64
import os
import time
from io import BytesIO
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

from auth import get_secret

READING_LEVELS = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
LANGUAGE_CODES = ["en", "es", "fr"]  # Extend as needed

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
STORAGE_BUCKET = "storybook-images"
AUDIO_BUCKET = "storybook-audio"
MAX_IMAGE_SIZE = 800
TARGET_BYTES = 100_000

# Flattened story content table (replaces book_pages, localized_story_versions, story_assets)
TABLE_STORY_CONTENT_FLAT = "story_content_flat"
PAGE_NUMBER_COLUMN = "page_number"  # Always order by this ascending for correct story sequence

# --- Image generation (Gemini); shared by Image Processor and Book Pages ---
SYSTEM_INSTRUCTIONS = (
    "You are a consistent storybook illustrator. CRITICAL: The main character must have the EXACT SAME face, hair, beard, and clothing in every image. "
    "Use the provided character description as a fixed design—do not vary it. No creative reinterpretation of the character. "
    "Quality: Standard. Aspect Ratio: 1024x1024. Never include text, words, or letters in the image. "
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

    sections = []
    sections.append(f"SCENE TO ILLUSTRATE: {story}" if story else "SCENE TO ILLUSTRATE: (illustrate the story moment)")
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


def get_reference_images(
    ref_file=None,
    selected_page_index: Optional[int] = None,
    pages=None,
    ref_image_url: Optional[str] = None,
) -> List[bytes]:
    """Collect reference images: uploaded file, selected in-session page, or published/uploaded page URL."""
    refs: List[bytes] = []
    if ref_file is not None:
        if hasattr(ref_file, "seek"):
            ref_file.seek(0)
        data = ref_file.read()
        if data:
            refs.append(data)
    if selected_page_index is not None and pages and 0 <= selected_page_index < len(pages):
        p = pages[selected_page_index]
        if p.get("image"):
            refs.append(p["image"])
    if ref_image_url and ref_image_url.strip():
        try:
            r = requests.get(ref_image_url, timeout=10)
            if r.ok and r.content:
                refs.append(r.content)
        except Exception:
            pass
    return refs


def get_gemini():
    """Get Google Gemini client (cached in session state)."""
    if st.session_state.gemini_client is None:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            st.error("Set GEMINI_API_KEY in .env for Nano Banana Pro.")
            return None
        try:
            from google import genai
            st.session_state.gemini_client = genai.Client(api_key=api_key)
        except ImportError:
            st.error("Install google-genai: pip install google-genai")
            return None
    return st.session_state.gemini_client


def _prepare_reference_images(reference_images: Optional[List[bytes]]) -> List[Image.Image]:
    """Validate and normalize reference images to RGB PIL Images. Skips any that fail."""
    prepared = []
    if not reference_images:
        return prepared
    for img_bytes in reference_images[:5]:
        try:
            img = Image.open(BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            prepared.append(img)
        except Exception:
            pass
    return prepared


def _generate_image_with_client(client, prompt: str, reference_images: Optional[List[bytes]] = None) -> Tuple[Optional[bytes], Optional[str]]:
    """Generate image using an existing Gemini client. Returns (image_bytes, error_message)."""
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
            return (None, reason)
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            fr = getattr(finish_reason, "name", str(finish_reason)).upper()
            if fr not in ("STOP", "END_TURN", "1", ""):
                return (None, f"Response finish reason: {finish_reason}")
        parts = candidate.content.parts
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                data = getattr(part.inline_data, "data", None)
                if data is not None:
                    if isinstance(data, bytes):
                        return (data, None)
                    try:
                        return (base64.b64decode(data), None)
                    except Exception:
                        return (data if isinstance(data, bytes) else bytes(data), None)
            img = getattr(part, "as_image", lambda: None)()
            if img is not None:
                buf = BytesIO()
                try:
                    img.save(buf, "PNG")
                except TypeError:
                    img.save(buf)
                return (buf.getvalue(), None)
        return (None, "No image in response")
    except Exception as e:
        return (None, str(e))


_GEMINI_MAX_RETRIES = 2


def generate_image_gemini(prompt: str, reference_images: Optional[List[bytes]] = None) -> Optional[bytes]:
    """Generate image via Nano Banana Pro (Gemini 3 Pro Image). Retries on transient errors."""
    client = get_gemini()
    if not client:
        return None
    last_error = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        result, error = _generate_image_with_client(client, prompt, reference_images)
        if result is not None:
            return result
        last_error = error or "Unknown error"
        if attempt < _GEMINI_MAX_RETRIES - 1:
            if any(code in last_error for code in ("400", "500", "503", "UNAVAILABLE")):
                time.sleep(2)
                continue
            break
        break
    ref_count = len(reference_images) if reference_images else 0
    diag = f"Prompt length: {len(prompt)} chars, reference images: {ref_count}"
    st.error(f"Nano Banana Pro generation failed: {last_error}")
    st.caption(f"Debug: {diag}")
    return None


def _build_batch_request_parts(prompt: str, ref_images: Optional[List[bytes]] = None) -> list:
    """Build contents parts for a Batch API request: text + optional base64 ref images."""
    parts = [{"text": prompt}]
    if ref_images:
        for img_bytes in ref_images[:5]:
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
            updates["audio_male_url"] = audio_male_url
        if audio_female_url is not None:
            updates["audio_female_url"] = audio_female_url
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


def upload_audio_to_storage(
    supabase, story_id: int, language_code: str, reading_level: str, page_index: int, audio_bytes: bytes, voice: str
) -> Optional[str]:
    """Upload audio to R2. voice is 'male' or 'female'. Returns public URL or None."""
    from storage_r2 import upload_audio

    path = f"{story_id}/{language_code}/{reading_level}/page_{page_index}_{voice}.mp3"
    return upload_audio(path, audio_bytes)


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
    """Fetch story_content_flat for (story_id, language_code, reading_level) where image_url is NULL or empty. Ordered by page_number."""
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
        return [
            row
            for row in rows
            if not (row.get("image_url") or "").strip()
        ]
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
                url = upload_audio_to_storage(
                    get_supabase(), story_id, language_code, reading_level, pn, data, voice
                )
                if url:
                    field = "audio_male_url" if voice == "male" else "audio_female_url"
                    if update_book_page(get_supabase(), row["id"], **{field: url}):
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


def run_audio_generator_view():
    """Audio Generator: batch create male/female audio via ElevenLabs for pages missing audio."""
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

    lang_options = sorted({v.get("language_code") for v in versions if v.get("language_code")}) or LANGUAGE_CODES
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

    # ElevenLabs settings: per level when "All levels", single set when "Single level"
    stability_default = 0.5
    similarity_boost_default = 0.75
    use_speaker_boost = False
    model_id = "eleven_flash_v2_5"
    output_format = "mp3_22050_32"
    optimize_streaming_latency = 1
    apply_text_normalization = "auto"

    def _get_ag_settings_for_level(lev):
        """Return speed, voice_male, voice_female for a level (from session state or defaults)."""
        if ag_all_levels_mode:
            male_idx = int(st.session_state.get(f"ag_voice_male_{lev}", 0))
            female_idx = int(st.session_state.get(f"ag_voice_female_{lev}", 0))
            return {
                "speed": float(st.session_state.get(f"ag_speed_{lev}", 1.0)),
                "voice_male": VOICES_MALE[male_idx][1],
                "voice_female": VOICES_FEMALE[female_idx][1],
            }
        else:
            male_idx = int(st.session_state.get("ag_voice_male", 0))
            female_idx = int(st.session_state.get("ag_voice_female", 0))
            return {
                "speed": float(st.session_state.get("ag_speed", 1.0)),
                "voice_male": VOICES_MALE[male_idx][1],
                "voice_female": VOICES_FEMALE[female_idx][1],
            }

    if ag_all_levels_mode:
        st.caption("Set speed and voices separately for each level. These are used when you generate audio for that level.")
        for lev in READING_LEVELS:
            with st.expander(f"Audio settings: {lev.replace('_', ' ').title()}", expanded=False):
                st.slider(
                    "Speed (0.7 = slower, 1.2 = faster)",
                    min_value=0.7,
                    max_value=1.2,
                    value=float(st.session_state.get(f"ag_speed_{lev}", 1.0)),
                    step=0.05,
                    key=f"ag_speed_{lev}",
                )
                st.selectbox(
                    "Male voice",
                    options=range(len(VOICES_MALE)),
                    format_func=lambda i: f"{VOICES_MALE[i][0]} — {VOICES_MALE[i][2]}",
                    key=f"ag_voice_male_{lev}",
                )
                st.selectbox(
                    "Female voice",
                    options=range(len(VOICES_FEMALE)),
                    format_func=lambda i: f"{VOICES_FEMALE[i][0]} — {VOICES_FEMALE[i][2]}",
                    key=f"ag_voice_female_{lev}",
                )
        st.caption("Model: eleven_flash_v2_5 · Output: mp3_22050_32 · Stability: 0.5 · Similarity: 0.75 · Latency: Normal")
    else:
        v1, v2 = st.columns(2)
        with v1:
            male_opt = st.selectbox(
                "Male voice",
                options=range(len(VOICES_MALE)),
                format_func=lambda i: f"{VOICES_MALE[i][0]} — {VOICES_MALE[i][2]}",
                key="ag_voice_male",
            )
            voice_male = VOICES_MALE[male_opt][1]
        with v2:
            female_opt = st.selectbox(
                "Female voice",
                options=range(len(VOICES_FEMALE)),
                format_func=lambda i: f"{VOICES_FEMALE[i][0]} — {VOICES_FEMALE[i][2]}",
                key="ag_voice_female",
            )
            voice_female = VOICES_FEMALE[female_opt][1]
        with st.expander("ElevenLabs audio settings", expanded=False):
            st.caption("Optimized for mobile: lightweight model, small files, balanced quality.")
            st.slider(
                "Speed (0.7 = slower, 1.2 = faster)",
                min_value=0.7,
                max_value=1.2,
                value=1.0,
                step=0.05,
                key="ag_speed",
            )
            st.caption("Model: eleven_flash_v2_5 · Output: mp3_22050_32 · Stability: 0.5 · Similarity: 0.75 · Latency: Normal")

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

    # Show which rows need audio (story + language + grade), like translator’s “Found X English pages”
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

    st.subheader("1. Preview first page")
    st.caption("Generate male and/or female audio for the first page. Listen and adjust settings before running the full batch.")
    if ag_all_levels_mode:
        preview_level = st.selectbox(
            "Preview using settings for",
            options=READING_LEVELS,
            key="ag_preview_settings_level",
            format_func=lambda x: x.replace("_", " ").title(),
        )
        first = (levels_missing.get(preview_level) or [None])[0] if levels_missing else None
    else:
        first = pages_missing[0] if pages_missing else None
    page_text = _get_page_text(first) if first else ""
    if ag_all_levels_mode:
        preview_settings = _get_ag_settings_for_level(preview_level)
        preview_voice_male = preview_settings["voice_male"]
        preview_voice_female = preview_settings["voice_female"]
        preview_speed = preview_settings["speed"]
    else:
        preview_voice_male = voice_male
        preview_voice_female = voice_female
        preview_speed = float(st.session_state.get("ag_speed", 1.0))
    pre1, pre2 = st.columns(2)
    with pre1:
        if st.button("Preview male voice", key="ag_preview_male"):
            if first and page_text:
                with st.spinner("Generating male preview..."):
                    male_bytes, _ = generate_elevenlabs_audio(
                        api_key, preview_voice_male, page_text, language_code,
                        stability=stability, similarity_boost=similarity_boost,
                        use_speaker_boost=use_speaker_boost, speed=preview_speed,
                        model_id=model_id, output_format=output_format,
                        optimize_streaming_latency=optimize_streaming_latency,
                        apply_text_normalization=apply_text_normalization,
                    )
                st.session_state["ag_preview_male_bytes"] = male_bytes
        if st.session_state.get("ag_preview_male_bytes"):
            st.audio(st.session_state["ag_preview_male_bytes"], format="audio/mp3")
            male_idx = int(st.session_state.get(f"ag_voice_male_{preview_level}", 0)) if ag_all_levels_mode else male_opt
            st.caption(f"Male ({VOICES_MALE[male_idx][0]})")
    with pre2:
        if st.button("Preview female voice", key="ag_preview_female"):
            if first and page_text:
                with st.spinner("Generating female preview..."):
                    female_bytes, _ = generate_elevenlabs_audio(
                        api_key, preview_voice_female, page_text, language_code,
                        stability=stability, similarity_boost=similarity_boost,
                        use_speaker_boost=use_speaker_boost, speed=preview_speed,
                        model_id=model_id, output_format=output_format,
                        optimize_streaming_latency=optimize_streaming_latency,
                        apply_text_normalization=apply_text_normalization,
                    )
                st.session_state["ag_preview_female_bytes"] = female_bytes
        if st.session_state.get("ag_preview_female_bytes"):
            st.audio(st.session_state["ag_preview_female_bytes"], format="audio/mp3")
            female_idx = int(st.session_state.get(f"ag_voice_female_{preview_level}", 0)) if ag_all_levels_mode else female_opt
            st.caption(f"Female ({VOICES_FEMALE[female_idx][0]})")

    st.subheader("2. Generate all")
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
            speed = float(st.session_state.get("ag_speed", 1.0))
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
                            api_key, voice_male, page_text, language_code,
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
                            api_key, voice_female, page_text, language_code,
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
        st.subheader("3. Preview generated batch (all levels)")
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
                        url = upload_audio_to_stories_path(
                            sb_save, b["story_id"], b["language_code"], reading_level_b, "male", page_index, b_bytes
                        )
                        if url and update_book_page(sb_save, row_id, audio_male_url=url, timing_male_json=t):
                            ok += 1
                    if entry.get("female"):
                        b_bytes, t = entry["female"]
                        url = upload_audio_to_stories_path(
                            sb_save, b["story_id"], b["language_code"], reading_level_b, "female", page_index, b_bytes
                        )
                        if url and update_book_page(sb_save, row_id, audio_female_url=url, timing_female_json=t):
                            ok += 1
            del st.session_state["ag_generated_batches"]
            st.success(f"Saved {ok} audio URLs to story_content_flat (all levels).")
            st.rerun()
    elif batch and batch.get("pages"):
        st.subheader("3. Preview generated batch")
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
                    url = upload_audio_to_stories_path(
                        sb_save, batch["story_id"], batch["language_code"], reading_level, "male", page_index, b
                    )
                    if url and update_book_page(sb_save, row_id, audio_male_url=url, timing_male_json=t):
                        ok += 1
                if entry.get("female"):
                    b, t = entry["female"]
                    url = upload_audio_to_stories_path(
                        sb_save, batch["story_id"], batch["language_code"], reading_level, "female", page_index, b
                    )
                    if url and update_book_page(sb_save, row_id, audio_female_url=url, timing_female_json=t):
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


def run_book_pages_view():
    """Render Book Pages: page selector + focus panel with edit text, regenerate image/audio, upload, delete."""
    st.title("Book Pages")
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
    bp_lang_options = sorted({v.get("language_code") for v in versions if v.get("language_code")}) or LANGUAGE_CODES
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
                    prompt = build_prompt(
                        (correction or "").strip(),
                        text_for_prompt,
                        ap, sp, cr, li, cp, fr,
                    )
                    refs = get_reference_images(ref_image_url=selected_ref_url)
                    with st.spinner("Generating image..."):
                        img = generate_image_gemini(prompt, refs if refs else None)
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

        # Male / Female audio
        st.markdown("**Male audio**")
        male_url = selected_row.get("audio_male_url")
        if male_url:
            st.audio(male_url, format="audio/mpeg")
        else:
            st.caption("—")
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Regenerate male", key=f"bp_regen_m_{row_id}"):
                api_key = get_secret("ELEVENLABS_API_KEY")
                if not api_key:
                    st.error("Set ELEVENLABS_API_KEY in .env for Regenerate audio.")
                else:
                    text_for_audio = (st.session_state.get(f"bp_text_{row_id}", new_text) or "").strip() or page_text
                    if not text_for_audio:
                        st.warning("Page text is empty.")
                    else:
                        voice_id = VOICES_MALE[bp_voice_male][1]
                        with st.spinner("Generating male audio..."):
                            audio_bytes, timing = generate_elevenlabs_audio(
                                api_key, voice_id, text_for_audio, language_code,
                                stability=0.5, similarity_boost=0.75, speed=bp_speed,
                            )
                        if audio_bytes:
                            url_audio = upload_audio_to_stories_path(sb, story_id, language_code, reading_level, "male", page_idx, audio_bytes)
                            if url_audio and update_book_page(sb, row_id, audio_male_url=url_audio, timing_male_json=timing):
                                st.success("Male audio regenerated and saved.")
                                st.rerun()
                            else:
                                st.error("Upload or save failed.")
                        else:
                            st.error("Audio generation failed.")
        with a2:
            if st.button("Upload male audio", key=f"bp_upload_m_{row_id}"):
                st.session_state.bp_editing_audio = (selected_row, "male")
                st.rerun()

        st.markdown("**Female audio**")
        female_url = selected_row.get("audio_female_url")
        if female_url:
            st.audio(female_url, format="audio/mpeg")
        else:
            st.caption("—")
        f1, f2 = st.columns(2)
        with f1:
            if st.button("Regenerate female", key=f"bp_regen_f_{row_id}"):
                api_key = get_secret("ELEVENLABS_API_KEY")
                if not api_key:
                    st.error("Set ELEVENLABS_API_KEY in .env for Regenerate audio.")
                else:
                    text_for_audio = (st.session_state.get(f"bp_text_{row_id}", new_text) or "").strip() or page_text
                    if not text_for_audio:
                        st.warning("Page text is empty.")
                    else:
                        voice_id = VOICES_FEMALE[bp_voice_female][1]
                        with st.spinner("Generating female audio..."):
                            audio_bytes, timing = generate_elevenlabs_audio(
                                api_key, voice_id, text_for_audio, language_code,
                                stability=0.5, similarity_boost=0.75, speed=bp_speed,
                            )
                        if audio_bytes:
                            url_audio = upload_audio_to_stories_path(sb, story_id, language_code, reading_level, "female", page_idx, audio_bytes)
                            if url_audio and update_book_page(sb, row_id, audio_female_url=url_audio, timing_female_json=timing):
                                st.success("Female audio regenerated and saved.")
                                st.rerun()
                            else:
                                st.error("Upload or save failed.")
                        else:
                            st.error("Audio generation failed.")
        with f2:
            if st.button("Upload female audio", key=f"bp_upload_f_{row_id}"):
                st.session_state.bp_editing_audio = (selected_row, "female")
                st.rerun()

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
