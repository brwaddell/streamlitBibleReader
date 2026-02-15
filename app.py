"""
Storybook Image Processor - Bulk image generation with approval workflow.
Images via Nano Banana Pro (Gemini). Extra-details hints via OpenAI GPT.
Review/approve/regenerate, then export to Supabase Storage + story_content_flat table.
"""

import base64
import os
from io import BytesIO
from typing import List, Optional

from PIL import Image

import streamlit as st
from dotenv import load_dotenv

from auth import get_secret, is_authenticated, logout, run_login_page
from lib import (
    fetch_batch_jobs_for_story,
    fetch_batch_jobs_for_version,
    fetch_image_for_display,
    fetch_book_pages,
    fetch_pages_missing_images,
    fetch_stories,
    get_supabase,
    get_story_grade_style,
    insert_batch_job,
    run_audio_generator_view,
    run_book_pages_view,
    update_batch_job_status,
    update_book_page,
    upsert_story_grade_style,
)
from stories_page import run_stories_view
from story_text_page import run_story_text_view
from translator_page import run_translator_view

load_dotenv()

READING_LEVELS = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]

GRADE_STYLE_DEFAULTS = {
    "grade_1": {
        "age_appropriateness": "Pre-school (ages 3–5). Simple, friendly, reassuring visuals.",
        "global_style": "Bright, simple flat-color illustrations, bold outlines, thick brushstrokes, very clear subjects, minimal background detail. High contrast and simplicity.",
        "character_ref": "Noah: elderly Middle Eastern man, long white beard to chest, weathered kind face with deep-set eyes, shoulder-length white hair, simple brown robe with rope belt, sandals. CRITICAL: This exact same character design must appear identically in every image.",
        "color_palette": "Earth tones, high contrast",
        "lighting": "Soft morning light",
        "framing": "Medium shot, centered subject, warm composition",
    },
    "grade_2": {
        "age_appropriateness": "Early reader (5–6 yrs). Clear focal points, engaging and easy to process.",
        "global_style": "Soft watercolor textures, hand-drawn charcoal outlines, gentle gradients, whimsical and warm atmosphere. Storybook feel.",
        "character_ref": "Noah: elderly Middle Eastern man, long white beard to chest, weathered kind face with deep-set eyes, shoulder-length white hair, simple brown robe with rope belt, sandals. CRITICAL: This exact same character design must appear identically in every image.",
        "color_palette": "Earth tones, warm browns and greens",
        "lighting": "Soft morning light",
        "framing": "Medium shot, centered subject, warm composition",
    },
    "grade_3": {
        "age_appropriateness": "Developing reader (7–8 yrs). More narrative detail while remaining approachable.",
        "global_style": "Richer watercolor textures, expressive linework, gentle gradients, warm storybook atmosphere. More environmental context.",
        "character_ref": "Noah: elderly Middle Eastern man, long white beard to chest, weathered kind face with deep-set eyes, shoulder-length white hair, simple brown robe with rope belt, sandals. CRITICAL: This exact same character design must appear identically in every image.",
        "color_palette": "Earth tones, warm browns and greens",
        "lighting": "Soft morning light",
        "framing": "Medium shot, balanced composition, warm storybook framing",
    },
    "grade_4": {
        "age_appropriateness": "Fluent reader (9–10 yrs). Sophisticated, more complex visuals.",
        "global_style": "Cinematic digital art, rich textures, detailed environmental storytelling, dramatic lighting (Chiaroscuro). Reverent tone.",
        "character_ref": "Noah: elderly Middle Eastern man, long white beard to chest, weathered kind face with deep-set eyes, shoulder-length white hair, simple brown robe with rope belt, sandals. CRITICAL: This exact same character design must appear identically in every image.",
        "color_palette": "Earth tones, rich and deep",
        "lighting": "Dramatic Chiaroscuro, reverent and epic",
        "framing": "Cinematic framing, rule of thirds, reverent composition",
    },
    "grade_5": {
        "age_appropriateness": "Independent reader (11+ yrs). Mature, nuanced visuals for older readers.",
        "global_style": "Cinematic digital art, rich textures, detailed environmental storytelling, dramatic lighting (Chiaroscuro). Reverent and epic tone.",
        "character_ref": "Noah: elderly Middle Eastern man, long white beard to chest, weathered kind face with deep-set eyes, shoulder-length white hair, simple brown robe with rope belt, sandals. CRITICAL: This exact same character design must appear identically in every image.",
        "color_palette": "Earth tones, rich and deep",
        "lighting": "Dramatic Chiaroscuro, reverent and epic",
        "framing": "Cinematic framing, rule of thirds, epic composition",
    },
}


def init_session_state():
    """Initialize session state keys."""
    defaults = {
        "story_id": None,
        "reading_level": None,
        "language_code": "en",
        "last_reading_level": None,
        "last_story_id": None,
        "supabase": None,
        "openai_client": None,
        "gemini_client": None,
        "ip_pending_images": {},  # row_id -> optimized image bytes (awaiting approve)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    grade_defaults = GRADE_STYLE_DEFAULTS["grade_1"]
    mapping = {
        "age_appropriateness": "age_appropriateness",
        "style_prompt": "global_style",
        "character_ref": "character_ref",
        "color_palette": "color_palette",
        "lighting": "lighting",
        "framing": "framing",
    }
    for session_key, defaults_key in mapping.items():
        if session_key not in st.session_state:
            st.session_state[session_key] = grade_defaults[defaults_key]


def get_openai():
    """Get OpenAI client (cached in session state)."""
    if st.session_state.openai_client is None:
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            st.error("Set OPENAI_API_KEY in .env")
            return None
        from openai import OpenAI
        st.session_state.openai_client = OpenAI(api_key=api_key)
    return st.session_state.openai_client


def build_style_string(
    age_appropriateness: str,
    style: str,
    character_ref: str,
    lighting: str,
    palette: str,
    framing: str = "",
) -> str:
    """Build Style section: [Age Appropriateness], [Global Style], [Character Ref], [Lighting], [Palette], [Framing]."""
    parts = [p for p in [age_appropriateness, style, character_ref, lighting, palette, framing] if p and p.strip()]
    return ", ".join(parts)


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


def generate_extra_details(
    client,
    story_text: str,
    character_ref: str,
    story_context: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """Call GPT to generate a 1-sentence visual description for the illustrator."""
    if not story_text or not story_text.strip():
        return None
    context_line = ""
    if story_context and story_context.strip():
        context_line = f"This story is: {story_context.strip()}. "
    prompt = (
        f'You are a storyboard artist for a children\'s storybook app. '
        f'{context_line}'
        f'Given this story text: "{story_text.strip()}", '
        f'write a 1-sentence visual description for an illustrator. '
        f'Focus on the character\'s action and the environment. '
        f'Use this character (keep consistent): {character_ref or "the main character"}. '
        f'Do not mention style, just the scene action.'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception as e:
        st.error(f"Failed to generate extra details: {e}")
        return None


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
            import requests
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


def _generate_image_with_client(client, prompt: str, reference_images: Optional[List[bytes]] = None) -> Optional[bytes]:
    """Generate image using an existing Gemini client (for background worker)."""
    try:
        from google.genai import types
        contents = [prompt]
        if reference_images:
            for img_bytes in reference_images[:5]:
                contents.append(Image.open(BytesIO(img_bytes)))
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        parts = response.candidates[0].content.parts
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                data = getattr(part.inline_data, "data", None)
                if data is not None:
                    if isinstance(data, bytes):
                        return data
                    try:
                        return base64.b64decode(data)
                    except Exception:
                        return data if isinstance(data, bytes) else bytes(data)
            img = getattr(part, "as_image", lambda: None)()
            if img is not None:
                buf = BytesIO()
                try:
                    img.save(buf, "PNG")
                except TypeError:
                    img.save(buf)
                return buf.getvalue()
        return None
    except Exception as e:
        return None  # Worker thread; don't use st.error here


def generate_image_gemini(prompt: str, reference_images: Optional[List[bytes]] = None) -> Optional[bytes]:
    """Generate image via Nano Banana Pro (Gemini 3 Pro Image)."""
    client = get_gemini()
    if not client:
        return None
    result = _generate_image_with_client(client, prompt, reference_images)
    if result is None:
        st.error("Nano Banana Pro generation failed.")
    return result


MAX_IMAGE_SIZE = 800
TARGET_BYTES = 100_000  # ~100KB for mobile


def optimize_image_for_mobile(image_bytes: bytes) -> bytes:
    """Resize to max 800x800 and convert to WebP, targeting ~100KB for mobile."""
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


def upload_to_storage(supabase, story_id: int, reading_level: str, page_index: int, image_bytes: bytes) -> Optional[str]:
    """Upload image to R2. Returns public URL or None."""
    from storage_r2 import upload_image

    path = f"{story_id}/{reading_level}/page_{page_index}.webp"
    return upload_image(path, image_bytes)


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


def main():
    st.set_page_config(page_title="Storybook Image Processor", layout="wide", initial_sidebar_state="expanded")

    if not is_authenticated():
        run_login_page()
        st.stop()

    init_session_state()

    st.sidebar.title("Storybook")
    page = st.sidebar.radio("Go to", ["Stories", "Story Text", "Image Processor", "Book Pages", "Translator", "Audio Generator"])
    st.sidebar.divider()
    if st.sidebar.button("Sign out"):
        logout()
        st.rerun()

    if page == "Stories":
        run_stories_view()
        return

    if page == "Story Text":
        run_story_text_view()
        return

    if page == "Book Pages":
        run_book_pages_view()
        return

    if page == "Audio Generator":
        run_audio_generator_view()
        return

    if page == "Translator":
        run_translator_view()
        return

    st.title("Image Processor")
    st.caption("Generate images for pages missing images. Add story text in Story Text first, then batch generate here.")

    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories()
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    # --- Filters ---
    st.header("1. Select story & version")
    language_code = "en"
    story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
    story_label = st.selectbox("Story", options=list(story_options.keys()), key="ip_story")
    story_id = story_options.get(story_label) if story_label else None

    ip_mode = st.radio(
        "Mode",
        options=["Single level", "All levels"],
        key="ip_mode",
        horizontal=True,
        help="Single level: one grade at a time. All levels: set style per grade, then batch all missing images for this story.",
    )
    all_levels_mode = ip_mode == "All levels"

    if not story_id:
        st.info("Select a story to continue.")
        return

    if all_levels_mode:
        # Fetch missing pages for every level
        levels_missing = {
            level: fetch_pages_missing_images(sb, story_id, language_code, level)
            for level in READING_LEVELS
        }
        total_missing = sum(len(p) for p in levels_missing.values())
        if total_missing == 0:
            st.success("No pages missing images for this story across any level.")
            st.caption("Add page text in Story Text first, or pick another story.")
            return
        # Summary: Grade 1: 12 missing · Grade 2: 12 missing · … · Total: 58
        parts = [f"**{level}**: {len(levels_missing[level])} missing" for level in READING_LEVELS]
        st.success(" · ".join(parts) + f" → **Total: {total_missing} pages**")
        # Current level for style/review: user picks via tabs/selector; default first level with missing
        if "ip_all_level" not in st.session_state or st.session_state["ip_all_level"] not in READING_LEVELS:
            st.session_state["ip_all_level"] = next(
                (lev for lev in READING_LEVELS if levels_missing[lev]), READING_LEVELS[0]
            )
        reading_level = st.session_state["ip_all_level"]
        pages_missing = levels_missing.get(reading_level, [])
    else:
        reading_level = st.selectbox("Reading level", options=READING_LEVELS, key="ip_reading_level")
        pages_missing = fetch_pages_missing_images(sb, story_id, language_code, reading_level)
        if not pages_missing:
            st.success("No pages missing images for this story + reading level.")
            st.caption("Add page text in Story Text first, or pick another story/level.")
            return
        st.success(f"**{len(pages_missing)}** pages missing images.")

    # Load style defaults for current (story_id, reading_level)
    story = next((s for s in stories if s.get("id") == story_id), None)
    if story_id and reading_level:
        defaults = dict(GRADE_STYLE_DEFAULTS.get(reading_level, GRADE_STYLE_DEFAULTS["grade_1"]))
        if story:
            for key in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                if story.get(key) not in (None, ""):
                    defaults[key] = story[key]
        grade_style = get_story_grade_style(sb, story_id, reading_level)
        if grade_style:
            for key in ["age_appropriateness", "global_style", "character_ref", "color_palette", "lighting", "framing"]:
                if grade_style.get(key) not in (None, ""):
                    defaults[key] = grade_style[key]
        # In all_levels_mode we don't persist to generic session_state; we use value= from defaults in the form
        if not all_levels_mode:
            for k, v in defaults.items():
                if k == "global_style":
                    if "style_prompt" not in st.session_state or not (st.session_state.get("style_prompt") or "").strip():
                        st.session_state["style_prompt"] = v
                elif k not in st.session_state or not (st.session_state.get(k) or "").strip():
                    st.session_state[k] = v

    # --- Image style controls ---
    st.header("2. Image style controls")
    if all_levels_mode:
        st.caption("Set style separately for each level. Select a level, edit, then Save. Batch submit uses saved styles.")
        level_label = st.selectbox(
            "Editing style for",
            options=READING_LEVELS,
            key="ip_all_level",
            format_func=lambda x: x.replace("_", " ").title(),
        )
        reading_level = level_label  # use selected level for form and save
        # Re-load defaults for the selected level (for value= below)
        defaults = dict(GRADE_STYLE_DEFAULTS.get(reading_level, GRADE_STYLE_DEFAULTS["grade_1"]))
        if story:
            for key in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                if story.get(key) not in (None, ""):
                    defaults[key] = story[key]
        grade_style = get_story_grade_style(sb, story_id, reading_level)
        if grade_style:
            for key in ["age_appropriateness", "global_style", "character_ref", "color_palette", "lighting", "framing"]:
                if grade_style.get(key) not in (None, ""):
                    defaults[key] = grade_style[key]
        age_appropriateness_val = defaults.get("age_appropriateness", "")
        style_prompt_val = defaults.get("global_style", "")
        character_ref_val = defaults.get("character_ref", "")
        color_palette_val = defaults.get("color_palette", "")
        lighting_val = defaults.get("lighting", "")
        framing_val = defaults.get("framing", "")
        # Force session_state for this level so widgets show the loaded values when user switches grade
        # (On Cloud, widget keys persist and value= is ignored once the key exists; without this, changing level doesn't update the form.)
        if st.session_state.get("ip_style_loaded_level") != reading_level:
            st.session_state["ip_style_loaded_level"] = reading_level
            st.session_state[f"ip_age_{reading_level}"] = age_appropriateness_val
            st.session_state[f"ip_style_{reading_level}"] = style_prompt_val
            st.session_state[f"ip_char_{reading_level}"] = character_ref_val
            st.session_state[f"ip_color_{reading_level}"] = color_palette_val
            st.session_state[f"ip_light_{reading_level}"] = lighting_val
            st.session_state[f"ip_frame_{reading_level}"] = framing_val
    else:
        age_appropriateness_val = st.session_state.get("age_appropriateness", "")
        style_prompt_val = st.session_state.get("style_prompt", "")
        character_ref_val = st.session_state.get("character_ref", "")
        color_palette_val = st.session_state.get("color_palette", "")
        lighting_val = st.session_state.get("lighting", "")
        framing_val = st.session_state.get("framing", "")

    with st.expander("Style controls", expanded=True):
        age_appropriateness = st.text_input(
            "Age appropriateness",
            value=age_appropriateness_val,
            key="age_appropriateness" if not all_levels_mode else f"ip_age_{reading_level}",
        )
        style_prompt = st.text_area(
            "Global style prompt",
            value=style_prompt_val,
            key="style_prompt" if not all_levels_mode else f"ip_style_{reading_level}",
        )
        character_ref = st.text_input(
            "Character reference",
            value=character_ref_val,
            key="character_ref" if not all_levels_mode else f"ip_char_{reading_level}",
        )
        color_palette = st.text_input("Color palette", value=color_palette_val, key="color_palette" if not all_levels_mode else f"ip_color_{reading_level}")
        lighting = st.text_input("Lighting", value=lighting_val, key="lighting" if not all_levels_mode else f"ip_light_{reading_level}")
        framing = st.text_input("Framing", value=framing_val, key="framing" if not all_levels_mode else f"ip_frame_{reading_level}")
        ref_file = st.file_uploader(
            "Reference image for character consistency (optional)",
            type=["png", "jpg", "jpeg"],
            key="ref_image",
        )
        all_pages = fetch_book_pages(sb, story_id, reading_level, language_code)
        published_with_images = [(r.get("page_index", i), r.get("image_url")) for i, r in enumerate(all_pages) if r.get("image_url")]
        ref_page_options = ["None"] + [f"Page {pi}" for pi, _ in published_with_images]
        ref_page_labels = {f"Page {pi}": url for pi, url in published_with_images}
        ref_selected = st.selectbox(
            "Use published page as reference",
            options=ref_page_options,
            key="ref_page_select" if not all_levels_mode else f"ip_refpage_{reading_level}",
        )
        selected_ref_url = ref_page_labels.get(ref_selected) if ref_selected != "None" else None
        if story_id and reading_level and st.button("Save style for this story & grade", key=f"save_style_{reading_level}" if all_levels_mode else "save_style"):
            if upsert_story_grade_style(sb, story_id, reading_level, {
                "age_appropriateness": age_appropriateness or "",
                "global_style": style_prompt or "",
                "character_ref": character_ref or "",
                "color_palette": color_palette or "",
                "lighting": lighting or "",
                "framing": framing or "",
            }):
                st.success(f"Style saved for {reading_level}.")
                st.rerun()

    # --- Batch status (check pending/running jobs) ---
    if all_levels_mode:
        batch_jobs = fetch_batch_jobs_for_story(sb, story_id)
    else:
        batch_jobs = fetch_batch_jobs_for_version(sb, story_id, reading_level)
    pending_jobs = [j for j in batch_jobs if j.get("status") in ("pending", "running")]

    # --- Batch generate ---
    st.header("3. Batch generate")
    st.caption("Generate the first page to use as a reference in style controls, then batch generate the rest.")

    def _check_batch_status(sb, job, pages_for_job, get_gemini_fn):
        """Check one batch job and apply results to pages_for_job; returns True if rerun needed."""
        check_job_key = job.get("batch_name")
        client = get_gemini_fn()
        if not client or not check_job_key:
            return False
        try:
            batch_job = client.batches.get(name=check_job_key)
            state = getattr(getattr(batch_job, "state", None), "name", None) or str(getattr(batch_job, "state", ""))
            if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_SUCCESS"):
                dest = getattr(batch_job, "dest", None) or getattr(batch_job, "destination", None)
                inlined = None
                if dest:
                    inlined = getattr(dest, "inlined_responses", None) or getattr(dest, "inlinedResponses", None)
                if inlined and pages_for_job:
                    if len(inlined) != len(pages_for_job):
                        st.warning("Page count changed since batch was submitted. Results may not match.")
                    pending = st.session_state.get("ip_pending_images", {})
                    ok_count = 0
                    failed_keys = []
                    for i, ir in enumerate(inlined):
                        if i >= len(pages_for_job):
                            break
                        row = pages_for_job[i]
                        row_id = row.get("id")
                        err = getattr(ir, "error", None)
                        if err:
                            failed_keys.append(f"page {row.get('page_index', i)}")
                            continue
                        resp = getattr(ir, "response", None) or ir
                        img_bytes = _extract_image_from_batch_response(resp)
                        if img_bytes and row_id:
                            opt = optimize_image_for_mobile(img_bytes)
                            pending[row_id] = opt
                            ok_count += 1
                        else:
                            failed_keys.append(f"page {row.get('page_index', i)}")
                    if failed_keys:
                        st.warning(f"Some pages failed or had no image: {', '.join(failed_keys)}")
                    st.session_state["ip_pending_images"] = pending
                    update_batch_job_status(sb, check_job_key, "succeeded")
                    st.success(f"Loaded {ok_count} images. Approve each in Review below.")
                else:
                    st.warning("No inline responses found or pages have changed.")
                return True
            elif state in ("JOB_STATE_FAILED", "JOB_STATE_FAILURE"):
                err = getattr(batch_job, "error", None) or ""
                update_batch_job_status(sb, check_job_key, "failed", error_message=str(err))
                st.error(f"Batch failed: {err}")
                return True
            elif "EXPIRED" in str(state) or "CANCELLED" in str(state):
                update_batch_job_status(sb, check_job_key, "expired" if "EXPIRED" in str(state) else "cancelled")
                st.warning("Job expired or was cancelled. Submit a new batch.")
                return True
            else:
                st.info("Still processing. Check again in a few minutes.")
                return True
        except Exception as e:
            st.error(f"Failed to check batch: {e}")
            return False

    if pending_jobs:
        st.info("You have batch jobs in progress. Return to this page later and click **Check batch status**.")
        for job in pending_jobs:
            created = job.get("created_at", "")[:19].replace("T", " ") if job.get("created_at") else "?"
            job_level = job.get("reading_level", "?")
            with st.expander(f"{job_level} – Batch {job.get('batch_name', '?')} – submitted {created} – Status: {job.get('status', '?')}"):
                st.caption(f"{job.get('page_count', 0)} pages. Images are generated asynchronously (~24h).")
                if st.button("Check batch status", key=f"ip_check_{job.get('batch_name', '')}"):
                    if all_levels_mode:
                        pages_for_job = fetch_pages_missing_images(sb, story_id, language_code, job_level)
                    else:
                        pages_for_job = pages_missing
                    if _check_batch_status(sb, job, pages_for_job, get_gemini):
                        st.rerun()

    gen_first_col, gen_all_col = st.columns(2)
    with gen_first_col:
        gen_first = st.button("Generate first page only", key="ip_gen_first")
    with gen_all_col:
        gen_all = st.button("Submit batch (50% cheaper, results in ~24h)", type="primary", key="ip_batch_gen")

    st.caption("Images are generated asynchronously. Return to this page later and click Check batch status.")

    if gen_first and pages_missing:
        client = get_gemini()
        if not client:
            st.error("Set GEMINI_API_KEY in .env.")
        else:
            first_row = pages_missing[0]
            page_text = (first_row.get("page_text") or first_row.get("text") or "").strip()
            refs = get_reference_images(ref_file, ref_image_url=selected_ref_url)
            with st.spinner("Generating first page..."):
                prompt = build_prompt(
                    "", page_text,
                    age_appropriateness or "", style_prompt or "", character_ref or "",
                    lighting or "", color_palette or "", framing or "",
                )
                img = generate_image_gemini(prompt, refs if refs else None)
            if img:
                opt = optimize_image_for_mobile(img)
                pending = st.session_state.get("ip_pending_images", {})
                pending[first_row["id"]] = opt
                st.session_state["ip_pending_images"] = pending
                st.success("First page generated. Go to Review to approve (export to R2 + Supabase), then use as reference.")
                st.rerun()
            else:
                st.error("Generation failed.")

    if gen_all:
        client = get_gemini()
        if not client:
            st.error("Set GEMINI_API_KEY in .env.")
        elif all_levels_mode and levels_missing:
            from google.genai import types
            refs = get_reference_images(ref_file, ref_image_url=selected_ref_url)
            gen_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            )
            submitted = []
            try:
                for level in READING_LEVELS:
                    pm = levels_missing.get(level, [])
                    if not pm:
                        continue
                    defaults = dict(GRADE_STYLE_DEFAULTS.get(level, GRADE_STYLE_DEFAULTS["grade_1"]))
                    if story:
                        for key in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                            if story.get(key) not in (None, ""):
                                defaults[key] = story[key]
                    grade_style = get_story_grade_style(sb, story_id, level)
                    if grade_style:
                        for key in ["age_appropriateness", "global_style", "character_ref", "color_palette", "lighting", "framing"]:
                            if grade_style.get(key) not in (None, ""):
                                defaults[key] = grade_style[key]
                    ap = defaults.get("age_appropriateness", "")
                    sp = defaults.get("global_style", "")
                    cr = defaults.get("character_ref", "")
                    cp = defaults.get("color_palette", "")
                    li = defaults.get("lighting", "")
                    fr = defaults.get("framing", "")
                    inline_requests = []
                    for row in pm:
                        page_text = (row.get("page_text") or row.get("text") or "").strip()
                        prompt = build_prompt("", page_text, ap, sp, cr, li, cp, fr)
                        parts = _build_batch_request_parts(prompt, refs if refs else None)
                        req = types.InlinedRequest(
                            contents=[{"parts": parts, "role": "user"}],
                            config=gen_config,
                        )
                        inline_requests.append(req)
                    batch_job = client.batches.create(
                        model="gemini-3-pro-image-preview",
                        src=inline_requests,
                        config={"display_name": f"story-{story_id}-{level}"},
                    )
                    batch_name = getattr(batch_job, "name", None) or str(batch_job)
                    if batch_name and insert_batch_job(sb, batch_name, story_id, level, len(pm)):
                        submitted.append(f"{level}: {len(pm)} pages")
                if submitted:
                    st.success("Submitted batch jobs: " + "; ".join(submitted) + ". Come back later and click Check batch status.")
                else:
                    st.warning("No levels had missing pages to submit.")
                st.rerun()
            except Exception as e:
                st.error(f"Batch submission failed: {e}")
        elif pages_missing:
            from google.genai import types
            refs = get_reference_images(ref_file, ref_image_url=selected_ref_url)
            gen_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            )
            inline_requests = []
            for row in pages_missing:
                page_text = (row.get("page_text") or row.get("text") or "").strip()
                prompt = build_prompt(
                    "",
                    page_text,
                    age_appropriateness or "",
                    style_prompt or "",
                    character_ref or "",
                    lighting or "",
                    color_palette or "",
                    framing or "",
                )
                parts = _build_batch_request_parts(prompt, refs if refs else None)
                req = types.InlinedRequest(
                    contents=[{"parts": parts, "role": "user"}],
                    config=gen_config,
                )
                inline_requests.append(req)
            try:
                batch_job = client.batches.create(
                    model="gemini-3-pro-image-preview",
                    src=inline_requests,
                    config={"display_name": f"story-{story_id}-{reading_level}"},
                )
                batch_name = getattr(batch_job, "name", None) or str(batch_job)
                if batch_name and insert_batch_job(sb, batch_name, story_id, reading_level, len(pages_missing)):
                    st.success(f"Batch submitted. Job ID: {batch_name}. Come back later and click Check batch status below.")
                else:
                    st.error("Failed to save batch job to database.")
                st.rerun()
            except Exception as e:
                st.error(f"Batch submission failed: {e}")

    # --- Review all pages: see images, approve (export to R2 + Supabase) or regenerate ---
    if all_levels_mode:
        review_level = st.selectbox(
            "Review level",
            options=READING_LEVELS,
            key="ip_review_level",
            format_func=lambda x: x.replace("_", " ").title(),
        )
    else:
        review_level = reading_level
    all_pages_for_review = fetch_book_pages(sb, story_id, review_level, language_code)
    pending = st.session_state.get("ip_pending_images", {})
    st.header("4. Review pages")
    st.caption("View generated images. Approve to export to R2 and save URL to Supabase. Regenerate to try again.")
    for row in all_pages_for_review:
        row_id = row.get("id")
        page_text = (row.get("page_text") or row.get("text") or "").strip()
        page_idx = row.get("page_index", row.get("page_number", 0))
        img_url = (row.get("image_url") or "").strip()
        has_published = bool(img_url)
        has_pending = row_id in pending
        page_label = f"✅ Page {page_idx}" if has_published else f"Page {page_idx}"
        with st.expander(page_label, expanded=not has_published):
            st.text(page_text or "(no text)")
            if has_pending:
                st.image(pending[row_id], caption=f"Page {page_idx} (pending approval)", use_container_width=False)
                st.caption("Approve to export to R2 and save to Supabase.")
            elif has_published:
                img_bytes = fetch_image_for_display(img_url)
                if img_bytes:
                    st.image(img_bytes, caption=f"Page {page_idx}", use_container_width=False)
            else:
                st.caption("No image yet.")
            edited_text = st.text_area("Page text", value=page_text, key=f"regen_text_{row_id}", height=60)
            correction = st.text_input("Correction (optional)", key=f"regen_corr_{row_id}", placeholder="e.g. Add a dragon in the background")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if has_pending and st.button("Approve (export to R2 + Supabase)", key=f"approve_btn_{row_id}", type="primary"):
                    opt = pending[row_id]
                    url = upload_to_storage(sb, story_id, review_level, int(page_idx), opt)
                    if url:
                        update_book_page(sb, row_id, image_url=url, page_text=(edited_text or page_text).strip())
                        del pending[row_id]
                        st.session_state["ip_pending_images"] = pending
                        st.success("Exported to R2 and saved to Supabase.")
                        st.rerun()
                    else:
                        st.error("Upload failed.")
            with btn_col2:
                if st.button("Regenerate" if (has_pending or has_published) else "Generate", key=f"regen_btn_{row_id}"):
                    extra = (correction or "").strip()
                    # Use style for review_level (in all_levels_mode we need that level's style)
                    if all_levels_mode:
                        gs = get_story_grade_style(sb, story_id, review_level)
                        defs = dict(GRADE_STYLE_DEFAULTS.get(review_level, GRADE_STYLE_DEFAULTS["grade_1"]))
                        if story:
                            for k in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                                if story.get(k) not in (None, ""):
                                    defs[k] = story[k]
                        if gs:
                            for k in ["age_appropriateness", "global_style", "character_ref", "color_palette", "lighting", "framing"]:
                                if gs.get(k) not in (None, ""):
                                    defs[k] = gs[k]
                        ap, sp, cr, li, cp, fr = defs.get("age_appropriateness", ""), defs.get("global_style", ""), defs.get("character_ref", ""), defs.get("lighting", ""), defs.get("color_palette", ""), defs.get("framing", "")
                    else:
                        ap, sp, cr, li, cp, fr = age_appropriateness or "", style_prompt or "", character_ref or "", lighting or "", color_palette or "", framing or ""
                    prompt = build_prompt(
                        extra,
                        (edited_text or page_text).strip(),
                        ap, sp, cr, li, cp, fr,
                    )
                    refs = get_reference_images(ref_file, ref_image_url=selected_ref_url)
                    with st.spinner("Generating..."):
                        img = generate_image_gemini(prompt, refs if refs else None)
                    if img:
                        opt = optimize_image_for_mobile(img)
                        pending[row_id] = opt
                        st.session_state["ip_pending_images"] = pending
                        st.success("Generated. Approve to export to R2 + Supabase.")
                        st.rerun()
                    else:
                        st.error("Generation failed.")


if __name__ == "__main__":
    main()
