"""
Storybook Image Processor - Bulk image generation with approval workflow.
Images via Nano Banana Pro (Gemini). Extra-details hints via OpenAI GPT.
Review/approve/regenerate, then export to Supabase Storage + story_content_flat table.
"""

import base64
import os
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image

import streamlit as st
from dotenv import load_dotenv

from auth import get_secret, is_authenticated, logout, run_login_page
from lib import (
    _build_batch_request_parts,
    _extract_image_from_batch_response,
    build_prompt,
    fetch_batch_jobs_for_version,
    fetch_book_pages,
    fetch_pages_missing_images,
    fetch_stories,
    generate_image_gemini,
    get_gemini,
    get_reference_images,
    get_supabase,
    get_story_grade_style,
    insert_batch_job,
    optimize_image_for_mobile,
    run_audio_generator_view,
    run_book_pages_view,
    update_batch_job_status,
    update_book_page,
    upload_image_to_storage,
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


# Cache TTL for Image Processor Supabase fetches (seconds)
_IP_CACHE_TTL = 60


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_fetch_stories(_cache_version: int):
    sb = get_supabase()
    return fetch_stories() if sb else []


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_fetch_pages_missing_images(
    story_id: int, language_code: str, reading_level: str, _cache_version: int, _cache_version_missing: int = 0
):
    sb = get_supabase()
    return fetch_pages_missing_images(sb, story_id, language_code, reading_level) if sb else []


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_get_story_grade_style(story_id: int, reading_level: str, _cache_version: int):
    sb = get_supabase()
    return get_story_grade_style(sb, story_id, reading_level) if sb else None


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_fetch_book_pages(story_id: int, reading_level: str, language_code: str, _cache_version: int):
    sb = get_supabase()
    return fetch_book_pages(sb, story_id, reading_level, language_code) if sb else []


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_fetch_batch_jobs_for_version(story_id: int, reading_level: str, _cache_version: int):
    sb = get_supabase()
    return fetch_batch_jobs_for_version(sb, story_id, reading_level) if sb else []


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
        "ip_cache_version": 0,  # bumped when full refetch needed
        "ip_pages_missing_version": 0,  # bumped on clear/approve so pages_missing count refreshes
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

    _v = st.session_state.get("ip_cache_version", 0)
    stories = _cached_fetch_stories(_v)
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    # --- Filters ---
    st.header("1. Select story & version")
    language_code = "en"
    story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
    story_label = st.selectbox("Story", options=list(story_options.keys()), key="ip_story")
    story_id = story_options.get(story_label) if story_label else None

    reading_level = st.selectbox("Reading level", options=READING_LEVELS, key="ip_reading_level", format_func=lambda x: x.replace("_", " ").title())

    if not story_id:
        st.info("Select a story to continue.")
        return

    # Pop just_approved / just_cleared early so we treat them correctly (avoids stale refetch)
    just_approved = st.session_state.pop("ip_just_approved", None) or {}
    just_cleared = set(st.session_state.pop("ip_just_cleared", None) or [])
    _v_missing = st.session_state.get("ip_pages_missing_version", 0)
    pages_missing = _cached_fetch_pages_missing_images(story_id, language_code, reading_level, _v, _v_missing)
    pages_missing_display = [r for r in pages_missing if r.get("id") not in just_approved]
    if pages_missing_display:
        st.success(f"**{len(pages_missing_display)}** pages missing images.")
    else:
        st.info("No pages missing images for this story + reading level. You can still set or save style below, or add page text in Story Text and return here.")

    # Load style defaults for current (story_id, reading_level)
    story = next((s for s in stories if s.get("id") == story_id), None)
    if story_id and reading_level:
        defaults = dict(GRADE_STYLE_DEFAULTS.get(reading_level, GRADE_STYLE_DEFAULTS["grade_1"]))
        if story:
            for key in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                if story.get(key) not in (None, ""):
                    defaults[key] = story[key]
        grade_style = _cached_get_story_grade_style(story_id, reading_level, _v)
        if grade_style:
            for key in ["age_appropriateness", "global_style", "character_ref", "color_palette", "lighting", "framing"]:
                if grade_style.get(key) not in (None, ""):
                    defaults[key] = grade_style[key]
        # When user changes story or reading level, reload style controls from that level's defaults/saved style.
        loaded_for = st.session_state.get("ip_style_loaded_for")
        current_for = f"{story_id}_{reading_level}"
        if loaded_for != current_for:
            st.session_state["ip_style_loaded_for"] = current_for
            st.session_state["age_appropriateness"] = defaults.get("age_appropriateness", "")
            st.session_state["style_prompt"] = defaults.get("global_style", "")
            st.session_state["character_ref"] = defaults.get("character_ref", "")
            st.session_state["color_palette"] = defaults.get("color_palette", "")
            st.session_state["lighting"] = defaults.get("lighting", "")
            st.session_state["framing"] = defaults.get("framing", "")
        else:
            # Same story+level: fill any missing keys so fields aren't empty on first load.
            if "age_appropriateness" not in st.session_state or st.session_state.get("age_appropriateness") == "":
                st.session_state["age_appropriateness"] = defaults.get("age_appropriateness", "")
            if "style_prompt" not in st.session_state or st.session_state.get("style_prompt") == "":
                st.session_state["style_prompt"] = defaults.get("global_style", "")
            if "character_ref" not in st.session_state or st.session_state.get("character_ref") == "":
                st.session_state["character_ref"] = defaults.get("character_ref", "")
            if "color_palette" not in st.session_state or st.session_state.get("color_palette") == "":
                st.session_state["color_palette"] = defaults.get("color_palette", "")
            if "lighting" not in st.session_state or st.session_state.get("lighting") == "":
                st.session_state["lighting"] = defaults.get("lighting", "")
            if "framing" not in st.session_state or st.session_state.get("framing") == "":
                st.session_state["framing"] = defaults.get("framing", "")

    # --- Image style controls ---
    st.header("2. Image style controls")
    with st.expander("Style controls", expanded=True):
        age_appropriateness = st.text_input("Age appropriateness", key="age_appropriateness")
        style_prompt = st.text_area("Global style prompt", key="style_prompt")
        character_ref = st.text_input("Character reference", key="character_ref")
        color_palette = st.text_input("Color palette", key="color_palette")
        lighting = st.text_input("Lighting", key="lighting")
        framing = st.text_input("Framing", key="framing")
        ref_file = st.file_uploader(
            "Reference image for character consistency (optional)",
            type=["png", "jpg", "jpeg"],
            key="ref_image",
        )
        all_pages = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
        published_with_images = [(r.get("page_index", i), r.get("image_url")) for i, r in enumerate(all_pages) if r.get("image_url") and r.get("id") not in just_cleared]
        # Merge just-approved pages so they appear in the ref dropdown immediately (avoids cache/read-after-write)
        id_to_page_index = {r.get("id"): r.get("page_index", i) for i, r in enumerate(all_pages)}
        for row_id, url in (just_approved or {}).items():
            if (url or "").strip() and row_id in id_to_page_index:
                pi = id_to_page_index[row_id]
                if not any(p[0] == pi for p in published_with_images):
                    published_with_images.append((pi, url.strip()))
        published_with_images.sort(key=lambda x: x[0])
        ref_page_options = ["None"] + [f"Page {pi}" for pi, _ in published_with_images]
        ref_page_labels = {f"Page {pi}": url for pi, url in published_with_images}
        ref_page_key = f"ref_page_select_{story_id}_{reading_level}"
        grade_style_for_ref = _cached_get_story_grade_style(story_id, reading_level, _v)
        # Initialize from saved style when this story+level key not set (per-context persistence)
        if ref_page_key not in st.session_state:
            default_ref = "None"
            if grade_style_for_ref is not None and grade_style_for_ref.get("reference_page_index") is not None:
                saved_ref_label = f"Page {grade_style_for_ref['reference_page_index']}"
                if saved_ref_label in ref_page_options:
                    default_ref = saved_ref_label
            st.session_state[ref_page_key] = default_ref
        # If current value is no longer in options (e.g. that page was removed), reset to None
        elif st.session_state.get(ref_page_key) not in ref_page_options:
            st.session_state[ref_page_key] = "None"
        ref_selected = st.selectbox(
            "Use published page as reference",
            options=ref_page_options,
            key=ref_page_key,
        )
        selected_ref_url = ref_page_labels.get(ref_selected) if ref_selected != "None" else None
        if story_id and reading_level and st.button("Save style for this story & grade", key="save_style"):
            ref_page_index = None
            if ref_selected and ref_selected != "None" and ref_selected.startswith("Page "):
                try:
                    ref_page_index = int(ref_selected.replace("Page ", "").strip())
                except ValueError:
                    pass
            if upsert_story_grade_style(sb, story_id, reading_level, {
                "age_appropriateness": age_appropriateness or "",
                "global_style": style_prompt or "",
                "character_ref": character_ref or "",
                "color_palette": color_palette or "",
                "lighting": lighting or "",
                "framing": framing or "",
                "reference_page_index": ref_page_index,
            }):
                st.success(f"Style saved for {reading_level}.")
                st.rerun()

    # --- Batch status (check pending/running jobs) ---
    batch_jobs = _cached_fetch_batch_jobs_for_version(story_id, reading_level, _v)
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
                    pages_for_job = pages_missing_display
                    if _check_batch_status(sb, job, pages_for_job, get_gemini):
                        st.rerun()

    gen_first_col, gen_all_col = st.columns(2)
    with gen_first_col:
        gen_first = st.button("Generate first page only", key="ip_gen_first")
    with gen_all_col:
        gen_all = st.button("Submit batch (50% cheaper, results in ~24h)", type="primary", key="ip_batch_gen")

    st.caption("Images are generated asynchronously. Return to this page later and click Check batch status.")

    if gen_first and pages_missing_display:
        client = get_gemini()
        if not client:
            st.error("Set GEMINI_API_KEY in .env.")
        else:
            first_row = pages_missing_display[0]
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

    if gen_all and pages_missing_display:
        client = get_gemini()
        if not client:
            st.error("Set GEMINI_API_KEY in .env.")
        else:
            from google.genai import types
            refs = get_reference_images(ref_file, ref_image_url=selected_ref_url)
            gen_config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            )
            inline_requests = []
            for row in pages_missing_display:
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
                if batch_name and insert_batch_job(sb, batch_name, story_id, reading_level, len(pages_missing_display)):
                    st.success(f"Batch submitted. Job ID: {batch_name}. Come back later and click Check batch status below.")
                else:
                    st.error("Failed to save batch job to database.")
                st.rerun()
            except Exception as e:
                st.error(f"Batch submission failed: {e}")

    # --- Review all pages: see images, approve (export to R2 + Supabase) or regenerate ---
    all_pages_for_review = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    # Merge just-approved URLs so UI shows them immediately (no refetch delay)
    if just_approved:
        for row in all_pages_for_review:
            rid = row.get("id")
            if rid in just_approved:
                row["image_url"] = just_approved[rid]
    # Apply just-cleared so UI shows removed image immediately (no refetch delay)
    if just_cleared:
        for row in all_pages_for_review:
            if row.get("id") in just_cleared:
                row["image_url"] = ""
    pending = st.session_state.get("ip_pending_images", {})
    st.header("4. Review pages")
    st.caption("View generated images. Approve to export to R2 and save URL to Supabase. Regenerate to try again. Clear image to remove and include the page in the next batch.")
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
                st.image(img_url, caption=f"Page {page_idx}", use_container_width=False)
            else:
                st.caption("No image yet.")
            edited_text = st.text_area(
                "Page text (for API only)",
                value=page_text,
                key=f"regen_text_{row_id}",
                height=60,
                help="Edit here to fix prompt/API errors. Only the text sent to the image API changes; your stored page text in the database is not updated.",
            )
            correction = st.text_input("Correction (optional)", key=f"regen_corr_{row_id}", placeholder="e.g. Add a dragon in the background")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if has_pending and st.button("Approve", key=f"approve_btn_{row_id}", type="primary"):
                    opt = pending[row_id]
                    url = upload_image_to_storage(sb, story_id, reading_level, int(page_idx), opt)
                    if url and (url or "").strip():
                        if update_book_page(sb, row_id, image_url=url):
                            del pending[row_id]
                            st.session_state["ip_pending_images"] = pending
                            # Bump only pages_missing so count updates; keep book_pages cached so UI shows new URL instantly via just_approved
                            st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                            just_approved = st.session_state.get("ip_just_approved") or {}
                            just_approved[row_id] = url
                            st.session_state["ip_just_approved"] = just_approved
                            st.success("Exported to R2 and saved to Supabase.")
                            st.rerun()
                        else:
                            st.error("Saved to R2 but failed to save URL to Supabase. Try again.")
                    else:
                        st.error("Upload failed.")
            with btn_col2:
                if st.button("Regenerate" if (has_pending or has_published) else "Generate", key=f"regen_btn_{row_id}"):
                    extra = (correction or "").strip()
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
            with btn_col3:
                if (has_pending or has_published) and st.button("Clear image", key=f"clear_btn_{row_id}", help="Remove image so this page is included in the next batch."):
                    if update_book_page(sb, row_id, image_url=""):
                        if row_id in pending:
                            del pending[row_id]
                            st.session_state["ip_pending_images"] = pending
                        # Bump only pages_missing so count updates; keep book_pages cached so UI shows cleared state instantly via just_cleared
                        st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                        just_cleared_ids = list(st.session_state.get("ip_just_cleared") or [])
                        just_cleared_ids.append(row_id)
                        st.session_state["ip_just_cleared"] = just_cleared_ids
                        st.success("Image cleared. Page will be included in the next batch.")
                        st.rerun()
                    else:
                        st.error("Failed to clear image.")


if __name__ == "__main__":
    main()
