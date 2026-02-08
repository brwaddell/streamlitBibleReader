"""
Storybook Image Processor - Bulk image generation with approval workflow.
Images via Nano Banana Pro (Gemini). Extra-details hints via OpenAI GPT.
Review/approve/regenerate, then export to Supabase Storage + book_pages table.
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image

import streamlit as st
from dotenv import load_dotenv

from auth import get_secret, is_authenticated, logout, run_login_page
from lib import fetch_stories, get_supabase, run_book_pages_view

load_dotenv()

READING_LEVELS = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
STORAGE_BUCKET = "storybook-images"

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


def new_page(text: str = "", extra_details: str = "") -> Dict:
    """Create a new page dict."""
    return {
        "text": text,
        "image": None,
        "prompt": "",
        "status": "pending",
        "correction": "",
        "extra_details": extra_details,
    }


def reset_to_fresh():
    """Clear workflow state so the app looks freshly opened."""
    keep = {"supabase", "openai_client", "gemini_client", "auth_client", "auth_tokens"}
    for key in list(st.session_state.keys()):
        if key not in keep:
            del st.session_state[key]
    init_session_state()
    st.rerun()


def init_session_state():
    """Initialize session state keys."""
    defaults = {
        "pages": [],
        "story_id": None,
        "reading_level": None,
        "last_reading_level": None,
        "stories": [],
        "supabase": None,
        "openai_client": None,
        "gemini_client": None,
        "ref_selected_page": "None",
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
    "Quality: Standard. Aspect Ratio: 1024x1024. "
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
    """Formula: [Extra Details]. Scene context: [Story Text]. Style: [Age], [Global Style], [Character Ref], [Lighting], [Palette]."""
    extra = (extra_details or "").strip()
    story = (story_text or "")[:500].strip()
    style_str = build_style_string(
        age_appropriateness or "",
        global_style or "",
        character_ref or "",
        lighting or "",
        palette or "",
        framing or "",
    )
    if extra:
        scene = f"{extra}. Scene context: {story}. Style: {style_str}. --no text, no words."
    else:
        scene = f"Scene context: {story}. Style: {style_str}. --no text, no words."
    return f"{SYSTEM_INSTRUCTIONS}{scene}"


def generate_extra_details(client, story_text: str, character_ref: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Call GPT to generate a 1-sentence visual description for the illustrator."""
    if not story_text or not story_text.strip():
        return None
    prompt = (
        f'You are a storyboard artist for a children\'s Bible app. '
        f'Given this story text: "{story_text.strip()}", '
        f'write a 1-sentence visual description for an illustrator. '
        f'Focus on the character\'s action and the environment. '
        f'Keep it consistent with our character: {character_ref or "the main character"}. '
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


def get_reference_images(ref_file=None, selected_page_index: Optional[int] = None, pages=None) -> List[bytes]:
    """Collect reference images: uploaded file + selected approved page image."""
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


def generate_image_gemini(prompt: str, reference_images: Optional[List[bytes]] = None) -> Optional[bytes]:
    """Generate image via Nano Banana Pro (Gemini 3 Pro Image) with optional reference images for character consistency."""
    client = get_gemini()
    if not client:
        return None
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
        st.error(f"Nano Banana Pro generation failed: {e}")
        return None


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
    """Upload image to Supabase Storage. Returns public URL or None."""
    path = f"{story_id}/{reading_level}/page_{page_index}.webp"
    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path, image_bytes, file_options={"content-type": "image/webp", "upsert": "true"}
        )
        public = supabase.storage.from_(STORAGE_BUCKET).get_public_url(path)
        return public
    except Exception as e:
        st.error(f"Upload failed for {path}: {e}")
        return None


def insert_book_page(supabase, story_id: int, reading_level: str, page_index: int, page_text: str, image_url: str):
    """Insert a row into book_pages."""
    row = {
        "story_id": story_id,
        "reading_level": reading_level,
        "page_index": page_index,
        "page_text": page_text,
        "image_url": image_url,
    }
    supabase.table("book_pages").insert(row).execute()


def main():
    st.set_page_config(page_title="Storybook Image Processor", layout="wide", initial_sidebar_state="expanded")

    if not is_authenticated():
        run_login_page()
        st.stop()

    init_session_state()

    st.sidebar.title("Storybook")
    page = st.sidebar.radio("Go to", ["Image Processor", "Book Pages"])
    st.sidebar.divider()
    if st.sidebar.button("Sign out"):
        logout()
        st.rerun()

    if page == "Book Pages":
        run_book_pages_view()
        return

    st.title("Storybook Image Processor")
    st.caption("Paste story text → split pages → fix punctuation → set style → generate → approve → export to Supabase")
    if st.button("🔄 Start Fresh", help="Clear all pages and images to start a new level/story"):
        reset_to_fresh()

    # --- Step 1: Story & Reading Level ---
    st.header("1. Select Story & Reading Level")
    col1, col2 = st.columns(2)
    with col1:
        stories = fetch_stories()
        if not stories:
            st.warning("No stories found. Create stories in Supabase first.")
        else:
            options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
            selected = st.selectbox("Story", options=list(options.keys()), key="story_select")
            st.session_state.story_id = options.get(selected) if selected else None
    with col2:
        st.session_state.reading_level = st.selectbox("Reading Level", READING_LEVELS, key="reading_level_select")

    rl = st.session_state.reading_level
    if rl and rl != st.session_state.get("last_reading_level"):
        defaults = GRADE_STYLE_DEFAULTS.get(rl, GRADE_STYLE_DEFAULTS["grade_1"])
        st.session_state["age_appropriateness"] = defaults["age_appropriateness"]
        st.session_state["style_prompt"] = defaults["global_style"]
        st.session_state["character_ref"] = defaults["character_ref"]
        st.session_state["color_palette"] = defaults["color_palette"]
        st.session_state["lighting"] = defaults["lighting"]
        st.session_state["framing"] = defaults["framing"]
        st.session_state["last_reading_level"] = rl

    if not st.session_state.story_id:
        st.stop()

    # --- Step 2: Create & Edit Pages ---
    st.header("2. Create & Edit Pages")
    st.caption("Add pages one at a time. Fix punctuation for sentence highlighting. Add extra details to help the image model.")

    for p in st.session_state.pages:
        if "extra_details" not in p:
            p["extra_details"] = p.get("per_page_override", "")

    if st.button("➕ Add Page", key="add_page"):
        st.session_state.pages = st.session_state.pages + [new_page()]
        st.rerun()

    if st.session_state.pages:
        for i, page in enumerate(st.session_state.pages):
            with st.container():
                cols = st.columns([20, 1])
                with cols[0]:
                    page["text"] = st.text_area(
                        f"Page {i} — Story text",
                        value=page["text"],
                        height=60,
                        key=f"page_text_{i}",
                        placeholder="The actual story text for this page. Fix punctuation for sentence highlighting.",
                    )
                    extra_col1, extra_col2 = st.columns([4, 1])
                    with extra_col1:
                        if "pending_extra_details" in st.session_state and i in st.session_state["pending_extra_details"]:
                            result = st.session_state["pending_extra_details"].pop(i)
                            page["extra_details"] = result
                            st.session_state[f"extra_details_{i}"] = result
                            if not st.session_state["pending_extra_details"]:
                                del st.session_state["pending_extra_details"]
                        page["extra_details"] = st.text_input(
                            f"Page {i} — Extra details for image (not in text)",
                            value=page.get("extra_details", ""),
                            key=f"extra_details_{i}",
                            placeholder="e.g. dragon in background, child looking surprised, warm sunset",
                        )
                    with extra_col2:
                        if st.button("✨ Generate", key=f"gen_extra_{i}", help="Auto-fill from story text using AI"):
                            if not page.get("extra_details", "").strip():
                                client = get_openai()
                                if client and page.get("text", "").strip():
                                    try:
                                        char_ref = st.session_state.get("character_ref", "")
                                        with st.spinner("Generating..."):
                                            result = generate_extra_details(client, page.get("text", ""), char_ref)
                                            if result:
                                                page["extra_details"] = result
                                                pending = st.session_state.get("pending_extra_details", {})
                                                pending[i] = result
                                                st.session_state["pending_extra_details"] = pending
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to generate: {e}")
                                elif not page.get("text", "").strip():
                                    st.warning("Add story text first.")
                with cols[1]:
                    if st.button("🗑", key=f"remove_{i}", help="Remove page"):
                        st.session_state.pages.pop(i)
                        st.rerun()
                st.divider()
    else:
        st.info("Click Add Page to get started.")

    # --- Image Controls ---
    with st.expander("4. Image Style Controls"):
        age_appropriateness = st.text_input(
            "Age appropriateness",
            placeholder="e.g. Pre-school (3–5), Early reader (5–6 yrs)",
            key="age_appropriateness",
        )
        style_prompt = st.text_area(
            "Global style prompt",
            placeholder="e.g. watercolor illustration, children's book style, whimsical",
            key="style_prompt",
        )
        character_ref = st.text_input(
            "Character / appearance reference (optional)",
            placeholder="e.g. young girl with red hair, wears blue dress",
            key="character_ref",
        )
        color_palette = st.text_input(
            "Color palette (optional)",
            placeholder="e.g. warm earth tones, muted blues",
            key="color_palette",
        )
        lighting = st.text_input(
            "Lighting Mood",
            placeholder="e.g. Soft overcast light",
            key="lighting",
        )
        framing = st.text_input(
            "Framing",
            placeholder="e.g. Medium shot, warm composition",
            key="framing",
        )
        ref_file = st.file_uploader(
            "Reference image for character consistency (optional)",
            type=["png", "jpg", "jpeg"],
            key="ref_image",
            help="Upload a sample image, or select an approved page below.",
        )
        approved_with_images = [
            (i, p) for i, p in enumerate(st.session_state.pages)
            if p.get("image") and p.get("status") == "approved"
        ]
        ref_page_options = ["None"] + [f"Page {i}" for i, _ in approved_with_images]
        ref_page_labels = {f"Page {i}": i for i, _ in approved_with_images}
        # Persist selection: only reset if saved value is no longer valid
        saved = st.session_state.ref_selected_page
        if saved not in ref_page_options:
            st.session_state.ref_selected_page = "None"
        default_idx = ref_page_options.index(st.session_state.ref_selected_page)
        ref_selected = st.selectbox(
            "Use approved page as reference",
            options=ref_page_options,
            index=default_idx,
            key="ref_page_select",
            help="Select an approved page image to use for character consistency.",
        )
        st.session_state.ref_selected_page = ref_selected
        selected_ref_index = ref_page_labels.get(ref_selected) if ref_selected != "None" else None
        style_preview = build_style_string(
            age_appropriateness, style_prompt, character_ref, lighting, color_palette, framing
        )
        if style_preview:
            st.caption(f"**Style (repeated every gen):** {style_preview[:150]}{'...' if len(style_preview) > 150 else ''}")

    # --- Generate & Review ---
    if st.session_state.pages:
        st.header("5. Generate & Review")
        for i, page in enumerate(st.session_state.pages):
            with st.container():
                preview = (page["text"][:50] + "…") if len(page.get("text", "")) > 50 else (page.get("text", "") or "(empty)")
                st.subheader(f"Page {i}: {preview}")
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    if page["image"]:
                        st.image(page["image"], use_container_width=True)
                    else:
                        st.info("No image yet")
                with col_b:
                    if page["status"] == "approved":
                        st.success("Approved")
                    elif page["image"]:
                        sub1, sub2 = st.columns(2)
                        with sub1:
                            if st.button("Approve", key=f"approve_{i}"):
                                page["status"] = "approved"
                                st.rerun()
                        with sub2:
                            correction = st.text_input("Regenerate with correction", key=f"correction_{i}")
                            if st.button("Regenerate", key=f"regen_{i}") and correction:
                                page["correction"] = correction
                                try:
                                    extra_with_correction = f"{page.get('extra_details', '')}. Correction: {correction}".strip()
                                    prompt = build_prompt(
                                        extra_with_correction,
                                        page.get("text", ""),
                                        age_appropriateness or "",
                                        style_prompt or "",
                                        character_ref or "",
                                        lighting or "",
                                        color_palette or "",
                                        framing or "",
                                    )
                                    with st.spinner("Regenerating..."):
                                        refs = get_reference_images(ref_file, selected_ref_index, st.session_state.pages)
                                        img = generate_image_gemini(prompt, refs if refs else None)
                                    if img:
                                        page["image"] = optimize_image_for_mobile(img)
                                        page["prompt"] = prompt
                                        page["status"] = "generated"
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Regeneration failed: {e}")
                    else:
                        if st.button("Generate", key=f"gen_{i}"):
                            page_text = page.get("text", "")
                            if not (page_text or "").strip():
                                st.warning("Add story text for this page first.")
                            else:
                                try:
                                    prompt = build_prompt(
                                        page.get("extra_details", ""),
                                        page_text,
                                        age_appropriateness or "",
                                        style_prompt or "",
                                        character_ref or "",
                                        lighting or "",
                                        color_palette or "",
                                        framing or "",
                                    )
                                    with st.spinner(f"Generating page {i}..."):
                                        refs = get_reference_images(ref_file, selected_ref_index, st.session_state.pages)
                                        img = generate_image_gemini(prompt, refs if refs else None)
                                    if img:
                                        page["image"] = optimize_image_for_mobile(img)
                                        page["prompt"] = prompt
                                        page["status"] = "generated"
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")
                st.divider()

        # --- Bulk Generate ---
        pending = [i for i, p in enumerate(st.session_state.pages) if not p["image"]]
        if pending:
            if st.button("Generate All Pending", key="gen_all"):
                prog = st.progress(0)
                for idx, i in enumerate(pending):
                    page = st.session_state.pages[i]
                    try:
                        prompt = build_prompt(
                            page.get("extra_details", ""),
                            page.get("text", ""),
                            age_appropriateness or "",
                            style_prompt or "",
                            character_ref or "",
                            lighting or "",
                            color_palette or "",
                            framing or "",
                        )
                        refs = get_reference_images(ref_file, selected_ref_index, st.session_state.pages)
                        img = generate_image_gemini(prompt, refs if refs else None)
                        if img:
                            page["image"] = optimize_image_for_mobile(img)
                            page["prompt"] = prompt
                            page["status"] = "generated"
                    except Exception as e:
                        st.error(f"Page {i} failed: {e}")
                    prog.progress((idx + 1) / len(pending))
                st.rerun()

        # --- Export ---
        st.header("6. Export to Supabase")
        approved = [p for p in st.session_state.pages if p["status"] == "approved"]
        if approved:
            st.info(f"{len(approved)} pages approved. Ready to export.")
            if st.button("Export to Supabase"):
                sb = get_supabase()
                sid = st.session_state.story_id
                rl = st.session_state.reading_level
                if sb and sid and rl:
                    prog = st.progress(0)
                    ok = 0
                    for idx, page in enumerate(st.session_state.pages):
                        if page["status"] != "approved" or not page["image"]:
                            continue
                        url = upload_to_storage(sb, sid, rl, idx, page["image"])
                        if url:
                            insert_book_page(sb, sid, rl, idx, page["text"], url)
                            ok += 1
                        prog.progress((idx + 1) / len(st.session_state.pages))
                    st.success(f"Exported {ok} pages to Supabase.")
        else:
            st.warning("Approve at least one page to export.")


if __name__ == "__main__":
    main()
