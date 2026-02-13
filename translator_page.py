"""
Translator Page: Localize existing English stories via OpenAI.
Select source story (English) → choose target language & reading level →
AI translates each page → side-by-side editor → save to Supabase (story_content_flat).
"""

from typing import List, Optional

import streamlit as st

from auth import get_secret
from lib import (
    READING_LEVELS,
    _get_page_text,
    fetch_pages_missing_translation,
    fetch_stories,
    get_supabase,
    insert_book_page,
)

# Target languages: (display name, language_code for DB + prompt)
TARGET_LANGUAGES = [
    ("Spanish", "es", "Spanish"),
    ("French", "fr", "French"),
    ("German", "de", "German"),
    ("Italian", "it", "Italian"),
    ("Portuguese", "pt", "Portuguese"),
    ("Dutch", "nl", "Dutch"),
    ("Japanese", "ja", "Japanese"),
    ("Korean", "ko", "Korean"),
]

# Reading level display for prompts (friendly names)
READING_LEVEL_LABELS = {
    "grade_1": "Pre-K / Grade 1 (ages 3–6)",
    "grade_2": "Grade 2 (ages 5–7)",
    "grade_3": "Grade 3 (ages 7–8)",
    "grade_4": "Grade 4 (ages 9–10)",
    "grade_5": "Grade 5 (ages 11+)",
}


def get_openai():
    """Get OpenAI client (cached in session state)."""
    if getattr(st.session_state, "translator_openai_client", None) is None:
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI
        st.session_state.translator_openai_client = OpenAI(api_key=api_key)
    return st.session_state.translator_openai_client


def translate_page_text(
    client,
    english_text: str,
    target_lang_name: str,
    reading_level_label: str,
    model: str = "gpt-4o",
) -> Optional[str]:
    """Call OpenAI to translate one page. Returns translated text or None."""
    if not english_text or not english_text.strip():
        return ""
    system_prompt = (
        f"You are a professional children's book translator. "
        f"Translate the following text into {target_lang_name} for a {reading_level_label} audience. "
        f"Maintain the whimsical tone, rhythm, and simplicity of the original."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": english_text.strip()},
            ],
            max_tokens=500,
        )
        text = resp.choices[0].message.content
        return (text or "").strip()
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return None


def run_translator_view():
    """Render the Translator page: story + grade + target language; translate only missing rows (like Audio Generator)."""
    st.title("Story Translator")
    st.caption("Source is always English. Select story, grade level, and target language; translate only pages that don't have a translation yet.")

    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories()
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    # --- Story ID + Mode + Grade level (if single) + Target language ---
    st.header("1. Select story & target")
    col1, col2 = st.columns(2)
    with col1:
        story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
        story_label = st.selectbox(
            "Story",
            options=list(story_options.keys()),
            key="tr_story",
        )
        story_id = story_options.get(story_label) if story_label else None
    with col2:
        target_options = [(name, code, prompt_name) for name, code, prompt_name in TARGET_LANGUAGES]
        target_lang_display = st.selectbox(
            "Target language",
            options=[t[0] for t in target_options],
            key="tr_target_lang",
        )
        target_entry = next(t for t in target_options if t[0] == target_lang_display)
        target_lang_code = target_entry[1]
        target_lang_prompt_name = target_entry[2]

    tr_all_levels = st.radio(
        "Mode",
        options=["Single level", "All levels"],
        key="tr_mode",
        horizontal=True,
        help="Single level: one grade at a time. All levels: translate missing pages for all 5 grades at once.",
    )
    all_levels_mode = tr_all_levels == "All levels"

    if not all_levels_mode:
        reading_level = st.selectbox(
            "Grade level",
            options=READING_LEVELS,
            format_func=lambda x: x.replace("_", " ").title(),
            key="tr_grade",
        )
    else:
        reading_level = None  # not used when all levels

    if not story_id:
        st.info("Select a story to continue.")
        return

    if all_levels_mode:
        levels_missing = {
            level: fetch_pages_missing_translation(sb, story_id, level, target_lang_code)
            for level in READING_LEVELS
        }
        total_missing = sum(len(p) for p in levels_missing.values())
        if total_missing == 0:
            st.info(
                f"No pages missing translation for **{target_lang_display}** across any grade (Story {story_id}). "
                "All English pages already have a translation, or there are no English pages."
            )
        else:
            parts = [f"**{lev}**: {len(levels_missing[lev])} missing" for lev in READING_LEVELS]
            st.success(" · ".join(parts) + f" → **Total: {total_missing} pages**")
            with st.expander("Pages to translate (by level)", expanded=False):
                for lev in READING_LEVELS:
                    pm = levels_missing.get(lev, [])
                    if pm:
                        st.caption(f"{lev}: {len(pm)} pages")
        pages_missing = []  # not used for single list when all-levels; we use levels_missing
    else:
        pages_missing = fetch_pages_missing_translation(sb, story_id, reading_level, target_lang_code)
        if pages_missing:
            st.success(
                f"**{len(pages_missing)}** pages missing translation for **{target_lang_display}** "
                f"(Story {story_id}, {reading_level}). These will be translated from English."
            )
            with st.expander("Pages to translate", expanded=True):
                for p in pages_missing:
                    txt = _get_page_text(p)
                    pn = p.get("page_index", p.get("page_number", "?"))
                    st.caption(f"Page {pn} — {txt[:70]}{'...' if len(txt) > 70 else ''}")
        else:
            st.info(
                f"No pages missing translation for **{target_lang_display}** (Story {story_id}, {reading_level}). "
                "All English pages already have a translation, or there are no English pages. Add English pages in Book Pages first."
            )

    # --- Translate (only missing pages) ---
    st.header("2. Translate")
    openai_client = get_openai()
    if not openai_client:
        st.error("Set OPENAI_API_KEY in .env to use the translator.")
        return

    can_translate = (all_levels_mode and total_missing > 0) if all_levels_mode else bool(pages_missing)
    if can_translate and st.button("Translate missing pages with GPT-4o", type="primary", key="tr_do_translate"):
        progress = st.progress(0)
        if all_levels_mode:
            levels_results: dict = {}
            total_pages = sum(len(levels_missing[lev]) for lev in READING_LEVELS)
            done = 0
            for level in READING_LEVELS:
                pm = levels_missing.get(level, [])
                if not pm:
                    continue
                level_label = READING_LEVEL_LABELS.get(level, level.replace("_", " ").title())
                results: List[dict] = []
                for i, page in enumerate(pm):
                    english_text = _get_page_text(page)
                    translated = translate_page_text(
                        openai_client,
                        english_text,
                        target_lang_prompt_name,
                        level_label,
                    )
                    image_url = page.get("image_url") or ""
                    page_index = page.get("page_index", page.get("page_number", i))
                    results.append({
                        "english_text": english_text,
                        "translated_text": translated if translated is not None else "",
                        "image_url": image_url,
                        "page_index": page_index,
                    })
                    done += 1
                    progress.progress(done / total_pages)
                levels_results[level] = results
            progress.progress(1.0)
            st.session_state["tr_result"] = {
                "story_id": story_id,
                "target_lang_code": target_lang_code,
                "levels": levels_results,
            }
        else:
            reading_level_label = READING_LEVEL_LABELS.get(
                reading_level,
                reading_level.replace("_", " ").title(),
            )
            results = []
            n = len(pages_missing)
            for i, page in enumerate(pages_missing):
                english_text = _get_page_text(page)
                translated = translate_page_text(
                    openai_client,
                    english_text,
                    target_lang_prompt_name,
                    reading_level_label,
                )
                image_url = page.get("image_url") or ""
                page_index = page.get("page_index", page.get("page_number", i))
                results.append({
                    "english_text": english_text,
                    "translated_text": translated if translated is not None else "",
                    "image_url": image_url,
                    "page_index": page_index,
                })
                progress.progress((i + 1) / n)
            progress.progress(1.0)
            st.session_state["tr_result"] = {
                "story_id": story_id,
                "target_lang_code": target_lang_code,
                "reading_level": reading_level,
                "pages": results,
            }
        st.success("Translation complete. Review and edit below, then save to Supabase.")
        st.rerun()

    # --- Side-by-side editor (after translation) ---
    result = st.session_state.get("tr_result")
    has_levels = result and "levels" in result and result.get("levels")
    has_pages = result and result.get("pages")
    if has_levels:
        if result.get("story_id") != story_id or result.get("target_lang_code") != target_lang_code:
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            result = None
            has_levels = False
    elif has_pages:
        if (
            result.get("story_id") != story_id
            or result.get("target_lang_code") != target_lang_code
            or result.get("reading_level") != reading_level
        ):
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            result = None
            has_pages = False

    if has_levels:
        st.header("3. Review & edit")
        st.caption("Edit the translated text on the right, then **Save to Supabase**. Select a level to review.")
        review_level = st.selectbox(
            "Review level",
            options=READING_LEVELS,
            key="tr_review_level",
            format_func=lambda x: x.replace("_", " ").title(),
        )
        pages_for_review = result["levels"].get(review_level, [])
        if "tr_edited" not in st.session_state:
            st.session_state.tr_edited = {}
        edited = st.session_state.tr_edited
        for p in pages_for_review:
            page_index = p["page_index"]
            key = (review_level, page_index)
            if key not in edited:
                edited[key] = p["translated_text"]
            st.subheader(f"Page {page_index} ({review_level})")
            col_en, col_tr = st.columns(2)
            with col_en:
                st.text_area("English", value=p["english_text"], height=120, disabled=True, key=f"tr_en_{review_level}_{page_index}")
            with col_tr:
                edited[key] = st.text_area(
                    "Translated (editable)",
                    value=edited[key],
                    height=120,
                    key=f"tr_edit_{review_level}_{page_index}",
                )
            st.divider()

        if st.button("Save to Supabase", type="primary", key="tr_save"):
            story_id_save = result["story_id"]
            target_lang_code_save = result["target_lang_code"]
            saved = 0
            for lev, level_pages in result["levels"].items():
                for p in level_pages:
                    page_index = p["page_index"]
                    key = (lev, page_index)
                    final_text = edited.get(key, p["translated_text"])
                    image_url = p.get("image_url") or None
                    if insert_book_page(
                        sb,
                        story_id=story_id_save,
                        language_code=target_lang_code_save,
                        reading_level=lev,
                        page_index=page_index,
                        page_text=final_text,
                        image_url=image_url,
                    ):
                        saved += 1
            if saved:
                st.success(f"Saved {saved} pages to Supabase (all levels).")
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            st.rerun()
        if st.button("Clear and translate again", key="tr_clear"):
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            st.rerun()

    elif has_pages:
        st.header("3. Review & edit")
        st.caption("Edit the translated text on the right. Click **Save to Supabase** when ready.")
        if "tr_edited" not in st.session_state:
            st.session_state.tr_edited = {p["page_index"]: p["translated_text"] for p in result["pages"]}
        edited = st.session_state.tr_edited
        for i, p in enumerate(result["pages"]):
            page_index = p["page_index"]
            if page_index not in edited:
                edited[page_index] = p["translated_text"]
            st.subheader(f"Page {page_index}")
            col_en, col_tr = st.columns(2)
            with col_en:
                st.text_area("English", value=p["english_text"], height=120, disabled=True, key=f"tr_en_{page_index}")
            with col_tr:
                edited[page_index] = st.text_area(
                    "Translated (editable)",
                    value=edited[page_index],
                    height=120,
                    key=f"tr_edit_{page_index}",
                )
            st.divider()

        if st.button("Save to Supabase", type="primary", key="tr_save"):
            story_id_save = result["story_id"]
            target_lang_code_save = result["target_lang_code"]
            reading_level_save = result["reading_level"]
            saved = 0
            for p in result["pages"]:
                page_index = p["page_index"]
                final_text = edited.get(page_index, p["translated_text"])
                image_url = p.get("image_url") or None
                if insert_book_page(
                    sb,
                    story_id=story_id_save,
                    language_code=target_lang_code_save,
                    reading_level=reading_level_save,
                    page_index=page_index,
                    page_text=final_text,
                    image_url=image_url,
                ):
                    saved += 1
            if saved:
                st.success(f"Saved {saved} pages to Supabase.")
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            st.rerun()
        if st.button("Clear and translate again", key="tr_clear"):
            if "tr_result" in st.session_state:
                del st.session_state["tr_result"]
            if "tr_edited" in st.session_state:
                del st.session_state["tr_edited"]
            st.rerun()
