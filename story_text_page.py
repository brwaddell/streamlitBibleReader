"""
Story Text Page: Paste full story, split by delimiter, save to story_content_flat.
Creates or replaces page text for a given story + reading level + language.
No images; use Image Processor to generate those.
"""

import streamlit as st

from lib import (
    LANGUAGE_CODES,
    READING_LEVELS,
    delete_book_pages_for_version,
    fetch_book_pages,
    fetch_stories,
    get_supabase,
    insert_book_page,
)


def run_story_text_view():
    """Render the Story Text page: paste, split by delimiter, save to Supabase."""
    st.title("Story Text")
    st.caption("Paste full story, split by delimiter, save to story_content_flat. Use Image Processor to generate images.")

    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories()
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    # --- Filters ---
    st.header("1. Select story & version")
    col1, col2, col3 = st.columns(3)
    with col1:
        story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
        story_label = st.selectbox("Story", options=list(story_options.keys()), key="st_story")
        story_id = story_options.get(story_label) if story_label else None
    with col2:
        reading_level = st.selectbox(
            "Reading level",
            options=READING_LEVELS,
            key="st_reading_level",
        )
    with col3:
        language_code = st.selectbox("Language", options=LANGUAGE_CODES, key="st_language")

    if not story_id:
        st.info("Select a story to continue.")
        return

    # --- Paste & split ---
    st.header("2. Paste & split")
    paste_raw = st.text_area(
        "Paste full story",
        placeholder="Paste the whole story. Use the delimiter below to separate pages (e.g. --- or one paragraph per line).",
        height=180,
        key="st_paste_raw",
    )
    delim_col, split_col = st.columns([2, 1])
    with delim_col:
        page_delimiter = st.text_input(
            "Page delimiter",
            value="---",
            key="st_page_delimiter",
            help="Split by this (e.g. ---). Type \\n or newline for one paragraph per line.",
        )
    with split_col:
        do_split = st.button("Preview split", type="primary", key="st_do_split")

    segments = st.session_state.get("st_segments", [])
    if do_split and paste_raw and paste_raw.strip():
        delim = (page_delimiter or "---").strip()
        if delim == "\\n" or delim.lower() == "newline":
            segments = [s.strip() for s in paste_raw.strip().splitlines() if s.strip()]
        else:
            segments = [s.strip() for s in paste_raw.strip().split(delim) if s.strip()]
        st.session_state["st_segments"] = segments
        if segments:
            st.success(f"Split into **{len(segments)}** pages. Review below, then click Save to Supabase.")
        else:
            st.warning("No non-empty segments. Check your delimiter.")
            st.session_state["st_segments"] = []
    elif do_split and (not paste_raw or not paste_raw.strip()):
        st.warning("Paste some story text first.")

    # Show segments for review
    if segments:
        st.header("3. Review pages")
        for i, seg in enumerate(segments):
            with st.expander(f"Page {i}", expanded=True):
                st.text(seg)
        st.divider()

    # --- Save ---
    st.header("4. Save to Supabase")
    save_to_all_levels = st.checkbox(
        "Save to all reading levels",
        key="st_save_all_levels",
        help="Apply the same split text to all 5 grades (grade_1–grade_5) for this story and language.",
    )
    st.caption(
        "Save will replace all existing pages for this story + reading level + language with the split result. "
        "Existing images will be lost; run Image Processor to regenerate."
    )

    # Confirmation when overwriting existing content
    pending = st.session_state.get("st_save_confirm_pending") and st.session_state.get("st_save_confirm_data")
    if pending:
        data = st.session_state["st_save_confirm_data"]
        levels_with_content = data.get("levels_with_content", [])
        st.warning(
            f"This story + language already has content for: **{', '.join(levels_with_content)}**. "
            "Overwriting will replace all pages and remove existing images. Confirm to proceed."
        )
        col1, col2, _ = st.columns([1, 1, 2])
        with col1:
            if st.button("Confirm overwrite", type="primary", key="st_confirm_overwrite"):
                sid = data["story_id"]
                lang = data["language_code"]
                levels_to_save = data["levels_to_save"]
                segs = data["segments"]
                total_saved = 0
                for lev in levels_to_save:
                    if delete_book_pages_for_version(sb, sid, lang, lev):
                        for i, seg in enumerate(segs):
                            if insert_book_page(sb, sid, lang, lev, i, seg, image_url=None):
                                total_saved += 1
                    else:
                        st.error(f"Failed to delete existing pages for {lev}.")
                        break
                else:
                    st.success(f"Saved {total_saved} pages to story_content_flat ({len(levels_to_save)} level(s)).")
                    if "st_segments" in st.session_state:
                        del st.session_state["st_segments"]
                del st.session_state["st_save_confirm_pending"]
                del st.session_state["st_save_confirm_data"]
                st.rerun()
        with col2:
            if st.button("Cancel", key="st_cancel_overwrite"):
                del st.session_state["st_save_confirm_pending"]
                del st.session_state["st_save_confirm_data"]
                st.rerun()
        st.stop()

    save_clicked = st.button("Save to Supabase", type="primary", key="st_save")
    if save_clicked:
        if not segments:
            st.warning("Split the story first to get pages.")
        else:
            levels_to_save = READING_LEVELS if save_to_all_levels else [reading_level]
            levels_with_content = [
                lev for lev in levels_to_save
                if fetch_book_pages(sb, story_id, lev, language_code)
            ]
            if levels_with_content:
                st.session_state["st_save_confirm_pending"] = True
                st.session_state["st_save_confirm_data"] = {
                    "story_id": story_id,
                    "language_code": language_code,
                    "levels_to_save": levels_to_save,
                    "segments": segments,
                    "levels_with_content": levels_with_content,
                }
                st.rerun()
            else:
                total_saved = 0
                for lev in levels_to_save:
                    if delete_book_pages_for_version(sb, story_id, language_code, lev):
                        for i, seg in enumerate(segments):
                            if insert_book_page(sb, story_id, language_code, lev, i, seg, image_url=None):
                                total_saved += 1
                    else:
                        st.error(f"Failed to delete existing pages for {lev}.")
                        break
                else:
                    st.success(f"Saved {total_saved} pages to story_content_flat ({len(levels_to_save)} level(s)).")
                    if "st_segments" in st.session_state:
                        del st.session_state["st_segments"]
                    st.rerun()
