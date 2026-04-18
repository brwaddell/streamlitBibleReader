"""
Storybook Image Processor - Bulk image generation with approval workflow.
Gemini image generation and/or Leonardo.ai; OpenAI scene prompts; export to R2 + story_content_flat.
"""

import streamlit as st
from dotenv import load_dotenv

from auth import is_authenticated, logout, run_login_page
from grade_style_defaults import GRADE_STYLE_DEFAULTS
from image_processor_page import run_image_processor_view
from lib import run_book_pages_view
from stories_page import run_stories_view
from story_setup_page import run_story_setup_view
from story_text_page import run_story_text_view

load_dotenv()


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
        "ip_pending_images": {},
        "ip_cache_version": 0,
        "ip_pages_missing_version": 0,
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


def main():
    st.set_page_config(page_title="Storybook Image Processor", layout="wide", initial_sidebar_state="expanded")

    if not is_authenticated():
        run_login_page()
        st.stop()

    init_session_state()

    st.sidebar.title("Storybook")
    page = st.sidebar.radio(
        "Go to",
        [
            "Story Titles",
            "Story Text Parser",
            "Story Setup",
            "Image Processor",
            "Story Content Manager",
        ],
    )
    st.sidebar.divider()
    if st.sidebar.button("Sign out"):
        logout()
        st.rerun()

    if page == "Story Titles":
        run_stories_view()
        return

    if page == "Story Text Parser":
        run_story_text_view()
        return

    if page == "Story Setup":
        run_story_setup_view()
        return

    if page == "Story Content Manager":
        run_book_pages_view()
        return

    run_image_processor_view()


if __name__ == "__main__":
    main()
