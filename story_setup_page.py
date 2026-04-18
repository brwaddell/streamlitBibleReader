"""
Story Setup wizard: English text → translation → audio in one guided flow.
Reuses Story Text Parser, Translator, and Audio Generator; optional auto-advance after saves.
"""

import streamlit as st

from lib import run_audio_generator_view
from story_text_page import run_story_text_view
from translator_page import apply_translator_target_to_audio_language, run_translator_view

# Must match st.radio option strings below (used for programmatic step jumps).
STEP_TEXT = "1 · English text"
STEP_TRANSLATE = "2 · Translate"
STEP_AUDIO = "3 · Audio"
STEPS = (STEP_TEXT, STEP_TRANSLATE, STEP_AUDIO)


def run_story_setup_view():
    """Wizard-style layout: three steps sharing the same tools as standalone pages."""
    st.title("Story Setup")
    st.caption(
        "Walk through **English pages**, **translation**, and **audio** in order. "
        "The same controls appear here as on Story Text Parser, Translator, and Audio Generator — "
        "without switching sidebar pages."
    )

    if "setup_wizard_step" not in st.session_state:
        st.session_state["setup_wizard_step"] = STEP_TEXT

    st.markdown("**Progress**")
    step = st.radio(
        "Story setup steps",
        STEPS,
        horizontal=True,
        key="setup_wizard_step",
        label_visibility="collapsed",
    )

    st.divider()

    if step == STEP_TEXT:
        st.markdown(f"### {STEP_TEXT}")
        st.caption("Paste or replace English `story_content_flat` pages, then continue to translation.")
        st.checkbox(
            "After a successful save, jump to **Translate**",
            value=st.session_state.get("wiz_auto_next_translate", True),
            key="wiz_auto_next_translate",
        )
        run_story_text_view(
            as_wizard_step=True,
            wizard_after_save_step=STEP_TRANSLATE,
        )
        if st.button("Continue to **Translate**", type="secondary", key="wiz_nav_to_translate"):
            st.session_state["setup_wizard_step"] = STEP_TRANSLATE
            st.rerun()

    elif step == STEP_TRANSLATE:
        st.markdown(f"### {STEP_TRANSLATE}")
        st.caption("Translate missing pages from English, review, and save. Then generate audio for that language.")
        st.checkbox(
            "After a successful translation save, jump to **Audio**",
            value=st.session_state.get("wiz_auto_next_audio", True),
            key="wiz_auto_next_audio",
        )
        run_translator_view(as_wizard_step=True, wizard_after_save_step=STEP_AUDIO)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back to **English text**", key="wiz_nav_to_text"):
                st.session_state["setup_wizard_step"] = STEP_TEXT
                st.rerun()
        with c2:
            if st.button("Continue to **Audio**", type="secondary", key="wiz_nav_to_audio"):
                apply_translator_target_to_audio_language()
                st.session_state["setup_wizard_step"] = STEP_AUDIO
                st.rerun()

    else:
        st.markdown(f"### {STEP_AUDIO}")
        st.caption("Generate ElevenLabs audio for pages that are still missing male/female tracks.")
        run_audio_generator_view(as_wizard_step=True)
        if st.button("Back to **Translate**", key="wiz_nav_back_translate"):
            st.session_state["setup_wizard_step"] = STEP_TRANSLATE
            st.rerun()
