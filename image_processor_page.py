"""Image Processor Streamlit view: Gemini and/or Leonardo, scene LLM, job queue."""
import time
from typing import Optional

import streamlit as st

import leonardo_client as leo
from auth import get_secret
from grade_style_defaults import GRADE_STYLE_DEFAULTS
from lib import (
    LEONARDO_MAX_CHARACTER_REFERENCE_REFS,
    MAX_REFERENCE_IMAGES,
    READING_LEVELS,
    _build_batch_request_parts,
    _extract_image_from_batch_response,
    build_leonardo_prompt,
    build_prompt,
    collect_reference_images,
    fetch_batch_jobs_for_version,
    fetch_book_pages,
    fetch_pages_missing_images,
    fetch_pending_image_generation_jobs,
    fetch_stories,
    generate_image_gemini,
    generate_image_leonardo,
    get_gemini,
    get_story_grade_style,
    get_supabase,
    insert_batch_job,
    insert_image_generation_job,
    optimize_image_for_mobile,
    parse_reference_images_json,
    reference_entries_from_text_slots,
    seed_reference_text_slots,
    update_batch_job_status,
    update_book_page,
    update_image_generation_job,
    upload_image_to_storage,
    upload_reference_image_to_storage,
    upsert_story_grade_style,
)
from scene_prompts import merge_negative_additions, merge_scene_to_extra_details, suggest_scene_prompt

_IP_CACHE_TTL = 60

_IP_REF_PENDING_SLOT_FILL = "ip_ref_pending_slot_fill"
_REF_GEN_EXTRA = (
    "Create one clear reference illustration (square), iconic and reusable for visual consistency across a children's book. "
    "No text, letters, or typography in the image."
)


def _row_id_str(rid) -> str:
    return str(rid) if rid is not None else ""


def _page_index_for_row(row: dict, fallback_i: int) -> int:
    v = row.get("page_index")
    if v is None:
        v = row.get("page_number")
    if v is None:
        v = fallback_i
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(fallback_i)
        except (TypeError, ValueError):
            return 0


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


def _bump_ip_cache_version():
    """Invalidate @st.cache_data fetches (book pages, grade style, etc.) after DB image changes."""
    st.session_state["ip_cache_version"] = st.session_state.get("ip_cache_version", 0) + 1


def _get_openai_client():
    if st.session_state.get("openai_client") is None:
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI

        st.session_state.openai_client = OpenAI(api_key=api_key)
    return st.session_state.openai_client


def run_image_processor_view():
    st.title("Image Processor")
    st.caption("Generate images for pages missing images. Add story text in Story Text Parser first, then generate here.")

    sb = get_supabase()
    if not sb:
        return

    _v = st.session_state.get("ip_cache_version", 0)
    stories = _cached_fetch_stories(_v)
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    language_code = "en"
    st.header("1. Select story & version")
    story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
    story_label = st.selectbox("Story", options=list(story_options.keys()), key="ip_story")
    story_id = story_options.get(story_label) if story_label else None
    reading_level = st.selectbox(
        "Reading level", options=READING_LEVELS, key="ip_reading_level", format_func=lambda x: x.replace("_", " ").title()
    )

    if not story_id:
        st.info("Select a story to continue.")
        return

    story_title = next((s.get("title", "") for s in stories if s.get("id") == story_id), "")

    just_approved = st.session_state.pop("ip_just_approved", None) or {}
    just_cleared = set(st.session_state.pop("ip_just_cleared", None) or [])
    just_cleared_ids = {_row_id_str(x) for x in just_cleared}
    # Until cached/DB book_pages shows new image_url, keep URL here so reference dropdown stays stable.
    _img_overlay = st.session_state.setdefault("ip_book_image_overlay", {})
    for _rid, _u in just_approved.items():
        if _u and _row_id_str(_rid):
            _img_overlay[_row_id_str(_rid)] = str(_u).strip()
    _v_missing = st.session_state.get("ip_pages_missing_version", 0)
    pages_missing = _cached_fetch_pages_missing_images(story_id, language_code, reading_level, _v, _v_missing)
    pages_missing_display = [r for r in pages_missing if r.get("id") not in just_approved]
    all_pages_for_counts = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    pending = st.session_state.get("ip_pending_images", {})
    pending_leo = fetch_pending_image_generation_jobs(sb, story_id, reading_level)

    n_missing = len(pages_missing_display)
    n_pending_approval = len(pending)
    n_leo_jobs = len(pending_leo)

    st.markdown(
        f"**Context:** story **{story_id}** · **{reading_level}** · **{language_code}** · "
        f"missing **{n_missing}** · pending approval **{n_pending_approval}** · Leonardo jobs **{n_leo_jobs}**"
    )

    if n_missing:
        st.success(f"**{n_missing}** pages missing images.")
    else:
        st.info("No pages missing images for this story + reading level. You can still set style or review.")

    story = next((s for s in stories if s.get("id") == story_id), None)
    grade_style = _cached_get_story_grade_style(story_id, reading_level, _v)
    if story_id and reading_level:
        defaults = dict(GRADE_STYLE_DEFAULTS.get(reading_level, GRADE_STYLE_DEFAULTS["grade_1"]))
        if story:
            for key in ["character_ref", "global_style", "age_appropriateness", "color_palette", "lighting", "framing"]:
                if story.get(key) not in (None, ""):
                    defaults[key] = story[key]
        if grade_style:
            for key in [
                "age_appropriateness",
                "global_style",
                "character_ref",
                "color_palette",
                "lighting",
                "framing",
                "character_reference_image_url",
                "reference_images",
                "default_image_provider",
            ]:
                if grade_style.get(key) not in (None, ""):
                    defaults[key] = grade_style[key]
            if grade_style.get("leonardo_seed") is not None:
                defaults["leonardo_seed"] = grade_style["leonardo_seed"]

        loaded_for = st.session_state.get("ip_style_loaded_for")
        current_for = f"{story_id}_{reading_level}"
        # Re-seed when story/level changes, or when ip_cache_version bumps (e.g. Settings save) so DB style
        # replaces stale session_state; otherwise non-empty widgets never pick up external updates.
        style_seeded_v = st.session_state.get("ip_style_seeded_cache_v")
        need_style_seed = loaded_for != current_for or style_seeded_v != _v
        if need_style_seed:
            st.session_state["ip_style_loaded_for"] = current_for
            st.session_state["ip_style_seeded_cache_v"] = _v
            st.session_state["age_appropriateness"] = defaults.get("age_appropriateness", "")
            st.session_state["style_prompt"] = defaults.get("global_style", "")
            st.session_state["character_ref"] = defaults.get("character_ref", "")
            st.session_state["color_palette"] = defaults.get("color_palette", "")
            st.session_state["lighting"] = defaults.get("lighting", "")
            st.session_state["framing"] = defaults.get("framing", "")
            dp = (defaults.get("default_image_provider") or "").strip().lower()
            st.session_state["ip_image_provider"] = dp if dp in ("gemini", "leonardo") else "leonardo"
            ls = defaults.get("leonardo_seed")
            st.session_state["ip_leo_seed_input"] = int(ls) if ls is not None else 0
            _ref_entries = parse_reference_images_json((grade_style or {}).get("reference_images"))
            if not _ref_entries and grade_style and (grade_style.get("character_reference_image_url") or "").strip():
                _ref_entries = [{"label": "Character", "url": grade_style["character_reference_image_url"].strip()}]
            seed_reference_text_slots(story_id, reading_level, MAX_REFERENCE_IMAGES, "ip_ref", _ref_entries)
        else:
            for sk, dk in [
                ("age_appropriateness", "age_appropriateness"),
                ("style_prompt", "global_style"),
                ("character_ref", "character_ref"),
                ("color_palette", "color_palette"),
                ("lighting", "lighting"),
                ("framing", "framing"),
            ]:
                if sk not in st.session_state or st.session_state.get(sk) == "":
                    st.session_state[sk] = defaults.get(dk, "")

    if "ip_image_provider" not in st.session_state:
        st.session_state["ip_image_provider"] = "leonardo"

    st.subheader("Checklist")
    c1, c2, c3, c4 = st.columns(4)
    _ri = parse_reference_images_json((grade_style or {}).get("reference_images"))
    _has_saved_refs = bool(_ri) or bool((grade_style or {}).get("character_reference_image_url"))
    with c1:
        st.caption("Reference image" + (" ✓" if _has_saved_refs else " — add URL in Settings"))
    with c2:
        st.caption("Style saved" + (" ✓" if grade_style else " (optional)"))
    with c3:
        st.caption("Scene LLM" + (" ✓" if st.session_state.get("ip_scene_suggestions") else " (optional)"))
    with c4:
        st.caption("Generate / queue")

    model_id = (get_secret("LEONARDO_MODEL_ID") or "").strip()
    api_key_leo = get_secret("LEONARDO_API_KEY")

    tab_settings, tab_workflow = st.tabs(["Settings", "Workflow"], key="ip_main_tabs")
    ref_file = None
    selected_ref_url = None
    leonardo_contrast: Optional[float] = None
    leo_side = leo.snap_leonardo_dimension(leo.DEFAULT_LEONARDO_WIDTH)

    with tab_settings:
        _pending_fill = st.session_state.pop(_IP_REF_PENDING_SLOT_FILL, None)
        if isinstance(_pending_fill, dict):
            _pf_sid = _pending_fill.get("story_id")
            _pf_lvl = _pending_fill.get("reading_level")
            if _pf_sid == story_id and _pf_lvl == reading_level:
                _pf_url = (_pending_fill.get("url") or "").strip()
                if _pf_url:
                    st.session_state[f"ip_ref_u_{story_id}_{reading_level}_0"] = _pf_url
                    if _pending_fill.get("label") is not None:
                        st.session_state[f"ip_ref_l_{story_id}_{reading_level}_0"] = (
                            str(_pending_fill.get("label") or "").strip() or "Reference"
                        )

        st.subheader("Style & reference")
        st.caption("Saved to `story_grade_styles` when you click **Save style**. Default image provider is set under **Workflow**.")
        age_appropriateness = st.text_input("Age appropriateness", key="age_appropriateness")
        style_prompt = st.text_area("Global style prompt", key="style_prompt")
        character_ref = st.text_area("Character reference", key="character_ref", height=100)
        color_palette = st.text_input("Color palette", key="color_palette")
        lighting = st.text_input("Lighting", key="lighting")
        framing = st.text_input("Framing", key="framing")
        leo_guidance = st.slider(
            "Leonardo guidance scale",
            1.0,
            20.0,
            6.0,
            0.5,
            key="ip_leo_guidance",
            help="Lower values (e.g. 5.5–6.5) are often softer for Phoenix storybook work.",
        )
        leo_neg_suffix = st.text_input(
            "Extra negative prompt (Leonardo)", key="ip_leo_neg_suffix", placeholder="e.g. scary, violent"
        )
        leo_cn_strength = st.selectbox(
            "Leonardo character reference strength",
            options=["Low", "Mid", "High"],
            index=0,
            key="ip_leo_cn_strength",
            help="Low reduces ref dominance so text (age/style) can steer tone more—good for young grades.",
        )
        _sq = list(leo.LEONARDO_SQUARE_PRESETS)
        _def_sq = leo.nearest_square_preset(leo.DEFAULT_LEONARDO_WIDTH)
        _sq_idx = _sq.index(_def_sq) if _def_sq in _sq else 3
        leo_side = st.selectbox(
            "Leonardo output size (square px; smaller = lower API cost)",
            options=_sq,
            index=_sq_idx,
            key="ip_leo_square",
            help="Matches mobile export (max 800px WebP). 800 aligns with default env.",
        )
        _leo_mid = (get_secret("LEONARDO_MODEL_ID") or "").strip()
        if _leo_mid and leo.is_phoenix_model(_leo_mid):
            _pc = st.selectbox(
                "Phoenix contrast (required for Phoenix + Alchemy)",
                options=["Low (3)", "Medium (3.5)", "High (4)"],
                index=0,
                key="ip_phoenix_contrast",
                help="Lower contrast is gentler; Phoenix API allows 3, 3.5, or 4 per docs.",
            )
            leonardo_contrast = {"Low (3)": 3.0, "Medium (3.5)": 3.5, "High (4)": 4.0}[_pc]
        st.number_input(
            "Leonardo seed (0 = random)",
            min_value=0,
            max_value=2_147_483_637,
            step=1,
            key="ip_leo_seed_input",
            help="Non-zero sends a fixed seed to Leonardo for consistency. Saved with style.",
        )

        st.markdown("**Reference image** (one URL; used for Gemini and Leonardo character reference)")
        st.caption("Paste a public HTTPS URL, upload to R2 (…/refs/), or generate below. Also saved as `character_reference_image_url`.")
        c_ra, c_rb = st.columns(2)
        with c_ra:
            st.text_input("Label", key=f"ip_ref_l_{story_id}_{reading_level}_0", placeholder="e.g. Character")
        with c_rb:
            st.text_input("Image URL", key=f"ip_ref_u_{story_id}_{reading_level}_0", placeholder="https://…")
        _ref_up_file = st.file_uploader(
            "Reference image file (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            key=f"ip_ref_upload_file_{story_id}_{reading_level}",
        )
        if st.button("Upload file to reference slot", key=f"ip_ref_upload_go_{story_id}_{reading_level}"):
            if _ref_up_file is None:
                st.warning("Choose a file first.")
            elif sb:
                _raw = _ref_up_file.read()
                if _raw:
                    _url = upload_reference_image_to_storage(sb, story_id, reading_level, 0, _raw)
                    if _url:
                        st.session_state[f"ip_ref_u_{story_id}_{reading_level}_0"] = _url
                        st.success("Uploaded — click **Save style** to persist.")
                        st.rerun()
                else:
                    st.warning("Empty file.")
            else:
                st.error("Supabase client not available.")

        ref_file = st.file_uploader(
            "Extra session reference upload (optional; prepended before saved URL)",
            type=["png", "jpg", "jpeg"],
            key="ref_image",
        )
        all_pages = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
        _img_overlay = st.session_state.setdefault("ip_book_image_overlay", {})
        for r in all_pages:
            rid = r.get("id")
            if rid is not None and (r.get("image_url") or "").strip():
                _img_overlay.pop(_row_id_str(rid), None)
        published_with_images = []
        for i, r in enumerate(all_pages):
            rid = r.get("id")
            if rid is not None and _row_id_str(rid) in just_cleared_ids:
                continue
            url = (r.get("image_url") or "").strip()
            if not url and rid is not None:
                url = (_img_overlay.get(_row_id_str(rid)) or "").strip()
            if not url:
                continue
            published_with_images.append((_page_index_for_row(r, i), url))
        published_with_images.sort(key=lambda x: x[0])
        ref_page_options = ["None"] + [f"Page {pi}" for pi, _ in published_with_images]
        ref_page_labels = {f"Page {pi}": url for pi, url in published_with_images}
        ref_page_key = f"ref_page_select_{story_id}_{reading_level}"
        grade_style_for_ref = _cached_get_story_grade_style(story_id, reading_level, _v)
        if ref_page_key not in st.session_state:
            default_ref = "None"
            if grade_style_for_ref is not None and grade_style_for_ref.get("reference_page_index") is not None:
                saved_ref_label = f"Page {grade_style_for_ref['reference_page_index']}"
                if saved_ref_label in ref_page_options:
                    default_ref = saved_ref_label
            st.session_state[ref_page_key] = default_ref
        elif ref_page_options and st.session_state.get(ref_page_key) not in ref_page_options:
            st.session_state[ref_page_key] = "None"
        ref_selected = st.selectbox(
            "Use published page as reference",
            options=ref_page_options,
            key=ref_page_key,
        )
        selected_ref_url = ref_page_labels.get(ref_selected) if ref_selected != "None" else None

        _provider_save = (st.session_state.get("ip_image_provider") or "leonardo").strip().lower()
        if _provider_save not in ("gemini", "leonardo"):
            _provider_save = "leonardo"

        if story_id and reading_level and st.button("Save style for this story & grade", key="save_style"):
            ref_page_index = None
            if ref_selected and ref_selected != "None" and ref_selected.startswith("Page "):
                try:
                    ref_page_index = int(ref_selected.replace("Page ", "").strip())
                except ValueError:
                    pass
            _persist_refs = reference_entries_from_text_slots(
                story_id, reading_level, MAX_REFERENCE_IMAGES, "ip_ref"
            )
            _first_url = _persist_refs[0]["url"] if _persist_refs else ""
            if upsert_story_grade_style(
                sb,
                story_id,
                reading_level,
                {
                    "age_appropriateness": age_appropriateness or "",
                    "global_style": style_prompt or "",
                    "character_ref": character_ref or "",
                    "color_palette": color_palette or "",
                    "lighting": lighting or "",
                    "framing": framing or "",
                    "reference_page_index": ref_page_index,
                    "default_image_provider": _provider_save,
                    "reference_images": _persist_refs,
                    "character_reference_image_url": _first_url,
                    "leonardo_seed": (int(st.session_state.get("ip_leo_seed_input", 0) or 0) or None),
                },
            ):
                _bump_ip_cache_version()
                st.success(f"Style saved for {reading_level}.")
                st.rerun()

        _gen_scope = f"{story_id}_{reading_level}"
        _ip_ref_bkey = f"ip_ref_gen_bytes_{_gen_scope}"
        st.subheader("Generate reference image")
        st.caption(
            "Uses the **Workflow** image provider and Leonardo seed above. Optional: merge character & style fields into the prompt."
        )
        st.text_area(
            "Prompt for reference image",
            key=f"ip_ref_gen_prompt_{_gen_scope}",
            height=100,
            placeholder="e.g. The tower under construction, wide stone ramp, workers in ancient Near Eastern dress",
        )
        st.checkbox(
            "Include character & style fields in prompt",
            value=True,
            key=f"ip_ref_gen_merge_{_gen_scope}",
        )
        _sq2 = list(leo.LEONARDO_SQUARE_PRESETS)
        _def_sq2 = leo.nearest_square_preset(leo.DEFAULT_LEONARDO_WIDTH)
        _sq_idx2 = _sq2.index(_def_sq2) if _def_sq2 in _sq2 else 3
        st.selectbox(
            "Leonardo output size (square px; only for Leonardo)",
            options=_sq2,
            index=_sq_idx2,
            key=f"ip_ref_gen_leo_square_{_gen_scope}",
        )
        _gcols = st.columns([1, 1, 1])
        with _gcols[0]:
            _do_gen = st.button("Generate", key=f"ip_ref_gen_go_{_gen_scope}")
        with _gcols[1]:
            _do_regen = st.button("Regenerate", key=f"ip_ref_gen_regen_{_gen_scope}")
        with _gcols[2]:
            _do_discard = st.button("Discard preview", key=f"ip_ref_gen_discard_{_gen_scope}")

        if _do_discard:
            st.session_state.pop(_ip_ref_bkey, None)
            st.rerun()

        if _do_gen or _do_regen:
            if _do_regen and _ip_ref_bkey not in st.session_state:
                st.info("Nothing to regenerate yet — click **Generate** first.")
            else:
                user_p = (st.session_state.get(f"ip_ref_gen_prompt_{_gen_scope}") or "").strip()
                if not user_p:
                    st.warning("Enter a prompt first.")
                else:
                    merge = bool(st.session_state.get(f"ip_ref_gen_merge_{_gen_scope}", True))
                    age = (st.session_state.get("age_appropriateness") or "").strip() if merge else ""
                    gstyle = (st.session_state.get("style_prompt") or "").strip() if merge else ""
                    cref = (st.session_state.get("character_ref") or "").strip() if merge else ""
                    light = (st.session_state.get("lighting") or "").strip() if merge else ""
                    pal = (st.session_state.get("color_palette") or "").strip() if merge else ""
                    frame = (st.session_state.get("framing") or "").strip() if merge else ""
                    provider = (st.session_state.get("ip_image_provider") or "leonardo").strip().lower()
                    if provider not in ("gemini", "leonardo"):
                        provider = "leonardo"
                    if provider == "gemini":
                        full = build_prompt(_REF_GEN_EXTRA, user_p, age, gstyle, cref, light, pal, frame, "")
                        with st.spinner("Gemini: generating reference…"):
                            out, cost_line = generate_image_gemini(full, None)
                    else:
                        api_key_r = (get_secret("LEONARDO_API_KEY") or "").strip()
                        model_id_r = (get_secret("LEONARDO_MODEL_ID") or "").strip()
                        if not api_key_r or not model_id_r:
                            st.error("Set LEONARDO_API_KEY and LEONARDO_MODEL_ID for Leonardo.")
                            out = None
                            cost_line = None
                        else:
                            pos, neg = build_leonardo_prompt(
                                _REF_GEN_EXTRA,
                                user_p,
                                age,
                                gstyle,
                                cref,
                                light,
                                pal,
                                frame,
                                "",
                                "",
                                reading_level=reading_level,
                            )
                            seed_in = int(st.session_state.get("ip_leo_seed_input", 0) or 0)
                            use_seed_r = seed_in if seed_in > 0 else None
                            leo_contrast_r = None
                            if leo.is_phoenix_model(model_id_r):
                                leo_contrast_r = leo.effective_contrast_for_model(model_id_r, None)
                            side_r = int(
                                st.session_state.get(f"ip_ref_gen_leo_square_{_gen_scope}", leo.DEFAULT_LEONARDO_WIDTH)
                            )
                            with st.spinner("Leonardo: generating reference…"):
                                out, cost_line = generate_image_leonardo(
                                    pos,
                                    neg,
                                    None,
                                    api_key=api_key_r,
                                    model_id=model_id_r,
                                    width=side_r,
                                    height=side_r,
                                    guidance_scale=6.0,
                                    seed=use_seed_r,
                                    controlnet_strength="Low",
                                    contrast=leo_contrast_r,
                                )
                    if out:
                        st.session_state[_ip_ref_bkey] = out
                        _costs = st.session_state.setdefault("ip_last_gen_cost", {})
                        if cost_line:
                            _costs["ref_preview"] = cost_line
                        st.success("Generated. Review below; **Approve** uploads to the reference slot or discard.")
                        st.rerun()

        _pending = st.session_state.get(_ip_ref_bkey)
        if _pending:
            st.image(_pending, caption="Preview (not saved until you approve)", use_container_width=True)
            _lcost = (st.session_state.get("ip_last_gen_cost") or {}).get("ref_preview")
            if _lcost:
                st.caption(_lcost)
            st.text_input(
                "Label for this reference",
                key=f"ip_ref_gen_ap_label_{_gen_scope}",
                placeholder="e.g. Character",
            )
            if st.button("Approve — upload to slot & fill URL", type="primary", key=f"ip_ref_gen_approve_{_gen_scope}"):
                _url = upload_reference_image_to_storage(sb, story_id, reading_level, 0, _pending)
                if _url:
                    _lab = (st.session_state.get(f"ip_ref_gen_ap_label_{_gen_scope}") or "").strip() or "Reference"
                    st.session_state[_IP_REF_PENDING_SLOT_FILL] = {
                        "story_id": story_id,
                        "reading_level": reading_level,
                        "slot": 0,
                        "label": _lab,
                        "url": _url,
                    }
                    st.session_state.pop(_ip_ref_bkey, None)
                    st.success("Uploaded to slot — click **Save style** to persist to the database.")
                    st.rerun()

    _saved_refs = reference_entries_from_text_slots(story_id, reading_level, MAX_REFERENCE_IMAGES, "ip_ref")
    refs, ref_prompt_note = collect_reference_images(
        ref_file=ref_file,
        ref_image_url=selected_ref_url,
        saved_labeled=_saved_refs if _saved_refs else None,
    )

    use_seed = int(st.session_state.get("ip_leo_seed_input", 0) or 0) or None

    def _check_batch_status_gemini(job, pages_for_job):
        check_job_key = job.get("batch_name")
        client = get_gemini()
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
                    pending_local = st.session_state.get("ip_pending_images", {})
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
                            pending_local[row_id] = opt
                            ok_count += 1
                        else:
                            failed_keys.append(f"page {row.get('page_index', i)}")
                    if failed_keys:
                        st.warning(f"Some pages failed or had no image: {', '.join(failed_keys)}")
                    st.session_state["ip_pending_images"] = pending_local
                    update_batch_job_status(sb, check_job_key, "succeeded")
                    st.success(f"Loaded {ok_count} images. Approve each in Review below.")
                else:
                    st.warning("No inline responses found or pages have changed.")
                return True
            if state in ("JOB_STATE_FAILED", "JOB_STATE_FAILURE"):
                err = getattr(batch_job, "error", None) or ""
                update_batch_job_status(sb, check_job_key, "failed", error_message=str(err))
                st.error(f"Batch failed: {err}")
                return True
            if "EXPIRED" in str(state) or "CANCELLED" in str(state):
                update_batch_job_status(sb, check_job_key, "expired" if "EXPIRED" in str(state) else "cancelled")
                st.warning("Job expired or was cancelled. Submit a new batch.")
                return True
            st.info("Still processing. Check again in a few minutes.")
            return True
        except Exception as e:
            st.error(f"Failed to check batch: {e}")
            return False

    with tab_workflow:
        image_provider = st.radio(
            "Image provider",
            options=["gemini", "leonardo"],
            format_func=lambda x: "Gemini" if x == "gemini" else "Leonardo.ai",
            horizontal=True,
            key="ip_image_provider",
        )
        st.header("3. Batch generate")
        batch_jobs = _cached_fetch_batch_jobs_for_version(story_id, reading_level, _v)
        pending_gemini_jobs = [j for j in batch_jobs if j.get("status") in ("pending", "running")]

        if image_provider == "gemini":
            st.caption("Gemini: generate first page, then submit batch (async ~24h).")
            if pending_gemini_jobs:
                st.info("Gemini batch jobs in progress.")
                for job in pending_gemini_jobs:
                    created = job.get("created_at", "")[:19].replace("T", " ") if job.get("created_at") else "?"
                    with st.expander(f"Batch {job.get('batch_name', '?')} — {created}"):
                        if st.button("Check batch status", key=f"ip_check_{job.get('batch_name', '')}"):
                            if _check_batch_status_gemini(job, pages_missing_display):
                                st.rerun()
            g1, g2 = st.columns(2)
            with g1:
                gen_first = st.button("Generate first page only", key="ip_gen_first")
            with g2:
                # st.button("Submit Gemini batch (50% cheaper, ~24h)", type="primary", key="ip_batch_gen")
                pass
            gen_all = False  # batch UI commented out; set True + uncomment button to re-enable

            if gen_first and pages_missing_display:
                client = get_gemini()
                if not client:
                    st.error("Set GEMINI_API_KEY in .env.")
                else:
                    first_row = pages_missing_display[0]
                    page_text = (first_row.get("page_text") or first_row.get("text") or "").strip()
                    sug = st.session_state.setdefault("ip_scene_suggestions", {}).get(first_row["id"], "")
                    with st.spinner("Generating first page..."):
                        prompt = build_prompt(
                            sug,
                            page_text,
                            age_appropriateness or "",
                            style_prompt or "",
                            character_ref or "",
                            lighting or "",
                            color_palette or "",
                            framing or "",
                            visual_reference_note=ref_prompt_note,
                        )
                        img, gen_cost = generate_image_gemini(prompt, refs if refs else None)
                    if img:
                        opt = optimize_image_for_mobile(img)
                        pend = st.session_state.get("ip_pending_images", {})
                        pend[first_row["id"]] = opt
                        st.session_state["ip_pending_images"] = pend
                        if gen_cost:
                            _cm = st.session_state.setdefault("ip_last_gen_cost", {})
                            _cm[str(first_row["id"])] = gen_cost
                        st.success("First page generated. Review below.")
                        st.rerun()
                    else:
                        st.error("Generation failed.")

            if gen_all and pages_missing_display:
                client = get_gemini()
                if not client:
                    st.error("Set GEMINI_API_KEY in .env.")
                else:
                    from google.genai import types

                    gen_config = types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="1:1"),
                    )
                    inline_requests = []
                    suggestions = st.session_state.setdefault("ip_scene_suggestions", {})
                    for row in pages_missing_display:
                        page_text = (row.get("page_text") or row.get("text") or "").strip()
                        extra = suggestions.get(row.get("id"), "")
                        prompt = build_prompt(
                            extra,
                            page_text,
                            age_appropriateness or "",
                            style_prompt or "",
                            character_ref or "",
                            lighting or "",
                            color_palette or "",
                            framing or "",
                            visual_reference_note=ref_prompt_note,
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
                            st.success(f"Batch submitted: {batch_name}")
                        else:
                            st.error("Failed to save batch job to database.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Batch submission failed: {e}")

        else:
            st.caption("Leonardo: one API job per page. Submit queue, then **Check all Leonardo jobs**.")
            if not api_key_leo or not model_id:
                st.warning("Set **LEONARDO_API_KEY** and **LEONARDO_MODEL_ID** in .env (model UUID from Leonardo).")
            if pending_leo:
                st.info(f"**{len(pending_leo)}** Leonardo job(s) pending.")
            l1, l2 = st.columns(2)
            with l1:
                gen_first = st.button("Generate first page only", key="ip_gen_first_leo")
            with l2:
                gen_all = st.button("Submit Leonardo queue", type="primary", key="ip_batch_leo")

            if gen_first and pages_missing_display and api_key_leo and model_id:
                first_row = pages_missing_display[0]
                page_text = (first_row.get("page_text") or first_row.get("text") or "").strip()
                sug = st.session_state.setdefault("ip_scene_suggestions", {}).get(first_row["id"], "")
                neg_suf = st.session_state.setdefault("ip_scene_negatives", {}).get(first_row["id"], "")
                pos, neg = build_leonardo_prompt(
                    sug,
                    page_text,
                    age_appropriateness or "",
                    style_prompt or "",
                    character_ref or "",
                    lighting or "",
                    color_palette or "",
                    framing or "",
                    user_negative_suffix=(leo_neg_suffix or "") + (f", {neg_suf}" if neg_suf else ""),
                    visual_reference_note=ref_prompt_note,
                    reading_level=reading_level,
                )
                with st.spinner("Leonardo: generating first page..."):
                    img, gen_cost = generate_image_leonardo(
                        pos,
                        neg,
                        refs if refs else None,
                        api_key=api_key_leo,
                        model_id=model_id,
                        width=int(leo_side),
                        height=int(leo_side),
                        guidance_scale=float(leo_guidance),
                        seed=use_seed,
                        controlnet_strength=leo_cn_strength,
                        contrast=leonardo_contrast,
                    )
                if img:
                    opt = optimize_image_for_mobile(img)
                    pend = st.session_state.get("ip_pending_images", {})
                    pend[first_row["id"]] = opt
                    st.session_state["ip_pending_images"] = pend
                    if gen_cost:
                        _cm = st.session_state.setdefault("ip_last_gen_cost", {})
                        _cm[str(first_row["id"])] = gen_cost
                    st.success("First page generated. Review below.")
                    st.rerun()

            if gen_all and pages_missing_display and api_key_leo and model_id:
                suggestions = st.session_state.setdefault("ip_scene_suggestions", {})
                neg_map = st.session_state.setdefault("ip_scene_negatives", {})
                controlnets = None
                try:
                    if refs:
                        controlnets = []
                        for chunk in refs[:LEONARDO_MAX_CHARACTER_REFERENCE_REFS]:
                            _iid = leo.upload_init_image_bytes(api_key_leo, chunk)
                            controlnets.extend(
                                leo.character_reference_controlnets(
                                    _iid, model_id=model_id, strength_type=leo_cn_strength
                                )
                            )
                    submitted = 0
                    for row in pages_missing_display:
                        page_text = (row.get("page_text") or row.get("text") or "").strip()
                        rid = row.get("id")
                        extra = suggestions.get(rid, "")
                        nxs = neg_map.get(rid, "")
                        pos, neg = build_leonardo_prompt(
                            extra,
                            page_text,
                            age_appropriateness or "",
                            style_prompt or "",
                            character_ref or "",
                            lighting or "",
                            color_palette or "",
                            framing or "",
                            user_negative_suffix=(leo_neg_suffix or "") + (f", {nxs}" if nxs else ""),
                            visual_reference_note=ref_prompt_note,
                            reading_level=reading_level,
                        )
                        gen_id = leo.create_generation(
                            api_key_leo,
                            pos[:1500],
                            model_id,
                            negative_prompt=neg[:1000],
                            width=int(leo_side),
                            height=int(leo_side),
                            num_images=1,
                            guidance_scale=float(leo_guidance),
                            seed=use_seed,
                            controlnets=controlnets,
                            contrast=leonardo_contrast,
                        )
                        if insert_image_generation_job(sb, story_id, reading_level, str(rid), gen_id, "leonardo"):
                            submitted += 1
                        time.sleep(0.35)
                    st.success(f"Submitted **{submitted}** Leonardo job(s). Use **Check all Leonardo jobs** below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Leonardo queue failed: {e}")

            if st.button("Check all Leonardo jobs", key="ip_check_all_leo"):
                if not api_key_leo:
                    st.error("Set LEONARDO_API_KEY.")
                else:
                    jobs = fetch_pending_image_generation_jobs(sb, story_id, reading_level)
                    pend = st.session_state.get("ip_pending_images", {})
                    cost_map = st.session_state.setdefault("ip_last_gen_cost", {})
                    ok = 0
                    for job in jobs:
                        gen = leo.get_generation(api_key_leo, job["external_generation_id"])
                        if not gen:
                            continue
                        stt = (gen.get("status") or "").upper()
                        jid = job.get("id")
                        cid = job.get("story_content_flat_id")
                        if stt == "COMPLETE":
                            imgs = gen.get("generated_images") or gen.get("generatedImages") or []
                            url = (imgs[0] or {}).get("url") if imgs else None
                            if url:
                                try:
                                    raw = leo.download_image(url)
                                    opt = optimize_image_for_mobile(raw)
                                    pend[str(cid)] = opt
                                    _cc = leo.format_generation_cost(gen)
                                    if _cc:
                                        cost_map[str(cid)] = _cc
                                    ok += 1
                                    update_image_generation_job(sb, str(jid), "complete")
                                except Exception as ex:
                                    update_image_generation_job(sb, str(jid), "failed", error_message=str(ex))
                            else:
                                update_image_generation_job(sb, str(jid), "failed", error_message="No image URL")
                        elif stt == "FAILED":
                            update_image_generation_job(sb, str(jid), "failed", error_message="Leonardo FAILED")
                    st.session_state["ip_pending_images"] = pend
                    st.success(f"Resolved jobs; loaded **{ok}** new image(s) into pending approval.")
                    st.rerun()

    all_pages_for_review = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    pending = st.session_state.get("ip_pending_images", {})
    if just_approved:
        for row in all_pages_for_review:
            rid = row.get("id")
            if rid in just_approved:
                row["image_url"] = just_approved[rid]
    if just_cleared_ids:
        for row in all_pages_for_review:
            if _row_id_str(row.get("id")) in just_cleared_ids:
                row["image_url"] = ""

    st.header("4. Review pages")
    work_queue_only = st.checkbox("Work queue only (no image or pending approval)", value=False, key="ip_work_queue")
    oa = _get_openai_client()
    if pages_missing_display and oa and st.button("Generate scene prompts for all missing pages (OpenAI)", key="ip_bulk_scenes"):
        suggestions = st.session_state.setdefault("ip_scene_suggestions", {})
        neg_map = st.session_state.setdefault("ip_scene_negatives", {})
        prog = st.progress(0.0)
        for i, row in enumerate(pages_missing_display):
            pt = (row.get("page_text") or row.get("text") or "").strip()
            data = suggest_scene_prompt(oa, pt, character_ref or "", style_prompt or "", story_title)
            if data:
                suggestions[row["id"]] = merge_scene_to_extra_details(data)
                neg_map[row["id"]] = merge_negative_additions(data)
            prog.progress((i + 1) / max(len(pages_missing_display), 1))
        st.session_state["_scene_ta_sync_ids"] = [r["id"] for r in pages_missing_display]
        st.success("Scene prompts filled. Edit per page below as needed.")
        st.rerun()

    if pending and st.checkbox(f"Confirm bulk approve **{len(pending)}** image(s)", key="ip_bulk_apr_conf"):
        if st.button("Approve all pending now", type="primary", key="ip_bulk_apr_go"):
            pend = dict(pending)
            errs = []
            for row_id, opt in list(pend.items()):
                row = next((r for r in all_pages_for_review if r.get("id") == row_id), None)
                if not row:
                    errs.append(str(row_id))
                    continue
                pidx = int(row.get("page_index", row.get("page_number", 0)))
                url = upload_image_to_storage(sb, story_id, reading_level, pidx, opt)
                if url and update_book_page(sb, row_id, image_url=url):
                    del pend[row_id]
                    st.session_state.setdefault("ip_book_image_overlay", {})[_row_id_str(row_id)] = str(url).strip()
                else:
                    errs.append(f"page {pidx}")
            st.session_state["ip_pending_images"] = pend
            st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
            _bump_ip_cache_version()
            if errs:
                st.warning(f"Some failed: {errs}")
            else:
                st.success("All pending images approved.")
            st.rerun()

    suggestions = st.session_state.setdefault("ip_scene_suggestions", {})
    neg_map = st.session_state.setdefault("ip_scene_negatives", {})

    # Streamlit forbids assigning widget-bound keys after the widget exists; sync before any text_area.
    for _rid in st.session_state.pop("_scene_ta_sync_ids", []):
        _sk = f"ip_scene_ta_{_rid}"
        _txt = st.session_state.get("ip_scene_suggestions", {}).get(_rid, "")
        st.session_state[_sk] = _txt

    for row in all_pages_for_review:
        row_id = row.get("id")
        page_text = (row.get("page_text") or row.get("text") or "").strip()
        page_idx = row.get("page_index", row.get("page_number", 0))
        img_url = (row.get("image_url") or "").strip()
        has_published = bool(img_url)
        has_pending = row_id in pending
        needs_work = (not has_published) or has_pending
        if work_queue_only and not needs_work:
            continue

        page_label = f"✅ Page {page_idx}" if has_published else f"Page {page_idx}"
        with st.expander(page_label, expanded=not has_published):
            st.text(page_text or "(no text)")
            scene_key = f"ip_scene_ta_{row_id}"
            if scene_key not in st.session_state:
                st.session_state[scene_key] = suggestions.get(row_id, "")
            scene_edit = st.text_area(
                "Scene prompt (for image API)",
                key=scene_key,
                height=70,
                help="LLM suggestion + your edits. Used as extra visual details for generation.",
            )
            suggestions[row_id] = scene_edit

            if oa:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Generate scene (LLM)", key=f"ip_scene_gen_{row_id}"):
                        data = suggest_scene_prompt(oa, page_text, character_ref or "", style_prompt or "", story_title)
                        if data:
                            merged = merge_scene_to_extra_details(data)
                            suggestions[row_id] = merged
                            neg_map[row_id] = merge_negative_additions(data)
                            st.session_state["_scene_ta_sync_ids"] = [row_id]
                            st.rerun()
                        else:
                            st.error("LLM scene generation failed.")
                with b2:
                    if st.button("Regenerate scene", key=f"ip_scene_reg_{row_id}"):
                        data = suggest_scene_prompt(oa, page_text, character_ref or "", style_prompt or "", story_title)
                        if data:
                            merged = merge_scene_to_extra_details(data)
                            suggestions[row_id] = merged
                            neg_map[row_id] = merge_negative_additions(data)
                            st.session_state["_scene_ta_sync_ids"] = [row_id]
                            st.rerun()

            img_col, act_col = st.columns([1, 1])
            with img_col:
                if has_pending:
                    st.image(pending[row_id], caption=f"Page {page_idx} (pending)", use_container_width=True)
                    _cost_ln = (st.session_state.get("ip_last_gen_cost") or {}).get(str(row_id))
                    if _cost_ln:
                        st.caption(_cost_ln)
                elif has_published:
                    st.image(img_url, caption=f"Page {page_idx}", use_container_width=True)
                else:
                    st.caption("No image yet.")

            edited_text = st.text_area(
                "Page text (for API only)",
                value=page_text,
                key=f"regen_text_{row_id}",
                height=60,
            )
            correction = st.text_input("Correction (optional)", key=f"regen_corr_{row_id}")
            extra_for_prompt = scene_edit or ""
            neg_suf_row = neg_map.get(row_id, "")
            corr_s = (correction or "").strip()
            extra_bits = ". ".join(p for p in [extra_for_prompt, corr_s] if p)
            gem_prompt = build_prompt(
                extra_bits,
                (edited_text or page_text).strip(),
                age_appropriateness or "",
                style_prompt or "",
                character_ref or "",
                lighting or "",
                color_palette or "",
                framing or "",
                visual_reference_note=ref_prompt_note,
            )
            leo_pos, leo_neg = build_leonardo_prompt(
                extra_for_prompt,
                (edited_text or page_text).strip(),
                age_appropriateness or "",
                style_prompt or "",
                character_ref or "",
                lighting or "",
                color_palette or "",
                framing or "",
                user_negative_suffix=(leo_neg_suffix or "") + (f", {neg_suf_row}" if neg_suf_row else ""),
                visual_reference_note=ref_prompt_note,
                reading_level=reading_level,
            )
            if corr_s:
                leo_pos = f"{leo_pos} {corr_s}".strip()
            with act_col:
                st.caption("Prompt preview")
                if image_provider == "leonardo":
                    st.text_area("Leonardo positive", value=leo_pos[:2000], height=100, disabled=True, key=f"pv_lp_{row_id}")
                    st.text_area("Leonardo negative", value=leo_neg[:1200], height=60, disabled=True, key=f"pv_ln_{row_id}")
                else:
                    st.text_area("Gemini prompt (excerpt)", value=gem_prompt[:2000], height=120, disabled=True, key=f"pv_gp_{row_id}")
                st.download_button(
                    "Download prompt snippet",
                    leo_pos + "\n---\n" + leo_neg if image_provider == "leonardo" else gem_prompt,
                    file_name=f"page_{page_idx}_prompt.txt",
                    key=f"dl_pr_{row_id}",
                )

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if has_pending and st.button("Approve", key=f"approve_btn_{row_id}", type="primary"):
                    opt = pending[row_id]
                    url = upload_image_to_storage(sb, story_id, reading_level, int(page_idx), opt)
                    if url and update_book_page(sb, row_id, image_url=url):
                        del pending[row_id]
                        _cm2 = st.session_state.setdefault("ip_last_gen_cost", {})
                        _cm2.pop(str(row_id), None)
                        st.session_state["ip_pending_images"] = pending
                        st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                        _bump_ip_cache_version()
                        ja = st.session_state.get("ip_just_approved") or {}
                        ja[row_id] = url
                        st.session_state["ip_just_approved"] = ja
                        st.success("Exported to R2 and saved.")
                        st.rerun()
                    else:
                        st.error("Upload or save failed.")
            with btn_col2:
                gen_label = "Regenerate" if (has_pending or has_published) else "Generate"
                if st.button(gen_label, key=f"regen_btn_{row_id}"):
                    text_src = (st.session_state.get(f"regen_text_{row_id}", edited_text) or page_text).strip()
                    extra = (st.session_state.get(f"ip_scene_ta_{row_id}", "") or "").strip()
                    corr = (st.session_state.get(f"regen_corr_{row_id}", "") or "").strip()
                    if image_provider == "gemini":
                        prompt = build_prompt(
                            f"{extra}. {corr}".strip() if extra or corr else corr,
                            text_src,
                            age_appropriateness or "",
                            style_prompt or "",
                            character_ref or "",
                            lighting or "",
                            color_palette or "",
                            framing or "",
                            visual_reference_note=ref_prompt_note,
                        )
                        with st.spinner("Generating..."):
                            img, gen_cost = generate_image_gemini(prompt, refs if refs else None)
                    else:
                        if not api_key_leo or not model_id:
                            st.error("Configure Leonardo API keys.")
                            img = None
                            gen_cost = None
                        else:
                            pos, neg = build_leonardo_prompt(
                                extra,
                                text_src,
                                age_appropriateness or "",
                                style_prompt or "",
                                character_ref or "",
                                lighting or "",
                                color_palette or "",
                                framing or "",
                                user_negative_suffix=(leo_neg_suffix or "")
                                + (f", {neg_map.get(row_id, '')}" if neg_map.get(row_id) else "")
                                + (f", {corr}" if corr else ""),
                                visual_reference_note=ref_prompt_note,
                                reading_level=reading_level,
                            )
                            with st.spinner("Leonardo..."):
                                img, gen_cost = generate_image_leonardo(
                                    pos,
                                    neg,
                                    refs if refs else None,
                                    api_key=api_key_leo,
                                    model_id=model_id,
                                    width=int(leo_side),
                                    height=int(leo_side),
                                    guidance_scale=float(leo_guidance),
                                    seed=use_seed,
                                    controlnet_strength=leo_cn_strength,
                                    contrast=leonardo_contrast,
                                )
                    if img:
                        opt = optimize_image_for_mobile(img)
                        pending[row_id] = opt
                        st.session_state["ip_pending_images"] = pending
                        if gen_cost:
                            _cm = st.session_state.setdefault("ip_last_gen_cost", {})
                            _cm[str(row_id)] = gen_cost
                        st.success("Generated. Approve to save.")
                        st.rerun()
                    elif image_provider == "gemini":
                        st.error("Generation failed.")
            with btn_col3:
                if (has_pending or has_published) and st.button("Clear image", key=f"clear_btn_{row_id}"):
                    if update_book_page(sb, row_id, image_url=""):
                        st.session_state.setdefault("ip_book_image_overlay", {}).pop(_row_id_str(row_id), None)
                        if row_id in pending:
                            del pending[row_id]
                            st.session_state["ip_pending_images"] = pending
                        st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                        _bump_ip_cache_version()
                        jc = list(st.session_state.get("ip_just_cleared") or [])
                        jc.append(row_id)
                        st.session_state["ip_just_cleared"] = jc
                        st.success("Image cleared.")
                        st.rerun()
                    else:
                        st.error("Failed to clear image.")
