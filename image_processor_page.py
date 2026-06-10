"""Image Processor: Leonardo storybook generation with optional reference library."""
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st

import leonardo_client as leo
from auth import get_secret
from grade_style_defaults import (
    character_ref_prompt_for_grade,
    location_ref_prompt_for_grade,
    series_style_prompt_for_grade,
)
from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS
from lib import (
    IMAGE_PROCESSOR_GRADES,
    PAGES_PER_IMAGE,
    apply_image_url_to_block,
    block_needs_image,
    build_simple_leonardo_prompt,
    combined_text_for_pages,
    fetch_book_pages,
    fetch_pages_missing_images,
    fetch_stories,
    find_character_ref_by_id,
    find_image_block_for_row_id,
    find_location_ref_by_id,
    generate_image_leonardo,
    generate_leonardo_reference_preview,
    get_approved_character_refs,
    get_approved_location_refs,
    get_approved_style_ref,
    get_saved_character_refs,
    get_saved_location_refs,
    get_saved_style_ref,
    get_story_grade_style,
    get_supabase,
    group_pages_into_image_blocks,
    optimize_image_for_mobile,
    page_number_for_row,
    save_storybook_references,
    storybook_references_from_grade_style,
    upload_image_to_storage,
    upload_typed_reference_image,
    _get_page_text,
)

_IP_CACHE_TTL = 60
_REF_NONE = ""


def _row_id_str(rid) -> str:
    return str(rid) if rid is not None else ""


def _scope_key(story_id: int, reading_level: str) -> str:
    return f"{story_id}_{reading_level}"


def _pending_ref_key(scope: str, kind: str, ref_id: str = "") -> str:
    suffix = ref_id if ref_id else kind
    return f"ip_ref_pending_{scope}_{kind}_{suffix}"


def _block_for_anchor(image_blocks: list, anchor_row: dict):
    aid = anchor_row.get("id")
    for block in image_blocks:
        if block["anchor_row"].get("id") == aid:
            return block
    members = [anchor_row]
    return {
        "anchor_row": anchor_row,
        "member_rows": members,
        "combined_text": combined_text_for_pages(members),
        "page_range_label": str(page_number_for_row(anchor_row)),
        "block_start": page_number_for_row(anchor_row),
    }


def _overlay_image_url_for_block(block: dict, url: str) -> None:
    overlay = st.session_state.setdefault("ip_book_image_overlay", {})
    u = str(url).strip()
    for member in block["member_rows"]:
        rid = _row_id_str(member.get("id"))
        if rid:
            overlay[rid] = u


def _approve_block_image(sb, story_id, reading_level, block, opt, pending: dict) -> bool:
    anchor = block["anchor_row"]
    row_id = anchor.get("id")
    pidx = page_number_for_row(anchor)
    url = upload_image_to_storage(sb, story_id, reading_level, pidx, opt)
    if not url:
        return False
    ok, _errs = apply_image_url_to_block(sb, block, url)
    if ok <= 0:
        return False
    pending.pop(row_id, None)
    _overlay_image_url_for_block(block, url)
    _cm2 = st.session_state.setdefault("ip_last_gen_cost", {})
    _cm2.pop(str(row_id), None)
    ja = st.session_state.get("ip_just_approved") or {}
    for member in block["member_rows"]:
        sid = _row_id_str(member.get("id"))
        if sid:
            ja[sid] = url
    st.session_state["ip_just_approved"] = ja
    return True


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
def _cached_fetch_book_pages(story_id: int, reading_level: str, language_code: str, _cache_version: int):
    sb = get_supabase()
    return fetch_book_pages(sb, story_id, reading_level, language_code) if sb else []


@st.cache_data(ttl=_IP_CACHE_TTL, show_spinner=False)
def _cached_get_story_grade_style(story_id: int, reading_level: str, _cache_version: int):
    sb = get_supabase()
    return get_story_grade_style(sb, story_id, reading_level) if sb else None


def _bump_ip_cache_version():
    st.session_state["ip_cache_version"] = st.session_state.get("ip_cache_version", 0) + 1


def _stash_storybook_refs(scope: str, refs: Dict[str, Any]) -> None:
    st.session_state[f"ip_refs_{scope}"] = refs


def _load_storybook_refs(scope: str, grade_style) -> Dict[str, Any]:
    refs = storybook_references_from_grade_style(grade_style)
    has_saved = bool(
        get_saved_style_ref(refs)
        or get_saved_character_refs(refs)
        or get_saved_location_refs(refs)
    )
    if has_saved:
        return refs
    stashed = st.session_state.get(f"ip_refs_{scope}")
    return stashed if isinstance(stashed, dict) else refs


def _just_approved_url(just_approved: Dict[Any, Any], row_id) -> str:
    if not just_approved or row_id is None:
        return ""
    return str(just_approved.get(row_id) or just_approved.get(_row_id_str(row_id)) or "").strip()


def _resolve_page_image_url(
    row: dict,
    overlay: Dict[str, str],
    just_approved: Dict[Any, Any],
) -> str:
    sid = _row_id_str(row.get("id"))
    return (
        (overlay.get(sid) or "").strip()
        or _just_approved_url(just_approved, row.get("id"))
        or (row.get("image_url") or "").strip()
    )


def _seed_grade_prompt_defaults(scope: str, reading_level: str) -> None:
    """Pre-fill reference prompt text areas from grade presets when story/grade changes."""
    loaded_for = st.session_state.get("ip_grade_prompts_loaded_for")
    if loaded_for == scope:
        return
    st.session_state["ip_grade_prompts_loaded_for"] = scope
    st.session_state[f"ip_style_prompt_{scope}"] = series_style_prompt_for_grade(reading_level)
    st.session_state[f"ip_char_prompt_{scope}"] = character_ref_prompt_for_grade(reading_level)
    st.session_state[f"ip_loc_prompt_{scope}"] = location_ref_prompt_for_grade(reading_level)


def _style_only_controlnets(refs: Dict[str, Any], model_id: str) -> Optional[List[Dict[str, Any]]]:
    style_ref = get_approved_style_ref(refs)
    if not style_ref:
        return None
    cn = leo.build_partial_storybook_controlnets(style_ref=style_ref, model_id=model_id)
    return cn or None


def _block_char_loc_ids(scope: str, row_id) -> tuple[str, str]:
    char_key = f"ip_char_{row_id}"
    loc_key = f"ip_loc_{row_id}"
    grade_char_key = f"ip_grade_char_{scope}"
    grade_loc_key = f"ip_grade_loc_{scope}"
    char_id = st.session_state.get(char_key, st.session_state.get(grade_char_key, _REF_NONE))
    loc_id = st.session_state.get(loc_key, st.session_state.get(grade_loc_key, _REF_NONE))
    return char_id or _REF_NONE, loc_id or _REF_NONE


def _blocks_ready_for_bulk_generate(
    image_blocks: list, pending: dict, scope: str
) -> List[dict]:
    ready: List[dict] = []
    for block in image_blocks:
        if not block_needs_image(block):
            continue
        row_id = block["anchor_row"].get("id")
        if row_id in pending:
            continue
        scene = (st.session_state.get(f"ip_scene_{row_id}") or "").strip()
        if not scene:
            continue
        ready.append(block)
    return ready


def _generate_block_image(
    api_key: str,
    model_id: str,
    refs: Dict[str, Any],
    scene_description: str,
    character_id: str,
    location_id: str,
    reading_level: str,
) -> tuple[Optional[bytes], Optional[str]]:
    style_ref = get_approved_style_ref(refs)
    character_ref = find_character_ref_by_id(refs, character_id) if character_id else None
    location_ref = find_location_ref_by_id(refs, location_id) if location_id else None
    controlnets = leo.build_partial_storybook_controlnets(
        style_ref, character_ref, location_ref, model_id=model_id
    )
    cn_arg = controlnets if controlnets else None
    pos, neg = build_simple_leonardo_prompt(
        scene_description,
        reading_level,
        character_ref=character_ref,
        location_ref=location_ref,
    )
    defaults = LEONARDO_GENERATION_DEFAULTS
    return generate_image_leonardo(
        pos,
        neg,
        None,
        api_key=api_key,
        model_id=model_id,
        width=int(defaults["width"]),
        height=int(defaults["height"]),
        guidance_scale=float(defaults["guidance_scale"]),
        controlnets=cn_arg,
        preset_style=str(defaults["presetStyle"]),
        alchemy=bool(defaults["alchemy"]),
    )


def _pop_pending_ref(key: str) -> None:
    st.session_state.pop(key, None)


def _get_pending_ref(key: str) -> Optional[Dict[str, Any]]:
    val = st.session_state.get(key)
    return val if isinstance(val, dict) else None


def _set_pending_ref(key: str, payload: Dict[str, Any]) -> None:
    st.session_state[key] = payload


def _approve_style_reference(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    pending: Dict[str, Any],
    scope: str,
) -> bool:
    url = upload_typed_reference_image(sb, story_id, reading_level, "style", "series", pending["bytes"])
    if not url:
        return False
    gen_id = (pending.get("generated_image_id") or "").strip()
    if not gen_id:
        st.error("Missing Leonardo image id for style reference.")
        return False
    refs["style"] = {
        "label": (pending.get("label") or "Series style").strip(),
        "prompt": (pending.get("prompt") or "").strip(),
        "url": url,
        "leonardo_init_image_id": gen_id,
        "approved": True,
    }
    if save_storybook_references(sb, story_id, reading_level, refs):
        _pop_pending_ref(_pending_ref_key(scope, "style"))
        _stash_storybook_refs(scope, refs)
        _bump_ip_cache_version()
        return True
    return False


def _approve_list_reference(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    pending: Dict[str, Any],
    scope: str,
    list_key: str,
    ref_type: str,
    api_key: str,
) -> bool:
    ref_id = (pending.get("ref_id") or "").strip() or uuid.uuid4().hex[:8]
    url = upload_typed_reference_image(sb, story_id, reading_level, ref_type, ref_id, pending["bytes"])
    if not url:
        return False
    if ref_type == "character":
        try:
            leo_id = leo.upload_init_image_bytes(api_key, pending["bytes"])
        except Exception as e:
            st.error(f"Leonardo character upload failed: {e}")
            return False
    else:
        leo_id = (pending.get("generated_image_id") or "").strip()
        if not leo_id:
            st.error("Missing Leonardo image id for location reference.")
            return False
    entry = {
        "id": f"{ref_type[0]}_{ref_id}",
        "label": (pending.get("label") or ref_type.title()).strip(),
        "prompt": (pending.get("prompt") or "").strip(),
        "url": url,
        "leonardo_init_image_id": leo_id,
        "approved": True,
    }
    items = list(refs.get(list_key) or [])
    items.append(entry)
    refs[list_key] = items
    if save_storybook_references(sb, story_id, reading_level, refs):
        _pop_pending_ref(_pending_ref_key(scope, ref_type, pending.get("pending_key_id") or ref_id))
        _stash_storybook_refs(scope, refs)
        _bump_ip_cache_version()
        return True
    return False


def _render_reference_library(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    api_key: str,
    model_id: str,
) -> Dict[str, Any]:
    scope = _scope_key(story_id, reading_level)
    _seed_grade_prompt_defaults(scope, reading_level)
    st.header("2. Reference images (optional)")
    st.caption(
        "Add style, character, and location references for this story and grade. "
        "Prompts below are pre-filled from grade presets (editable). "
        "Generate a preview, approve to save, then optionally pick them when generating scenes."
    )

    # --- Style (0 or 1) ---
    st.subheader("Style reference")
    approved_style = get_saved_style_ref(refs)
    if approved_style:
        st.image(approved_style["url"], caption=approved_style.get("label") or "Series style", width=200)
        if st.button("Remove style reference", key=f"ip_ref_del_style_{scope}"):
            refs["style"] = None
            if save_storybook_references(sb, story_id, reading_level, refs):
                _bump_ip_cache_version()
                st.rerun()
    else:
        sk = _pending_ref_key(scope, "style")
        pending = _get_pending_ref(sk)
        st.text_input("Label", value="Series style", key=f"ip_style_label_{scope}")
        st.text_area("Prompt", key=f"ip_style_prompt_{scope}", height=120)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Generate", key=f"ip_style_gen_{scope}"):
                if not api_key:
                    st.error("Set LEONARDO_API_KEY in .env.")
                else:
                    prompt = (st.session_state.get(f"ip_style_prompt_{scope}") or "").strip()
                    if not prompt:
                        st.warning("Enter a prompt.")
                    else:
                        with st.spinner("Generating style reference…"):
                            img, cost, gen_id = generate_leonardo_reference_preview(
                                prompt, api_key=api_key, model_id=model_id, reading_level=reading_level
                            )
                        if img:
                            _set_pending_ref(
                                sk,
                                {
                                    "bytes": img,
                                    "generated_image_id": gen_id,
                                    "label": st.session_state.get(f"ip_style_label_{scope}", "Series style"),
                                    "prompt": prompt,
                                },
                            )
                            st.rerun()
        if pending:
            st.image(pending["bytes"], caption="Style preview (pending)", width=200)
            if pending.get("generated_image_id"):
                st.caption(f"Leonardo id: {pending['generated_image_id']}")
            with c2:
                if st.button("Approve", key=f"ip_style_appr_{scope}", type="primary"):
                    if _approve_style_reference(sb, story_id, reading_level, refs, pending, scope):
                        st.success("Style reference saved.")
                        st.rerun()
            with c3:
                if st.button("Discard", key=f"ip_style_disc_{scope}"):
                    _pop_pending_ref(sk)
                    st.rerun()

    # --- Characters ---
    st.subheader("Characters")
    for entry in get_saved_character_refs(refs):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.image(entry["url"], caption=entry.get("label") or entry.get("id"), width=160)
        with col_b:
            if st.button("Delete", key=f"ip_ref_del_char_{scope}_{entry.get('id')}"):
                refs["characters"] = [c for c in (refs.get("characters") or []) if c.get("id") != entry.get("id")]
                if save_storybook_references(sb, story_id, reading_level, refs):
                    _bump_ip_cache_version()
                    st.rerun()

    draft_id = st.session_state.setdefault(f"ip_char_draft_id_{scope}", f"c_{uuid.uuid4().hex[:8]}")
    pk = _pending_ref_key(scope, "character", draft_id)
    char_pending = _get_pending_ref(pk)
    st.text_input("Character name", key=f"ip_char_label_{scope}")
    st.text_area("Character prompt", key=f"ip_char_prompt_{scope}", height=70)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        if st.button("Generate", key=f"ip_char_gen_{scope}"):
            if not api_key:
                st.error("Set LEONARDO_API_KEY in .env.")
            else:
                prompt = (st.session_state.get(f"ip_char_prompt_{scope}") or "").strip()
                label = (st.session_state.get(f"ip_char_label_{scope}") or "").strip()
                if not prompt or not label:
                    st.warning("Enter a name and prompt.")
                else:
                    style_cn = _style_only_controlnets(refs, model_id)
                    if not style_cn:
                        st.info("Approve a style reference first for character refs that match the series look.")
                    with st.spinner("Generating character reference…"):
                        img, cost, gen_id = generate_leonardo_reference_preview(
                            prompt,
                            api_key=api_key,
                            model_id=model_id,
                            reading_level=reading_level,
                            controlnets=style_cn,
                        )
                    if img:
                        _set_pending_ref(
                            pk,
                            {
                                "bytes": img,
                                "generated_image_id": gen_id,
                                "label": label,
                                "prompt": prompt,
                                "ref_id": draft_id.split("_", 1)[-1],
                                "pending_key_id": draft_id,
                            },
                        )
                        st.rerun()
    if char_pending:
        st.image(char_pending["bytes"], caption="Character preview (pending)", width=160)
        with cc2:
            if st.button("Approve", key=f"ip_char_appr_{scope}", type="primary"):
                if _approve_list_reference(
                    sb, story_id, reading_level, refs, char_pending, scope, "characters", "character", api_key
                ):
                    st.session_state[f"ip_char_draft_id_{scope}"] = f"c_{uuid.uuid4().hex[:8]}"
                    st.success("Character reference saved.")
                    st.rerun()
        with cc3:
            if st.button("Discard", key=f"ip_char_disc_{scope}"):
                _pop_pending_ref(pk)
                st.rerun()

    # --- Locations ---
    st.subheader("Locations")
    for entry in get_saved_location_refs(refs):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.image(entry["url"], caption=entry.get("label") or entry.get("id"), width=160)
        with col_b:
            if st.button("Delete", key=f"ip_ref_del_loc_{scope}_{entry.get('id')}"):
                refs["locations"] = [loc for loc in (refs.get("locations") or []) if loc.get("id") != entry.get("id")]
                if save_storybook_references(sb, story_id, reading_level, refs):
                    _bump_ip_cache_version()
                    st.rerun()

    loc_draft_id = st.session_state.setdefault(f"ip_loc_draft_id_{scope}", f"l_{uuid.uuid4().hex[:8]}")
    lk = _pending_ref_key(scope, "location", loc_draft_id)
    loc_pending = _get_pending_ref(lk)
    st.text_input("Location name", key=f"ip_loc_label_{scope}")
    st.text_area("Location prompt", key=f"ip_loc_prompt_{scope}", height=70)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if st.button("Generate", key=f"ip_loc_gen_{scope}"):
            if not api_key:
                st.error("Set LEONARDO_API_KEY in .env.")
            else:
                prompt = (st.session_state.get(f"ip_loc_prompt_{scope}") or "").strip()
                label = (st.session_state.get(f"ip_loc_label_{scope}") or "").strip()
                if not prompt or not label:
                    st.warning("Enter a name and prompt.")
                else:
                    style_cn = _style_only_controlnets(refs, model_id)
                    if not style_cn:
                        st.info("Approve a style reference first for location refs that match the series look.")
                    with st.spinner("Generating location reference…"):
                        img, cost, gen_id = generate_leonardo_reference_preview(
                            prompt,
                            api_key=api_key,
                            model_id=model_id,
                            reading_level=reading_level,
                            controlnets=style_cn,
                        )
                    if img:
                        _set_pending_ref(
                            lk,
                            {
                                "bytes": img,
                                "generated_image_id": gen_id,
                                "label": label,
                                "prompt": prompt,
                                "ref_id": loc_draft_id.split("_", 1)[-1],
                                "pending_key_id": loc_draft_id,
                            },
                        )
                        st.rerun()
    if loc_pending:
        st.image(loc_pending["bytes"], caption="Location preview (pending)", width=160)
        with lc2:
            if st.button("Approve", key=f"ip_loc_appr_{scope}", type="primary"):
                if _approve_list_reference(
                    sb, story_id, reading_level, refs, loc_pending, scope, "locations", "location", api_key
                ):
                    st.session_state[f"ip_loc_draft_id_{scope}"] = f"l_{uuid.uuid4().hex[:8]}"
                    st.success("Location reference saved.")
                    st.rerun()
        with lc3:
            if st.button("Discard", key=f"ip_loc_disc_{scope}"):
                _pop_pending_ref(lk)
                st.rerun()

    return refs


def run_image_processor_view():
    st.title("Image Processor")
    st.caption(
        f"Generate one Leonardo image per {PAGES_PER_IMAGE}-page block (same URL saved on each page in the block). "
        "Optional style/character/location references are managed below per story and grade."
    )

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
    if st.session_state.get("ip_reading_level") not in IMAGE_PROCESSOR_GRADES:
        st.session_state["ip_reading_level"] = IMAGE_PROCESSOR_GRADES[0]
    story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
    story_label = st.selectbox("Story", options=list(story_options.keys()), key="ip_story")
    story_id = story_options.get(story_label) if story_label else None
    reading_level = st.selectbox(
        "Reading level",
        options=list(IMAGE_PROCESSOR_GRADES),
        key="ip_reading_level",
        format_func=lambda x: x.replace("_", " ").title(),
    )

    if not story_id:
        st.info("Select a story to continue.")
        return

    scope = _scope_key(story_id, reading_level)
    grade_style = _cached_get_story_grade_style(story_id, reading_level, _v)
    refs = _load_storybook_refs(scope, grade_style)

    api_key_leo = (get_secret("LEONARDO_API_KEY") or "").strip()
    model_id = (get_secret("LEONARDO_MODEL_ID") or "").strip() or str(LEONARDO_GENERATION_DEFAULTS["modelId"])
    if not api_key_leo:
        st.warning("Set **LEONARDO_API_KEY** in `.env` to generate images.")

    refs = _render_reference_library(sb, story_id, reading_level, refs, api_key_leo, model_id)
    _stash_storybook_refs(scope, refs)

    just_approved = st.session_state.pop("ip_just_approved", None) or {}
    just_cleared = set(st.session_state.pop("ip_just_cleared", None) or [])
    just_cleared_ids = {_row_id_str(x) for x in just_cleared}
    _img_overlay = st.session_state.setdefault("ip_book_image_overlay", {})
    for _rid, _u in just_approved.items():
        sid = _row_id_str(_rid)
        if _u and sid:
            _img_overlay[sid] = str(_u).strip()

    _v_missing = st.session_state.get("ip_pages_missing_version", 0)
    pages_missing = _cached_fetch_pages_missing_images(story_id, language_code, reading_level, _v, _v_missing)
    pages_missing_display = [
        r for r in pages_missing if not _resolve_page_image_url(r, _img_overlay, just_approved)
    ]
    all_pages_for_counts = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    image_blocks = group_pages_into_image_blocks(all_pages_for_counts)
    missing_blocks = [b for b in image_blocks if block_needs_image(b)]
    n_pages_in_missing_blocks = sum(len(b["member_rows"]) for b in missing_blocks)
    pending = st.session_state.get("ip_pending_images", {})

    n_missing = len(pages_missing_display)
    n_pending_approval = len(pending)

    st.markdown(
        f"**Context:** story **{story_id}** · **{reading_level}** · **{language_code}** · "
        f"images missing **{n_missing}** ({n_pages_in_missing_blocks} pages) · "
        f"pending approval **{n_pending_approval}**"
    )

    if n_missing:
        st.success(
            f"**{n_missing}** image(s) missing — covers **{n_pages_in_missing_blocks}** pages "
            f"({PAGES_PER_IMAGE} pages share each image)."
        )
    else:
        st.info("No images missing for this story + reading level. You can still regenerate or review.")

    char_options = get_approved_character_refs(refs)
    loc_options = get_approved_location_refs(refs)
    char_ids = [_REF_NONE] + [c["id"] for c in char_options]
    loc_ids = [_REF_NONE] + [loc["id"] for loc in loc_options]
    char_labels = {_REF_NONE: "None", **{c["id"]: c.get("label") or c["id"] for c in char_options}}
    loc_labels = {_REF_NONE: "None", **{loc["id"]: loc.get("label") or loc["id"] for loc in loc_options}}

    all_pages_for_review = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    for row in all_pages_for_review:
        resolved = _resolve_page_image_url(row, _img_overlay, just_approved)
        if _row_id_str(row.get("id")) in just_cleared_ids:
            row["image_url"] = ""
        elif resolved:
            row["image_url"] = resolved

    image_blocks_review = group_pages_into_image_blocks(all_pages_for_review)

    st.header("3. Generate scene images")
    st.caption(
        "Character and location references strongly affect output — leave as **None** unless a scene needs them. "
        "Scene description is always primary."
    )
    grade_char_key = f"ip_grade_char_{scope}"
    grade_loc_key = f"ip_grade_loc_{scope}"
    if grade_char_key not in st.session_state:
        st.session_state[grade_char_key] = _REF_NONE
    if grade_loc_key not in st.session_state:
        st.session_state[grade_loc_key] = _REF_NONE
    gc1, gc2 = st.columns(2)
    with gc1:
        st.selectbox(
            "Default character for new blocks",
            options=char_ids,
            format_func=lambda x: char_labels.get(x, x),
            key=grade_char_key,
        )
    with gc2:
        st.selectbox(
            "Default location for new blocks",
            options=loc_ids,
            format_func=lambda x: loc_labels.get(x, x),
            key=grade_loc_key,
        )

    work_queue_only = st.checkbox("Work queue only (no image or pending approval)", value=False, key="ip_work_queue")

    bulk_ready = _blocks_ready_for_bulk_generate(image_blocks_review, pending, scope)
    if bulk_ready:
        st.caption(
            f"**{len(bulk_ready)}** block(s) ready for bulk generate "
            f"(missing image, scene description filled, not already pending)."
        )
    if bulk_ready and st.checkbox(
        f"Confirm bulk generate **{len(bulk_ready)}** block(s) into pending queue",
        key="ip_bulk_gen_conf",
    ):
        if st.button("Generate all missing now", type="primary", key="ip_bulk_gen_go"):
            if not api_key_leo:
                st.error("Set LEONARDO_API_KEY in .env.")
            else:
                prog = st.progress(0.0, text="Starting bulk generate…")
                errs: List[str] = []
                ok_count = 0
                total = len(bulk_ready)
                for i, block in enumerate(bulk_ready):
                    row_id = block["anchor_row"].get("id")
                    page_range = block["page_range_label"]
                    prog.progress(
                        i / total,
                        text=f"Generating {i + 1}/{total}: pages {page_range}…",
                    )
                    scene = (st.session_state.get(f"ip_scene_{row_id}") or "").strip()
                    char_id, loc_id = _block_char_loc_ids(scope, row_id)
                    img, gen_cost = _generate_block_image(
                        api_key_leo,
                        model_id,
                        refs,
                        scene,
                        char_id,
                        loc_id,
                        reading_level,
                    )
                    if img:
                        opt = optimize_image_for_mobile(img)
                        pending[row_id] = opt
                        if gen_cost:
                            _cm = st.session_state.setdefault("ip_last_gen_cost", {})
                            _cm[str(row_id)] = gen_cost
                        ok_count += 1
                    else:
                        errs.append(f"pages {page_range}")
                st.session_state["ip_pending_images"] = pending
                prog.progress(1.0, text="Bulk generate finished.")
                if errs:
                    st.warning(f"Generated **{ok_count}/{total}**. Failed: {', '.join(errs)}")
                else:
                    st.success(f"Generated **{ok_count}** image(s) into pending queue. Review below, then bulk approve.")
                st.rerun()

    if pending and st.checkbox(f"Confirm bulk approve **{len(pending)}** image(s)", key="ip_bulk_apr_conf"):
        if st.button("Approve all pending now", type="primary", key="ip_bulk_apr_go"):
            pend = dict(pending)
            errs = []
            for row_id, opt in list(pend.items()):
                block = find_image_block_for_row_id(image_blocks_review, row_id)
                if not block:
                    errs.append(str(row_id))
                    continue
                if _approve_block_image(sb, story_id, reading_level, block, opt, pend):
                    pass
                else:
                    errs.append(f"pages {block['page_range_label']}")
            st.session_state["ip_pending_images"] = pend
            st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
            _bump_ip_cache_version()
            if errs:
                st.warning(f"Some failed: {errs}")
            else:
                st.success("All pending images approved (URL copied to each page in each block).")
            st.rerun()

    for block in image_blocks_review:
        row = block["anchor_row"]
        row_id = row.get("id")
        page_text = block["combined_text"]
        page_range = block["page_range_label"]
        member_urls = [_resolve_page_image_url(m, _img_overlay, just_approved) for m in block["member_rows"]]
        img_url = member_urls[0] if member_urls else ""
        has_published = bool(img_url) and all(member_urls)
        has_pending = row_id in pending
        needs_work = block_needs_image(block) or has_pending
        if work_queue_only and not needs_work:
            continue

        page_label = f"✅ Pages {page_range}" if has_published else f"Pages {page_range}"
        with st.expander(page_label, expanded=not has_published):
            for member in block["member_rows"]:
                pn = page_number_for_row(member)
                mt = _get_page_text(member)
                st.caption(f"**Page {pn}:** {(mt or '(no text)')[:200]}{'…' if len(mt or '') > 200 else ''}")

            scene_key = f"ip_scene_{row_id}"
            if scene_key not in st.session_state:
                st.session_state[scene_key] = page_text or ""
            st.text_area("Scene Description", key=scene_key, height=100)

            char_key = f"ip_char_{row_id}"
            loc_key = f"ip_loc_{row_id}"
            if char_key not in st.session_state:
                st.session_state[char_key] = st.session_state.get(grade_char_key, _REF_NONE)
            if loc_key not in st.session_state:
                st.session_state[loc_key] = st.session_state.get(grade_loc_key, _REF_NONE)

            st.selectbox(
                "Character",
                options=char_ids,
                format_func=lambda x: char_labels.get(x, x),
                key=char_key,
            )
            st.selectbox(
                "Location",
                options=loc_ids,
                format_func=lambda x: loc_labels.get(x, x),
                key=loc_key,
            )

            img_col, act_col = st.columns([1, 1])
            with img_col:
                if has_pending:
                    st.image(
                        pending[row_id],
                        caption=f"Pages {page_range} (pending — saves to all {len(block['member_rows'])} pages)",
                        use_container_width=True,
                    )
                    _cost_ln = (st.session_state.get("ip_last_gen_cost") or {}).get(str(row_id))
                    if _cost_ln:
                        st.caption(_cost_ln)
                elif has_published:
                    st.image(img_url, caption=f"Pages {page_range}", use_container_width=True)
                else:
                    st.caption("No image yet.")

            with act_col:
                if st.button("Generate", key=f"gen_btn_{row_id}", type="primary"):
                    if not api_key_leo:
                        st.error("Set LEONARDO_API_KEY in .env.")
                    else:
                        scene = (st.session_state.get(scene_key) or "").strip()
                        if not scene:
                            st.warning("Enter a scene description.")
                        else:
                            char_id, loc_id = _block_char_loc_ids(scope, row_id)
                            with st.spinner("Leonardo: generating…"):
                                img, gen_cost = _generate_block_image(
                                    api_key_leo,
                                    model_id,
                                    refs,
                                    scene,
                                    char_id,
                                    loc_id,
                                    reading_level,
                                )
                            if img:
                                opt = optimize_image_for_mobile(img)
                                pending[row_id] = opt
                                st.session_state["ip_pending_images"] = pending
                                if gen_cost:
                                    _cm = st.session_state.setdefault("ip_last_gen_cost", {})
                                    _cm[str(row_id)] = gen_cost
                                st.success("Generated. Approve to save on all pages in this block.")
                                st.rerun()

                if has_pending and st.button("Approve", key=f"approve_btn_{row_id}"):
                    if _approve_block_image(sb, story_id, reading_level, block, pending[row_id], pending):
                        st.session_state["ip_pending_images"] = pending
                        st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                        _bump_ip_cache_version()
                        st.success(f"Exported to R2 and saved on pages {page_range}.")
                        st.rerun()
                    else:
                        st.error("Upload or save failed.")

                if (has_pending or has_published) and st.button("Clear image", key=f"clear_btn_{row_id}"):
                    ok, _ = apply_image_url_to_block(sb, block, "")
                    if ok > 0:
                        overlay = st.session_state.setdefault("ip_book_image_overlay", {})
                        for member in block["member_rows"]:
                            overlay.pop(_row_id_str(member.get("id")), None)
                        if row_id in pending:
                            del pending[row_id]
                            st.session_state["ip_pending_images"] = pending
                        st.session_state["ip_pages_missing_version"] = st.session_state.get("ip_pages_missing_version", 0) + 1
                        _bump_ip_cache_version()
                        jc = list(st.session_state.get("ip_just_cleared") or [])
                        for member in block["member_rows"]:
                            mid = member.get("id")
                            if mid is not None:
                                jc.append(mid)
                        st.session_state["ip_just_cleared"] = jc
                        st.success(f"Image cleared from pages {page_range}.")
                        st.rerun()
                    else:
                        st.error("Failed to clear image.")
