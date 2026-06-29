"""Image Processor: Leonardo storybook generation with optional reference library."""
import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional

import streamlit as st

import leonardo_client as leo
from auth import get_secret
from grade_style_defaults import (
    character_ref_prompt_for_grade,
    grade_scene_settings_for_prompt,
    series_style_prompt_for_grade,
)
from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS, STYLE_CONTROLNET_SCENE_STRENGTH
from image_pipeline import (
    CREDITS_EXHAUSTED_ERROR,
    approve_pipeline_item,
    create_pipeline_run,
    discover_work_items,
    requeue_pipeline_item,
    resume_pipeline,
    run_pipeline,
)
from scene_prompts import (
    generate_all_scene_descriptions,
    generate_scene_description_paragraph,
    generate_style_scene_description,
)
from translator_page import get_openai
from lib import (
    IMAGE_PROCESSOR_GRADES,
    PAGES_PER_IMAGE,
    apply_image_url_to_block,
    block_needs_image,
    build_simple_leonardo_prompt,
    combined_text_for_pages,
    fetch_book_pages,
    fetch_pages_missing_images,
    fetch_pipeline_failed_items,
    fetch_pipeline_review_queue,
    get_active_pipeline_run,
    pipeline_tables_ready,
    PIPELINE_SCHEMA_SETUP_MSG,
    update_pipeline_item,
    fetch_stories,
    find_character_ref_by_id,
    find_image_block_for_row_id,
    formatted_page_text_block,
    generate_image_leonardo,
    generate_leonardo_reference_preview,
    get_approved_character_refs,
    get_approved_style_ref,
    get_saved_character_refs,
    get_saved_style_ref,
    get_story_grade_style,
    get_supabase,
    group_pages_into_image_blocks,
    optimize_image_for_mobile,
    page_number_for_row,
    pending_image_display_url,
    save_storybook_references,
    storybook_references_from_grade_style,
    upload_image_to_storage,
    upload_typed_reference_image,
    _get_page_text,
)

_IP_CACHE_TTL = 60
_REF_NONE = ""

# region agent log
_DEBUG_LOG_PATH = "/Users/benwaddell/cursor projects/storybook-image-processor/.cursor/debug-81a77e.log"
_DEBUG_SESSION_ID = "81a77e"


def _dbg_ui(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
# endregion


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
    _cached_get_story_grade_style.clear()


def _stash_storybook_refs(scope: str, refs: Dict[str, Any]) -> None:
    st.session_state[f"ip_refs_{scope}"] = refs


def _load_storybook_refs(scope: str, grade_style) -> Dict[str, Any]:
    return storybook_references_from_grade_style(grade_style)


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
    """Pre-fill prompt text areas from grade presets when story/grade changes."""
    loaded_for = st.session_state.get("ip_grade_prompts_loaded_for")
    if loaded_for == scope:
        return
    st.session_state["ip_grade_prompts_loaded_for"] = scope
    style_key = _style_scene_session_key(scope)
    if not (st.session_state.get(style_key) or "").strip():
        st.session_state[style_key] = series_style_prompt_for_grade(reading_level)
    st.session_state[f"ip_char_prompt_{scope}"] = character_ref_prompt_for_grade(reading_level)


def _style_only_controlnets(refs: Dict[str, Any], model_id: str) -> Optional[List[Dict[str, Any]]]:
    style_ref = get_approved_style_ref(refs)
    if not style_ref:
        return None
    cn = leo.build_partial_storybook_controlnets(style_ref=style_ref, model_id=model_id)
    return cn or None


def _block_char_id(scope: str, row_id) -> str:
    char_key = f"ip_char_{row_id}"
    grade_char_key = f"ip_grade_char_{scope}"
    char_id = st.session_state.get(char_key, st.session_state.get(grade_char_key, _REF_NONE))
    return char_id or _REF_NONE


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


def _story_meta(stories: list, story_id: int) -> Dict[str, str]:
    for s in stories:
        if s.get("id") == story_id:
            return {
                "title": (s.get("title") or "").strip(),
                "description": (s.get("description") or "").strip(),
            }
    return {"title": "", "description": ""}


def _ref_text_for_block(
    refs: Dict[str, Any], scope: str, row_id,
) -> str:
    char_id = _block_char_id(scope, row_id)
    ref = find_character_ref_by_id(refs, char_id) if char_id else None
    if not ref:
        return ""
    label = (ref.get("label") or "").strip()
    prompt = (ref.get("prompt") or "").strip()
    if label and prompt:
        return f"{label}: {prompt}"
    return label or prompt


def _all_character_refs_text(refs: Dict[str, Any]) -> str:
    lines: List[str] = []
    for entry in get_approved_character_refs(refs):
        label = (entry.get("label") or "").strip()
        prompt = (entry.get("prompt") or "").strip()
        if label and prompt:
            lines.append(f"- {label}: {prompt}")
        elif label:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _block_page_text(block: dict) -> str:
    """Per-page labeled text for this block (preferred over combined_text alone)."""
    members = block.get("member_rows") or []
    labeled = formatted_page_text_block(members)
    if labeled:
        return labeled
    return (block.get("combined_text") or "").strip()


def _block_prompt_payload(block: dict) -> dict:
    pages = []
    for member in block.get("member_rows") or []:
        pages.append(
            {
                "page_number": page_number_for_row(member),
                "text": _get_page_text(member),
            }
        )
    page_text = _block_page_text(block)
    return {
        "block_start": block.get("block_start"),
        "page_range_label": block.get("page_range_label") or "",
        "combined_text": page_text or block.get("combined_text") or "",
        "pages": pages,
    }


def _style_scene_session_key(scope: str) -> str:
    return f"ip_story_style_scene_{scope}"


def _get_style_scene(scope: str) -> str:
    return (st.session_state.get(_style_scene_session_key(scope)) or "").strip()


def _generate_style_scene_for_story(
    blocks: List[dict],
    *,
    story_title: str,
    story_summary: str,
    reading_level: str,
    grade_style: Optional[dict],
) -> Optional[str]:
    client = get_openai()
    if not client or not blocks:
        return None
    payloads = [_block_prompt_payload(b) for b in blocks]
    return generate_style_scene_description(
        client,
        payloads,
        story_title=story_title,
        story_summary=story_summary,
        default_style_scene=series_style_prompt_for_grade(reading_level),
        grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
    )


def _generate_scenes_for_full_story(
    blocks: List[dict],
    *,
    story_title: str,
    story_summary: str,
    reading_level: str,
    grade_style: Optional[dict],
    refs: Dict[str, Any],
    scope: str,
) -> Optional[Dict[int, str]]:
    client = get_openai()
    if not client or not blocks:
        return None
    payloads = [_block_prompt_payload(b) for b in blocks]
    return generate_all_scene_descriptions(
        client,
        payloads,
        story_title=story_title,
        story_summary=story_summary,
        character_refs=_all_character_refs_text(refs),
        style_scene=_get_style_scene(scope),
        grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
        pages_per_image=PAGES_PER_IMAGE,
    )


def _generate_scene_for_block(
    block: dict,
    *,
    story_title: str,
    story_summary: str,
    reading_level: str,
    grade_style: Optional[dict],
    refs: Dict[str, Any],
    scope: str,
) -> Optional[str]:
    client = get_openai()
    if not client:
        return None
    row_id = block["anchor_row"].get("id")
    page_text = _block_page_text(block)
    return generate_scene_description_paragraph(
        client,
        page_text,
        story_title=story_title,
        story_summary=story_summary,
        character_ref=_ref_text_for_block(refs, scope, row_id),
        style_scene=_get_style_scene(scope),
        grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
        page_range_label=block.get("page_range_label") or "",
    )


def _generate_block_image(
    api_key: str,
    model_id: str,
    refs: Dict[str, Any],
    scene_description: str,
    character_id: str,
    reading_level: str,
    style_scene: str = "",
    page_context: str = "",
    page_range_label: str = "",
) -> tuple[Optional[bytes], Optional[str]]:
    style_ref = get_approved_style_ref(refs)
    character_ref = find_character_ref_by_id(refs, character_id) if character_id else None
    controlnets = leo.build_partial_storybook_controlnets(
        style_ref,
        character_ref,
        None,
        model_id=model_id,
        style_strength=STYLE_CONTROLNET_SCENE_STRENGTH if style_ref else None,
    )
    cn_arg = controlnets if controlnets else None
    pos, neg = build_simple_leonardo_prompt(
        scene_description,
        reading_level,
        style_scene=style_scene or None,
        has_style_controlnet=bool(style_ref),
        page_context=page_context or None,
        page_range_label=page_range_label or None,
        character_ref=character_ref,
        location_ref=None,
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


def _refs_with_style_cleared(refs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "style": None,
        "characters": list(refs.get("characters") or []),
        "locations": list(refs.get("locations") or []),
    }


def _remove_saved_style_reference(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    scope: str,
) -> bool:
    cleared = _refs_with_style_cleared(refs)
    if not save_storybook_references(sb, story_id, reading_level, cleared):
        return False
    refs.clear()
    refs.update(cleared)
    _pop_pending_ref(_pending_ref_key(scope, "style"))
    _stash_storybook_refs(scope, refs)
    _bump_ip_cache_version()
    return True


def _generate_style_ref_preview(
    scope: str,
    *,
    api_key: str,
    model_id: str,
    reading_level: str,
) -> bool:
    prompt = _get_style_scene(scope)
    if not prompt:
        st.warning("Enter or generate a style scene description first.")
        return False
    with st.spinner("Generating style reference…"):
        img, _cost, gen_id = generate_leonardo_reference_preview(
            prompt,
            api_key=api_key,
            model_id=model_id,
            reading_level=reading_level,
            ref_kind="style",
        )
    if not img:
        st.error("Style reference generation failed.")
        return False
    sk = _pending_ref_key(scope, "style")
    _set_pending_ref(
        sk,
        {
            "bytes": img,
            "generated_image_id": gen_id,
            "label": st.session_state.get(f"ip_style_label_{scope}", "Series style"),
            "prompt": prompt,
        },
    )
    return True


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


def _render_style_scene_section(
    sb,
    story_id: int,
    reading_level: str,
    scope: str,
    refs: Dict[str, Any],
    *,
    story_meta: Dict[str, str],
    grade_style: Optional[dict],
    image_blocks: List[dict],
    api_key: str,
    model_id: str,
) -> Dict[str, Any]:
    _seed_grade_prompt_defaults(scope, reading_level)
    st.header("2. Story style scene")
    st.caption(
        "ChatGPT reads the **full story** plus your **grade default scene settings** and writes one "
        "style scene description. This text is sent to Leonardo with every illustration and used for the optional style reference image."
    )

    grade_settings = grade_scene_settings_for_prompt(reading_level, grade_style)
    with st.expander("Grade default scene settings (ChatGPT uses these as a baseline)"):
        st.text(grade_settings or "(No grade settings found.)")

    style_scene_key = _style_scene_session_key(scope)
    openai_client = get_openai()
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        if not openai_client:
            st.caption("Set **OPENAI_API_KEY** in `.env` to generate.")
        elif image_blocks and st.button(
            "Generate from full story",
            key="ip_gen_style_scene",
            type="primary",
        ):
            with st.spinner("ChatGPT: reading full story and writing style scene…"):
                style_scene = _generate_style_scene_for_story(
                    image_blocks,
                    story_title=story_meta["title"],
                    story_summary=story_meta["description"],
                    reading_level=reading_level,
                    grade_style=grade_style,
                )
            if style_scene:
                st.session_state[style_scene_key] = style_scene
                st.success("Style scene generated. Review below, then generate illustration moments in section 4.")
                st.rerun()
            else:
                st.error("Could not generate style scene. Check OPENAI_API_KEY and that the story has page text.")

    st.text_area(
        "Style scene description",
        key=style_scene_key,
        height=140,
        help="Story-specific visual world for Leonardo: places, lighting, age-appropriate mood.",
    )

    st.subheader("Style reference image (optional)")
    st.caption("Generate a reference image from the style scene above. Regenerate until you are happy, then approve.")
    sk = _pending_ref_key(scope, "style")
    pending = _get_pending_ref(sk)
    approved_style = get_saved_style_ref(refs)
    st.text_input("Label", value="Series style", key=f"ip_style_label_{scope}")

    if approved_style and not pending:
        st.image(approved_style["url"], caption=approved_style.get("label") or "Series style", width=200)
        rm_col, rep_col = st.columns(2)
        with rm_col:
            if st.button("Remove style reference", key=f"ip_ref_del_style_{scope}"):
                if _remove_saved_style_reference(sb, story_id, reading_level, refs, scope):
                    st.success("Style reference removed.")
                    st.rerun()
                else:
                    st.error("Failed to remove style reference.")
        with rep_col:
            if st.button("Replace style image", key=f"ip_ref_replace_style_{scope}"):
                if not api_key:
                    st.error("Set LEONARDO_API_KEY in .env.")
                elif _remove_saved_style_reference(sb, story_id, reading_level, refs, scope):
                    if _generate_style_ref_preview(
                        scope, api_key=api_key, model_id=model_id, reading_level=reading_level
                    ):
                        st.rerun()
    elif pending:
        st.image(pending["bytes"], caption="Style preview (pending — approve or regenerate)", width=200)
        if pending.get("generated_image_id"):
            st.caption(f"Leonardo id: {pending['generated_image_id']}")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Regenerate", key=f"ip_style_regen_{scope}"):
                if not api_key:
                    st.error("Set LEONARDO_API_KEY in .env.")
                elif _generate_style_ref_preview(
                    scope, api_key=api_key, model_id=model_id, reading_level=reading_level
                ):
                    st.rerun()
        with b2:
            if st.button("Approve", key=f"ip_style_appr_{scope}", type="primary"):
                if _approve_style_reference(sb, story_id, reading_level, refs, pending, scope):
                    st.success("Style reference saved.")
                    st.rerun()
        with b3:
            if st.button("Discard", key=f"ip_style_disc_{scope}"):
                _pop_pending_ref(sk)
                st.rerun()
    else:
        if st.button("Generate style image", key=f"ip_style_gen_{scope}"):
            if not api_key:
                st.error("Set LEONARDO_API_KEY in .env.")
            elif _generate_style_ref_preview(
                scope, api_key=api_key, model_id=model_id, reading_level=reading_level
            ):
                st.rerun()

    return refs


def _render_reference_library(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    api_key: str,
    model_id: str,
) -> Dict[str, Any]:
    scope = _scope_key(story_id, reading_level)
    st.header("3. Character references (optional)")
    st.caption(
        "Add character references for this story and grade. "
        "Generate a preview, approve to save, then optionally pick them when generating scenes."
    )

    # --- Characters ---
    for entry in get_saved_character_refs(refs):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.image(entry["url"], caption=entry.get("label") or entry.get("id"), width=160)
        with col_b:
            if st.button("Delete", key=f"ip_ref_del_char_{scope}_{entry.get('id')}"):
                refs["characters"] = [c for c in (refs.get("characters") or []) if c.get("id") != entry.get("id")]
                if save_storybook_references(sb, story_id, reading_level, refs):
                    _stash_storybook_refs(scope, refs)
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
                            ref_kind="character",
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

    return refs


def _render_batch_automation_section(sb, stories: List[dict]) -> None:
    st.header("0. Batch automation")
    st.caption(
        "Scan all stories for missing images (grades 1–4, English), run the full pipeline "
        "(style scene, refs, illustration moments, Leonardo), then review and approve here."
    )

    ready, setup_msg = pipeline_tables_ready(sb)
    if not ready:
        st.warning(setup_msg or PIPELINE_SCHEMA_SETUP_MSG)
        with st.expander("SQL to run in Supabase", expanded=True):
            st.markdown(
                "Open your Supabase project → **SQL Editor** → New query, then paste and run each file from this repo:"
            )
            st.code("supabase_image_pipeline.sql", language=None)
            st.code("supabase_story_grade_styles_style_scene.sql", language=None)
        return

    if "ip_batch_work_items" not in st.session_state:
        st.session_state["ip_batch_work_items"] = None

    scan_col, _ = st.columns([1, 3])
    with scan_col:
        if st.button("Scan all stories for missing images", key="ip_batch_scan"):
            items = discover_work_items(sb, language_code="en")
            # region agent log
            _dbg_ui(
                run_id="discover",
                hypothesis_id="H1",
                location="image_processor_page.py:_render_batch_automation_section",
                message="Scan button completed",
                data={"itemsCount": len(items)},
            )
            # endregion
            st.session_state["ip_batch_work_items"] = items
            st.rerun()

    work_items = st.session_state.get("ip_batch_work_items")
    if work_items is not None:
        if not work_items:
            st.info("No missing images found across all stories and grades.")
        else:
            total_blocks = sum(w.missing_block_count for w in work_items)
            st.markdown(f"**{len(work_items)}** story/grade version(s) · **{total_blocks}** missing block(s)")
            rows = [
                {
                    "Story": w.story_title,
                    "ID": w.story_id,
                    "Grade": w.reading_level.replace("_", " "),
                    "Missing blocks": w.missing_block_count,
                }
                for w in work_items
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    active_run = get_active_pipeline_run(sb)
    if active_run:
        run_id = str(active_run["id"])
        status = active_run.get("status") or ""
        st.markdown(
            f"**Active run:** `{run_id[:8]}…` · status **{status}** · "
            f"done **{active_run.get('items_done', 0)}** / **{active_run.get('items_total', 0)}** · "
            f"failed **{active_run.get('items_failed', 0)}**"
        )
        if active_run.get("leonardo_spend_note"):
            st.caption(active_run["leonardo_spend_note"][:500])

        if status == "credits_exhausted":
            st.error(active_run.get("last_error") or CREDITS_EXHAUSTED_ERROR)
            if st.button("Resume batch", key="ip_batch_resume", type="primary"):
                prog = st.progress(0.0, text="Resuming batch…")
                status_box = st.empty()

                def _on_progress(msg: str, done: int, total: int) -> None:
                    frac = (done / total) if total else 0.0
                    prog.progress(min(1.0, frac), text=msg)
                    status_box.caption(f"{done}/{total} — {msg}")

                result = resume_pipeline(sb, run_id, progress_callback=_on_progress)
                prog.progress(1.0, text="Done")
                if result.status == "credits_exhausted":
                    st.error(result.last_error or CREDITS_EXHAUSTED_ERROR)
                elif result.status == "completed":
                    st.success(result.message or "Batch complete.")
                    st.session_state["ip_batch_work_items"] = discover_work_items(sb, language_code="en")
                    st.rerun()
                else:
                    st.warning(result.message or f"Run ended: {result.status}")

        elif status == "running":
            st.warning("A batch run has queued items remaining.")
            if st.button("Continue batch", key="ip_batch_continue", type="primary"):
                prog = st.progress(0.0, text="Continuing batch…")
                status_box = st.empty()

                def _on_progress_cont(msg: str, done: int, total: int) -> None:
                    frac = (done / total) if total else 0.0
                    prog.progress(min(1.0, frac), text=msg)
                    status_box.caption(f"{done}/{total} — {msg}")

                result = run_pipeline(sb, run_id, progress_callback=_on_progress_cont)
                prog.progress(1.0, text="Done")
                if result.status == "credits_exhausted":
                    st.error(result.last_error or CREDITS_EXHAUSTED_ERROR)
                elif result.status == "completed":
                    st.success(result.message or "Batch complete.")
                    st.session_state["ip_batch_work_items"] = discover_work_items(sb, language_code="en")
                    st.rerun()
                else:
                    st.warning(result.message or f"Run ended: {result.status}")

    if work_items:
        total_blocks = sum(w.missing_block_count for w in work_items)
        confirm = st.checkbox(
            f"Confirm batch run on **{total_blocks}** missing block(s)",
            key="ip_batch_confirm",
        )
        if confirm and st.button("Run batch pipeline", key="ip_batch_run", type="primary"):
            if active_run and active_run.get("status") in ("running", "credits_exhausted"):
                st.error("An active batch run exists. Use Continue or Resume above.")
            else:
                run_id = create_pipeline_run(sb, work_items, language_code="en")
                # region agent log
                _dbg_ui(
                    run_id=str(run_id or "no_run"),
                    hypothesis_id="H3",
                    location="image_processor_page.py:_render_batch_automation_section",
                    message="Run batch clicked",
                    data={
                        "workItemsCount": len(work_items),
                        "totalBlocksRequested": total_blocks,
                        "runIdCreated": bool(run_id),
                    },
                )
                # endregion
                if not run_id:
                    st.error("Nothing to enqueue (missing blocks may already be in the review queue).")
                else:
                    prog = st.progress(0.0, text="Starting batch…")
                    status_box = st.empty()

                    def _on_progress(msg: str, done: int, total: int) -> None:
                        frac = (done / total) if total else 0.0
                        prog.progress(min(1.0, frac), text=msg)
                        status_box.caption(f"{done}/{total} — {msg}")

                    result = run_pipeline(sb, run_id, progress_callback=_on_progress)
                    prog.progress(1.0, text="Done")
                    if result.status == "credits_exhausted":
                        st.error(result.last_error or CREDITS_EXHAUSTED_ERROR)
                    elif result.status == "completed":
                        st.success(
                            f"Batch complete — **{result.items_done}** generated, "
                            f"**{result.items_failed}** failed. Review below."
                        )
                        st.session_state["ip_batch_work_items"] = discover_work_items(sb, language_code="en")
                    else:
                        st.warning(result.message or f"Run ended: {result.status}")

    st.subheader("Review queue")
    tab_pending, tab_failed = st.tabs(["Pending approval", "Failed"])

    with tab_pending:
        review_items = fetch_pipeline_review_queue(sb, language_code="en")
        # region agent log
        block_counts: Dict[str, int] = {}
        for _it in review_items:
            _bk = f"{_it.get('story_id')}_{_it.get('reading_level')}_{_it.get('block_start')}"
            block_counts[_bk] = block_counts.get(_bk, 0) + 1
        _dbg_ui(
            run_id="review-render",
            hypothesis_id="H2",
            location="image_processor_page.py:review_queue",
            message="Review queue loaded",
            data={
                "count": len(review_items),
                "duplicateBlocks": {k: v for k, v in block_counts.items() if v > 1},
                "items": [
                    {
                        "id": str(it.get("id")),
                        "storyId": it.get("story_id"),
                        "readingLevel": it.get("reading_level"),
                        "blockStart": it.get("block_start"),
                        "status": it.get("status"),
                        "pendingUrl": (it.get("pending_image_url") or "")[:120],
                        "updatedAt": it.get("updated_at"),
                    }
                    for it in review_items
                ],
            },
        )
        # endregion
        if not review_items:
            st.info("No images pending approval.")
        else:
            story_titles = {s["id"]: s.get("title") or f"Story {s['id']}" for s in stories}
            if st.button(f"Approve all ({len(review_items)})", key="ip_batch_approve_all", type="primary"):
                ok = 0
                for it in review_items:
                    if approve_pipeline_item(sb, it, stories=stories):
                        ok += 1
                st.success(f"Approved **{ok}** image(s).")
                st.session_state["ip_pages_missing_version"] = (
                    st.session_state.get("ip_pages_missing_version", 0) + 1
                )
                st.rerun()
            for it in review_items:
                sid = it.get("story_id")
                title = story_titles.get(sid, f"Story {sid}")
                label = (
                    f"{title} · {it.get('reading_level', '').replace('_', ' ')} · "
                    f"pages {it.get('page_range_label') or it.get('block_start')}"
                )
                with st.expander(label, expanded=False):
                    url = pending_image_display_url(it)
                    if url:
                        st.image(url, caption="Pending approval", width=320)
                    if it.get("illustration_moment"):
                        st.caption(it["illustration_moment"][:300])
                    if it.get("leonardo_cost"):
                        st.caption(it["leonardo_cost"])
                    ac1, ac2, ac3 = st.columns(3)
                    item_id = str(it["id"])
                    with ac1:
                        if st.button("Approve", key=f"ip_batch_appr_{item_id}", type="primary"):
                            if approve_pipeline_item(sb, it, stories=stories):
                                st.success("Approved.")
                                st.session_state["ip_pages_missing_version"] = (
                                    st.session_state.get("ip_pages_missing_version", 0) + 1
                                )
                                st.rerun()
                            else:
                                st.error("Approve failed.")
                    with ac2:
                        if st.button("Regenerate", key=f"ip_batch_regen_{item_id}"):
                            run_id = str(it.get("run_id") or "")
                            # region agent log
                            _dbg_ui(
                                run_id="regen-click",
                                hypothesis_id="H3",
                                location="image_processor_page.py:regenerate",
                                message="Regenerate clicked",
                                data={
                                    "itemId": item_id,
                                    "runId": run_id,
                                    "blockStart": it.get("block_start"),
                                    "oldPendingUrl": (it.get("pending_image_url") or "")[:120],
                                    "updatedAt": it.get("updated_at"),
                                },
                            )
                            # endregion
                            if run_id and requeue_pipeline_item(sb, it, run_id):
                                result = run_pipeline(sb, run_id)
                                # region agent log
                                refreshed = fetch_pipeline_review_queue(
                                    sb,
                                    language_code="en",
                                    story_id=int(sid) if sid is not None else None,
                                )
                                _dbg_ui(
                                    run_id="regen-done",
                                    hypothesis_id="H1",
                                    location="image_processor_page.py:regenerate",
                                    message="Regenerate pipeline finished",
                                    data={
                                        "itemId": item_id,
                                        "pipelineStatus": result.status,
                                        "reviewItemsAfter": [
                                            {
                                                "id": str(r.get("id")),
                                                "blockStart": r.get("block_start"),
                                                "pendingUrl": (r.get("pending_image_url") or "")[:120],
                                                "updatedAt": r.get("updated_at"),
                                            }
                                            for r in refreshed
                                            if r.get("block_start") == it.get("block_start")
                                        ],
                                    },
                                )
                                # endregion
                                if result.status == "credits_exhausted":
                                    st.error(result.last_error or CREDITS_EXHAUSTED_ERROR)
                                else:
                                    st.success("Regenerated — check review queue.")
                                st.rerun()
                            else:
                                st.error("Could not requeue.")
                    with ac3:
                        if st.button("Discard", key=f"ip_batch_disc_{item_id}"):
                            # region agent log
                            _dbg_ui(
                                run_id="discard-click",
                                hypothesis_id="H4",
                                location="image_processor_page.py:discard",
                                message="Discard clicked",
                                data={
                                    "itemId": item_id,
                                    "blockStart": it.get("block_start"),
                                    "pendingUrl": (it.get("pending_image_url") or "")[:120],
                                },
                            )
                            # endregion
                            update_pipeline_item(sb, item_id, status="superseded")
                            st.rerun()

    with tab_failed:
        failed_items = fetch_pipeline_failed_items(sb, language_code="en")
        if not failed_items:
            st.info("No failed items.")
        else:
            story_titles = {s["id"]: s.get("title") or f"Story {s['id']}" for s in stories}
            for it in failed_items:
                sid = it.get("story_id")
                title = story_titles.get(sid, f"Story {sid}")
                st.markdown(
                    f"**{title}** · {it.get('reading_level')} · pages {it.get('page_range_label')} — "
                    f"{it.get('error_message') or 'unknown error'}"
                )
                if st.button("Retry", key=f"ip_batch_retry_{it['id']}"):
                    run_id = str(it.get("run_id") or "")
                    if run_id and requeue_pipeline_item(sb, it, run_id):
                        result = run_pipeline(sb, run_id)
                        if result.status == "credits_exhausted":
                            st.error(result.last_error or CREDITS_EXHAUSTED_ERROR)
                        st.rerun()

    st.divider()


def run_image_processor_view():
    st.title("Image Processor")
    st.caption(
        f"Generate one Leonardo image per {PAGES_PER_IMAGE}-page block (same URL saved on each page in the block). "
        "Optional character references are managed per story and grade."
    )

    sb = get_supabase()
    if not sb:
        return

    _v = st.session_state.get("ip_cache_version", 0)
    stories = _cached_fetch_stories(_v)
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return

    _render_batch_automation_section(sb, stories)

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
    all_pages_for_review = _cached_fetch_book_pages(story_id, reading_level, language_code, _v)
    image_blocks_review = group_pages_into_image_blocks(all_pages_for_review)
    story_meta = _story_meta(stories, story_id)

    refs = _load_storybook_refs(scope, grade_style)
    api_key_leo = (get_secret("LEONARDO_API_KEY") or "").strip()
    model_id = (get_secret("LEONARDO_MODEL_ID") or "").strip() or str(LEONARDO_GENERATION_DEFAULTS["modelId"])
    if not api_key_leo:
        st.warning("Set **LEONARDO_API_KEY** in `.env` to generate images.")

    refs = _render_style_scene_section(
        sb,
        story_id,
        reading_level,
        scope,
        refs,
        story_meta=story_meta,
        grade_style=grade_style,
        image_blocks=image_blocks_review,
        api_key=api_key_leo,
        model_id=model_id,
    )
    openai_client = get_openai()

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
    image_blocks = image_blocks_review
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
    char_ids = [_REF_NONE] + [c["id"] for c in char_options]
    char_labels = {_REF_NONE: "None", **{c["id"]: c.get("label") or c["id"] for c in char_options}}

    for row in all_pages_for_review:
        resolved = _resolve_page_image_url(row, _img_overlay, just_approved)
        if _row_id_str(row.get("id")) in just_cleared_ids:
            row["image_url"] = ""
        elif resolved:
            row["image_url"] = resolved

    st.header("4. Generate scene images")
    st.caption(
        "Generate a **style scene** in section 2 first. Each block needs an **illustration moment** below; "
        "Leonardo receives the style scene plus that moment. Character references are optional."
    )
    grade_char_key = f"ip_grade_char_{scope}"
    if grade_char_key not in st.session_state:
        st.session_state[grade_char_key] = _REF_NONE
    st.selectbox(
        "Default character for new blocks",
        options=char_ids,
        format_func=lambda x: char_labels.get(x, x),
        key=grade_char_key,
    )

    work_queue_only = st.checkbox("Work queue only (no image or pending approval)", value=False, key="ip_work_queue")

    if not _get_style_scene(scope):
        st.info("Generate a **style scene** in section 2 before generating images.")

    if openai_client and _get_style_scene(scope) and image_blocks_review and st.button(
        f"Generate illustration moments for all blocks ({len(image_blocks_review)})",
        key="ip_bulk_scene_gen",
    ):
        with st.spinner("ChatGPT: writing illustration moments for each block…"):
            scenes_by_block = _generate_scenes_for_full_story(
                image_blocks_review,
                story_title=story_meta["title"],
                story_summary=story_meta["description"],
                reading_level=reading_level,
                grade_style=grade_style,
                refs=refs,
                scope=scope,
            )
        if not scenes_by_block:
            st.error("Could not generate scene descriptions. Check OPENAI_API_KEY and try again.")
        else:
            applied = 0
            missing: List[str] = []
            for block in image_blocks_review:
                bs = block.get("block_start")
                row_id = block["anchor_row"].get("id")
                scene = scenes_by_block.get(bs) if bs is not None else None
                if scene:
                    st.session_state[f"ip_scene_{row_id}"] = scene
                    applied += 1
                else:
                    missing.append(block.get("page_range_label") or str(bs))
            if applied == len(image_blocks_review):
                st.success(
                    f"Generated **{applied}** illustration moments. "
                    "Review, edit if needed, then generate images."
                )
            elif applied:
                st.warning(
                    f"Generated **{applied}/{len(image_blocks_review)}** scene descriptions. "
                    f"Missing blocks: {', '.join(missing)} — use per-block regenerate or try again."
                )
            else:
                st.error("ChatGPT returned no usable scene descriptions.")
            st.rerun()

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
                    char_id = _block_char_id(scope, row_id)
                    img, gen_cost = _generate_block_image(
                        api_key_leo,
                        model_id,
                        refs,
                        scene,
                        char_id,
                        reading_level,
                        style_scene=_get_style_scene(scope),
                        page_context=_block_page_text(block),
                        page_range_label=page_range,
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
                st.session_state[scene_key] = ""
            gen_scene_col, _ = st.columns([1, 3])
            with gen_scene_col:
                if st.button(
                    "Regenerate moment",
                    key=f"gen_scene_{row_id}",
                    disabled=not openai_client or not _get_style_scene(scope),
                    help="Writes an illustration moment for this block using the style scene and page text.",
                ):
                    with st.spinner("ChatGPT: writing illustration moment…"):
                        scene = _generate_scene_for_block(
                            block,
                            story_title=story_meta["title"],
                            story_summary=story_meta["description"],
                            reading_level=reading_level,
                            grade_style=grade_style,
                            refs=refs,
                            scope=scope,
                        )
                    if scene:
                        st.session_state[scene_key] = scene
                        st.rerun()
                    else:
                        st.error("Could not generate scene description.")
            st.text_area(
                "Illustration moment",
                key=scene_key,
                height=100,
                help="What happens in this illustration (action and composition). Leonardo also receives the labeled page text for this block and the style scene from section 2.",
            )

            char_key = f"ip_char_{row_id}"
            if char_key not in st.session_state:
                st.session_state[char_key] = st.session_state.get(grade_char_key, _REF_NONE)

            st.selectbox(
                "Character",
                options=char_ids,
                format_func=lambda x: char_labels.get(x, x),
                key=char_key,
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
                        if not _get_style_scene(scope):
                            st.warning("Generate a style scene in section 2 first.")
                        elif not scene:
                            st.warning("Enter an illustration moment.")
                        else:
                            char_id = _block_char_id(scope, row_id)
                            with st.spinner("Leonardo: generating…"):
                                img, gen_cost = _generate_block_image(
                                    api_key_leo,
                                    model_id,
                                    refs,
                                    scene,
                                    char_id,
                                    reading_level,
                                    style_scene=_get_style_scene(scope),
                                    page_context=_block_page_text(block),
                                    page_range_label=block.get("page_range_label") or "",
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
