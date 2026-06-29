"""Headless batch image pipeline (no Streamlit)."""
from __future__ import annotations

import os
import time
import uuid
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import leonardo_client as leo
from auth import get_secret
from grade_style_defaults import (
    character_ref_prompt_for_grade,
    grade_scene_settings_for_prompt,
    series_style_prompt_for_grade,
)
from leonardo_series_config import LEONARDO_GENERATION_DEFAULTS, STYLE_CONTROLNET_SCENE_STRENGTH
from scene_prompts import (
    analyze_character_reference_need,
    generate_all_scene_descriptions,
    generate_scene_description_paragraph,
    generate_style_scene_description,
)

from lib import (
    IMAGE_PROCESSOR_GRADES,
    PIPELINE_ITEM_ACTIVE_STATUSES,
    append_pipeline_spend_note,
    apply_image_url_to_block,
    block_needs_image,
    build_simple_leonardo_prompt,
    fetch_book_pages,
    fetch_next_queued_pipeline_item,
    fetch_pages_missing_images,
    fetch_pipeline_item,
    fetch_pipeline_items_by_run,
    fetch_pipeline_review_queue,
    fetch_stories,
    find_character_ref_by_id,
    find_image_block_for_row_id,
    formatted_page_text_block,
    generate_image_leonardo,
    generate_leonardo_reference_preview,
    get_approved_character_refs,
    get_approved_style_ref,
    get_pipeline_run,
    get_style_scene_text,
    get_story_grade_style,
    group_pages_into_image_blocks,
    has_active_pipeline_item_for_block,
    insert_pipeline_items,
    insert_pipeline_run,
    optimize_image_for_mobile,
    page_number_for_row,
    save_storybook_references,
    save_style_scene_text,
    storybook_references_from_grade_style,
    update_pipeline_item,
    update_pipeline_run,
    upload_image_to_storage,
    upload_pending_block_image,
    upload_typed_reference_image,
    _get_page_text,
)

CREDITS_EXHAUSTED_ERROR = (
    "ERROR: Leonardo API credits exhausted. Add credits and click Resume."
)

ProgressCallback = Callable[[str, int, int], None]

# region agent log
_DEBUG_LOG_PATH = "/Users/benwaddell/cursor projects/storybook-image-processor/.cursor/debug-81a77e.log"
_DEBUG_SESSION_ID = "81a77e"


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
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


@dataclass
class WorkItem:
    story_id: int
    story_title: str
    reading_level: str
    language_code: str
    missing_block_count: int


@dataclass
class PipelineRunResult:
    run_id: str
    status: str
    items_done: int = 0
    items_failed: int = 0
    items_total: int = 0
    last_error: str = ""
    message: str = ""


def _openai_client():
    key = (get_secret("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=key)


def _leonardo_config() -> Tuple[str, str]:
    api_key = (get_secret("LEONARDO_API_KEY") or "").strip()
    model_id = (get_secret("LEONARDO_MODEL_ID") or "").strip() or str(
        LEONARDO_GENERATION_DEFAULTS["modelId"]
    )
    return api_key, model_id


def _batch_budget_usd() -> Optional[float]:
    raw = (os.getenv("LEONARDO_BATCH_BUDGET_USD") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def block_page_text(block: dict) -> str:
    members = block.get("member_rows") or []
    labeled = formatted_page_text_block(members)
    if labeled:
        return labeled
    return (block.get("combined_text") or "").strip()


def block_prompt_payload(block: dict) -> dict:
    pages = []
    for member in block.get("member_rows") or []:
        pages.append(
            {
                "page_number": page_number_for_row(member),
                "text": _get_page_text(member),
            }
        )
    page_text = block_page_text(block)
    return {
        "block_start": block.get("block_start"),
        "page_range_label": block.get("page_range_label") or "",
        "combined_text": page_text or block.get("combined_text") or "",
        "pages": pages,
    }


def generate_block_image(
    api_key: str,
    model_id: str,
    refs: Dict[str, Any],
    scene_description: str,
    character_id: str,
    reading_level: str,
    style_scene: str = "",
    page_context: str = "",
    page_range_label: str = "",
) -> Tuple[Optional[bytes], Optional[str]]:
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


def discover_work_items(sb, *, language_code: str = "en") -> List[WorkItem]:
    stories = fetch_stories()
    items: List[WorkItem] = []
    total_levels_scanned = 0
    total_missing_blocks = 0
    total_blocks_possible = 0
    versions_with_pages = 0
    # region agent log
    version_metrics: List[Dict[str, Any]] = []
    # endregion
    for story in stories:
        story_id = story.get("id")
        if story_id is None:
            continue
        title = (story.get("title") or "").strip() or f"Story {story_id}"
        for level in IMAGE_PROCESSOR_GRADES:
            total_levels_scanned += 1
            pages = fetch_book_pages(sb, story_id, level, language_code)
            if not pages:
                continue
            versions_with_pages += 1
            blocks = group_pages_into_image_blocks(pages)
            total_blocks_possible += len(blocks)
            missing = [b for b in blocks if block_needs_image(b)]
            total_missing_blocks += len(missing)
            # region agent log
            version_metrics.append(
                {
                    "storyId": int(story_id),
                    "readingLevel": level,
                    "pagesCount": len(pages),
                    "blocksTotal": len(blocks),
                    "blocksMissing": len(missing),
                }
            )
            # endregion
            if not missing:
                continue
            items.append(
                WorkItem(
                    story_id=int(story_id),
                    story_title=title,
                    reading_level=level,
                    language_code=language_code,
                    missing_block_count=len(missing),
                )
            )
    # region agent log
    _dbg(
        run_id="discover",
        hypothesis_id="H1",
        location="image_pipeline.py:discover_work_items",
        message="Discovery summary by block count",
        data={
            "storiesCount": len(stories),
            "levelsScanned": total_levels_scanned,
            "versionsWithPages": versions_with_pages,
            "itemsCount": len(items),
            "blocksPossibleTotal": total_blocks_possible,
            "missingBlocksTotal": total_missing_blocks,
            "gradesIncluded": list(IMAGE_PROCESSOR_GRADES),
            "languageCode": language_code,
            "versionMetrics": version_metrics[:300],
        },
    )
    # endregion
    return items


def should_skip_block(
    sb,
    block: dict,
    *,
    story_id: int,
    reading_level: str,
    language_code: str,
    exclude_item_id: Optional[str] = None,
    ignore_pending_review: bool = False,
) -> Tuple[bool, str]:
    if not block_needs_image(block):
        return True, "already_has_image"
    block_start = block.get("block_start")
    if block_start is None:
        return True, "invalid_block"
    active_statuses = PIPELINE_ITEM_ACTIVE_STATUSES
    if ignore_pending_review:
        active_statuses = tuple(
            s for s in PIPELINE_ITEM_ACTIVE_STATUSES if s != "pending_review"
        )
    if has_active_pipeline_item_for_block(
        sb,
        story_id,
        reading_level,
        language_code,
        int(block_start),
        exclude_item_id=exclude_item_id,
        active_statuses=active_statuses,
    ):
        return True, "active_pipeline_item"
    return False, ""


def supersede_pending_review_items(
    sb,
    *,
    language_code: str = "en",
    story_ids: Optional[Set[int]] = None,
) -> int:
    """Mark pending_review items as superseded so blocks can be re-enqueued."""
    items = fetch_pipeline_review_queue(sb, language_code=language_code)
    count = 0
    for item in items:
        sid = item.get("story_id")
        if sid is None:
            continue
        if story_ids is not None and int(sid) not in story_ids:
            continue
        update_pipeline_item(sb, str(item["id"]), status="superseded")
        count += 1
    return count


def collect_processable_blocks(
    sb,
    work_items: List[WorkItem],
    *,
    limit: Optional[int] = None,
    ignore_pending_review: bool = False,
) -> Tuple[List[dict], Dict[str, int]]:
    """Return pipeline row dicts for missing blocks not blocked by active pipeline items."""
    rows: List[dict] = []
    skipped_active = 0
    skipped_not_missing = 0
    for wi in work_items:
        pages = fetch_book_pages(sb, wi.story_id, wi.reading_level, wi.language_code)
        blocks = group_pages_into_image_blocks(pages)
        for block in blocks:
            if not block_needs_image(block):
                skipped_not_missing += 1
                continue
            skip, reason = should_skip_block(
                sb,
                block,
                story_id=wi.story_id,
                reading_level=wi.reading_level,
                language_code=wi.language_code,
                ignore_pending_review=ignore_pending_review,
            )
            if skip:
                if reason == "active_pipeline_item":
                    skipped_active += 1
                continue
            anchor = block["anchor_row"]
            rows.append(
                {
                    "story_id": wi.story_id,
                    "reading_level": wi.reading_level,
                    "language_code": wi.language_code,
                    "block_start": block["block_start"],
                    "anchor_row_id": str(anchor.get("id")),
                    "page_range_label": block.get("page_range_label") or "",
                    "status": "queued",
                }
            )
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    stats = {
        "skipped_active": skipped_active,
        "skipped_not_missing": skipped_not_missing,
        "processable": len(rows),
    }
    return rows, stats


def create_pipeline_run(
    sb,
    work_items: List[WorkItem],
    *,
    language_code: str = "en",
    limit: Optional[int] = None,
    ignore_pending_review: bool = False,
) -> Optional[str]:
    requested_blocks = sum(int(w.missing_block_count or 0) for w in work_items)
    rows, stats = collect_processable_blocks(
        sb,
        work_items,
        limit=limit,
        ignore_pending_review=ignore_pending_review,
    )
    skipped_active = stats["skipped_active"]
    skipped_not_missing = stats["skipped_not_missing"]
    if not rows:
        return None
    run_id = insert_pipeline_run(sb, language_code=language_code, items_total=len(rows))
    if not run_id:
        return None
    for row in rows:
        row["run_id"] = run_id
    inserted = insert_pipeline_items(sb, rows)
    # region agent log
    _dbg(
        run_id=run_id,
        hypothesis_id="H3",
        location="image_pipeline.py:create_pipeline_run",
        message="Queue creation summary",
        data={
            "workItemsCount": len(work_items),
            "requestedBlocksFromScan": requested_blocks,
            "rowsAttempted": len(rows),
            "rowsInserted": len(inserted),
            "skippedActive": skipped_active,
            "skippedNotMissing": skipped_not_missing,
        },
    )
    # endregion
    if len(inserted) != len(rows):
        update_pipeline_run(sb, run_id, status="cancelled", last_error="Failed to enqueue items")
        return None
    return run_id


def _story_meta(stories: List[dict], story_id: int) -> Dict[str, str]:
    for s in stories:
        if s.get("id") == story_id:
            return {
                "title": (s.get("title") or "").strip(),
                "description": (s.get("description") or "").strip(),
            }
    return {"title": "", "description": ""}


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


def _auto_approve_style_reference(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    *,
    style_scene: str,
    api_key: str,
    model_id: str,
) -> bool:
    if get_approved_style_ref(refs):
        return True
    img, _cost, gen_id = generate_leonardo_reference_preview(
        style_scene,
        api_key=api_key,
        model_id=model_id,
        reading_level=reading_level,
        ref_kind="style",
    )
    if not img or not gen_id:
        return False
    url = upload_typed_reference_image(sb, story_id, reading_level, "style", "series", img)
    if not url:
        return False
    refs["style"] = {
        "label": "Series style",
        "prompt": style_scene,
        "url": url,
        "leonardo_init_image_id": gen_id,
        "approved": True,
    }
    return save_storybook_references(sb, story_id, reading_level, refs)


def _auto_approve_character_reference(
    sb,
    story_id: int,
    reading_level: str,
    refs: Dict[str, Any],
    *,
    label: str,
    prompt: str,
    api_key: str,
    model_id: str,
) -> Optional[str]:
    if get_approved_character_refs(refs):
        return get_approved_character_refs(refs)[0].get("id")
    img, _cost, gen_id = generate_leonardo_reference_preview(
        prompt,
        api_key=api_key,
        model_id=model_id,
        reading_level=reading_level,
        ref_kind="character",
    )
    if not img:
        return None
    ref_id = uuid.uuid4().hex[:8]
    url = upload_typed_reference_image(sb, story_id, reading_level, "character", ref_id, img)
    if not url:
        return None
    try:
        leo_id = leo.upload_init_image_bytes(api_key, img)
    except Exception:
        leo_id = (gen_id or "").strip()
    if not leo_id:
        return None
    entry_id = f"c_{ref_id}"
    entry = {
        "id": entry_id,
        "label": label or "Character",
        "prompt": prompt,
        "url": url,
        "leonardo_init_image_id": leo_id,
        "approved": True,
    }
    items = list(refs.get("characters") or [])
    items.append(entry)
    refs["characters"] = items
    if save_storybook_references(sb, story_id, reading_level, refs):
        return entry_id
    return None


@dataclass
class _VersionContext:
    style_scene: str = ""
    character_ref_id: str = ""
    refs: Dict[str, Any] = field(default_factory=dict)
    prepared: bool = False


def _ensure_version_prep(
    sb,
    run_id: str,
    story_id: int,
    reading_level: str,
    language_code: str,
    *,
    openai_client,
    api_key: str,
    model_id: str,
    stories: List[dict],
    spend_tracker: Dict[str, float],
    budget_usd: Optional[float],
) -> _VersionContext:
    ctx = _VersionContext()
    grade_style = get_story_grade_style(sb, story_id, reading_level)
    ctx.refs = storybook_references_from_grade_style(grade_style)

    style_scene = get_style_scene_text(grade_style)
    if not style_scene:
        pages = fetch_book_pages(sb, story_id, reading_level, language_code)
        blocks = group_pages_into_image_blocks(pages)
        meta = _story_meta(stories, story_id)
        style_scene = generate_style_scene_description(
            openai_client,
            [_block_prompt_payload_from_block(b) for b in blocks],
            story_title=meta["title"],
            story_summary=meta["description"],
            default_style_scene=series_style_prompt_for_grade(reading_level),
            grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
        ) or series_style_prompt_for_grade(reading_level)
        save_style_scene_text(sb, story_id, reading_level, style_scene)
    ctx.style_scene = style_scene

    if not get_approved_style_ref(ctx.refs):
        ok = _auto_approve_style_reference(
            sb,
            story_id,
            reading_level,
            ctx.refs,
            style_scene=style_scene,
            api_key=api_key,
            model_id=model_id,
        )
        if not ok:
            raise RuntimeError(f"Failed to auto-approve style reference for story {story_id} {reading_level}")
        ctx.refs = storybook_references_from_grade_style(
            get_story_grade_style(sb, story_id, reading_level)
        )

    if get_approved_character_refs(ctx.refs):
        ctx.character_ref_id = get_approved_character_refs(ctx.refs)[0].get("id") or ""
    else:
        pages = fetch_book_pages(sb, story_id, reading_level, language_code)
        blocks = group_pages_into_image_blocks(pages)
        analysis = analyze_character_reference_need(
            openai_client,
            story_text="\n\n".join(block_page_text(b) for b in blocks),
            style_scene=style_scene,
            grade_character_defaults=character_ref_prompt_for_grade(reading_level),
            blocks=[block_prompt_payload(b) for b in blocks],
        )
        if analysis.get("needs_character_ref"):
            prompt = (analysis.get("character_prompt") or "").strip()
            if not prompt:
                prompt = character_ref_prompt_for_grade(reading_level)
            label = (analysis.get("character_label") or "Character").strip()
            char_id = _auto_approve_character_reference(
                sb,
                story_id,
                reading_level,
                ctx.refs,
                label=label,
                prompt=prompt,
                api_key=api_key,
                model_id=model_id,
            )
            if char_id:
                ctx.character_ref_id = char_id
                ctx.refs = storybook_references_from_grade_style(
                    get_story_grade_style(sb, story_id, reading_level)
                )

    # Include generating items — the current block is set to generating before prep runs.
    version_items = [
        it
        for it in fetch_pipeline_items_by_run(sb, run_id)
        if it.get("story_id") == story_id
        and it.get("reading_level") == reading_level
        and it.get("status") in ("queued", "generating")
    ]
    needs_moments = [it for it in version_items if not (it.get("illustration_moment") or "").strip()]
    # region agent log
    _dbg(
        run_id=run_id,
        hypothesis_id="H4",
        location="image_pipeline.py:_ensure_version_prep",
        message="Moment prep counts",
        data={
            "storyId": story_id,
            "readingLevel": reading_level,
            "versionItemsCount": len(version_items),
            "needsMomentsCount": len(needs_moments),
        },
    )
    # endregion
    if needs_moments:
        pages = fetch_book_pages(sb, story_id, reading_level, language_code)
        all_blocks = group_pages_into_image_blocks(pages)
        block_by_start = {b["block_start"]: b for b in all_blocks}
        payloads = []
        for it in needs_moments:
            bs = it.get("block_start")
            block = block_by_start.get(bs)
            if block:
                payloads.append(block_prompt_payload(block))
        # region agent log
        _dbg(
            run_id=run_id,
            hypothesis_id="H5",
            location="image_pipeline.py:_ensure_version_prep",
            message="Moment payload mapping",
            data={
                "storyId": story_id,
                "readingLevel": reading_level,
                "needsMomentsBlockStarts": [it.get("block_start") for it in needs_moments],
                "payloadCount": len(payloads),
                "availableBlockStarts": sorted(list(block_by_start.keys()))[:80],
            },
        )
        # endregion
        if payloads:
            meta = _story_meta(stories, story_id)
            scenes = generate_all_scene_descriptions(
                openai_client,
                payloads,
                story_title=meta["title"],
                story_summary=meta["description"],
                character_refs=_all_character_refs_text(ctx.refs),
                style_scene=style_scene,
                grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
            ) or {}
            # region agent log
            _dbg(
                run_id=run_id,
                hypothesis_id="H4",
                location="image_pipeline.py:_ensure_version_prep",
                message="Scene generation response keys",
                data={
                    "storyId": story_id,
                    "readingLevel": reading_level,
                    "sceneKeys": sorted([int(k) for k in scenes.keys()])[:120],
                    "sceneCount": len(scenes),
                },
            )
            # endregion
            for it in needs_moments:
                bs = it.get("block_start")
                moment = scenes.get(int(bs)) if bs is not None else None
                if not moment:
                    block = block_by_start.get(bs)
                    if block and openai_client:
                        moment = generate_scene_description_paragraph(
                            openai_client,
                            block_prompt_payload(block)["combined_text"],
                            story_title=meta["title"],
                            story_summary=meta["description"],
                            character_ref=_all_character_refs_text(ctx.refs),
                            style_scene=style_scene,
                            grade_scene_settings=grade_scene_settings_for_prompt(
                                reading_level, grade_style
                            ),
                            page_range_label=block.get("page_range_label") or "",
                        )
                if moment:
                    update_pipeline_item(
                        sb,
                        str(it["id"]),
                        illustration_moment=moment,
                        style_scene=style_scene,
                        character_ref_id=ctx.character_ref_id or None,
                    )

    ctx.prepared = True
    return ctx


def _ensure_item_illustration_moment(
    sb,
    item: dict,
    *,
    openai_client,
    stories: List[dict],
    style_scene: str,
    refs: Dict[str, Any],
    grade_style: Optional[dict],
    language_code: str,
) -> str:
    """Return illustration moment for one item, generating via ChatGPT if needed."""
    existing = (item.get("illustration_moment") or "").strip()
    if existing:
        return existing
    story_id = int(item["story_id"])
    reading_level = item["reading_level"]
    block_start = int(item["block_start"])
    pages = fetch_book_pages(sb, story_id, reading_level, language_code)
    blocks = group_pages_into_image_blocks(pages)
    block = next((b for b in blocks if b.get("block_start") == block_start), None)
    if not block or not openai_client:
        return ""
    meta = _story_meta(stories, story_id)
    moment = generate_scene_description_paragraph(
        openai_client,
        block_prompt_payload(block)["combined_text"],
        story_title=meta["title"],
        story_summary=meta["description"],
        character_ref=_all_character_refs_text(refs),
        style_scene=style_scene,
        grade_scene_settings=grade_scene_settings_for_prompt(reading_level, grade_style),
        page_range_label=block.get("page_range_label") or "",
    )
    if moment:
        update_pipeline_item(
            sb,
            str(item["id"]),
            illustration_moment=moment,
            style_scene=style_scene,
        )
    return (moment or "").strip()


def _block_prompt_payload_from_block(block: dict) -> dict:
    return block_prompt_payload(block)


def _track_spend(
    sb,
    run_id: str,
    cost_note: Optional[str],
    spend_tracker: Dict[str, float],
    budget_usd: Optional[float],
) -> bool:
    """Returns False if soft budget exceeded."""
    if cost_note:
        append_pipeline_spend_note(sb, run_id, cost_note)
    if budget_usd is None:
        return True
    # Best-effort USD tracking from cost note strings is limited; rely on Leonardo hard cap primarily.
    return True


def run_pipeline(
    sb,
    run_id: str,
    *,
    resume: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineRunResult:
    run = get_pipeline_run(sb, run_id)
    if not run:
        return PipelineRunResult(run_id, "cancelled", message="Run not found")
    if run.get("status") == "credits_exhausted" and not resume:
        return PipelineRunResult(
            run_id,
            "credits_exhausted",
            last_error=run.get("last_error") or CREDITS_EXHAUSTED_ERROR,
            message="Run paused — resume to continue",
        )
    if run.get("status") not in ("running", "credits_exhausted"):
        return PipelineRunResult(
            run_id,
            run.get("status") or "cancelled",
            message=f"Run is {run.get('status')}",
        )

    if resume or run.get("status") == "credits_exhausted":
        update_pipeline_run(sb, run_id, status="running", last_error=None)

    # Recover items stuck in generating after an interrupted Streamlit session.
    for stuck in fetch_pipeline_items_by_run(sb, run_id, status="generating"):
        update_pipeline_item(sb, str(stuck["id"]), status="queued")

    openai_client = _openai_client()
    if not openai_client:
        update_pipeline_run(sb, run_id, status="cancelled", last_error="OPENAI_API_KEY not set")
        return PipelineRunResult(run_id, "cancelled", last_error="OPENAI_API_KEY not set")

    api_key, model_id = _leonardo_config()
    if not api_key:
        update_pipeline_run(sb, run_id, status="cancelled", last_error="LEONARDO_API_KEY not set")
        return PipelineRunResult(run_id, "cancelled", last_error="LEONARDO_API_KEY not set")

    stories = fetch_stories()
    budget_usd = _batch_budget_usd()
    spend_tracker: Dict[str, float] = {"total": 0.0}
    version_cache: Dict[Tuple[int, str], _VersionContext] = {}
    prepared_versions: Set[Tuple[int, str]] = set()

    items_total = int(run.get("items_total") or 0)
    items_done = int(run.get("items_done") or 0)
    items_failed = int(run.get("items_failed") or 0)

    while True:
        item = fetch_next_queued_pipeline_item(sb, run_id)
        if not item:
            update_pipeline_run(
                sb,
                run_id,
                status="completed",
                items_done=items_done,
                items_failed=items_failed,
            )
            return PipelineRunResult(
                run_id,
                "completed",
                items_done=items_done,
                items_failed=items_failed,
                items_total=items_total,
                message="Batch complete",
            )

        item_id = str(item["id"])
        story_id = int(item["story_id"])
        reading_level = item["reading_level"]
        language_code = item.get("language_code") or "en"
        block_start = int(item["block_start"])
        version_key = (story_id, reading_level)

        if progress_callback:
            progress_callback(
                f"Story {story_id} · {reading_level} · block {block_start}",
                items_done,
                items_total,
            )

        pages = fetch_book_pages(sb, story_id, reading_level, language_code)
        blocks = group_pages_into_image_blocks(pages)
        block = next((b for b in blocks if b.get("block_start") == block_start), None)
        if not block:
            update_pipeline_item(sb, item_id, status="failed", error_message="Block not found")
            items_failed += 1
            update_pipeline_run(sb, run_id, items_failed=items_failed, items_done=items_done)
            continue

        skip, reason = should_skip_block(
            sb,
            block,
            story_id=story_id,
            reading_level=reading_level,
            language_code=language_code,
            exclude_item_id=item_id,
        )
        if skip:
            update_pipeline_item(sb, item_id, status="skipped", error_message=reason)
            items_done += 1
            update_pipeline_run(sb, run_id, items_done=items_done, items_failed=items_failed)
            continue

        update_pipeline_item(sb, item_id, status="generating")

        try:
            if version_key not in prepared_versions:
                vctx = _ensure_version_prep(
                    sb,
                    run_id,
                    story_id,
                    reading_level,
                    language_code,
                    openai_client=openai_client,
                    api_key=api_key,
                    model_id=model_id,
                    stories=stories,
                    spend_tracker=spend_tracker,
                    budget_usd=budget_usd,
                )
                version_cache[version_key] = vctx
                prepared_versions.add(version_key)
            else:
                vctx = version_cache[version_key]

            refreshed = fetch_pipeline_item(sb, item_id) or item
            moment = (refreshed.get("illustration_moment") or "").strip()
            if not moment:
                grade_style = get_story_grade_style(sb, story_id, reading_level)
                moment = _ensure_item_illustration_moment(
                    sb,
                    refreshed,
                    openai_client=openai_client,
                    stories=stories,
                    style_scene=vctx.style_scene,
                    refs=vctx.refs,
                    grade_style=grade_style,
                    language_code=language_code,
                )
            if not moment:
                # region agent log
                _dbg(
                    run_id=run_id,
                    hypothesis_id="H4",
                    location="image_pipeline.py:run_pipeline",
                    message="Moment missing after prep and fallback",
                    data={
                        "itemId": item_id,
                        "storyId": story_id,
                        "readingLevel": reading_level,
                        "blockStart": block_start,
                        "pageRangeLabel": item.get("page_range_label"),
                    },
                )
                # endregion
                raise RuntimeError("Missing illustration moment after prep")

            time.sleep(1.5)
            img, cost_note = generate_block_image(
                api_key,
                model_id,
                vctx.refs,
                moment,
                vctx.character_ref_id,
                reading_level,
                style_scene=vctx.style_scene,
                page_context=block_page_text(block),
                page_range_label=block.get("page_range_label") or "",
            )
            if not img:
                raise RuntimeError("Leonardo returned no image")

            _track_spend(sb, run_id, cost_note, spend_tracker, budget_usd)
            pending_url = upload_pending_block_image(
                sb, story_id, reading_level, block_start, img, item_id=item_id
            )
            if not pending_url:
                raise RuntimeError("Failed to upload pending image")

            # region agent log
            _dbg(
                run_id=run_id,
                hypothesis_id="H1",
                location="image_pipeline.py:run_pipeline",
                message="Pending image uploaded",
                data={
                    "itemId": item_id,
                    "blockStart": block_start,
                    "pendingUrl": pending_url[:160],
                    "usesUniquePath": f"/pending/{item_id}.webp" in pending_url,
                    "imageBytesLen": len(img),
                    "imageHash8": hashlib.sha256(img).hexdigest()[:8],
                },
            )
            # endregion

            update_pipeline_item(
                sb,
                item_id,
                status="pending_review",
                pending_image_url=pending_url,
                leonardo_cost=cost_note,
                style_scene=vctx.style_scene,
                illustration_moment=moment,
                character_ref_id=vctx.character_ref_id or None,
                error_message=None,
            )
            items_done += 1
            update_pipeline_run(sb, run_id, items_done=items_done, items_failed=items_failed)

        except leo.LeonardoCreditsExhaustedError as e:
            update_pipeline_item(sb, item_id, status="queued", error_message=None)
            update_pipeline_run(
                sb,
                run_id,
                status="credits_exhausted",
                last_error=CREDITS_EXHAUSTED_ERROR,
                items_done=items_done,
                items_failed=items_failed,
            )
            return PipelineRunResult(
                run_id,
                "credits_exhausted",
                items_done=items_done,
                items_failed=items_failed,
                items_total=items_total,
                last_error=CREDITS_EXHAUSTED_ERROR,
                message=str(e),
            )
        except Exception as e:
            update_pipeline_item(
                sb,
                item_id,
                status="failed",
                error_message=str(e)[:500],
            )
            items_failed += 1
            update_pipeline_run(sb, run_id, items_failed=items_failed, items_done=items_done)
            continue


def resume_pipeline(
    sb,
    run_id: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineRunResult:
    run = get_pipeline_run(sb, run_id)
    if not run:
        return PipelineRunResult(run_id, "cancelled", message="Run not found")
    if run.get("status") != "credits_exhausted":
        return PipelineRunResult(
            run_id,
            run.get("status") or "cancelled",
            message="Only credits_exhausted runs can be resumed",
        )
    return run_pipeline(sb, run_id, resume=True, progress_callback=progress_callback)


def approve_pipeline_item(
    sb,
    item: dict,
    *,
    stories: Optional[List[dict]] = None,
) -> bool:
    """Approve a pending_review item: copy image to production path and update story_content_flat."""
    story_id = int(item["story_id"])
    reading_level = item["reading_level"]
    language_code = item.get("language_code") or "en"
    block_start = int(item["block_start"])
    pending_url = (item.get("pending_image_url") or "").strip()
    if not pending_url:
        return False

    pages = fetch_book_pages(sb, story_id, reading_level, language_code)
    blocks = group_pages_into_image_blocks(pages)
    block = next((b for b in blocks if b.get("block_start") == block_start), None)
    if not block:
        return False

    from lib import fetch_image_for_display, pending_image_display_url

    img_bytes = fetch_image_for_display(pending_image_display_url(item))
    if not img_bytes:
        return False
    opt = optimize_image_for_mobile(img_bytes)
    anchor = block["anchor_row"]
    pidx = page_number_for_row(anchor)
    url = upload_image_to_storage(sb, story_id, reading_level, pidx, opt)
    if not url:
        return False
    ok, _errs = apply_image_url_to_block(sb, block, url)
    if ok <= 0:
        return False
    update_pipeline_item(sb, str(item["id"]), status="approved")
    return True


def requeue_pipeline_item(sb, item: dict, run_id: str) -> bool:
    """Supersede a pending/failed item and enqueue a fresh generation."""
    old_id = str(item["id"])
    update_pipeline_item(sb, old_id, status="superseded")
    new_row = {
        "run_id": run_id,
        "story_id": item["story_id"],
        "reading_level": item["reading_level"],
        "language_code": item.get("language_code") or "en",
        "block_start": item["block_start"],
        "anchor_row_id": item["anchor_row_id"],
        "page_range_label": item.get("page_range_label") or "",
        "status": "queued",
    }
    inserted = insert_pipeline_items(sb, [new_row])
    if not inserted:
        return False
    # region agent log
    _dbg(
        run_id="requeue",
        hypothesis_id="H3",
        location="image_pipeline.py:requeue_pipeline_item",
        message="Requeued pipeline item",
        data={
            "supersededId": old_id,
            "newId": str(inserted[0].get("id")),
            "blockStart": item.get("block_start"),
            "oldPendingUrl": (item.get("pending_image_url") or "")[:120],
        },
    )
    # endregion
    update_pipeline_run(
        sb,
        run_id,
        status="running",
        items_total=int(get_pipeline_run(sb, run_id).get("items_total") or 0) + 1,
    )
    return True
