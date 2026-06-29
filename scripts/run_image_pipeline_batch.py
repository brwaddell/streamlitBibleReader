#!/usr/bin/env python3
"""
Headless batch image pipeline for remaining missing story images.

Uses image_pipeline.py (Leonardo + OpenAI prep) outside Streamlit.

Examples:
  python3 scripts/run_image_pipeline_batch.py --scan-only
  python3 scripts/run_image_pipeline_batch.py --dry-run
  python3 scripts/run_image_pipeline_batch.py --run --limit 1
  python3 scripts/run_image_pipeline_batch.py --run
  python3 scripts/run_image_pipeline_batch.py --resume RUN_UUID
"""

import argparse
import os
import sys
from typing import List, Optional, Set

from dotenv import load_dotenv
from supabase import create_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lib
from image_pipeline import (
    CREDITS_EXHAUSTED_ERROR,
    WorkItem,
    collect_processable_blocks,
    create_pipeline_run,
    discover_work_items,
    resume_pipeline,
    run_pipeline,
    supersede_pending_review_items,
)
from lib import block_needs_image, fetch_book_pages, fetch_pipeline_review_queue, group_pages_into_image_blocks


def _load_env() -> None:
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def _require_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    sb = create_client(url, key)
    lib.get_supabase = lambda: sb
    return sb


def _parse_story_ids(values: Optional[List[str]]) -> Optional[Set[int]]:
    if not values:
        return None
    return {int(v) for v in values}


def _filter_work_items(
    work_items: List[WorkItem],
    story_ids: Optional[Set[int]],
) -> List[WorkItem]:
    if story_ids is None:
        return work_items
    return [w for w in work_items if w.story_id in story_ids]


def _print_scan(
    sb,
    work_items: List[WorkItem],
    *,
    language_code: str,
    story_ids: Optional[Set[int]],
) -> None:
    total_blocks = sum(w.missing_block_count for w in work_items)
    pending = fetch_pipeline_review_queue(sb, language_code=language_code)
    if story_ids is not None:
        pending = [p for p in pending if int(p.get("story_id") or 0) in story_ids]

    rows, stats = collect_processable_blocks(sb, work_items)
    after_supersede, after_stats = collect_processable_blocks(
        sb, work_items, ignore_pending_review=True
    )
    unique_stories = sorted({w.story_id for w in work_items})

    print("=== Missing image inventory ===")
    print(f"  language_code          = {language_code}")
    print(f"  unique stories         = {len(unique_stories)}")
    print(f"  story/grade versions   = {len(work_items)}")
    print(f"  missing blocks         = {total_blocks}")
    print(f"  pages (3 per block)    = {total_blocks * 3}")
    print(f"  pending review (bad)   = {len(pending)}")
    print(f"  processable now        = {stats['processable']}")
    print(f"  after supersede        = {after_stats['processable']}")
    print(f"  blocked (active item)  = {stats['skipped_active']}")
    print()

    if pending:
        print("=== Pending review (superseded on --run unless --keep-pending) ===")
        for item in pending:
            print(
                f"  story_id={item.get('story_id')} | {item.get('reading_level')} | "
                f"pages {item.get('page_range_label') or item.get('block_start')}"
            )
        print()

    print("=== By story/grade ===")
    for w in sorted(work_items, key=lambda x: (x.story_id, x.reading_level)):
        pages = fetch_book_pages(sb, w.story_id, w.reading_level, w.language_code)
        blocks = group_pages_into_image_blocks(pages)
        missing_blocks = [b for b in blocks if block_needs_image(b)]
        ranges = [b.get("page_range_label", "?") for b in missing_blocks]
        shown = ", ".join(ranges[:8])
        if len(ranges) > 8:
            shown += "..."
        title = w.story_title[:40]
        print(
            f"  story_id={w.story_id} | {w.reading_level} | "
            f"{w.missing_block_count} blocks | {title} | {shown}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate missing story images via the headless image pipeline."
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Print missing-image inventory and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show blocks that would be enqueued (after superseding pending review).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Supersede bad pending-review items, enqueue missing blocks, and run pipeline.",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a credits_exhausted pipeline run.",
    )
    parser.add_argument(
        "--keep-pending",
        action="store_true",
        help="Do not supersede pending_review items before --run.",
    )
    parser.add_argument(
        "--story-id",
        type=int,
        action="append",
        dest="story_ids",
        help="Restrict to one or more story_id values (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of blocks to enqueue/process.",
    )
    parser.add_argument(
        "--language-code",
        default="en",
        help="Language code to process (default: en).",
    )
    args = parser.parse_args(argv)

    if not any([args.scan_only, args.dry_run, args.run, args.resume]):
        args.scan_only = True

    _load_env()
    sb = _require_supabase()
    story_ids = _parse_story_ids(args.story_ids)
    language_code = (args.language_code or "en").strip() or "en"

    if args.resume:
        run_id = args.resume.strip()
        print(f"Resuming pipeline run {run_id}...")

        def _on_progress(msg: str, done: int, total: int) -> None:
            print(f"  [{done}/{total}] {msg}")

        result = resume_pipeline(sb, run_id, progress_callback=_on_progress)
        print(f"Status: {result.status}")
        if result.message:
            print(result.message)
        if result.last_error:
            print(result.last_error, file=sys.stderr)
        if result.status == "credits_exhausted":
            print(f"Re-run with: python3 scripts/run_image_pipeline_batch.py --resume {run_id}")
            return 2
        return 0 if result.status == "completed" else 1

    work_items = _filter_work_items(
        discover_work_items(sb, language_code=language_code),
        story_ids,
    )

    if args.scan_only:
        _print_scan(sb, work_items, language_code=language_code, story_ids=story_ids)
        return 0

    ignore_pending = not args.keep_pending
    preview_rows, preview_stats = collect_processable_blocks(
        sb,
        work_items,
        limit=args.limit,
        ignore_pending_review=ignore_pending,
    )

    if args.dry_run:
        if ignore_pending:
            pending_count = fetch_pipeline_review_queue(sb, language_code=language_code)
            if story_ids is not None:
                pending_count = [
                    p for p in pending_count if int(p.get("story_id") or 0) in story_ids
                ]
            print(f"Would supersede {len(pending_count)} pending_review item(s).")
        print(f"Would enqueue {len(preview_rows)} block(s) (limit={args.limit or 'none'})")
        for row in preview_rows:
            print(
                f"  story_id={row['story_id']} | {row['reading_level']} | "
                f"pages {row.get('page_range_label') or row.get('block_start')}"
            )
        return 0

    if args.run:
        if ignore_pending:
            n = supersede_pending_review_items(
                sb, language_code=language_code, story_ids=story_ids
            )
            if n:
                print(f"Superseded {n} pending_review item(s) for regeneration.")

        rows, stats = collect_processable_blocks(
            sb,
            work_items,
            limit=args.limit,
            ignore_pending_review=False,
        )
        if not rows:
            print("No blocks to process.")
            return 0

        run_id = create_pipeline_run(
            sb,
            work_items,
            language_code=language_code,
            limit=args.limit,
            ignore_pending_review=False,
        )
        if not run_id:
            print("Failed to create pipeline run.", file=sys.stderr)
            return 1

        print(f"Created pipeline run {run_id} with {len(rows)} block(s). Running...")

        def _on_progress(msg: str, done: int, total: int) -> None:
            print(f"  [{done}/{total}] {msg}")

        result = run_pipeline(sb, run_id, progress_callback=_on_progress)
        print(f"Status: {result.status}")
        if result.message:
            print(result.message)
        if result.last_error:
            print(result.last_error, file=sys.stderr)
        if result.status == "credits_exhausted":
            print(CREDITS_EXHAUSTED_ERROR)
            print(f"Re-run with: python3 scripts/run_image_pipeline_batch.py --resume {run_id}")
            return 2
        print("Review new images in Streamlit → Image Processor → section 0.")
        return 0 if result.status == "completed" else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
