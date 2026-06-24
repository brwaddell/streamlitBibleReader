#!/usr/bin/env python3
"""
Regenerate male English audio for story pages using ElevenLabs and update Supabase.

Defaults:
- language_code = "en"
- ElevenLabs voice_id = "bV9ai9Wem8olqrkR49Zw"
- speed = 0.75

This script:
- Selects rows from story_content_flat with language_code = "en"
- Re-generates male audio for each page using the fixed voice/speed
- Uploads audio to R2 via existing helpers
- Updates audio_male_url and timing_male_json in story_content_flat

Safety features:
- --dry-run    : do not call ElevenLabs or write any changes
- --limit N    : process at most N rows (after filters)
- --story-id N : restrict to a single story_id
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Set
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from supabase import create_client

# Ensure project root is on sys.path when script is run directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib import (
    ELEVENLABS_VOICE_MALE_EN,
    PAGE_TEXT_COLUMN,
    TABLE_STORY_CONTENT_FLAT,
    _get_page_text,
    approve_audio_for_page,
    generate_elevenlabs_audio,
    page_number_for_row,
)


load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY_MALE") or ""

# Fixed ElevenLabs settings for this migration
ELEVENLABS_SPEED = 0.75
DEFAULT_PAGE_SIZE = 500
DEFAULT_FAILED_LOG = "scripts/regenerate_male_audio_failed.jsonl"


def _load_row_ids_from_log(path: str) -> List[str]:
    ids: List[str] = []
    seen: Set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(row.get("id") or "").strip()
            if rid and rid not in seen:
                seen.add(rid)
                ids.append(rid)
    return ids


def fetch_rows_by_ids(supabase, row_ids: List[str]):
    rows = []
    chunk_size = 100
    for i in range(0, len(row_ids), chunk_size):
        chunk = row_ids[i : i + chunk_size]
        r = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select(
                "id, story_id, language_code, reading_level, page_number, "
                f"{PAGE_TEXT_COLUMN}, audio_male_url"
            )
            .in_("id", chunk)
            .execute()
        )
        rows.extend(r.data or [])
    order = {rid: idx for idx, rid in enumerate(row_ids)}
    rows.sort(key=lambda row: order.get(str(row.get("id")), 10**9))
    return rows


def append_failed_log(path: str, row: dict, error: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entry = {
        "id": row.get("id"),
        "story_id": row.get("story_id"),
        "reading_level": row.get("reading_level"),
        "page_number": page_number_for_row(row),
        "error": error,
        "ts": int(time.time()),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _require_supabase() -> "Client":
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _require_elevenlabs_key(dry_run: bool) -> Optional[str]:
    if dry_run:
        # In dry-run we never call ElevenLabs, so key is optional.
        return ELEVENLABS_API_KEY or None
    if not ELEVENLABS_API_KEY:
        print("Missing ELEVENLABS_API_KEY (or ELEVENLABS_API_KEY_MALE) in environment.", file=sys.stderr)
        sys.exit(1)
    return ELEVENLABS_API_KEY


def fetch_target_rows(
    supabase,
    story_id: Optional[int],
    limit: Optional[int],
    start_offset: int = 0,
    min_url_age_hours: Optional[float] = None,
):
    """
    Fetch story_content_flat rows to process.

    Filters:
    - language_code = 'en'
    - optional story_id
    Ordered for deterministic processing:
    - story_id, reading_level, page_number
    """
    rows = []
    offset = max(0, int(start_offset or 0))
    page_size = DEFAULT_PAGE_SIZE
    cap = limit if (limit is not None and limit > 0) else None

    while True:
        batch_size = min(page_size, cap - len(rows)) if cap is not None else page_size
        if batch_size <= 0:
            break

        q = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select(
                "id, story_id, language_code, reading_level, page_number, "
                f"{PAGE_TEXT_COLUMN}, audio_male_url"
            )
            .eq("language_code", "en")
            .order("story_id")
            .order("reading_level")
            .order("page_number")
            .range(offset, offset + batch_size - 1)
        )
        if story_id is not None:
            q = q.eq("story_id", story_id)

        r = q.execute()
        batch = r.data or []
        if not batch:
            break

        rows.extend(batch)
        offset += len(batch)

        if len(batch) < batch_size:
            # Last page.
            break
        if cap is not None and len(rows) >= cap:
            rows = rows[:cap]
            break

    if min_url_age_hours is not None and min_url_age_hours > 0:
        rows = [
            row
            for row in rows
            if not _url_updated_within_hours((row.get("audio_male_url") or "").strip(), min_url_age_hours)
        ]

    return rows


def _url_updated_within_hours(url: Optional[str], hours: float) -> bool:
    """True when audio URL has ?v=<unix_ts> newer than the cutoff."""
    if not url or hours <= 0:
        return False
    try:
        query = parse_qs(urlparse(url).query)
        raw = (query.get("v") or [None])[0]
        if not raw:
            return False
        ts = int(raw)
    except (TypeError, ValueError):
        return False
    return ts >= int(time.time() - (hours * 3600))


def regenerate_row(
    supabase,
    elevenlabs_key: str,
    row: dict,
    dry_run: bool = False,
    skip_recent_hours: Optional[float] = None,
    force: bool = False,
) -> str:
    """Regenerate male audio for a single row. Returns True on success."""
    row_id = row.get("id")
    story_id = row.get("story_id")
    language_code = (row.get("language_code") or "en").strip() or "en"
    reading_level = (row.get("reading_level") or "").strip()
    page_num = page_number_for_row(row)
    text = _get_page_text(row)

    if not row_id or not story_id:
        print(f"  [skip] row missing id or story_id: {row}")
        return "skipped"
    if not text:
        print(f"  [skip] story {story_id} page {page_num}: empty text")
        return "skipped"

    old_url = (row.get("audio_male_url") or "").strip() or None
    if not force and skip_recent_hours and _url_updated_within_hours(old_url, skip_recent_hours):
        print(
            f"  [skip-recent] story {story_id}, reading_level={reading_level}, page={page_num}, "
            f"row_id={row_id} (audio_male_url updated within last {skip_recent_hours:g}h)"
        )
        return "skipped_recent"

    if dry_run:
        print(
            f"  [dry-run] would regenerate male audio for story {story_id}, "
            f"reading_level={reading_level}, page={page_num}, row_id={row_id}"
        )
        return "ok"

    audio_bytes, timing_json = generate_elevenlabs_audio(
        api_key=elevenlabs_key,
        voice_id=ELEVENLABS_VOICE_MALE_EN,
        text=text,
        language_code=language_code,
        speed=ELEVENLABS_SPEED,
    )
    if not audio_bytes:
        print(
            f"  [error] ElevenLabs failed for story {story_id}, "
            f"reading_level={reading_level}, page={page_num}, row_id={row_id}"
        )
        return "failed"

    url = approve_audio_for_page(
        supabase,
        row_id=row_id,
        story_id=story_id,
        language_code=language_code,
        reading_level=reading_level,
        gender="male",
        page_index=page_num,
        audio_bytes=audio_bytes,
        timing_json=timing_json,
        old_url=old_url,
    )
    if not url:
        print(
            f"  [error] approve_audio_for_page failed for story {story_id}, "
            f"reading_level={reading_level}, page={page_num}, row_id={row_id}"
        )
        return "failed"

    print(
        f"  [ok] story {story_id}, reading_level={reading_level}, page={page_num}, "
        f"row_id={row_id}, new_male_url={url[:80]}..."
    )
    return "ok"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate male English audio for story pages using ElevenLabs and update Supabase."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rows that would be processed without calling ElevenLabs or updating Supabase.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to process (after filters). Use this for small test batches.",
    )
    parser.add_argument(
        "--story-id",
        type=int,
        default=None,
        help="Restrict processing to a single story_id. Combine with --limit for a very small test batch.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Skip the first N matching rows (ordered by story_id, reading_level, page_number).",
    )
    parser.add_argument(
        "--skip-recent-hours",
        type=float,
        default=None,
        help="Skip rows whose audio_male_url has ?v= timestamp within the last N hours (already updated).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process rows even if audio_male_url was updated recently (ignores --skip-recent-hours).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between ElevenLabs calls (rate-limit safety). Default 1.0.",
    )
    parser.add_argument(
        "--failed-log",
        default=None,
        help=f"Append failed rows as JSONL for retry (default: {DEFAULT_FAILED_LOG} when not dry-run).",
    )
    parser.add_argument(
        "--retry-from-log",
        default=None,
        help="Only process row ids listed in a prior --failed-log JSONL file.",
    )
    parser.add_argument(
        "--no-failed-log",
        action="store_true",
        help="Do not write a failed-rows log file.",
    )
    parser.add_argument(
        "--min-url-age-hours",
        type=float,
        default=None,
        help=(
            "Only process rows whose audio_male_url is missing or older than N hours "
            "(useful to retry failures after a partial run without a failed log)."
        ),
    )
    args = parser.parse_args(argv)

    supabase = _require_supabase()
    eleven_key = _require_elevenlabs_key(dry_run=args.dry_run)

    print("Regenerating male audio for English pages in story_content_flat")
    print(f"  voice_id  = {ELEVENLABS_VOICE_MALE_EN}")
    print(f"  speed     = {ELEVENLABS_SPEED}")
    print(f"  dry_run   = {args.dry_run}")
    print(f"  limit     = {args.limit if args.limit is not None else 'none'}")
    print(f"  story_id  = {args.story_id if args.story_id is not None else 'all'}")
    print(f"  offset    = {args.start_offset}")
    print(
        f"  skip_recent_hours = "
        f"{args.skip_recent_hours if args.skip_recent_hours is not None else 'none'}"
    )
    print(f"  force     = {args.force}")
    print(f"  sleep     = {args.sleep}s")
    print(
        f"  min_url_age_hours = "
        f"{args.min_url_age_hours if args.min_url_age_hours is not None else 'none'}"
    )

    failed_log_path = None
    if not args.dry_run and not args.no_failed_log:
        failed_log_path = args.failed_log or DEFAULT_FAILED_LOG
        if args.retry_from_log:
            print(f"  retry_log = {args.retry_from_log}")
        else:
            print(f"  failed_log = {failed_log_path}")

    if args.retry_from_log:
        row_ids = _load_row_ids_from_log(args.retry_from_log)
        if not row_ids:
            print(f"No row ids found in {args.retry_from_log}", file=sys.stderr)
            return 1
        rows = fetch_rows_by_ids(supabase, row_ids)
        print(f"Retrying {len(rows)} row(s) from log.")
    else:
        rows = fetch_target_rows(
            supabase,
            story_id=args.story_id,
            limit=args.limit,
            start_offset=args.start_offset,
            min_url_age_hours=args.min_url_age_hours,
        )
    total = len(rows)
    print(f"Found {total} row(s) to consider.")

    processed = 0
    ok = 0
    skipped = 0
    skipped_recent = 0
    failed = 0

    for row in rows:
        processed += 1
        print(
            f"\nRow {processed}/{total}: "
            f"story_id={row.get('story_id')}, reading_level={row.get('reading_level')}, "
            f"page={page_number_for_row(row)}, id={row.get('id')}"
        )
        result = regenerate_row(
            supabase=supabase,
            elevenlabs_key=eleven_key or "",
            row=row,
            dry_run=args.dry_run,
            skip_recent_hours=args.skip_recent_hours,
            force=args.force or bool(args.retry_from_log),
        )
        if result == "ok":
            ok += 1
        elif result == "skipped_recent":
            skipped_recent += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
            if failed_log_path and not args.dry_run:
                append_failed_log(failed_log_path, row, result)

        if not args.dry_run and args.sleep > 0 and processed < total:
            time.sleep(args.sleep)

    print(
        "\nDone.\n"
        f"  processed: {processed}\n"
        f"  ok       : {ok}\n"
        f"  skipped  : {skipped}\n"
        f"  skipped_recent: {skipped_recent}\n"
        f"  failed   : {failed}\n"
    )
    if failed and failed_log_path and not args.dry_run:
        print(
            "Retry failures with:\n"
            f"  python3 scripts/regenerate_male_audio_en_batch.py "
            f"--retry-from-log {failed_log_path}\n"
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

