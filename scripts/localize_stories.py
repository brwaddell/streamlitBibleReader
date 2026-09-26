#!/usr/bin/env python3
"""
Translate English story_content_flat pages and record male/female audio.

Writes new language rows (page text + copied image_url) via OpenAI, then
ElevenLabs with-timestamps audio for pages missing a male or female track.
Existing rows and existing audio are skipped, so the script can be resumed.

Requires SUPABASE_URL, SUPABASE_SERVICE_KEY, and — unless --dry-run —
OPENAI_API_KEY for translation and ELEVENLABS_API_KEY for audio.
"""

import argparse
import os
import sys
import time
from typing import Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from languages import (
    READING_LEVEL_LABELS,
    TRANSLATION_TARGET_CODES,
    elevenlabs_language_code,
    language_label,
    translation_system_prompt,
)
from lib import (
    ELEVENLABS_TTS_MODEL_ID,
    ELEVENLABS_TTS_OUTPUT_FORMAT,
    ELEVENLABS_VOICE_FEMALE_DEFAULT,
    PAGE_NUMBER_COLUMN,
    PAGE_TEXT_COLUMN,
    TABLE_STORY_CONTENT_FLAT,
    _get_page_text,
    approve_audio_for_page,
    audio_tts_speed_for_grade,
    elevenlabs_voice_male_for_language,
    generate_elevenlabs_audio,
    insert_book_page,
    page_number_for_row,
)


PageKey = Tuple[int, str, int]


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        print(f"Missing {name} in the environment.", file=sys.stderr)
        sys.exit(1)
    return value


def _fetch_pages(supabase, language_code: str, story_id: Optional[int], reading_level: Optional[str]) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    page_size = 500
    columns = (
        f"id, story_id, language_code, reading_level, {PAGE_NUMBER_COLUMN}, "
        f"{PAGE_TEXT_COLUMN}, image_url, audio_male_url, audio_female_url"
    )
    while True:
        query = (
            supabase.table(TABLE_STORY_CONTENT_FLAT)
            .select(columns)
            .eq("language_code", language_code)
            .order("story_id")
            .order("reading_level")
            .order(PAGE_NUMBER_COLUMN)
            .range(offset, offset + page_size - 1)
        )
        if story_id is not None:
            query = query.eq("story_id", story_id)
        if reading_level:
            query = query.eq("reading_level", reading_level)
        batch = query.execute().data or []
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < page_size:
            break
    return rows


def _page_key(row: dict) -> PageKey:
    return (
        int(row["story_id"]),
        str(row.get("reading_level") or ""),
        int(page_number_for_row(row)),
    )


def _existing_keys(rows: Iterable[dict]) -> Set[PageKey]:
    return {_page_key(row) for row in rows}


def _translate_text(client: OpenAI, english_text: str, target_code: str, reading_level: str, model: str) -> str:
    label = READING_LEVEL_LABELS.get(reading_level, reading_level.replace("_", " ").title())
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": translation_system_prompt(target_code, label)},
            {"role": "user", "content": english_text.strip()},
        ],
        max_tokens=1200,
    )
    return (response.choices[0].message.content or "").strip()


def _has_audio(row: dict, gender: str) -> bool:
    url = row.get("audio_male_url") if gender == "male" else row.get("audio_female_url")
    return bool((url or "").strip())


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    supabase = create_client(_require_env("SUPABASE_URL"), _require_env("SUPABASE_SERVICE_KEY"))
    targets = [code.strip().lower() for code in args.languages.split(",") if code.strip()]
    invalid = [code for code in targets if code not in TRANSLATION_TARGET_CODES]
    if invalid:
        print(
            f"Unknown language code(s): {', '.join(invalid)}. "
            f"Choose from {', '.join(TRANSLATION_TARGET_CODES)}.",
            file=sys.stderr,
        )
        return 1

    english_pages = _fetch_pages(supabase, "en", args.story_id, args.reading_level)
    print(f"English pages in scope: {len(english_pages)}")
    if not english_pages:
        return 0

    do_translate = args.translate or not args.audio
    do_audio = args.audio or not args.translate
    openai_client = None
    eleven_key = None
    if not args.dry_run and do_translate:
        openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    if not args.dry_run and do_audio:
        eleven_key = _require_env("ELEVENLABS_API_KEY")

    failures = 0
    for code in targets:
        print(f"\n=== {language_label(code)} ===")
        existing_rows = _fetch_pages(supabase, code, args.story_id, args.reading_level)
        existing = _existing_keys(existing_rows)
        missing = [row for row in english_pages if _page_key(row) not in existing]
        print(f"  existing rows: {len(existing_rows)}  missing translations: {len(missing)}")

        translated = 0
        if do_translate:
            work = missing[: args.limit] if args.limit else missing
            for index, source in enumerate(work, start=1):
                key = _page_key(source)
                english_text = _get_page_text(source)
                if not english_text:
                    print(f"  skip empty English page {key}")
                    continue
                if args.dry_run:
                    print(f"  [dry-run] translate story={key[0]} {key[1]} page={key[2]}")
                    translated += 1
                    continue
                try:
                    text = _translate_text(openai_client, english_text, code, key[1], args.model)
                except Exception as exc:
                    failures += 1
                    print(f"  translation failed {key}: {exc}", file=sys.stderr)
                    continue
                if not text:
                    failures += 1
                    print(f"  empty translation {key}", file=sys.stderr)
                    continue
                ok = insert_book_page(
                    supabase,
                    story_id=key[0],
                    language_code=code,
                    reading_level=key[1],
                    page_index=key[2],
                    page_text=text,
                    image_url=(source.get("image_url") or None),
                )
                if not ok:
                    failures += 1
                    print(f"  insert failed {key}", file=sys.stderr)
                    continue
                translated += 1
                print(f"  saved {index}/{len(work)} story={key[0]} {key[1]} page={key[2]}")
                if args.sleep:
                    time.sleep(args.sleep)
            print(f"  translations written: {translated}")

        if not do_audio:
            continue

        audio_rows = _fetch_pages(supabase, code, args.story_id, args.reading_level)
        pending = [row for row in audio_rows if not _has_audio(row, "male") or not _has_audio(row, "female")]
        if args.limit:
            pending = pending[: args.limit]
        print(f"  pages missing male and/or female audio: {len(pending)}")
        recorded = 0
        tts_lang = elevenlabs_language_code(code)
        voice_male = elevenlabs_voice_male_for_language(code)
        for index, row in enumerate(pending, start=1):
            page_text = _get_page_text(row)
            page_index = page_number_for_row(row)
            story_id = int(row["story_id"])
            reading_level = str(row.get("reading_level") or "")
            speed = audio_tts_speed_for_grade(reading_level)
            if not page_text or row.get("id") is None:
                continue
            if args.dry_run:
                need = []
                if not _has_audio(row, "male"):
                    need.append("male")
                if not _has_audio(row, "female"):
                    need.append("female")
                print(
                    f"  [dry-run] audio {','.join(need)} story={story_id} {reading_level} "
                    f"page={page_index} lang={tts_lang}"
                )
                recorded += 1
                continue
            for gender, voice_id in (("male", voice_male), ("female", ELEVENLABS_VOICE_FEMALE_DEFAULT)):
                if _has_audio(row, gender):
                    continue
                audio_bytes, timing = generate_elevenlabs_audio(
                    eleven_key,
                    voice_id,
                    page_text,
                    tts_lang,
                    stability=0.5,
                    similarity_boost=0.75,
                    use_speaker_boost=False,
                    speed=speed,
                    model_id=ELEVENLABS_TTS_MODEL_ID,
                    output_format=ELEVENLABS_TTS_OUTPUT_FORMAT,
                    optimize_streaming_latency=0,
                    apply_text_normalization="auto",
                )
                if not audio_bytes:
                    failures += 1
                    print(
                        f"  audio failed {gender} story={story_id} {reading_level} page={page_index}",
                        file=sys.stderr,
                    )
                    continue
                old_url = row.get("audio_male_url") if gender == "male" else row.get("audio_female_url")
                url = approve_audio_for_page(
                    supabase,
                    row_id=row["id"],
                    story_id=story_id,
                    language_code=code,
                    reading_level=reading_level,
                    gender=gender,
                    page_index=page_index,
                    audio_bytes=audio_bytes,
                    timing_json=timing,
                    old_url=old_url or None,
                )
                if not url:
                    failures += 1
                    print(
                        f"  upload failed {gender} story={story_id} {reading_level} page={page_index}",
                        file=sys.stderr,
                    )
                    continue
                if gender == "male":
                    row["audio_male_url"] = url
                else:
                    row["audio_female_url"] = url
                if args.sleep:
                    time.sleep(args.sleep)
            recorded += 1
            print(f"  audio {index}/{len(pending)} story={story_id} {reading_level} page={page_index}")
        print(f"  audio pages handled: {recorded}")

    if failures:
        print(f"\nFinished with {failures} failure(s).", file=sys.stderr)
        return 1
    print("\nFinished.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate stories and record ElevenLabs audio.")
    parser.add_argument(
        "--languages",
        default=",".join(TRANSLATION_TARGET_CODES),
        help="Comma-separated language codes (default: every target except English).",
    )
    parser.add_argument("--story-id", type=int, default=None)
    parser.add_argument("--reading-level", default=None, help="e.g. grade_1")
    parser.add_argument("--translate", action="store_true", help="Only translate (skip audio).")
    parser.add_argument("--audio", action="store_true", help="Only record audio for rows that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Count work and do not call APIs or write rows.")
    parser.add_argument("--limit", type=int, default=None, help="Max pages per language for each phase.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between API calls.")
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
