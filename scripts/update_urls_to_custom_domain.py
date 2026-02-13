#!/usr/bin/env python3
"""
Update story_content_flat URLs to use custom domains:
  - image.aptreelearning.com for images
  - audio.aptreelearning.com for audio
Replaces Supabase or R2 pub URLs with your custom domain URLs.
"""
import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
IMAGE_DOMAIN = os.getenv("IMAGE_DOMAIN", "https://image.aptreelearning.com").rstrip("/")
AUDIO_DOMAIN = os.getenv("AUDIO_DOMAIN", "https://audio.aptreelearning.com").rstrip("/")

# Supabase: .../public/<bucket>/<path>
SUPABASE_PATH_RE = re.compile(r"/storage/v1/object/public/([^/]+)/(.+)$")


def extract_path(url: str) -> Optional[str]:
    """Extract object path from Supabase, R2 pub, or custom domain URL."""
    if not url or not url.strip():
        return None
    url = url.strip()
    # Supabase
    m = SUPABASE_PATH_RE.search(url)
    if m:
        return m.group(2)
    # Any other URL: path is everything after the domain (first slash)
    parsed = urlparse(url)
    path = (parsed.path or "").lstrip("/")
    return path if path else None


def is_image_url(url: str) -> bool:
    """Heuristic: .webp or known image path patterns."""
    if not url:
        return False
    return ".webp" in url.lower() or "storybook-images" in url


def is_audio_url(url: str) -> bool:
    """Heuristic: .mp3 or known audio path patterns."""
    if not url:
        return False
    return ".mp3" in url.lower() or "storybook-audio" in url


def rewrite_url(url: str) -> Optional[str]:
    """Rewrite URL to use custom domain. Returns new URL or None if unchanged/unrecognized."""
    path = extract_path(url)
    if not path:
        return None
    if is_image_url(url):
        return f"{IMAGE_DOMAIN}/{path}"
    if is_audio_url(url):
        return f"{AUDIO_DOMAIN}/{path}"
    return None


def main():
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY]):
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        return 1

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    r = supabase.table("story_content_flat").select("id, image_url, audio_male_url, audio_female_url").execute()
    rows = r.data or []
    print(f"Found {len(rows)} rows")
    print(f"Image domain: {IMAGE_DOMAIN}")
    print(f"Audio domain: {AUDIO_DOMAIN}")

    updated = 0
    for row in rows:
        rid = row.get("id")
        if not rid:
            continue
        updates = {}

        for field in ("image_url", "audio_male_url", "audio_female_url"):
            url = (row.get(field) or "").strip()
            if not url:
                continue
            new_url = rewrite_url(url)
            if new_url and new_url != url:
                updates[field] = new_url
                print(f"  Row {rid} {field}: {url[:50]}... -> {new_url[:60]}...")

        if updates:
            try:
                supabase.table("story_content_flat").update(updates).eq("id", rid).execute()
                updated += 1
            except Exception as e:
                print(f"  Row {rid}: DB update failed: {e}")

    print(f"\nDone. Updated {updated} rows.")
    return 0


if __name__ == "__main__":
    exit(main())
