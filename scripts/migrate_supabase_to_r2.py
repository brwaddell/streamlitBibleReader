#!/usr/bin/env python3
"""
Migrate images and audio from Supabase Storage to Cloudflare R2.
Downloads each file from Supabase, uploads to R2, updates story_content_flat URLs.
Run once; skips URLs that already point to R2.
"""
import os
import re
from typing import Optional, Tuple

import boto3
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_IMAGES = os.getenv("R2_BUCKET_IMAGES", "storybook-images")
R2_BUCKET_AUDIO = os.getenv("R2_BUCKET_AUDIO", "storybook-audio")
R2_PUBLIC_URL_IMAGES = os.getenv("R2_PUBLIC_URL_IMAGES", "").rstrip("/")
R2_PUBLIC_URL_AUDIO = os.getenv("R2_PUBLIC_URL_AUDIO", "").rstrip("/")

# Supabase public URL: .../storage/v1/object/public/<bucket>/<path>
SUPABASE_PATH_RE = re.compile(r"/storage/v1/object/public/([^/]+)/(.+)$")


def is_supabase_url(url: str) -> bool:
    return url and "supabase" in url and "/storage/" in url


def extract_path_from_supabase_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (bucket, path) or (None, None)."""
    m = SUPABASE_PATH_RE.search(url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def download(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def main():
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY]):
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        return 1
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PUBLIC_URL_IMAGES, R2_PUBLIC_URL_AUDIO]):
        print("Missing R2 credentials or public URLs. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PUBLIC_URL_IMAGES, R2_PUBLIC_URL_AUDIO")
        return 1

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    s3 = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )

    r = supabase.table("story_content_flat").select("id, image_url, audio_male_url, audio_female_url").execute()
    rows = r.data or []
    print(f"Found {len(rows)} rows in story_content_flat")

    updated = 0
    errors = 0
    for row in rows:
        rid = row.get("id")
        if not rid:
            continue
        updates = {}

        for field, bucket_name, r2_bucket, base_url, content_type in [
            ("image_url", "storybook-images", R2_BUCKET_IMAGES, R2_PUBLIC_URL_IMAGES, "image/webp"),
            ("audio_male_url", "storybook-audio", R2_BUCKET_AUDIO, R2_PUBLIC_URL_AUDIO, "audio/mpeg"),
            ("audio_female_url", "storybook-audio", R2_BUCKET_AUDIO, R2_PUBLIC_URL_AUDIO, "audio/mpeg"),
        ]:
            url = (row.get(field) or "").strip()
            if not url or not is_supabase_url(url):
                continue

            sb_bucket, path = extract_path_from_supabase_url(url)
            if not sb_bucket or not path:
                print(f"  Row {rid} {field}: could not parse path from {url}")
                errors += 1
                continue

            data = download(url)
            if not data:
                errors += 1
                continue

            try:
                s3.put_object(Bucket=r2_bucket, Key=path, Body=data, ContentType=content_type)
                new_url = f"{base_url}/{path}"
                updates[field] = new_url
                print(f"  Row {rid} {field}: migrated -> {new_url[:60]}...")
            except Exception as e:
                print(f"  Row {rid} {field}: R2 upload failed: {e}")
                errors += 1

        if updates:
            try:
                supabase.table("story_content_flat").update(updates).eq("id", rid).execute()
                updated += 1
            except Exception as e:
                print(f"  Row {rid}: DB update failed: {e}")
                errors += 1

    print(f"\nDone. Updated {updated} rows, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
