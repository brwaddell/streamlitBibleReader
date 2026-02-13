"""Cloudflare R2 storage for images and audio. S3-compatible via boto3."""
from typing import Optional

from auth import get_secret


_r2_client = None


def _get_r2_client():
    """Lazy-init boto3 S3 client configured for R2."""
    global _r2_client
    if _r2_client is not None:
        return _r2_client
    account_id = get_secret("R2_ACCOUNT_ID")
    access_key = get_secret("R2_ACCESS_KEY_ID")
    secret_key = get_secret("R2_SECRET_ACCESS_KEY")
    if not account_id or not access_key or not secret_key:
        return None
    import boto3

    _r2_client = boto3.client(
        service_name="s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return _r2_client


def _ensure_image_domain(url: str) -> str:
    """Ensure image URLs use image.aptreelearning.com, not bare aptreelearning.com."""
    if not url:
        return url
    u = url.rstrip("/")
    if "aptreelearning.com" in u and "image.aptreelearning.com" not in u:
        return "https://image.aptreelearning.com"
    return u


def _ensure_audio_domain(url: str) -> str:
    """Ensure audio URLs use audio.aptreelearning.com, not bare aptreelearning.com."""
    if not url:
        return url
    u = url.rstrip("/")
    if "aptreelearning.com" in u and "audio.aptreelearning.com" not in u:
        return "https://audio.aptreelearning.com"
    return u


def upload_image(path: str, image_bytes: bytes) -> Optional[str]:
    """Upload image to R2. Returns public URL or None."""
    client = _get_r2_client()
    bucket = get_secret("R2_BUCKET_IMAGES", "storybook-images")
    base_url = get_secret("R2_PUBLIC_URL_IMAGES")
    if not client or not base_url:
        return None
    base_url = _ensure_image_domain(base_url)
    try:
        client.put_object(
            Bucket=bucket,
            Key=path,
            Body=image_bytes,
            ContentType="image/webp",
        )
        base = base_url.rstrip("/")
        return f"{base}/{path}"
    except Exception as e:
        import streamlit as st
        st.error(f"R2 upload failed for {path}: {e}")
        return None


def upload_audio(path: str, audio_bytes: bytes) -> Optional[str]:
    """Upload audio to R2. Returns public URL or None."""
    client = _get_r2_client()
    bucket = get_secret("R2_BUCKET_AUDIO", "storybook-audio")
    base_url = get_secret("R2_PUBLIC_URL_AUDIO")
    if not client or not base_url:
        return None
    base_url = _ensure_audio_domain(base_url)
    try:
        client.put_object(
            Bucket=bucket,
            Key=path,
            Body=audio_bytes,
            ContentType="audio/mpeg",
        )
        base = base_url.rstrip("/")
        return f"{base}/{path}"
    except Exception as e:
        import streamlit as st
        st.error(f"R2 audio upload failed for {path}: {e}")
        return None
