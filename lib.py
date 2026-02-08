"""Shared utilities for Storybook Image Processor."""
import os
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st
from PIL import Image

from auth import get_secret

READING_LEVELS = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
STORAGE_BUCKET = "storybook-images"
MAX_IMAGE_SIZE = 800
TARGET_BYTES = 100_000


def get_supabase():
    """Get Supabase client (cached in session state)."""
    if getattr(st.session_state, "supabase", None) is None:
        url = get_secret("SUPABASE_URL")
        key = get_secret("SUPABASE_SERVICE_KEY")
        if not url or not key:
            st.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
            return None
        from supabase import create_client
        st.session_state.supabase = create_client(url, key)
    return st.session_state.supabase


def fetch_stories():
    """Fetch stories from Supabase for dropdown."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        r = sb.table("stories").select("id, title, description").order("created_at", desc=True).execute()
        return r.data or []
    except Exception as e:
        st.error(f"Failed to fetch stories: {e}")
        return []


def fetch_book_pages(supabase, story_id: int, reading_level: str, search: str = ""):
    """Fetch book_pages filtered by story_id and reading_level, optionally search page_text."""
    try:
        q = (
            supabase.table("book_pages")
            .select("*")
            .eq("story_id", story_id)
            .eq("reading_level", reading_level)
            .order("page_index")
        )
        r = q.execute()
        rows = r.data or []
        if search and search.strip():
            s = search.strip().lower()
            rows = [row for row in rows if s in (row.get("page_text") or "").lower()]
        return rows
    except Exception as e:
        st.error(f"Failed to fetch book pages: {e}")
        return []


def delete_book_page(supabase, row_id):
    """Delete a book_page row by id."""
    try:
        supabase.table("book_pages").delete().eq("id", row_id).execute()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def update_book_page(supabase, row_id: int, page_text: Optional[str] = None, image_url: Optional[str] = None):
    """Update a book_page row by id."""
    try:
        updates = {}
        if page_text is not None:
            updates["page_text"] = page_text
        if image_url is not None:
            updates["image_url"] = image_url
        if not updates:
            return True
        supabase.table("book_pages").update(updates).eq("id", row_id).execute()
        return True
    except Exception as e:
        st.error(f"Update failed: {e}")
        return False


def optimize_image_for_mobile(image_bytes: bytes) -> bytes:
    """Resize to max 800x800 and convert to WebP."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if w > MAX_IMAGE_SIZE or h > MAX_IMAGE_SIZE:
        ratio = min(MAX_IMAGE_SIZE / w, MAX_IMAGE_SIZE / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    for quality in [85, 80, 75, 70, 65, 60]:
        buf.seek(0)
        buf.truncate(0)
        img.save(buf, "WEBP", quality=quality)
        if buf.tell() <= TARGET_BYTES:
            break
    buf.seek(0)
    return buf.read()


def upload_image_to_storage(supabase, story_id: int, reading_level: str, page_index: int, image_bytes: bytes) -> Optional[str]:
    """Upload image to Supabase Storage. Returns public URL or None."""
    path = f"{story_id}/{reading_level}/page_{page_index}.webp"
    try:
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path, image_bytes, file_options={"content-type": "image/webp", "upsert": "true"}
        )
        return supabase.storage.from_(STORAGE_BUCKET).get_public_url(path)
    except Exception as e:
        st.error(f"Upload failed for {path}: {e}")
        return None


@st.dialog("Replace image", width="medium")
def replace_image_modal(row: dict, story_id: int, reading_level: str):
    """Modal to replace image: upload to Supabase Storage and update book_pages.image_url."""
    st.caption(
        "The new image will be uploaded to Supabase Storage and the book_pages row will be updated with the new URL."
    )
    new_img = st.file_uploader("Choose image", type=["png", "jpg", "jpeg", "webp"], key="bp_modal_upload")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload & update", type="primary"):
            if not new_img:
                st.error("Select an image first.")
            else:
                data = new_img.read()
                opt = optimize_image_for_mobile(data)
                url = upload_image_to_storage(
                    get_supabase(), story_id, reading_level, row["page_index"], opt
                )
                if url and update_book_page(get_supabase(), row["id"], image_url=url):
                    st.success("Image uploaded to Supabase. URL updated in book_pages.")
                    if "bp_editing_image" in st.session_state:
                        del st.session_state.bp_editing_image
                    st.rerun()
    with col2:
        if st.button("Remove image", type="secondary"):
            if update_book_page(get_supabase(), row["id"], image_url=""):
                st.success("Image removed.")
                if "bp_editing_image" in st.session_state:
                    del st.session_state.bp_editing_image
                st.rerun()


def run_book_pages_view():
    """Render the Book Pages CRUD view in an Airtable-like table format."""
    st.title("Book Pages")
    st.caption("View, search, and delete exported book pages from Supabase.")
    sb = get_supabase()
    if not sb:
        return
    stories = fetch_stories()
    if not stories:
        st.warning("No stories found. Create stories in Supabase first.")
        return
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        story_options = {f"{s['title']} (id: {s['id']})": s["id"] for s in stories}
        story_label = st.selectbox("Story", options=list(story_options.keys()), key="bp_story")
        story_id = story_options.get(story_label) if story_label else None
    with col2:
        reading_level = st.selectbox("Reading Level", READING_LEVELS, key="bp_reading_level")
    with col3:
        search = st.text_input("Search page text", placeholder="Filter by page text...", key="bp_search")
    if not story_id:
        st.info("Select a story to load book pages.")
        return
    rows = fetch_book_pages(sb, story_id, reading_level, search)
    if not rows:
        st.info("No book pages found for this story and reading level.")
        return

    # Table header
    h1, h2, h3, h4 = st.columns([0.5, 3, 1, 0.3])
    with h1:
        st.markdown("**Page**")
    with h2:
        st.markdown("**Page Text**")
    with h3:
        st.markdown("**Preview**")
    with h4:
        st.markdown("**🗑️**")
    st.divider()

    # Table rows
    edited_texts = {}
    for r in rows:
        c1, c2, c3, c4 = st.columns([0.5, 3, 1, 0.3])
        with c1:
            st.write(r.get("page_index"))
        with c2:
            edited_texts[r["id"]] = st.text_area(
                "Text",
                value=r.get("page_text") or "",
                height=80,
                key=f"bp_text_{r['id']}",
                label_visibility="collapsed",
            )
        with c3:
            img_url = r.get("image_url") or ""
            if img_url:
                try:
                    st.image(img_url, width=80)
                except Exception:
                    st.caption("(image)")
            else:
                st.caption("No image")
            if st.button("Replace", key=f"img_{r['id']}", help="Click to replace image — uploads to Supabase, updates book_pages"):
                st.session_state.bp_editing_image = r
                st.rerun()
        with c4:
            if st.button("🗑️", key=f"del_{r['id']}", help="Delete row"):
                st.session_state.bp_pending_delete = [r["id"]]
                st.rerun()

    # Replace image modal
    if st.session_state.get("bp_editing_image"):
        replace_image_modal(
            st.session_state.bp_editing_image, story_id, reading_level
        )

    # Save text edits
    text_changes = [
        (r["id"], edited_texts[r["id"]])
        for r in rows
        if edited_texts[r["id"]] != (r.get("page_text") or "")
    ]
    if text_changes and st.button("💾 Save text edits"):
        for rid, new_text in text_changes:
            update_book_page(sb, rid, page_text=new_text)
        st.success(f"Saved {len(text_changes)} edit(s).")
        st.rerun()

    # Delete confirmation
    pending = st.session_state.get("bp_pending_delete", [])
    if pending:
        st.warning(f"Delete {len(pending)} row(s)? This cannot be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete"):
                for rid in pending:
                    delete_book_page(sb, rid)
                del st.session_state.bp_pending_delete
                st.success(f"Deleted {len(pending)} row(s).")
                st.rerun()
        with c2:
            if st.button("Cancel"):
                del st.session_state.bp_pending_delete
                st.rerun()

    st.caption(
        "Edit text and click Save. Click the preview to replace image (uploads to Supabase, updates book_pages). "
        "Click 🗑️ to delete (with confirmation)."
    )
