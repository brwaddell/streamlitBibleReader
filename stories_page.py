"""Stories CRUD: list view with expandable cards; edit title, description, free, published per story."""

import streamlit as st

from lib import delete_story, fetch_stories, get_supabase, insert_story, update_story


def run_stories_view():
    """Render a simple Stories list: each story in an expander with inline edit and free/published toggles."""
    st.title("Story Titles")
    st.caption("Edit title, description, and set Free / Published per story.")

    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories() or []
    search_q = st.text_input(
        "Search",
        value=st.session_state.get("stories_search_q", ""),
        placeholder="Search title or description...",
        key="stories_search_q",
    ).strip().lower()

    if search_q:
        stories = [
            s for s in stories
            if search_q in (s.get("title") or "").lower() or search_q in (s.get("description") or "").lower()
        ]
    st.caption(f"Showing {len(stories)} story" + ("s" if len(stories) != 1 else "") + ".")

    # --- Add new story ---
    with st.expander("Add new story", expanded=False):
        new_title = st.text_input("Title", key="new_story_title", placeholder="Story title")
        new_description = st.text_area("Description", key="new_story_desc", placeholder="Optional description", height=80)
        c1, c2 = st.columns(2)
        with c1:
            new_free = st.checkbox("Free", value=False, key="new_story_free")
        with c2:
            new_published = st.checkbox("Published", value=False, key="new_story_published")
        if st.button("Create story", type="primary", key="create_story_btn"):
            if not (new_title and new_title.strip()):
                st.warning("Title is required.")
            else:
                created = insert_story(sb, {
                    "title": new_title.strip(),
                    "description": (new_description or "").strip(),
                    "free": new_free,
                    "published": new_published,
                })
                if created:
                    st.success(f"Created story: {new_title.strip()}")
                    st.rerun()
                # else insert_story already showed error

    # --- List: one expander per story ---
    for s in stories:
        sid = s.get("id")
        title = s.get("title") or ""
        description = s.get("description") or ""
        free = bool(s.get("free"))
        published = bool(s.get("published"))
        # Initialize checkbox session state from DB only when key is missing, so toggles aren't reset on every rerun
        if f"story_free_{sid}" not in st.session_state:
            st.session_state[f"story_free_{sid}"] = free
        if f"story_published_{sid}" not in st.session_state:
            st.session_state[f"story_published_{sid}"] = published
        label = f"ID {sid} — {title}" if title else f"ID {sid} (no title)"
        with st.expander(label, expanded=False):
            edit_title = st.text_input("Title", value=title, key=f"story_title_{sid}")
            edit_description = st.text_area("Description", value=description, key=f"story_desc_{sid}", height=80)
            c1, c2 = st.columns(2)
            with c1:
                edit_free = st.checkbox("Free", value=free, key=f"story_free_{sid}")
            with c2:
                edit_published = st.checkbox("Published", value=published, key=f"story_published_{sid}")
            col_save, col_del, _ = st.columns([1, 1, 2])
            with col_save:
                if st.button("Save", key=f"save_story_{sid}"):
                    if not (edit_title and edit_title.strip()):
                        st.warning("Title is required.")
                    elif update_story(sb, sid, {
                        "title": edit_title.strip(),
                        "description": (edit_description or "").strip(),
                        "free": edit_free,
                        "published": edit_published,
                    }):
                        # Clear checkbox state so next run we re-init from DB and show saved values
                        st.session_state.pop(f"story_free_{sid}", None)
                        st.session_state.pop(f"story_published_{sid}", None)
                        st.success("Saved.")
                        st.rerun()
            with col_del:
                if st.button("Delete", key=f"del_story_{sid}"):
                    if delete_story(sb, sid):
                        st.success("Deleted.")
                        st.rerun()
                    # else delete_story already showed error

    if not stories:
        st.info("No stories yet. Add one above or clear the search.")
