"""Stories CRUD: simple inline create, update, and delete via st.data_editor."""

import pandas as pd
import streamlit as st

from lib import delete_story, fetch_stories, get_supabase, insert_story, update_story


def _normalize_id_series(df: pd.DataFrame) -> pd.Series:
    """Return IDs as pandas nullable Int64 (or empty series if missing)."""
    if "id" not in df.columns:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(df["id"], errors="coerce").astype("Int64")


def run_stories_view():
    """Render a simple Stories CRUD editor in one table."""
    st.title("Stories")
    st.caption("Simple CRUD for title and description.")

    sb = get_supabase()
    if not sb:
        return

    stories = fetch_stories() or []
    base_df = pd.DataFrame(
        [{"id": s.get("id"), "title": s.get("title") or "", "description": s.get("description") or ""} for s in stories]
    )
    if base_df.empty:
        base_df = pd.DataFrame(columns=["id", "title", "description"])

    # Search, filter, sort controls
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        search_q = st.text_input(
            "Search",
            value=st.session_state.get("stories_search_q", ""),
            placeholder="Search title or description...",
            key="stories_search_q",
        ).strip()
    with c2:
        description_filter = st.selectbox(
            "Filter",
            options=["All", "Has description", "No description"],
            key="stories_desc_filter",
        )
    with c3:
        sort_col = st.selectbox("Sort by", options=["id", "title", "description"], key="stories_sort_col")
    with c4:
        sort_dir = st.selectbox("Order", options=["Ascending", "Descending"], key="stories_sort_dir")

    working_df = base_df.copy()
    if search_q:
        q = search_q.lower()
        working_df = working_df[
            working_df["title"].astype(str).str.lower().str.contains(q, na=False)
            | working_df["description"].astype(str).str.lower().str.contains(q, na=False)
        ]

    if description_filter == "Has description":
        working_df = working_df[working_df["description"].astype(str).str.strip() != ""]
    elif description_filter == "No description":
        working_df = working_df[working_df["description"].astype(str).str.strip() == ""]

    ascending = sort_dir == "Ascending"
    if sort_col == "id":
        working_df = working_df.assign(_id_sort=pd.to_numeric(working_df["id"], errors="coerce")).sort_values(
            "_id_sort", ascending=ascending, na_position="last"
        ).drop(columns=["_id_sort"])
    else:
        working_df = working_df.sort_values(
            sort_col,
            ascending=ascending,
            key=lambda s: s.astype(str).str.lower(),
            na_position="last",
        )
    st.caption(f"Showing {len(working_df)} of {len(base_df)} stories.")

    edited_df = st.data_editor(
        working_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "title": st.column_config.TextColumn("Title", required=True, width="medium"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
        key="stories_editor",
    )

    if st.button("Save changes", type="primary"):
        edited_df = edited_df.fillna("")
        base_df_norm = base_df.fillna("")

        old_ids_series = _normalize_id_series(base_df_norm).dropna()
        new_ids_series = _normalize_id_series(edited_df).dropna()
        old_ids = set(int(v) for v in old_ids_series.tolist())
        new_ids = set(int(v) for v in new_ids_series.tolist())
        delete_ids = old_ids - new_ids

        # Inserts are rows without an ID.
        new_row_mask = _normalize_id_series(edited_df).isna()
        to_insert = edited_df[new_row_mask]

        # Updates are rows with ID that changed.
        existing_mask = _normalize_id_series(edited_df).notna()
        to_update = edited_df[existing_mask].copy()
        to_update["id"] = pd.to_numeric(to_update["id"], errors="coerce").astype("Int64")
        base_by_id = base_df_norm.copy()
        base_by_id["id"] = pd.to_numeric(base_by_id["id"], errors="coerce").astype("Int64")
        base_by_id = base_by_id.dropna(subset=["id"]).set_index("id")

        inserted_count = 0
        updated_count = 0
        deleted_count = 0

        for _, row in to_insert.iterrows():
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()
            if not title:
                continue
            if insert_story(sb, {"title": title, "description": description}):
                inserted_count += 1

        for _, row in to_update.iterrows():
            sid = row.get("id")
            if pd.isna(sid):
                continue
            sid = int(sid)
            if sid not in base_by_id.index:
                continue
            old = base_by_id.loc[sid]
            new_title = str(row.get("title", "")).strip()
            new_description = str(row.get("description", "")).strip()
            if not new_title:
                st.warning(f"Story {sid}: title is required. Skipped update.")
                continue
            old_title = str(old.get("title", "")).strip()
            old_description = str(old.get("description", "")).strip()
            if new_title != old_title or new_description != old_description:
                if update_story(sb, sid, {"title": new_title, "description": new_description}):
                    updated_count += 1

        for sid in delete_ids:
            if delete_story(sb, sid):
                deleted_count += 1

        st.success(f"Saved changes. Inserted {inserted_count}, updated {updated_count}, deleted {deleted_count}.")
        st.rerun()
