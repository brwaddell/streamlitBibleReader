"""Supabase Auth for Storybook Image Processor."""
import os
from typing import Tuple

import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    """Get config from st.secrets (Community Cloud) or os.getenv (local)."""
    try:
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except (FileNotFoundError, st.errors.StreamlitAPIException):
        pass
    return os.getenv(key, default)


def get_auth_client():
    """Get Supabase client with anon key for auth (login/session)."""
    if getattr(st.session_state, "auth_client", None) is None:
        url = get_secret("SUPABASE_URL")
        anon_key = get_secret("SUPABASE_ANON_KEY")
        if not url or not anon_key:
            return None
        from supabase import create_client
        client = create_client(url, anon_key)
        # Restore session from session state (persists across Streamlit reruns)
        tokens = st.session_state.get("auth_tokens")
        if tokens:
            try:
                client.auth.set_session(tokens["access_token"], tokens["refresh_token"])
            except Exception:
                st.session_state.auth_tokens = None
        st.session_state.auth_client = client
    return st.session_state.auth_client


def get_session():
    """Get current auth session if logged in."""
    client = get_auth_client()
    if not client:
        return None
    try:
        session = client.auth.get_session()
        if session:
            # Persist tokens for next run
            st.session_state.auth_tokens = {
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
            }
        return session
    except Exception:
        return None


def get_profile_id() -> str:
    """Get current user's profile id (auth user id) for subscription/profiles lookups."""
    session = get_session()
    if not session or not session.user:
        return ""
    return str(session.user.id) if session.user.id else ""


def is_authenticated() -> bool:
    """Check if user has a valid session."""
    return get_session() is not None


def login(email: str, password: str) -> Tuple[bool, str]:
    """Sign in with email/password. Returns (success, error_message)."""
    client = get_auth_client()
    if not client:
        return False, "Auth not configured (missing SUPABASE_URL or SUPABASE_ANON_KEY)."
    try:
        resp = client.auth.sign_in_with_password({"email": email, "password": password})
        if resp.session:
            st.session_state.auth_tokens = {
                "access_token": resp.session.access_token,
                "refresh_token": resp.session.refresh_token,
            }
        return True, ""
    except Exception as e:
        return False, str(e)


def logout():
    """Sign out and clear auth state."""
    client = get_auth_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    for key in ("auth_client", "auth_tokens"):
        if key in st.session_state:
            del st.session_state[key]


def run_login_page() -> bool:
    """Show login form. Returns True if successfully logged in (rerun), else False (stops)."""
    st.title("Storybook Image Processor")
    st.caption("Sign in to continue")
    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted and email and password:
            ok, err = login(email.strip(), password)
            if ok:
                st.success("Signed in.")
                st.rerun()
            else:
                st.error(err or "Login failed.")
    st.caption("Create an account in Supabase Dashboard → Authentication → Users.")
    return False
