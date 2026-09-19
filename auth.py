"""Admin login for Storybook Image Processor.

Uses a single username/password from Streamlit secrets or environment
variables. Supabase is not used for authentication.
"""
import hmac
import os
from typing import Tuple

import streamlit as st

SESSION_AUTH_KEY = "admin_authenticated"
SESSION_USER_KEY = "admin_username"


def get_secret(key: str, default: str = "") -> str:
    """Get config from st.secrets (Community Cloud) or os.getenv (local)."""
    try:
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except (FileNotFoundError, st.errors.StreamlitAPIException):
        pass
    return os.getenv(key, default)


def _configured_credentials() -> Tuple[str, str]:
    username = (get_secret("ADMIN_USERNAME") or "").strip()
    password = get_secret("ADMIN_PASSWORD") or ""
    return username, password


def is_auth_configured() -> bool:
    """True when admin username and password are both set."""
    username, password = _configured_credentials()
    return bool(username and password)


def credentials_match(username: str, password: str, expected_user: str, expected_pass: str) -> bool:
    """Constant-time compare of submitted credentials against configured admin account."""
    user_ok = hmac.compare_digest(username.strip().encode("utf-8"), expected_user.encode("utf-8"))
    pass_ok = hmac.compare_digest(password.encode("utf-8"), expected_pass.encode("utf-8"))
    return user_ok and pass_ok


def is_authenticated() -> bool:
    """Check if the current Streamlit session is signed in as admin."""
    return bool(st.session_state.get(SESSION_AUTH_KEY))


def get_admin_username() -> str:
    """Username of the signed-in admin, or empty if not authenticated."""
    if not is_authenticated():
        return ""
    return str(st.session_state.get(SESSION_USER_KEY) or "")


def login(username: str, password: str) -> Tuple[bool, str]:
    """Sign in with the configured admin account. Returns (success, error_message)."""
    expected_user, expected_pass = _configured_credentials()
    if not expected_user or not expected_pass:
        return False, "Admin login is not configured (set ADMIN_USERNAME and ADMIN_PASSWORD)."
    if credentials_match(username, password, expected_user, expected_pass):
        st.session_state[SESSION_AUTH_KEY] = True
        st.session_state[SESSION_USER_KEY] = expected_user
        return True, ""
    return False, "Invalid username or password."


def logout():
    """Sign out and clear admin auth state."""
    for key in (SESSION_AUTH_KEY, SESSION_USER_KEY):
        st.session_state.pop(key, None)


def run_login_page() -> bool:
    """Show login form. Returns True if successfully logged in (rerun), else False (stops)."""
    st.title("Storybook Image Processor")
    st.caption("Admin sign in")

    if not is_auth_configured():
        st.error(
            "Admin login is not configured. Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` "
            "in Streamlit secrets or your `.env` file."
        )
        return False

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if not username or not password:
                st.error("Enter username and password.")
            else:
                ok, err = login(username, password)
                if ok:
                    st.success("Signed in.")
                    st.rerun()
                else:
                    st.error(err or "Login failed.")
    return False
