"""Tests for local admin login (no Supabase Auth)."""
import unittest
from unittest.mock import patch

import auth


class CredentialsMatchTests(unittest.TestCase):
    def test_matching_credentials(self):
        self.assertTrue(auth.credentials_match("admin", "secret", "admin", "secret"))

    def test_username_is_stripped(self):
        self.assertTrue(auth.credentials_match("  admin  ", "secret", "admin", "secret"))

    def test_wrong_password(self):
        self.assertFalse(auth.credentials_match("admin", "nope", "admin", "secret"))

    def test_wrong_username(self):
        self.assertFalse(auth.credentials_match("other", "secret", "admin", "secret"))

    def test_empty_rejected(self):
        self.assertFalse(auth.credentials_match("", "", "admin", "secret"))


class LoginLogoutTests(unittest.TestCase):
    def _secrets(self, key, default=""):
        values = {"ADMIN_USERNAME": "admin", "ADMIN_PASSWORD": "secret"}
        return values.get(key, default)

    def test_login_success_sets_session(self):
        with patch.object(auth, "get_secret", side_effect=self._secrets), patch.object(
            auth.st, "session_state", {}
        ):
            ok, err = auth.login("admin", "secret")
            self.assertTrue(ok)
            self.assertEqual(err, "")
            self.assertTrue(auth.st.session_state[auth.SESSION_AUTH_KEY])
            self.assertEqual(auth.st.session_state[auth.SESSION_USER_KEY], "admin")
            self.assertTrue(auth.is_authenticated())
            self.assertEqual(auth.get_admin_username(), "admin")

    def test_login_rejects_bad_password(self):
        with patch.object(auth, "get_secret", side_effect=self._secrets), patch.object(
            auth.st, "session_state", {}
        ):
            ok, err = auth.login("admin", "wrong")
            self.assertFalse(ok)
            self.assertEqual(err, "Invalid username or password.")
            self.assertFalse(auth.is_authenticated())

    def test_login_requires_configured_secrets(self):
        with patch.object(auth, "get_secret", return_value=""), patch.object(
            auth.st, "session_state", {}
        ):
            ok, err = auth.login("admin", "secret")
            self.assertFalse(ok)
            self.assertIn("not configured", err)

    def test_logout_clears_session(self):
        with patch.object(auth, "get_secret", side_effect=self._secrets), patch.object(
            auth.st, "session_state", {}
        ):
            auth.login("admin", "secret")
            auth.logout()
            self.assertFalse(auth.is_authenticated())
            self.assertEqual(auth.get_admin_username(), "")


if __name__ == "__main__":
    unittest.main()
