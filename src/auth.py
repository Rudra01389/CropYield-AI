"""Demonstration authentication using Streamlit session state (NOT production-grade security).

The login screen offers one-click access to the demo farmer account: anyone who opens the app can press the button, so
this protects nothing and is meant only to demonstrate login / session / logout handling in an academic prototype.

- Credentials live in auth_config.py (salted PBKDF2 hash, no plain-text password check in code).
- Only the username is kept in st.session_state; the password is never stored by this module.
- Nothing is written to browser local storage, cookies or a database.
"""
import hashlib
import hmac

import streamlit as st

import auth_config

AUTH_KEY = "auth_username"
_DUMMY_SALT = "00" * 16
_DUMMY_HASH = "00" * 32


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
                               auth_config.PBKDF2_ITERATIONS).hex()


def authenticate(username: str, password: str):
    """Return the farmer profile (dict incl. 'username') for valid credentials, else None.
    A hash is computed even for unknown usernames so both failure cases take similar time."""
    key = (username or "").strip().lower()
    rec = auth_config.DEMO_USERS.get(key)
    given = _hash(password or "", rec["salt"] if rec else _DUMMY_SALT)
    expected = rec["password_hash"] if rec else _DUMMY_HASH
    if rec is not None and hmac.compare_digest(given, expected):
        return {"username": key, **rec["profile"]}
    return None


def login(user: dict) -> None:
    st.session_state[AUTH_KEY] = user["username"]


def demo_login() -> bool:
    """One-click sign-in as the demo farmer. The account is still authenticated through authenticate()
    (salted-hash check against auth_config), not by simply setting a flag. Returns True on success."""
    creds = auth_config.DEMO_CREDENTIALS_HINT
    user = authenticate(creds["username"], creds["password"])
    if user is None:
        return False
    login(user)
    return True


def is_authenticated() -> bool:
    u = st.session_state.get(AUTH_KEY)
    return isinstance(u, str) and u in auth_config.DEMO_USERS


def current_user():
    """Profile of the signed-in farmer, or None."""
    if not is_authenticated():
        return None
    u = st.session_state[AUTH_KEY]
    return {"username": u, **auth_config.DEMO_USERS[u]["profile"]}


def logout() -> None:
    """Clear the whole authenticated session (login, navigation, form and chart state)."""
    st.session_state.clear()
