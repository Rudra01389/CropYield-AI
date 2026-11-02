"""Tests for the one-click demonstration login (src/auth.py, src/login.py, auth_config.py)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth_config  # noqa: E402
from src import auth, data  # noqa: E402

CRED = auth_config.DEMO_CREDENTIALS_HINT
PAGES = ["Dashboard", "Predict Yield", "Data & Trends", "Model Performance"]
BUTTON = "Continue as Demo Farmer"


def new_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def enter(at):
    """Click the one-click demo button on the login screen."""
    return at.button(key="demo_login").click().run()


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def is_login_screen(at):
    return (len(at.sidebar.radio) == 0 and "Welcome, Farmer!" in page_text(at) and len(at.get("button")) == 1
            and at.button(key="demo_login").label == BUTTON)


# ------------------------------------------------------------------ config + auth logic
def test_demo_account_matches_the_specification():
    rec = auth_config.DEMO_USERS["samplefarmer"]["profile"]
    assert rec == {"name": "Ramesh Patil", "district": "Pune", "state": "Maharashtra", "crop": "Rice",
                   "season": "Kharif", "land_area_ha": 2.5}
    assert CRED["username"] == "samplefarmer"


def test_password_is_still_stored_only_as_a_salted_hash():
    rec = auth_config.DEMO_USERS["samplefarmer"]
    assert "password" not in rec and len(rec["password_hash"]) == 64 and len(rec["salt"]) == 32
    assert CRED["password"] not in str(rec)


def test_credential_check_behind_the_button_still_works():
    user = auth.authenticate("samplefarmer", CRED["password"])
    assert user["name"] == "Ramesh Patil" and user["username"] == "samplefarmer" and "password" not in user
    assert auth.authenticate("samplefarmer", "wrong") is None
    assert auth.authenticate("nobody", CRED["password"]) is None
    assert auth.authenticate("", "") is None and auth.authenticate(None, None) is None


def test_demo_login_authenticates_through_the_hash_check(monkeypatch):
    """If the stored hash no longer matches, the one-click login must fail rather than silently let anyone in."""
    users = {k: dict(v) for k, v in auth_config.DEMO_USERS.items()}
    users["samplefarmer"]["password_hash"] = "0" * 64
    monkeypatch.setattr(auth_config, "DEMO_USERS", users)
    assert auth.demo_login() is False


# ------------------------------------------------------------------ login screen
def test_login_screen_has_only_the_demo_button():
    at = new_app().run()
    assert is_login_screen(at) and not at.exception
    assert len(at.text_input) == 0                                   # no username / password fields
    text = page_text(at)
    assert "Demonstration login" in text and "not secure production authentication" in text
    assert "Show demo credentials" not in text and CRED["password"] not in text
    assert "Ramesh Patil" in text and "Demo farmer" in text           # who the button opens
    assert not any(e.label == "Show demo credentials" for e in at.get("expander"))


def test_unauthenticated_visitor_cannot_see_any_protected_content():
    at = new_app().run()
    text = page_text(at)
    for leaked in ["Total historical records", "Predict Yield", "Model Performance", "Farmer name", "Data & Trends"]:
        assert leaked not in text, leaked
    assert len(at.get("download_button")) == 0 and "auth_username" not in at.session_state


def test_visitor_must_click_the_button_to_enter():
    at = new_app().run()
    at.run()                                                         # reruns alone never authenticate
    assert is_login_screen(at) and "auth_username" not in at.session_state
    at = enter(at)
    assert not is_login_screen(at) and at.session_state["auth_username"] == "samplefarmer"


# ------------------------------------------------------------------ after login
def test_button_opens_dashboard_with_welcome_and_farmer_profile():
    at = enter(new_app().run())
    assert not at.exception and not at.error and len(at.sidebar.radio) == 1
    assert at.sidebar.radio[0].value == "Dashboard"
    text = page_text(at)
    assert "Welcome, Ramesh Patil!" in text
    for expected in ["Ramesh Patil", "Pune, Maharashtra", "Rice · Kharif", "2.5 hectares"]:
        assert expected in text, expected
    assert CRED["password"] not in str(at.session_state)             # no password anywhere in session state


@pytest.mark.parametrize("page", PAGES)
def test_every_page_is_reachable_after_login(page):
    at = enter(new_app().run())
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception and not at.error
    assert at.sidebar.radio[0].value == page


def test_predict_page_is_prefilled_with_the_farmers_district_and_land():
    at = enter(new_app().run())
    at.sidebar.radio[0].set_value("Predict Yield").run()
    assert at.selectbox[0].value == "PUNE"
    text = page_text(at)
    assert "Rice · Kharif" in text and "your land 2.5 ha" in text
    at.button[0].click().run()
    labels = [m.label for m in at.metric]
    assert "Predicted rice yield" in labels and any(l.startswith("Your 2.5 ha") for l in labels)


def test_land_area_option_feeds_2_5_ha_into_the_model_and_warns():
    at = enter(new_app().run())
    at.sidebar.radio[0].set_value("Predict Yield").run()
    at.checkbox(key="use_land").check().run()
    area = next(n for n in at.number_input if n.label.startswith("Cultivated area"))
    assert area.value == 2.5
    at.button[0].click().run()
    assert at.warning and not at.exception
    assert not any(l.startswith("Your 2.5 ha") for l in [m.label for m in at.metric])


# ------------------------------------------------------------------ logout
def test_logout_returns_to_login_and_clears_session_state():
    at = enter(new_app().run())
    at.sidebar.radio[0].set_value("Data & Trends").run()
    assert at.session_state["auth_username"] == "samplefarmer"
    at.button(key="logout_btn").click().run()
    assert is_login_screen(at) and not at.exception
    assert "auth_username" not in at.session_state and "nav" not in at.session_state
    assert "Total historical records" not in page_text(at)


def test_after_logout_the_button_is_needed_again():
    at = enter(new_app().run())
    at.button(key="logout_btn").click().run()
    at.run()
    assert is_login_screen(at)                                       # still the login screen, nothing leaked
    at = enter(at)                                                   # clicking again works and starts on the Dashboard
    assert at.sidebar.radio[0].value == "Dashboard" and "Welcome, Ramesh Patil!" in page_text(at)


def test_tampering_with_session_state_does_not_grant_access():
    at = new_app().run()
    at.session_state["auth_username"] = "not-a-real-user"
    at.run()
    assert is_login_screen(at)


# ------------------------------------------------------------------ existing features still work behind the login
def test_existing_features_work_after_login():
    at = enter(new_app().run())
    at.sidebar.radio[0].set_value("Predict Yield").run()
    at.button[0].click().run()
    assert len(at.get("download_button")) == 1                      # CSV report
    assert any("Understanding the Prediction" in m.value for m in at.markdown)   # interpretation + importance
    at.sidebar.radio[0].set_value("Data & Trends").run()
    at.selectbox(key="map_pick").set_value("PUNE").run()             # district map highlight
    expected = data.load_dataset().query("district == 'PUNE'").yield_t_ha.mean()
    assert next(m.value for m in at.metric if m.label == "Pune mean yield") == f"{expected:.2f} t/ha"
    assert not at.exception
