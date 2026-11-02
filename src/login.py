"""Login screen: one-click demonstration access (see src/auth.py - demo authentication, not production security)."""
import streamlit as st

import auth_config
from src import auth, ui


def render():
    user = auth_config.DEMO_USERS[auth_config.DEMO_CREDENTIALS_HINT["username"]]["profile"]
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        ui.login_art()
    with right:
        with st.container(key="card_login"):
            ui.login_heading("Explore the app with the demonstration account.")
            ui.demo_farmer_preview(user)
            st.button("Continue as Demo Farmer", icon=":material/arrow_forward:", key="demo_login", type="primary",
                      width="stretch", on_click=auth.demo_login)
            ui.demo_note()
    ui.footer()
