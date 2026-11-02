"""Reusable UI pieces: CSS injection, page header, callouts, chips, chart styling."""
from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

FOREST, GREEN, LEAF, AMBER, BLUE, SLATE, MINT = "#1B5E3A", "#2E8B57", "#7CB342", "#E0A100", "#2F6F9F", "#5B6B62", "#BFE3CC"

# one colour per model, used by every chart so a model always looks the same
MODEL_COLORS = {
    "ANN (all features)": FOREST, "ANN (no weather)": LEAF,
    "Linear Regression": BLUE, "Linear Regression (no weather)": "#8FB8D6",
    "District-mean baseline": AMBER,
}

_CSS = Path(__file__).with_name("style.css")


def inject_css():
    try:
        css = _CSS.read_text()
    except OSError:
        css = ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str, badge: str = ""):
    b = f'<span class="badge">{escape(badge)}</span>' if badge else ""
    st.markdown(f'<div class="page-head"><div class="ico">{icon}</div><div><h1>{escape(title)}{b}</h1>'
                f'<p>{escape(subtitle)}</p></div></div>', unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    cap = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(f'<div class="sec"><h3>{escape(title)}</h3>{cap}</div>', unsafe_allow_html=True)


def callout(kind: str, title: str, body: str):
    icon = {"warn": "⚠️", "good": "✅", "info": "ℹ️"}[kind]
    st.markdown(f'<div class="callout {kind}"><div class="ci">{icon}</div><div><b>{escape(title)}</b>{escape(body)}</div></div>',
                unsafe_allow_html=True)


def chips(items, muted=False):
    cls = "chip n" if muted else "chip"
    st.markdown('<div class="chips">' + "".join(f'<span class="{cls}">{escape(str(i))}</span>' for i in items) + "</div>",
                unsafe_allow_html=True)


def mini_cards(cards, cols):
    """cards: list of (title, description). Rendered in `cols` columns."""
    columns = st.columns(cols)
    for i, (t, d) in enumerate(cards):
        columns[i % cols].markdown(f'<div class="mini"><div class="t">{escape(t)}</div><div class="d">{escape(d)}</div></div>',
                                   unsafe_allow_html=True)


def definition_list(items):
    """items: list of (name, description) rendered as compact rows."""
    st.markdown("".join(
        f'<div class="def"><div class="dn">{escape(n)}</div><div class="dd">{escape(d)}</div></div>' for n, d in items),
        unsafe_allow_html=True)


def steps(items):
    st.markdown("".join(f'<div class="step"><div class="num">{i}</div><div class="txt">{escape(t)}</div></div>'
                        for i, t in enumerate(items, 1)), unsafe_allow_html=True)


def footer():
    st.markdown('<div class="foot">CropYield AI · Soft Computing mini project · Academic prototype, not a production '
                'forecasting system. Predictions are statistical estimates.</div>', unsafe_allow_html=True)


def style_chart(chart: alt.Chart, height: int = 320):
    """Apply the dashboard look to an Altair chart (use with st.altair_chart(..., theme=None))."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=SLATE, titleColor="#14281D", titleFontWeight=600, titleFontSize=12,
                            gridColor="#E6ECE7", domainColor="#C9D4CC", tickColor="#C9D4CC", labelFontSize=11)
            .configure_legend(labelColor="#14281D", titleColor=SLATE, labelFontSize=11, orient="top", direction="horizontal",
                              titleFontSize=11, symbolType="circle")
            .configure_title(color="#14281D", fontSize=13, anchor="start"))


def show(chart, height: int = 320):
    st.altair_chart(style_chart(chart, height), width="stretch", theme=None)
