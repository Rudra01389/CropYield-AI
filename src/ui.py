"""Reusable UI pieces: CSS injection, headers, cards, gauge, tables, sidebar parts, chart styling."""
import math
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


# ------------------------------------------------------------------------------ sidebar
def sidebar_brand():
    st.markdown('<div class="brand"><div class="logo">🌱</div><div><div class="name"><span>Crop</span>Yield AI</div>'
                '<div class="tag">Smart Agricultural Analytics</div></div></div>', unsafe_allow_html=True)


def sidebar_footer(rows, note):
    """rows: list of (label, value) shown in a small project-information card."""
    body = "".join(f'<div class="sfl">{escape(k)}</div><div class="sfv">{escape(v)}</div>' for k, v in rows)
    st.markdown(f'<div class="sfoot"><div class="sft">Project information</div>{body}</div>', unsafe_allow_html=True)
    st.caption(note)


# ------------------------------------------------------------------------------- headers
def page_header(icon: str, title: str, subtitle: str, badge: str = ""):
    """Title + subtitle on the left, optional badge on the right (icon kept for API compatibility)."""
    b = f'<span class="pbadge">{escape(badge)}</span>' if badge else ""
    st.markdown(f'<div class="phead"><div class="ptxt"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>{b}</div>',
                unsafe_allow_html=True)


def greeting(title: str, subtitle: str):
    st.markdown(f'<div class="greet"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    cap = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(f'<div class="sec"><h3>{escape(title)}</h3>{cap}</div>', unsafe_allow_html=True)


def card_title(title: str, hint: str = ""):
    h = f'<span class="cth">{escape(hint)}</span>' if hint else ""
    st.markdown(f'<div class="ct"><span>{escape(title)}</span>{h}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------- cards
def kpi_cards(items):
    """items: (icon, label, value, sub[, tone]) per card; tone is green | purple | blue | amber."""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        icon, label, value, sub = it[:4]
        tone = it[4] if len(it) > 4 else "green"
        col.markdown(f'<div class="kpi"><div class="kt"><div class="klbl">{escape(label)}</div>'
                     f'<div class="kico {tone}">{icon}</div></div><div class="kval">{escape(str(value))}</div>'
                     f'<div class="ksub">{escape(sub)}</div></div>', unsafe_allow_html=True)


def pill(text: str):
    return f'<span class="gpill">📍 {escape(text)}</span>'


def stat_row(items):
    """items: (label, value) small stats in one row (HTML string returned for embedding in a card)."""
    return '<div class="srow">' + "".join(
        f'<div class="sitem"><div class="sl">{escape(k)}</div><div class="sv">{escape(v)}</div></div>' for k, v in items) + "</div>"


def info_list(items):
    """items: (icon, title, text, tone)."""
    st.markdown("".join(
        f'<div class="irow"><div class="iico {t}">{i}</div><div><div class="it">{escape(a)}</div>'
        f'<div class="id">{escape(b)}</div></div></div>' for i, a, b, t in items), unsafe_allow_html=True)


def html_table(headers, rows):
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def gauge(value: float, vmax: float, center: str, caption: str, legend):
    """Segmented semicircle gauge. `value/vmax` fills the arc; legend = [(label, text, colour)]."""
    n, cx, cy, r1, r2 = 40, 150, 148, 90, 128
    filled = max(0, min(n, round(value / vmax * n)))
    lines = []
    for i in range(n):
        th = math.pi - i * math.pi / (n - 1)
        x1, y1, x2, y2 = cx + r1 * math.cos(th), cy - r1 * math.sin(th), cx + r2 * math.cos(th), cy - r2 * math.sin(th)
        if i < filled:                                  # light leaf-green -> forest across the filled part
            f = i / max(1, filled - 1)
            col = f"rgb({int(124 + (27 - 124) * f)},{int(179 + (94 - 179) * f)},{int(66 + (58 - 66) * f)})"
        else:
            col = "#E3EAE5"
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="7" '
                     f'stroke-linecap="round"/>')
    leg = "".join(f'<span class="lg"><i style="background:{c}"></i>{escape(a)}: <b>{escape(b)}</b></span>' for a, b, c in legend)
    st.markdown(
        f'<div class="gauge"><svg viewBox="0 0 300 166" xmlns="http://www.w3.org/2000/svg">{"".join(lines)}'
        f'<text x="150" y="130" text-anchor="middle" font-size="31" font-weight="700" fill="#14281D">{escape(center)}</text>'
        f'<text x="150" y="152" text-anchor="middle" font-size="11.5" fill="#5B6B62">{escape(caption)}</text></svg>'
        f'<div class="glegend">{leg}</div></div>', unsafe_allow_html=True)


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


def bullets(items):
    st.markdown('<ul class="bl">' + "".join(f"<li>{escape(t)}</li>" for t in items) + "</ul>", unsafe_allow_html=True)


def footer():
    st.markdown('<div class="foot">CropYield AI · Soft Computing mini project · Academic prototype, not a production '
                'forecasting system. Predictions are statistical estimates.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------- charts
def style_chart(chart: alt.Chart, height: int = 320):
    """Apply the dashboard look to an Altair chart (use with st.altair_chart(..., theme=None))."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=SLATE, titleColor="#14281D", titleFontWeight=600, titleFontSize=12,
                            gridColor="#EEF2EF", domainColor="#C9D4CC", tickColor="#C9D4CC", labelFontSize=11)
            .configure_legend(labelColor="#14281D", titleColor=SLATE, labelFontSize=11, orient="top", direction="horizontal",
                              titleFontSize=11, symbolType="circle")
            .configure_title(color="#14281D", fontSize=13, anchor="start"))


def show(chart, height: int = 320):
    st.altair_chart(style_chart(chart, height), width="stretch", theme=None)
