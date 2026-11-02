"""Reusable UI pieces: CSS injection, headers, cards, gauge, tables, sidebar parts, chart styling."""
import math
import random
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


# ------------------------------------------------------------------------- login / profile
def initials(name: str) -> str:
    parts = [w for w in name.split() if w]
    return "".join(w[0].upper() for w in parts[:2]) or "?"


def login_art():
    """Original agricultural illustration (sky, sun, hills, rice field with converging furrows) drawn as inline SVG."""
    rnd = random.Random(7)
    hz = 310
    svg = ['<svg viewBox="0 0 600 680" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">',
           '<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#CDEBD8"/>'
           '<stop offset="1" stop-color="#F7FBF2"/></linearGradient>'
           '<linearGradient id="fld" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#A5D6A7"/>'
           '<stop offset=".45" stop-color="#4FA36F"/><stop offset="1" stop-color="#1F7A48"/></linearGradient>'
           '<radialGradient id="sun"><stop offset="0" stop-color="#FFE9A8" stop-opacity=".95"/>'
           '<stop offset="1" stop-color="#FFE9A8" stop-opacity="0"/></radialGradient></defs>',
           f'<rect width="600" height="{hz + 20}" fill="url(#sky)"/>',
           '<circle cx="480" cy="130" r="125" fill="url(#sun)"/><circle cx="480" cy="130" r="38" fill="#F6C945"/>',
           '<g fill="#fff" opacity=".85"><ellipse cx="130" cy="120" rx="52" ry="14"/><ellipse cx="160" cy="108" rx="34" ry="12"/>'
           '<ellipse cx="340" cy="170" rx="46" ry="12"/><ellipse cx="365" cy="160" rx="28" ry="10"/></g>',
           f'<path d="M0 {hz} Q90 {hz-60} 190 {hz-25} T380 {hz-30} T600 {hz-45} L600 {hz+30} L0 {hz+30}Z" fill="#BFDFC9"/>',
           f'<path d="M0 {hz+8} Q120 {hz-38} 250 {hz-8} T600 {hz-15} L600 {hz+40} L0 {hz+40}Z" fill="#9CCFA8"/>',
           f'<path d="M0 {hz+20} L600 {hz+20} L600 680 L0 680Z" fill="url(#fld)"/>']
    # soil bands that widen towards the viewer, then furrows converging on the horizon
    y, step = hz + 20, 8.0
    band = 0
    while y < 680:
        if band % 2 == 0:
            svg.append(f'<rect x="0" y="{y:.1f}" width="600" height="{step:.1f}" fill="#124428" opacity=".10"/>')
        y += step
        step *= 1.22
        band += 1
    for x in range(-1000, 1600, 70):
        svg.append(f'<line x1="300" y1="{hz + 20}" x2="{x}" y2="680" stroke="#124428" stroke-opacity=".22" stroke-width="1.6"/>')
    for _ in range(300):                                   # little rice plants, larger when closer
        yy = rnd.uniform(hz + 30, 675)
        t = (yy - hz) / (680 - hz)
        h = 3 + t * 18
        xx = rnd.uniform(0, 600)
        svg.append(f'<path d="M{xx:.1f} {yy:.1f} q{-h*.25:.1f} {-h*.6:.1f} {-h*.45:.1f} {-h:.1f} M{xx:.1f} {yy:.1f} '
                   f'q{h*.05:.1f} {-h*.7:.1f} {h*.05:.1f} {-h*1.1:.1f} M{xx:.1f} {yy:.1f} q{h*.25:.1f} {-h*.6:.1f} {h*.45:.1f} {-h:.1f}" '
                   f'stroke="#E9F7EC" stroke-opacity=".65" stroke-width="{0.8 + t * 1.4:.1f}" fill="none" stroke-linecap="round"/>')
    svg.append("</svg>")
    st.markdown(
        '<div class="lgn-art">' + "".join(svg) +
        '<div class="lgn-top"><div class="lgn-logo">🌱</div><div><div class="lgn-name"><span>Crop</span>Yield AI</div>'
        '<div class="lgn-tag">Smart Agricultural Analytics</div></div></div>'
        '<div class="lgn-head">AI-powered rice yield prediction for Maharashtra</div>'
        '<div class="lgn-pills"><span>🗺️ District yield map</span><span>🧠 ANN yield estimates</span>'
        '<span>🛰️ NASA POWER weather</span></div></div>', unsafe_allow_html=True)


def login_heading(subtitle: str):
    st.markdown(f'<div class="lgn-mark">🌱</div><h1 class="lgn-title">Welcome, Farmer!</h1>'
                f'<p class="lgn-sub">{escape(subtitle)}</p>', unsafe_allow_html=True)


def demo_farmer_preview(profile: dict):
    """Shows whose sample account the button opens (read from auth_config, nothing typed by the visitor)."""
    st.markdown(f'<div class="lgn-who"><div class="av">{escape(initials(profile["name"]))}</div><div>'
                f'<div class="wn">{escape(profile["name"])} <span>Demo farmer</span></div>'
                f'<div class="wd">{escape(profile["district"])}, {escape(profile["state"])} · {escape(profile["crop"])} '
                f'({escape(profile["season"])}) · {profile["land_area_ha"]:g} ha</div></div></div>', unsafe_allow_html=True)


def demo_note():
    st.markdown('<div class="demo-note">🧪 <b>Demonstration login.</b> This one-click sample account is for an academic project '
                'and is <b>not secure production authentication</b>: anyone who opens the app can enter. '
                'It uses no real credentials or personal data.</div>', unsafe_allow_html=True)


def sidebar_profile(profile: dict):
    st.markdown(f'<div class="sprof"><div class="av">{escape(initials(profile["name"]))}</div><div>'
                f'<div class="pn">{escape(profile["name"])}</div>'
                f'<div class="pl">{escape(profile["district"])}, {escape(profile["state"])}</div></div></div>',
                unsafe_allow_html=True)


def profile_card(profile: dict):
    """Farmer profile: name, district and state, crop and season, cultivated land area."""
    items = [("Farmer name", profile["name"]), ("District & state", f'{profile["district"]}, {profile["state"]}'),
             ("Crop & season", f'{profile["crop"]} · {profile["season"]}'), ("Cultivated land", f'{profile["land_area_ha"]:g} hectares')]
    cells = "".join(f'<div class="pcell"><div class="pcl">{escape(k)}</div><div class="pcv">{escape(v)}</div></div>' for k, v in items)
    st.markdown(f'<div class="pcard"><div class="pav">{escape(initials(profile["name"]))}</div><div class="pgrid">{cells}</div>'
                f'<span class="pdemo">Demo account</span></div>', unsafe_allow_html=True)
