"""
Carbon Crunch — Receipt Intelligence
=====================================
Modern landing-page + tool for OCR receipt extraction with carbon insights.
Run with:   streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.preprocess import (
    enhance_contrast, denoise, deskew, resize_for_ocr,
    estimate_blur, estimate_brightness, to_grayscale,
)
from src.ocr import run_ocr, get_full_text, get_average_confidence
from src.extractor import (
    extract_date, extract_store_name, extract_total, extract_items,
)
from src.confidence import adjust_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Carbon Crunch · Receipt Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Carbon coefficient — rough industry average kg CO₂e per ₹/$ spent on retail.
# (Used here as an illustrative estimate; real Carbon Crunch uses category-
#  specific factors from emissions databases like Climatiq / DEFRA.)
CARBON_KG_PER_UNIT = 0.041


# ═══════════════════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500&display=swap');

/* ─── RESET ─────────────────────────────────────────────────────────────── */
* { box-sizing: border-box; }
.stApp {
    background:
      radial-gradient(ellipse 1000px 700px at 80% -10%, rgba(0, 200, 150, 0.10) 0%, transparent 60%),
      radial-gradient(ellipse 700px 500px at 0% 100%, rgba(0, 200, 150, 0.06) 0%, transparent 60%),
      #050d0a;
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, sans-serif;
    color: #e5f5ec;
}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important; height: 0 !important;
}
#MainMenu, footer { visibility: hidden; }
.block-container {
    padding: 1rem 2rem 4rem !important;
    max-width: 1240px !important;
}
/* Kill Streamlit's auto anchor link icons next to headings */
.stMarkdown a[href^="#"], h1 a, h2 a, h3 a, h4 a,
[data-testid="stHeaderActionElements"], [class*="StyledLinkIcon"] {
    display: none !important; visibility: hidden !important;
}
.stMarkdown p { margin: 0 !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ─── NAVBAR ────────────────────────────────────────────────────────────── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.3rem;
    background: rgba(8, 18, 14, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 200, 150, 0.12);
    border-radius: 14px;
    margin-bottom: 3rem;
}
.brand { display: flex; align-items: center; gap: 0.7rem; }
.brand-mark {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #00c896, #00a884);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 16px rgba(0, 200, 150, 0.4);
}
.brand-mark svg { width: 19px; height: 19px; }
.brand-text {
    font-size: 1rem; font-weight: 700; color: #ecfdf5;
    letter-spacing: -0.01em; line-height: 1.2;
}
.brand-text .sub {
    color: #4ade80; font-weight: 500;
    margin-left: 0.5rem; font-size: 0.8rem; opacity: 0.85;
}
.nav-pill {
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.25);
    color: #4ade80;
    padding: 0.32rem 0.85rem; border-radius: 999px;
    font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.status-dot {
    width: 7px; height: 7px; background: #00c896; border-radius: 50%;
    box-shadow: 0 0 8px #00c896;
    animation: pulse 2s ease-in-out infinite;
    display: inline-block; margin-right: 0.4rem; vertical-align: middle;
}
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ─── HERO ──────────────────────────────────────────────────────────────── */
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.25);
    color: #4ade80;
    padding: 0.4rem 1rem; border-radius: 999px;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-h1 {
    font-size: 3.4rem; font-weight: 800; line-height: 1.08;
    letter-spacing: -0.04em; color: #f0fdf4;
    margin: 0 0 1.3rem 0;
}
.hero-h1 .accent {
    background: linear-gradient(135deg, #00c896 0%, #4ade80 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #94d4b3; font-size: 1.1rem; line-height: 1.6;
    margin: 0 0 2rem 0; max-width: 480px; opacity: 0.9;
}
.hero-cta {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #00c896, #00a884);
    color: #052218; font-weight: 700; font-size: 0.95rem;
    padding: 0.85rem 1.6rem; border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 200, 150, 0.3);
    cursor: pointer; transition: all 0.2s; border: none;
    text-decoration: none;
}
.hero-cta:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(0, 200, 150, 0.45);
}
.hero-trust {
    display: flex; gap: 1.5rem; margin-top: 2rem;
    color: #6b8a7c; font-size: 0.78rem;
}
.hero-trust strong { color: #ecfdf5; font-weight: 600; }

/* Hero illustration */
.hero-art {
    position: relative; height: 360px; width: 100%;
    display: flex; align-items: center; justify-content: center;
}

/* ─── UPLOAD SECTION ────────────────────────────────────────────────────── */
.section-anchor {
    display: block; height: 0; visibility: hidden;
    margin-top: -2rem; padding-top: 2rem;
}
.section-eyebrow {
    color: #4ade80; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    text-align: center; margin-bottom: 0.7rem;
}
.section-title {
    color: #f0fdf4; font-size: 1.85rem; font-weight: 700;
    letter-spacing: -0.02em; text-align: center;
    margin: 0 0 0.6rem 0;
}
.section-sub {
    color: #94d4b3; font-size: 0.95rem;
    text-align: center; margin: 0 auto 2rem auto;
    max-width: 520px; opacity: 0.85;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(8, 18, 14, 0.5) !important;
    border: 2px dashed rgba(0, 200, 150, 0.35) !important;
    border-radius: 16px !important;
    padding: 3rem 2rem !important;
    transition: all 0.25s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0, 200, 150, 0.7) !important;
    background: rgba(0, 200, 150, 0.05) !important;
    box-shadow: 0 0 30px rgba(0, 200, 150, 0.15);
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div { color: #94d4b3 !important; }

/* ─── BUTTONS ───────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #00c896, #00a884) !important;
    color: #052218 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    box-shadow: 0 4px 14px rgba(0, 200, 150, 0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0, 200, 150, 0.4) !important;
}

/* ─── RESULT CARD ───────────────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, rgba(0, 200, 150, 0.04), rgba(8, 18, 14, 0.6));
    border: 1px solid rgba(0, 200, 150, 0.18);
    border-radius: 18px;
    padding: 1.8rem;
    backdrop-filter: blur(20px);
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 200, 150, 0.6), transparent);
}
.result-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; padding-bottom: 1rem;
    border-bottom: 1px solid rgba(0, 200, 150, 0.1);
}
.result-title {
    color: #ecfdf5; font-size: 1.1rem; font-weight: 700;
    letter-spacing: -0.01em;
}
.result-status {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0, 200, 150, 0.12);
    color: #4ade80; padding: 0.3rem 0.8rem;
    border-radius: 999px; font-size: 0.7rem; font-weight: 700;
    border: 1px solid rgba(0, 200, 150, 0.25);
}

.field {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.95rem 0;
    border-bottom: 1px solid rgba(0, 200, 150, 0.06);
}
.field:last-child { border-bottom: none; }
.field-label {
    color: #6b8a7c; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
}
.field-value {
    color: #ecfdf5; font-size: 1rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
}
.field-value.big {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #00c896, #4ade80);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Confidence bar */
.conf-bar-wrap {
    display: flex; align-items: center; gap: 0.7rem;
    width: 180px;
}
.conf-bar {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 999px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.4s ease;
}
.conf-bar-fill.high { background: linear-gradient(90deg, #00c896, #4ade80); }
.conf-bar-fill.mid  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.conf-bar-fill.low  { background: linear-gradient(90deg, #ef4444, #f87171); }
.conf-pct {
    color: #ecfdf5; font-size: 0.78rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; min-width: 45px;
    text-align: right;
}

/* Carbon highlight */
.carbon-banner {
    margin-top: 1.5rem; padding: 1.1rem 1.3rem;
    background: linear-gradient(135deg, rgba(0, 200, 150, 0.12), rgba(74, 222, 128, 0.06));
    border: 1px solid rgba(0, 200, 150, 0.3);
    border-left: 3px solid #00c896;
    border-radius: 12px;
    display: flex; align-items: center; gap: 1rem;
}
.carbon-icon {
    width: 42px; height: 42px;
    background: rgba(0, 200, 150, 0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.carbon-icon svg { width: 22px; height: 22px; color: #4ade80; }
.carbon-text { flex: 1; }
.carbon-label {
    color: #6ee7b7; font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.carbon-value {
    color: #ecfdf5; font-size: 1.3rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.carbon-value .unit {
    color: #6b8a7c; font-size: 0.75rem;
    font-weight: 500; margin-left: 0.4rem;
}

/* Image preview */
.preview-wrap {
    background: rgba(8, 18, 14, 0.5);
    border: 1px solid rgba(0, 200, 150, 0.12);
    border-radius: 16px;
    padding: 1.2rem;
}
.preview-label {
    color: #6b8a7c; font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 0.9rem;
}
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(0, 200, 150, 0.15);
}

/* ─── HOW IT WORKS ──────────────────────────────────────────────────────── */
.step-card {
    background: rgba(8, 18, 14, 0.55);
    border: 1px solid rgba(0, 200, 150, 0.1);
    border-radius: 16px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    height: 100%;
    transition: all 0.25s;
    position: relative;
}
.step-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 200, 150, 0.3);
    box-shadow: 0 12px 30px rgba(0, 200, 150, 0.1);
}
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #00c896, #00a884);
    color: #052218; font-weight: 800; font-size: 0.95rem;
    border-radius: 10px;
    margin: 0 auto 1rem auto;
    box-shadow: 0 4px 14px rgba(0, 200, 150, 0.3);
}
.step-title {
    color: #ecfdf5; font-size: 1.05rem; font-weight: 700;
    margin: 0 0 0.5rem 0; letter-spacing: -0.01em;
}
.step-text {
    color: #94d4b3; font-size: 0.88rem; line-height: 1.55;
    opacity: 0.85;
}

/* ─── FOOTER ────────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center; color: #4b6359; font-size: 0.78rem;
    padding: 2.5rem 0 0.5rem;
    border-top: 1px solid rgba(0, 200, 150, 0.06);
    margin-top: 4rem;
}
.app-footer strong { color: #4ade80; font-weight: 600; }

/* ─── EMPTY STATE ───────────────────────────────────────────────────────── */
.empty {
    text-align: center; padding: 4rem 2rem; color: #6b8a7c;
}
.empty .icon-wrap {
    width: 56px; height: 56px;
    margin: 0 auto 1rem;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
}
.empty svg { width: 26px; height: 26px; color: #4ade80; }
.empty-title {
    color: #d1fae5; font-size: 1rem; font-weight: 500;
    margin-bottom: 0.4rem;
}
.empty-sub { color: #6b8a7c; font-size: 0.83rem; }

/* Spinner color */
.stSpinner > div > div { border-top-color: #00c896 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def conf_class(score: float) -> str:
    if score >= 0.7: return "high"
    if score >= 0.5: return "mid"
    return "low"


def render_field(label: str, value: str) -> str:
    val = value if value not in (None, "") else "—"
    return (f'<div class="field"><span class="field-label">{label}</span>'
            f'<span class="field-value">{val}</span></div>')


def render_field_with_conf(label: str, value: str, conf: float) -> str:
    val = value if value not in (None, "") else "—"
    cls = conf_class(conf)
    pct = int(conf * 100)
    return (
        f'<div class="field"><span class="field-label">{label}</span>'
        f'<div style="display:flex;align-items:center;gap:1rem">'
        f'<span class="field-value">{val}</span>'
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar"><div class="conf-bar-fill {cls}" '
        f'style="width:{pct}%"></div></div>'
        f'<span class="conf-pct">{pct}%</span>'
        f'</div></div></div>'
    )


# Pre-warm the OCR reader ONCE per Streamlit session — saves 5-8 sec on every
# subsequent upload because the model weights stay in memory.
@st.cache_resource(show_spinner=False)
def _warm_ocr_reader():
    from src.ocr import _get_reader
    return _get_reader()


# Cache extraction results per file (by content hash) — re-uploading the
# same file is instant. We cache ONLY the lightweight text data, never images.
@st.cache_data(show_spinner=False, max_entries=100)
def _cached_extract(file_bytes: bytes, filename: str) -> dict:
    """Returns the lightweight extraction dict (no images)."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_orig is None:
        return {"error": "Cannot decode image"}

    # Optimized preprocessing — skip expensive denoising for sharp images
    gray = to_grayscale(img_orig)
    blur_score = estimate_blur(gray)

    img = resize_for_ocr(img_orig, target_height=1200, max_height=1800)
    img = enhance_contrast(img)
    if blur_score < 200:                  # only denoise if needed
        img = denoise(img)
    img, _ = deskew(img)

    lines = run_ocr(img)
    if not lines:
        return {"error": "No text detected"}

    avg_ocr = get_average_confidence(lines)
    store_name, store_raw = extract_store_name(lines)
    date_res = extract_date(lines)
    total_res = extract_total(lines)
    items_raw = extract_items(lines)

    date_v = date_res[0] if date_res else None
    total_v = total_res[0] if total_res else None

    store_c = adjust_confidence("store_name", store_name, store_raw)
    date_c  = adjust_confidence("date", date_v, date_res[1] if date_res else 0)
    total_c = adjust_confidence("total_amount", total_v, total_res[1] if total_res else 0)

    # Overall confidence — weighted average
    overall = (store_c * 0.25 + date_c * 0.25 + total_c * 0.5)

    # Carbon estimate
    carbon = 0.0
    try:
        if total_v: carbon = float(total_v) * CARBON_KG_PER_UNIT
    except (TypeError, ValueError):
        carbon = 0.0

    return {
        "filename": filename,
        "store_name": store_name,
        "date": date_v,
        "total_amount": total_v,
        "store_conf": store_c,
        "date_conf":  date_c,
        "total_conf": total_c,
        "overall_conf": overall,
        "avg_ocr": avg_ocr,
        "carbon_kg": round(carbon, 3),
        "n_items": len(items_raw),
        "n_lines": len(lines),
    }


def process_file(file_bytes: bytes, filename: str) -> dict:
    """Public entry point — uses cached extraction + decodes image fresh."""
    result = _cached_extract(file_bytes, filename)
    if "error" in result:
        return result
    # Decode image fresh (fast, ~50ms) — keep arrays out of the cache
    arr = np.frombuffer(file_bytes, np.uint8)
    img_orig = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return {**result, "_original_image": img_orig}


# Warm the OCR reader on app start (cached — only runs once per session)
_warm_ocr_reader()


# ═══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════
LOGO_SVG = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2C8 6 4 9 4 14C4 18.4 7.6 22 12 22C16.4 22 20 18.4 20 14C20 9 16 6 12 2Z"
        fill="#052218" stroke="#ecfdf5" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M12 11C10 13 8 15 8 17" stroke="#4ade80" stroke-width="1.6"
        stroke-linecap="round"/>
</svg>
"""

st.markdown(f"""
<div class="nav">
  <div class="brand">
    <div class="brand-mark">{LOGO_SVG}</div>
    <div class="brand-text">Carbon Crunch<span class="sub">Receipt Intelligence</span></div>
  </div>
  <span class="nav-pill"><span class="status-dot"></span>Pipeline Live</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — HERO  (left text · right illustration)
# ═══════════════════════════════════════════════════════════════════════════════
hero_l, hero_r = st.columns([1.1, 1], gap="large")

with hero_l:
    st.markdown("""
    <div class="hero-eyebrow">◆ AI · OCR · CARBON ANALYTICS</div>
    <div class="hero-h1">
      Turn receipts into <span class="accent">carbon insights</span> instantly.
    </div>
    <div class="hero-sub">
      Upload receipts and get structured data with carbon estimates in seconds.
      Confidence-aware extraction, audit-grade outputs, ready for your emissions
      dashboard.
    </div>
    <a href="#upload" class="hero-cta">
      Upload Receipt → Get Insights
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
      </svg>
    </a>
    <div class="hero-trust">
      <span><strong>OCR</strong> EasyOCR engine</span>
      <span><strong>Privacy</strong> on-device</span>
      <span><strong>Audit</strong> confidence-tagged</span>
    </div>
    """, unsafe_allow_html=True)

with hero_r:
    st.markdown("""
    <div class="hero-art">
      <svg viewBox="0 0 400 360" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%">
        <defs>
          <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#00c896" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#00c896" stop-opacity="0"/>
          </linearGradient>
          <linearGradient id="g2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0a1f17"/>
            <stop offset="100%" stop-color="#081813"/>
          </linearGradient>
        </defs>
        <g transform="translate(60 30) rotate(-8 110 160)">
          <path d="M0 0 L220 0 L220 280 L210 290 L195 280 L180 290 L165 280 L150 290 L135 280 L120 290 L105 280 L90 290 L75 280 L60 290 L45 280 L30 290 L15 280 L0 290 Z" fill="url(%23g2)" stroke="rgba(0,200,150,0.4)" stroke-width="1.5"/>
          <line x1="20" y1="40"  x2="200" y2="40"  stroke="#4ade80" stroke-width="2.5"/>
          <line x1="20" y1="60"  x2="160" y2="60"  stroke="rgba(255,255,255,0.4)" stroke-width="1.5"/>
          <line x1="20" y1="90"  x2="100" y2="90"  stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="120" y1="90" x2="200" y2="90"  stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="20" y1="115" x2="100" y2="115" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="120" y1="115" x2="200" y2="115" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="20" y1="140" x2="100" y2="140" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="120" y1="140" x2="200" y2="140" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
          <line x1="20" y1="180" x2="200" y2="180" stroke="rgba(0,200,150,0.4)" stroke-width="1.2" stroke-dasharray="3"/>
          <line x1="20" y1="210" x2="100" y2="210" stroke="rgba(74,222,128,0.7)" stroke-width="2.5"/>
          <line x1="120" y1="210" x2="200" y2="210" stroke="rgba(74,222,128,0.9)" stroke-width="3"/>
        </g>
        <g transform="translate(220 60)">
          <rect x="0" y="0" width="140" height="70" rx="12" fill="rgba(8,18,14,0.95)" stroke="rgba(0,200,150,0.5)" stroke-width="1.2"/>
          <circle cx="18" cy="22" r="6" fill="#4ade80"/>
          <rect x="32" y="16" width="80" height="5" rx="2" fill="rgba(255,255,255,0.5)"/>
          <rect x="32" y="26" width="50" height="4" rx="2" fill="rgba(255,255,255,0.25)"/>
          <line x1="14" y1="44" x2="126" y2="44" stroke="rgba(0,200,150,0.15)"/>
          <text x="14" y="60" font-family="monospace" font-size="11" fill="#4ade80" font-weight="700">986.50</text>
          <text x="120" y="60" font-family="monospace" font-size="9" fill="rgba(74,222,128,0.7)" text-anchor="end">94%</text>
        </g>
        <g transform="translate(240 200)">
          <rect x="0" y="0" width="150" height="80" rx="12" fill="rgba(8,18,14,0.95)" stroke="rgba(0,200,150,0.5)" stroke-width="1.2"/>
          <g transform="translate(14 16)">
            <path d="M14 0C9 5 4 9 4 16C4 22 9 26 14 26C19 26 24 22 24 16C24 9 19 5 14 0Z" fill="rgba(0,200,150,0.2)" stroke="#4ade80" stroke-width="1.5"/>
            <path d="M14 12C12 14 10 16 10 18" stroke="#4ade80" stroke-width="1.5" stroke-linecap="round"/>
          </g>
          <text x="48" y="22" font-family="Inter" font-size="9" fill="rgba(255,255,255,0.5)" font-weight="700" letter-spacing="2">CO2 EST.</text>
          <text x="48" y="44" font-family="monospace" font-size="18" fill="#4ade80" font-weight="700">40.4 kg</text>
          <rect x="14" y="58" width="120" height="6" rx="3" fill="rgba(255,255,255,0.06)"/>
          <rect x="14" y="58" width="78" height="6" rx="3" fill="#00c896" opacity="0.7"/>
        </g>
        <circle cx="200" cy="180" r="3" fill="#00c896" opacity="0.6"/>
        <circle cx="220" cy="170" r="2" fill="#00c896" opacity="0.4"/>
        <circle cx="240" cy="155" r="2.5" fill="#00c896" opacity="0.5"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — UPLOAD SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div id="upload" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top:4rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">◇ STEP 1 — UPLOAD</div>
<div class="section-title">Drag & drop your receipts</div>
<div class="section-sub">
  Supports PNG, JPG, JPEG. Process one or many — extraction takes
  about 5 seconds per receipt.
</div>
""", unsafe_allow_html=True)

up_l, up_c, up_r = st.columns([1, 6, 1])
with up_c:
    files = st.file_uploader(
        label="upload receipts",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — PROCESSING + 4 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if files:
    process_l, process_c, process_r = st.columns([1, 6, 1])
    with process_c:
        if "results" not in st.session_state or \
           st.session_state.get("file_signature") != [f.name for f in files]:
            with st.spinner(f"◆ Extracting data from {len(files)} receipt(s)…"):
                results = []
                for f in files:
                    results.append(process_file(f.read(), f.name))
                    time.sleep(0.05)
                st.session_state["results"] = results
                st.session_state["file_signature"] = [f.name for f in files]
        results = st.session_state["results"]

    st.markdown('<div style="margin-top:3rem"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">◇ STEP 2 — INSIGHTS</div>
    <div class="section-title">Extracted data + carbon estimates</div>
    <div class="section-sub">
      Every field comes with a calibrated confidence score. Carbon estimate
      uses an industry-average emission coefficient — replace with your own
      category-specific factors in production.
    </div>
    """, unsafe_allow_html=True)

    for idx, r in enumerate(results):
        if "error" in r:
            st.error(f"❌ {r.get('filename', 'file')}: {r['error']}")
            continue

        st.markdown(f'<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
        col_img, col_data = st.columns([1, 1.15], gap="large")

        # LEFT — image preview
        with col_img:
            st.markdown(
                f'<div class="preview-wrap">'
                f'<div class="preview-label">◆ Receipt · {r["filename"]}</div>',
                unsafe_allow_html=True,
            )
            st.image(cv2.cvtColor(r["_original_image"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # RIGHT — extracted data card
        with col_data:
            overall = r["overall_conf"]
            ocls = conf_class(overall)
            opct = int(overall * 100)

            html = (
                '<div class="result-card">'
                '<div class="result-header">'
                '<div class="result-title">Extracted Data</div>'
                f'<div class="result-status">'
                f'<span style="width:6px;height:6px;background:#00c896;'
                f'border-radius:50%;box-shadow:0 0 8px #00c896"></span>'
                f'Confidence {opct}%'
                f'</div>'
                '</div>'
                + render_field_with_conf("Store", r["store_name"], r["store_conf"])
                + render_field_with_conf("Date", r["date"] or "Not detected", r["date_conf"])
                + render_field_with_conf("Total", r["total_amount"] or "Not detected", r["total_conf"])
                + render_field("Items detected", str(r["n_items"]))
                +
                '<div class="carbon-banner">'
                '<div class="carbon-icon">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
                ' stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 2C8 6 4 9 4 14C4 18.4 7.6 22 12 22C16.4 22 20 18.4 20 14C20 9 16 6 12 2Z"/>'
                '<path d="M12 11C10 13 8 15 8 17"/>'
                '</svg>'
                '</div>'
                '<div class="carbon-text">'
                '<div class="carbon-label">CARBON ESTIMATE</div>'
                f'<div class="carbon-value">{r["carbon_kg"]:.2f}<span class="unit">kg CO₂e</span></div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

            payload = {
                "file": r["filename"],
                "store_name":   {"value": r["store_name"], "confidence": round(r["store_conf"], 3)},
                "date":         {"value": r["date"],       "confidence": round(r["date_conf"], 3)},
                "total_amount": {"value": r["total_amount"], "confidence": round(r["total_conf"], 3)},
                "carbon_estimate_kg_co2e": r["carbon_kg"],
                "overall_confidence": round(r["overall_conf"], 3),
            }
            st.download_button(
                "↓ Download JSON",
                data=json.dumps(payload, indent=2, ensure_ascii=False),
                file_name=f"{Path(r['filename']).stem}.json",
                mime="application/json",
                key=f"dl_{idx}",
                use_container_width=True,
            )

else:
    # Empty state under uploader
    st.markdown("""
    <div class="empty">
      <div class="icon-wrap">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div class="empty-title">No receipts uploaded yet</div>
      <div class="empty-sub">Drop one above to see extracted insights here</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="margin-top:5rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-eyebrow">◇ THE WORKFLOW</div>
<div class="section-title">How it works</div>
<div class="section-sub">
  Three steps from a paper receipt to an emissions-ready data row.
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="large")
with s1:
    st.markdown("""
    <div class="step-card">
      <div class="step-num">1</div>
      <div class="step-title">Upload</div>
      <div class="step-text">
        Drop a single receipt or a batch. Supports JPG, PNG, BMP, TIFF, WebP.
        Images are processed locally — nothing leaves your machine.
      </div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown("""
    <div class="step-card">
      <div class="step-num">2</div>
      <div class="step-title">Extract</div>
      <div class="step-text">
        OpenCV preprocessing fixes noise, lighting, skew. EasyOCR reads the text
        with per-line confidence scores. Regex + heuristics pull out the fields.
      </div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown("""
    <div class="step-card">
      <div class="step-num">3</div>
      <div class="step-title">Analyze</div>
      <div class="step-text">
        Each field gets a calibrated trust score, low-confidence items are
        flagged for review, and a CO₂ estimate is computed against industry
        emission factors.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
  Built for <strong>Carbon Crunch</strong> · AI OCR Pipeline · April 2026
  <br/>
  <span style='font-size:0.7rem;opacity:0.7'>
    Crafted by Tirupathamma Guntru · EasyOCR · OpenCV · Streamlit
  </span>
</div>
""", unsafe_allow_html=True)
