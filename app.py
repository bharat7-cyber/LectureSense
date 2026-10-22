"""
LectureSense — AI Study Workspace
================================

PDF + MP4  ->  faster-whisper (word timestamps)
           ->  src/inference_temporal_v4.py  (checkpoints/temporal_v4_best_model.pt)
           ->  timeline.json + rendered slides
           ->  Study Workspace (video player, slide timeline, notes, transcript).

The V4 model, decoder and every inference flag are untouched. This file adds the
Study Workspace and learning tools around the existing alignment pipeline.

Run with:
    streamlit run app.py
"""

import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import torch
import streamlit as st

# Streamlit >=1.50 deprecates use_container_width in favour of width="stretch".
import inspect as _inspect
_FULL = ({"width": "stretch"} if "width" in _inspect.signature(st.button).parameters
         else {"use_container_width": True})

# faster-whisper is imported lazily inside _load_whisper_pipeline() so the app
# still boots without it and a broken install surfaces as a friendly error.

# ============================================================
# INTERNAL PATHS (never surfaced in the UI)
# ============================================================

_BASE_DIR = Path(__file__).resolve().parent
_INFERENCE_SCRIPT = _BASE_DIR / "LectureSense_V11_Inference.py"
_TUNED_CONFIG_PATH = _BASE_DIR / "checkpoints" / "v4_2_decoder_tuned_config.json"
_CHECKPOINT_PATH = _BASE_DIR / "checkpoints" / "temporal_v4_best_model.pt"

_RUN_DIR = _BASE_DIR / "runs" / "session"
_TIMELINE_PATH = _RUN_DIR / "timeline.json"
_SLIDES_DIR = _RUN_DIR / "rendered_slides"
_UPLOAD_DIR = _RUN_DIR / "uploads"
_LOG_PATH = _RUN_DIR / "pipeline.log"
_VIDEO_NAME = "lecture.mp4"

_EXAMPLE_DIR = _BASE_DIR / "demo_v11"
_EXAMPLE_TIMELINE_PATH = _EXAMPLE_DIR / "timeline.json"
_EXAMPLE_SLIDES_DIR = _EXAMPLE_DIR / "rendered_slides"

_INFERENCE_TIMEOUT_S = 1800
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# faster-whisper config (affects transcript generation only)
# ============================================================
# FAST WHISPER CONFIG — RTX 2050 / 4GB VRAM
# ============================================================

_WHISPER_MODEL_SIZE = "distil-small.en"

# Automatically use CUDA when available.
_WHISPER_DEVICE = "auto"

# RTX 2050 compatible.
_WHISPER_GPU_COMPUTE_TYPE = "float16"
_WHISPER_CPU_COMPUTE_TYPE = "int8"

# Keep this at 8 for stable 4GB VRAM usage.
_WHISPER_BATCH_SIZE = 8

# Greedy decoding = significantly faster.
_WHISPER_BEAM_SIZE = 1

# VAD removes long silent regions before decoding.
_WHISPER_VAD_MIN_SILENCE_MS = 500

# Explicit language avoids language-identification work.
_WHISPER_LANGUAGE = "en"

# Persistent transcript cache.
_WHISPER_CACHE_DIR = _BASE_DIR / "runs" / "transcript_cache"
# ============================================================
# PAGE CONFIG + STYLE
# ============================================================

st.set_page_config(
    page_title="LectureSense",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --paper:#080B12; --surface:#111722; --surface-2:#161E2C; --ink:#F5F7FB;
        --stone:#A7B0C0; --line:#252D3A; --line-strong:#344052;
        --accent:#7C83FF; --accent-soft:#171C34; --accent-line:#4A52C9;
        --amber:#F59E0B; --amber-soft:#2A2110;
        --good:#34D399; --good-soft:#102A22; --bad:#F87171; --bad-soft:#2D1618;
        --radius-s:8px; --radius-m:12px; --radius-l:18px;
    }
    html, body, [class*="css"] { font-family:'Inter',-apple-system,sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    #MainMenu, header, footer, [data-testid="stStatusWidget"],
    [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility:hidden; height:0; }
    .block-container { padding:1.6rem 1.4rem 3rem; max-width:1280px; }
    [data-testid="stWidgetLabel"] { display:none; }
    hr { border-color:var(--line); margin:.9rem 0; }

    /* ---- Brand bar ---- */
    .ls-brand { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem; }
    .ls-logo { display:flex; align-items:center; gap:.6rem; font-weight:800; letter-spacing:.04em; font-size:.95rem; }
    .ls-logo .glyph { width:28px; height:28px; border-radius:8px; display:grid; place-items:center;
        background:linear-gradient(135deg,#7C83FF,#A78BFA); color:#0B0E17; font-size:.9rem; }
    .ls-hero h1 { font-size:2.3rem; font-weight:800; letter-spacing:-.02em; line-height:1.12; margin:.2rem 0 .6rem; }
    .ls-hero h1 span { background:linear-gradient(90deg,#7C83FF,#C4B5FD); -webkit-background-clip:text; color:transparent; }
    .ls-hero p { color:var(--stone); font-size:1rem; max-width:620px; line-height:1.6; margin:0 0 1.4rem; }
    .ls-features { display:flex; flex-wrap:wrap; gap:.5rem; margin-bottom:1.6rem; }
    .ls-feature { font-size:.78rem; color:var(--stone); border:1px solid var(--line); background:var(--surface);
        padding:.32rem .7rem; border-radius:999px; }

    /* ---- Upload ---- */
    div[data-testid="stFileUploader"] { background:var(--surface); border:1.5px dashed var(--line-strong);
        border-radius:var(--radius-l); padding:.4rem .4rem .9rem; transition:border-color .15s, background .15s; }
    div[data-testid="stFileUploader"]:hover { border-color:var(--accent); background:var(--accent-soft); }
    [data-testid="stFileUploaderDropzone"] { background:var(--surface) !important; border:1.5px dashed #3A4558 !important; }
    [data-testid="stFileUploaderDropzone"] * { color:#E8EDF5 !important; }
    [data-testid="stFileUploaderDropzone"] button { background:#171C26 !important; color:#fff !important;
        border:1px solid #343D4D !important; border-radius:9px !important; font-weight:600 !important; }
    [data-testid="stFileUploaderFile"] { background:var(--accent-soft) !important; border-radius:var(--radius-s) !important; }
    [data-testid="stFileUploaderFileName"] { color:var(--ink) !important; font-weight:500; }
    .ls-step { display:flex; align-items:center; gap:.55rem; padding:.85rem .9rem 0; font-size:.9rem; font-weight:600; }
    .ls-step .num { display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px;
        border-radius:50%; background:var(--accent-soft); color:var(--accent); font-size:.74rem; font-weight:700; }
    .ls-step.done .num { background:var(--good-soft); color:var(--good); }
    .ls-step-sub { padding:.15rem .9rem 0 2.7rem; font-size:.78rem; color:var(--stone); }
    .ls-filemeta { font-size:.78rem; color:var(--stone); padding:.35rem .2rem 0; }

    /* ---- Buttons ---- */
    .stButton { margin:0 !important; }
    .stButton > button { border-radius:var(--radius-s); font-weight:600; border:1px solid var(--line-strong);
        background:var(--surface); color:var(--ink); padding:.55rem 1rem; transition:all .12s ease; }
    .stButton > button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); background:var(--accent-soft); }
    .stButton > button:disabled { color:#4A5568; border-color:var(--line); background:transparent; }
    button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background:var(--accent) !important; color:#0B0E17 !important; border:1px solid var(--accent) !important; }
    button[kind="primary"]:hover:not(:disabled), button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
        background:#9298FF !important; color:#0B0E17 !important; transform:translateY(-1px);
        box-shadow:0 6px 18px rgba(124,131,255,.35); }
    .stDownloadButton > button { background:var(--surface); color:var(--ink); border:1px solid var(--line-strong);
        font-weight:600; border-radius:var(--radius-s); }
    .stDownloadButton > button:hover { border-color:var(--accent); color:var(--accent); }
    .ls-hint { font-size:.84rem; color:var(--stone); margin:.2rem 0; }

    /* ---- Processing card ---- */
    .ls-proc { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-l); padding:1.4rem 1.5rem; }
    .ls-proc h3 { margin:0 0 .2rem; font-size:1.15rem; font-weight:700; }
    .ls-proc .sub { color:var(--stone); font-size:.86rem; margin-bottom:1rem; }
    .ls-pstep { display:flex; align-items:center; gap:.7rem; padding:.55rem 0; font-size:.92rem; color:var(--stone); border-top:1px solid var(--line); }
    .ls-pstep:first-of-type { border-top:none; }
    .ls-pstep .ic { width:22px; height:22px; border-radius:50%; display:grid; place-items:center; font-size:.72rem;
        background:var(--line); color:var(--stone); flex:none; }
    .ls-pstep.active { color:var(--ink); font-weight:600; }
    .ls-pstep.active .ic { background:var(--accent); color:#0B0E17; }
    .ls-pstep.done { color:var(--ink); }
    .ls-pstep.done .ic { background:var(--good-soft); color:var(--good); }
    .ls-pstep .meta { margin-left:auto; font-size:.78rem; color:var(--stone); font-variant-numeric:tabular-nums; font-weight:500; }
    .ls-bar { height:6px; background:var(--line); border-radius:999px; overflow:hidden; margin-top:1rem; }
    .ls-bar > div { height:100%; background:linear-gradient(90deg,#7C83FF,#A78BFA); border-radius:999px; transition:width .25s ease; }
    .ls-bar.indet > div { width:35%; animation:ls-slide 1.3s ease-in-out infinite; }
    @keyframes ls-slide { 0%{margin-left:-35%} 100%{margin-left:100%} }

    /* ---- Workspace header / stats ---- */
    .ls-pill { display:inline-flex; align-items:center; gap:.45rem; font-size:.8rem; font-weight:600; color:var(--good);
        background:var(--good-soft); padding:.32rem .85rem .32rem .65rem; border-radius:999px; }
    .ls-pill .dot { width:6px; height:6px; border-radius:50%; background:var(--good); }
    .ls-stats { display:flex; gap:.5rem; flex-wrap:wrap; margin:.2rem 0 .9rem; }
    .ls-stat { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-m); padding:.5rem .85rem; }
    .ls-stat b { display:block; font-size:1.05rem; font-variant-numeric:tabular-nums; }
    .ls-stat span { font-size:.72rem; color:var(--stone); text-transform:uppercase; letter-spacing:.05em; }

    /* ---- Timeline strip ---- */
    .ls-strip { display:flex; height:12px; gap:2px; margin:.2rem 0 .3rem; }
    .ls-strip div { border-radius:3px; background:var(--line-strong); min-width:3px; }
    .ls-strip div.seen { background:#3B4378; }
    .ls-strip div.on { background:var(--accent); }
    .ls-strip-label { display:flex; justify-content:space-between; font-size:.78rem; color:var(--stone); margin-bottom:1rem; }

    /* ---- Panels ---- */
    .ls-label { font-size:.74rem; font-weight:700; letter-spacing:.06em; color:var(--stone); text-transform:uppercase; margin:0 0 .5rem; }
    .ls-card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-l); padding:1rem; }
    .ls-slide-frame { border:1px solid var(--line); border-radius:var(--radius-m); overflow:hidden; background:#000; }
    .ls-slide-frame img { display:block; width:100%; }
    .ls-empty { border:1px dashed var(--line-strong); border-radius:var(--radius-m); padding:2.2rem 1rem; text-align:center;
        color:var(--stone); font-size:.88rem; background:var(--surface); }
    .ls-slide-title { font-size:1.3rem; font-weight:700; margin:.7rem 0 .1rem; }
    .ls-slide-time { font-size:.82rem; color:var(--stone); font-variant-numeric:tabular-nums; margin-bottom:.4rem; }
    div[data-testid="stVideo"] video { border-radius:var(--radius-m); border:1px solid var(--line); background:#000; }

    .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { background:transparent; padding:8px 14px; color:var(--stone); font-weight:500; font-size:.88rem; }
    .stTabs [aria-selected="true"] { color:var(--ink) !important; font-weight:600; }
    .stTabs [data-baseweb="tab-highlight"] { background:var(--accent) !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top:1rem; }

    .ls-keypoints { margin:0; padding-left:1.1rem; }
    .ls-keypoints li { font-size:.98rem; line-height:1.7; margin-bottom:.65rem; }
    .ls-chip { display:inline-block; background:var(--accent-soft); color:#B9BEFF; border:1px solid var(--accent-line);
        font-size:.8rem; font-weight:600; padding:.28rem .7rem; border-radius:999px; margin:0 .35rem .4rem 0; }
    .ls-block { background:#0D131D; border:1px solid var(--line); color:#D8DEE9; font-size:.92rem; line-height:1.7;
        padding:1rem; border-radius:var(--radius-m); white-space:pre-wrap; }
    .ls-tline { font-size:.92rem; line-height:1.6; color:var(--ink); padding:.15rem 0 .35rem; }

    div[data-testid="stTextInput"] input { border-radius:var(--radius-s) !important; border:1px solid var(--line-strong) !important;
        background:var(--surface) !important; color:var(--ink) !important; }
    div[data-testid="stTextInput"] > div { background:transparent !important; }
    div[data-testid="stAlert"] { border-radius:var(--radius-m); }

    /* ---- Study workspace: tabs, notes, concepts, Q&A, assistant ---- */
    .stTabs [data-baseweb="tab-list"] { overflow-x:auto; scrollbar-width:none; }
    .stTabs [data-baseweb="tab"] { white-space:nowrap; }
    .ls-note-head { display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:.4rem;
        margin:0 0 .8rem; }
    .ls-note-head h3 { margin:0; font-size:1.15rem; font-weight:700; }
    .ls-note-head span { font-size:.8rem; color:var(--stone); font-variant-numeric:tabular-nums; }
    .ls-sec { font-size:.72rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--accent);
        margin:1.1rem 0 .5rem; }
    .ls-take { display:flex; gap:.7rem; background:var(--surface); border:1px solid var(--line);
        border-radius:var(--radius-m); padding:.75rem .9rem; margin-bottom:.5rem; }
    .ls-take .n { flex:none; width:22px; height:22px; border-radius:50%; background:var(--accent-soft); color:var(--accent);
        font-size:.74rem; font-weight:700; display:grid; place-items:center; margin-top:.1rem; }
    .ls-take .tx { font-size:.95rem; line-height:1.65; flex:1; }
    .ls-take .tm, .ls-ref { flex:none; font-size:.72rem; font-weight:600; color:#B9BEFF; background:var(--accent-soft);
        border:1px solid var(--accent-line); border-radius:999px; padding:.1rem .55rem; height:fit-content;
        font-variant-numeric:tabular-nums; white-space:nowrap; }
    .ls-concept { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-m);
        padding:.85rem .95rem .6rem; margin-top:.2rem; }
    .ls-term { font-size:1rem; font-weight:700; margin-bottom:.25rem; }
    .ls-def { font-size:.9rem; line-height:1.6; color:#D8DEE9; }
    .ls-also { font-size:.76rem; color:var(--stone); margin-top:.45rem; }
    .ls-ans-lead { font-size:.92rem; color:var(--ink); line-height:1.6; margin-bottom:.55rem; }
    .ls-src { background:var(--surface-2); border:1px solid var(--line); border-left:3px solid var(--accent);
        border-radius:var(--radius-s); padding:.6rem .8rem; margin-bottom:.45rem; font-size:.9rem; line-height:1.6; }
    .ls-src-kind { font-size:.7rem; color:var(--stone); margin-left:.5rem; text-transform:uppercase; letter-spacing:.05em; }
    .ls-src .ls-ref { display:inline-block; margin-bottom:.3rem; }
    .ls-q { display:flex; justify-content:flex-end; margin:1rem 0 .5rem; }
    .ls-q div { max-width:85%; background:var(--accent); color:#0B0E17; font-weight:500; font-size:.92rem;
        padding:.55rem .9rem; border-radius:16px 16px 4px 16px; line-height:1.5; }
    .ls-a { background:var(--surface); border:1px solid var(--line); border-radius:16px 16px 16px 4px;
        padding:.8rem .95rem .5rem; max-width:96%; }
    .ls-a-who { font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--accent);
        margin-bottom:.35rem; }
    div[data-testid="stForm"] { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius-m);
        padding:.8rem .9rem; }
    [data-testid="stFormSubmitButton"] > button { border-radius:var(--radius-s); font-weight:600; width:100%; }
    @media (max-width: 900px) {
        .ls-q div { max-width:95%; }
        .ls-take { flex-wrap:wrap; }
    }

    /* ---- AI study notes ---- */
    .ls-ai-badge { display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:700;
        color:#C4B5FD; background:var(--accent-soft); border:1px solid var(--accent-line); border-radius:999px;
        padding:.15rem .6rem; margin-bottom:.6rem; }
    .ls-ai-summary { font-size:1rem; line-height:1.7; background:linear-gradient(135deg,#171C34,#111722);
        border:1px solid var(--accent-line); border-radius:var(--radius-m); padding:.9rem 1rem; }
    .ls-defrow { display:flex; gap:.8rem; padding:.6rem 0; border-bottom:1px solid var(--line); }
    .ls-defrow:last-child { border-bottom:none; }
    .ls-defrow .k { flex:none; width:28%; min-width:110px; font-weight:700; font-size:.92rem; color:#B9BEFF; }
    .ls-defrow .v { flex:1; font-size:.92rem; line-height:1.65; color:#D8DEE9; }
    .ls-bul { margin:0; padding-left:1.15rem; }
    .ls-bul li { font-size:.95rem; line-height:1.7; margin-bottom:.4rem; }
    .ls-formula { background:#0D131D; border:1px solid var(--line); border-left:3px solid var(--amber);
        border-radius:var(--radius-s); padding:.55rem .85rem; margin-bottom:.45rem;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9rem; color:#FCD9A0; overflow-x:auto; }
    .ls-example { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--good);
        border-radius:var(--radius-s); padding:.6rem .85rem; margin-bottom:.45rem; font-size:.92rem; line-height:1.65; }
    @media (max-width: 900px) {
        .ls-defrow { flex-direction:column; gap:.15rem; }
        .ls-defrow .k { width:auto; }
    }

    /* ---- Responsive ---- */
    @media (max-width: 900px) {
        .block-container { padding:1rem .8rem 2.5rem; }
        .ls-hero h1 { font-size:1.7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

for _key, _default in {
    "result": None,
    "slides_dir": None,
    "video_path": None,
    "error": None,
    "error_details": None,
    "pos": 0,
    "video_start": 0,
    "from_example": False,
    "reviewed_visits": set(),
    "quiz_items": [],
    "quiz_answers": {},
    "quiz_score": 0,
    "quiz_submitted": False,
    "flash_index": 0,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ============================================================
# NOTE-GENERATION HELPERS (derived strictly from the real timeline)
# ============================================================

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on",
    "for", "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "we", "you", "your", "our",
    "they", "their", "he", "she", "his", "her", "i", "my", "me", "us", "them", "not",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "should",
    "may", "might", "shall", "must", "let", "lets", "just", "about", "into", "over",
    "also", "there", "here", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "any", "some", "no", "nor", "than", "too", "very", "one", "two",
    "now", "up", "out", "off", "again", "further", "because", "while", "each", "other",
    "such", "only", "own", "same", "s", "t", "re", "ve", "ll",
    "okay", "um", "uh", "yeah", "like", "going", "get", "got", "gonna", "actually",
    "basically", "kind", "sort", "thing", "things", "really", "right", "well",
}
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_SIGNAL_WORDS = (
    "important", "remember", "note that", "key", "essentially", "crucial",
    "must", "always", "never", "therefore", "in summary", "to summarize",
    "the main", "critical", "significant",
)


def _extract_key_points(transcript_text: str, max_points: int = 5):
    """Extractive key points: real sentences from the transcript."""
    if not transcript_text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(transcript_text) if s.strip()]
    if not sentences:
        return []

    def score(sentence: str):
        lower = sentence.lower()
        signal = sum(1 for w in _SIGNAL_WORDS if w in lower)
        return signal * 2 + min(len(sentence.split()), 30) / 30.0

    ranked = sorted(sentences, key=score, reverse=True)
    chosen, seen = [], set()
    for s in ranked:
        if s not in seen and len(s.split()) >= 4:
            chosen.append(s)
            seen.add(s)
        if len(chosen) >= max_points:
            break
    return [s for s in sentences if s in seen] or chosen


def _extract_keywords(text: str, max_keywords: int = 8):
    """Frequency-based keywords from real transcript + OCR text."""
    if not text:
        return []
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 3]
    if not words:
        return []
    counts = Counter(words)
    common = [w for w, _ in counts.most_common(max_keywords * 2)]
    repeated = [w for w in common if counts[w] > 1]
    ranked = repeated if len(repeated) >= 3 else common
    return [w.capitalize() for w in ranked[:max_keywords]]


def _build_visits(timeline: list):
    """Chronological slide 'visits': consecutive timeline segments that share
    a slide_id are merged. If the lecturer returns to an earlier slide it is a
    new visit, so every entry maps to one real, contiguous span of the video."""
    segs = sorted(
        (s for s in timeline if isinstance(s, dict)),
        key=lambda s: float(s.get("start_time") or 0),
    )
    visits = []
    for seg in segs:
        sid = seg.get("slide_id")
        if visits and visits[-1]["slide_id"] == sid:
            visits[-1]["segments"].append(seg)
        else:
            visits.append({"slide_id": sid, "segments": [seg]})

    for v in visits:
        segments = v["segments"]
        transcript = " ".join(
            (s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip()
        )
        ocr_parts = []
        for s in segments:
            ocr = s.get("ocr_text") or s.get("slide_ocr")
            if ocr and ocr.strip():
                ocr_parts.append(ocr.strip())
        ocr_text = "\n".join(dict.fromkeys(ocr_parts))
        v.update(
            transcript_text=transcript,
            ocr_text=ocr_text,
            key_points=_extract_key_points(transcript),
            keywords=_extract_keywords(transcript + "\n" + ocr_text),
            start_time=min(float(s.get("start_time") or 0) for s in segments),
            end_time=max(float(s.get("end_time") or 0) for s in segments),
            slide_index=segments[0].get("slide_index", 0) or 0,
            slide_title=segments[0].get("slide_title", "") or "",
        )
    return visits


def _compile_downloadable_notes(visits: list) -> str:
    lines = ["LECTURE STUDY NOTES", "=" * 40, ""]
    for i, v in enumerate(visits, start=1):
        lines.append(f"{i}. Slide {v['slide_id']}  ({_clock(v['start_time'])} - {_clock(v['end_time'])})")
        lines.append("-" * 40)
        if v["keywords"]:
            lines.append("Keywords: " + ", ".join(v["keywords"]))
        if v["key_points"]:
            lines.append("\nKey points:")
            lines += [f"  • {p}" for p in v["key_points"]]
        if v["ocr_text"]:
            lines += ["\nSlide content:", v["ocr_text"]]
        lines += ["\nTranscript:", v["transcript_text"] or "(no transcript captured)", "", ""]
    return "\n".join(lines)


# ============================================================
# ============================================================
# TEACHING-NOTES UI
# ============================================================

st.markdown(
    """
    <style>
    .ls-teach-section {
        margin:.8rem 0 .45rem;
        font-size:.86rem;
        font-weight:800;
        letter-spacing:.06em;
        text-transform:uppercase;
        color:var(--accent);
    }
    .ls-teach-card {
        background:var(--surface);
        border:1px solid var(--line);
        border-radius:var(--radius-m);
        padding:1rem 1.05rem;
        margin-bottom:.75rem;
    }
    .ls-teach-lead {
        color:var(--ink);
        font-size:.98rem;
        line-height:1.7;
    }
    .ls-teach-step {
        display:flex;
        gap:.7rem;
        align-items:flex-start;
        padding:.48rem 0;
        border-bottom:1px solid var(--line);
    }
    .ls-teach-step:last-child { border-bottom:0; }
    .ls-teach-step .num {
        min-width:24px;
        height:24px;
        border-radius:7px;
        display:grid;
        place-items:center;
        background:var(--accent-soft);
        color:var(--accent);
        font-size:.75rem;
        font-weight:800;
    }
    .ls-teach-step .text {
        color:var(--ink);
        line-height:1.6;
        font-size:.92rem;
    }
    .ls-teach-def {
        padding:.7rem 0;
        border-bottom:1px solid var(--line);
    }
    .ls-teach-def:last-child { border-bottom:0; }
    .ls-teach-def .term {
        font-weight:800;
        color:var(--ink);
        margin-bottom:.2rem;
    }
    .ls-teach-def .definition {
        color:var(--ink);
        line-height:1.6;
    }
    .ls-teach-def .easy {
        color:var(--stone);
        line-height:1.55;
        margin-top:.25rem;
        font-size:.9rem;
    }
    .ls-teach-bullet {
        color:var(--ink);
        line-height:1.6;
        margin:.38rem 0;
    }
    .ls-teach-revision {
        background:var(--accent-soft);
        border:1px solid var(--accent-line);
        border-radius:var(--radius-m);
        padding:.9rem 1rem;
        margin-top:.45rem;
    }
    .ls-teach-revision .row {
        padding:.25rem 0;
        color:var(--ink);
        line-height:1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# GENERAL HELPERS
# ============================================================

def _esc(text) -> str:
    return html.escape(str(text or ""))


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _parse_clock(text: str):
    """'83', '1:23' or '01:02:03' -> seconds, else None."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3 or any(p < 0 for p in parts):
        return None
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def _format_mavils_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0

    total_ms = int(round(float(seconds) * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1_000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _find_slide_image(slides_dir, slide_index, slide_title=""):
    if not slides_dir or not Path(slides_dir).exists():
        return None
    images = sorted(
        (p for p in Path(slides_dir).iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    )
    if not images:
        return None
    if slide_title:
        for p in images:
            if p.stem == slide_title:
                return p
    try:
        idx = int(slide_index)
    except (TypeError, ValueError):
        return None
    return images[idx] if 0 <= idx < len(images) else None


def _load_timeline_json(path: Path):
    if not path.exists():
        return None, "No timeline was produced for this lecture."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "The generated timeline could not be read."
    if not isinstance(data, dict) or "timeline" not in data:
        return None, "The generated timeline is missing expected data."
    return data, None


def _pair_fingerprint(pdf_bytes: bytes, video_bytes: bytes) -> str:
    """Fingerprint uploads AND pipeline/config so old timelines are not reused."""
    h = hashlib.sha256()
    h.update(b"LectureSense-V4.2-tuned-pipeline-v1")
    h.update(len(pdf_bytes).to_bytes(8, "little"))
    h.update(pdf_bytes[:1_000_000])
    h.update(len(video_bytes).to_bytes(8, "little"))
    h.update(video_bytes[:1_000_000])

    # Small config and inference sources are hashed by content; large model
    # checkpoints/support modules are invalidated by size + modification time.
    source_paths = (
        _INFERENCE_SCRIPT,
        _TUNED_CONFIG_PATH,
        _CHECKPOINT_PATH,
        _BASE_DIR / "src" / "inference_temporal_v4.py",
        _BASE_DIR / "src" / "inference.py",
        _BASE_DIR / "src" / "boundary_dp_v4.py",
        _BASE_DIR / "src" / "video_visual_alignment.py",
    )
    for path in source_paths:
        h.update(str(path.name).encode("utf-8", errors="replace"))
        if not path.exists():
            h.update(b"<missing>")
            continue
        stat = path.stat()
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        if path in (_INFERENCE_SCRIPT, _TUNED_CONFIG_PATH):
            h.update(path.read_bytes())
    return h.hexdigest()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _find_example_video():
    if _EXAMPLE_DIR.exists():
        for p in sorted(_EXAMPLE_DIR.glob("*.mp4")):
            return p
    return None


def _reset_to_upload():
    st.session_state.update(
        result=None, slides_dir=None, video_path=None, error=None,
        error_details=None, pos=0, video_start=0, from_example=False, lecture_notes=None,
    )


def _select_visit(index: int, seek_to=None, visits=None):
    """Select a slide visit, mark it reviewed and seek the video."""
    st.session_state.pos = max(0, int(index))
    reviewed = st.session_state.setdefault("reviewed_visits", set())
    reviewed.add(st.session_state.pos)
    if seek_to is not None:
        st.session_state.video_start = int(seek_to)
    elif visits is not None and 0 <= st.session_state.pos < len(visits):
        st.session_state.video_start = int(visits[st.session_state.pos]["start_time"])


# ============================================================
# STUDY WORKSPACE HELPERS (retrieval strictly over the real timeline data)
# ============================================================

_QA_STOP = _STOPWORDS | {
    "explain", "tell", "mean", "means", "meaning", "lecture", "slide", "slides", "topic", "discuss",
    "discussed", "talk", "talked", "said", "say", "define", "definition", "describe", "give", "please",
    "show", "find", "mentioned", "covered", "cover", "simply", "simple", "professor", "lecturer", "time",
}


def _stem(word: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def _terms(text: str):
    return [_stem(w) for w in _WORD_RE.findall((text or "").lower()) if w not in _QA_STOP]


def _build_units(visits: list):
    """Searchable evidence units: real transcript sentences and slide-text lines."""
    units = []
    for vi, v in enumerate(visits):
        key_points = set(v["key_points"])
        for seg in v["segments"]:
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            t = float(seg.get("start_time") or 0)
            for sent in _SENTENCE_SPLIT_RE.split(txt):
                sent = sent.strip()
                if len(sent.split()) >= 4:
                    units.append({"visit": vi, "slide_id": v["slide_id"], "t": t, "text": sent,
                                  "kind": "Transcript · key point" if sent in key_points else "Transcript",
                                  "terms": set(_terms(sent))})
        for line in v["ocr_text"].splitlines():
            line = line.strip()
            if len(line.split()) >= 2:
                units.append({"visit": vi, "slide_id": v["slide_id"], "t": v["start_time"], "text": line,
                              "kind": "Slide text", "terms": set(_terms(line))})
    return units


def _retrieve(question: str, units: list, boost_visit=None, k: int = 3):
    q = set(_terms(question))
    if not q:
        return []
    df = Counter()
    for u in units:
        for t in u["terms"] & q:
            df[t] += 1
    n = max(len(units), 1)
    scored = []
    for u in units:
        m = u["terms"] & q
        if not m or len(m) / len(q) < 0.34:
            continue
        score = sum(math.log(1 + n / df[t]) for t in m) / (1 + 0.02 * len(u["terms"]))
        if boost_visit is not None and u["visit"] == boost_visit:
            score *= 1.3
        scored.append((score, u))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, u in scored:
        if u["text"] in seen:
            continue
        seen.add(u["text"])
        out.append(u)
        if len(out) >= k:
            break
    return out


def _intent(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(summar\w*|recap|overview|tl;?dr)\b", q) or (
        re.search(r"\b(this|current)\s+(slide|topic|part|section|segment)\b", q)
        and re.search(r"\b(explain|simple|simply|about|covers?|main|what)\b", q)
    ):
        return "summary"
    if re.search(r"\b(where|when|which slide|what time|timestamp|at what point)\b", q):
        return "locate"
    if re.search(r"\b(explain|simply|simple|eli5|what is|what's|what are|define|definition|meaning|means?|why|how)\b", q):
        return "explain"
    return "answer"


def _point_time(visit: dict, point: str) -> float:
    """Start time of the transcript segment that contains a key point."""
    probe = point[:40]
    for seg in visit["segments"]:
        if probe and probe in (seg.get("text") or ""):
            return float(seg.get("start_time") or visit["start_time"])
    return visit["start_time"]


_NOT_FOUND = ("I couldn't find that in this lecture's transcript or slide text, so I won't guess. "
              "Try different wording, a term from Key Concepts, or the search box above.")


def _compose_answer(question: str, visits: list, units: list, pos: int):
    intent = _intent(question)
    cur = visits[pos]
    if intent == "summary":
        pts = cur["key_points"]
        if not pts:
            return {"lead": f"There isn't enough transcript on Slide {cur['slide_id']} to summarise it.", "sources": []}
        return {
            "lead": f"Summary of Slide {cur['slide_id']} ({_clock(cur['start_time'])} – {_clock(cur['end_time'])}), "
                    "using only what was said there:",
            "sources": [{"visit": pos, "slide_id": cur["slide_id"], "t": _point_time(cur, p), "text": p,
                         "kind": "Key point"} for p in pts[:5]],
        }
    if intent == "locate":
        hits = _retrieve(question, units, k=14)
        picked, seen = [], set()
        for h in hits:
            if h["visit"] not in seen:
                seen.add(h["visit"])
                picked.append(h)
            if len(picked) == 4:
                break
        if not picked:
            return {"lead": _NOT_FOUND, "sources": []}
        picked.sort(key=lambda h: h["t"])
        return {"lead": "That comes up at these points in the lecture (earliest first):", "sources": picked}
    hits = _retrieve(question, units, boost_visit=pos, k=3)
    if not hits:
        return {"lead": _NOT_FOUND, "sources": []}
    lead = "Here's what the lecture says about it:" if intent == "explain" else "Based on the lecture:"
    return {"lead": lead, "sources": hits}


def _ask(question: str, hist_key: str, visits: list, units: list, pos: int):
    hist = st.session_state.setdefault(hist_key, [])
    hist.append({"q": question, "ans": _compose_answer(question, visits, units, pos)})
    del hist[:-30]


def _submit_question(input_key: str, hist_key: str, visits: list, units: list, pos: int):
    q = (st.session_state.get(input_key) or "").strip()
    if q:
        _ask(q, hist_key, visits, units, pos)


def _clear_history(hist_key: str):
    st.session_state[hist_key] = []


def _render_answer(ans: dict, prefix: str):
    parts = [f'<div class="ls-ans-lead">{_esc(ans["lead"])}</div>']
    for s in ans["sources"]:
        parts.append(
            f'<div class="ls-src"><span class="ls-ref">Slide {_esc(s["slide_id"])} · {_clock(s["t"])}</span>'
            f'<span class="ls-src-kind">{_esc(s["kind"])}</span><div>{_esc(s["text"])}</div></div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)
    srcs = ans["sources"]
    for row in range(0, len(srcs), 3):
        chunk = srcs[row:row + 3]
        cols = st.columns(3, gap="small")
        for ci, s in enumerate(chunk):
            cols[ci].button(f"▶ {_clock(s['t'])} · Slide {s['slide_id']}", key=f"{prefix}_{row + ci}", **_FULL,
                            on_click=_select_visit, args=(s["visit"],), kwargs={"seek_to": s["t"]})


def _render_history(hist_key: str, assistant: bool, newest_first: bool):
    hist = st.session_state.get(hist_key, [])
    order = list(range(len(hist)))
    if newest_first:
        order.reverse()
        order = order[:6]
    for i in order:
        m = hist[i]
        st.markdown(f'<div class="ls-q"><div>{_esc(m["q"])}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ls-a"><div class="ls-a-who">{"Lecture assistant" if assistant else "From the lecture"}</div>',
                    unsafe_allow_html=True)
        _render_answer(m["ans"], f"{hist_key}_{i}")
        st.markdown("</div>", unsafe_allow_html=True)


def _slide_number(value):
    """Return a comparable numeric slide id from common labels."""
    text = str(value or "").strip().lower()
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _lecture_note_sections(notes: dict):
    """Return normalized concept sections from the AI lecture-note source."""
    if not isinstance(notes, dict):
        return []
    sections = notes.get("sections", [])
    return sections if isinstance(sections, list) else []


def _section_matches_visit(section: dict, visit: dict):
    """Match a concept section to a slide using explicit source_slides metadata."""
    visit_num = _slide_number(visit.get("slide_id"))
    if visit_num is None:
        return False
    source_slides = section.get("source_slides", [])
    if not isinstance(source_slides, list):
        return False
    nums = {_slide_number(x) for x in source_slides}
    return visit_num in {x for x in nums if x is not None}


def _relevant_note_sections(cur: dict, visits: list, notes: dict | None = None):
    """Find concept sections explicitly grounded in the current aligned slide."""
    notes = notes if notes is not None else st.session_state.get("lecture_notes")
    sections = _lecture_note_sections(notes)
    return [sec for sec in sections if _section_matches_visit(sec, cur)]


def _ensure_lecture_notes_available(visits: list):
    """Use the existing Lecture Notes pipeline before downstream study tools."""
    notes = st.session_state.get("lecture_notes")
    if notes:
        return notes
    if not _notes_api_key():
        return None
    generated, err = _generate_lecture_notes(visits)
    if generated:
        st.session_state.lecture_notes = generated
        return generated
    st.session_state.lecture_notes_error = err
    return None


def _clean_items(values, limit=None):
    """Deduplicate short source-derived strings while preserving order."""
    out = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def _structured_concepts(cur: dict, visits: list):
    """Concept cards derived only from the structured Gemini Lecture Notes."""
    notes = st.session_state.get("lecture_notes")
    sections = _relevant_note_sections(cur, visits, notes)
    cards = []

    for sec in sections:
        definition_map = {}
        for d in sec.get("definitions", []) or []:
            if not isinstance(d, dict):
                continue
            term = str(d.get("term", "")).strip()
            definition = str(d.get("definition", "")).strip()
            easy = str(d.get("easy_explanation", "")).strip()
            if term:
                definition_map[term.lower()] = (term, definition, easy)

        concept_names = _clean_items(sec.get("important_concepts", []), limit=12)
        concept_names.extend(
            _clean_items([x[0] for x in definition_map.values()], limit=12)
        )
        concept_names = _clean_items(concept_names, limit=12)

        for term in concept_names:
            term_info = definition_map.get(term.lower())
            if term_info:
                label, definition, easy = term_info
                explanation = definition or easy
            else:
                label = term
                definition = ""
                easy = ""
                explanation = ""

            if not explanation:
                explanation = str(sec.get("basic_idea", "")).strip() or str(sec.get("explanation", "")).strip()

            source_slides = sec.get("source_slides", []) or [cur.get("slide_id")]
            source_num = _slide_number(source_slides[0])
            source_visit = next(
                (i for i, v in enumerate(visits) if _slide_number(v.get("slide_id")) == source_num),
                next((i for i, v in enumerate(visits) if v is cur), 0),
            )
            source_time = visits[source_visit].get("start_time", cur.get("start_time", 0)) if visits else cur.get("start_time", 0)

            cards.append({
                "term": label,
                "def": explanation,
                "definition": definition,
                "easy_explanation": easy,
                "slide_id": ", ".join(map(str, source_slides)) or cur.get("slide_id"),
                "t": source_time,
                "visit": source_visit,
                "section": sec,
            })

    deduped = []
    seen = set()
    for card in cards:
        key = card["term"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    return deduped


def _concept_cards(cur: dict, visits: list):
    """Compatibility wrapper: downstream UI now receives structured concepts."""
    return _structured_concepts(cur, visits)


def _grounded_terms(cur: dict, visits: list, limit=8):
    """Key Terms sourced from concept names and definitions, not frequency counts."""
    cards = _structured_concepts(cur, visits)
    return _clean_items([c["term"] for c in cards], limit=limit)


def _grounded_revision_points(cur: dict, visits: list, limit=5):
    """Quick revision from AI-generated concept notes only."""
    sections = _relevant_note_sections(cur, visits)
    values = []
    for sec in sections:
        values.extend(sec.get("quick_revision", []) or [])
    for sec in sections:
        values.extend(sec.get("exam_points", []) or [])
    for sec in sections:
        values.extend(sec.get("what_we_know", []) or [])
    return _clean_items(values, limit=limit)


def _grounded_exam_material(cur: dict, visits: list):
    """Structured exam material from concept notes, preserving source categories."""
    sections = _relevant_note_sections(cur, visits)
    definitions = []
    steps = []
    examples = []
    exam_points = []
    confusions = []
    formulas = []
    concepts = []

    for sec in sections:
        concepts.extend(sec.get("important_concepts", []) or [])
        definitions.extend(sec.get("definitions", []) or [])
        steps.extend(sec.get("steps", []) or [])
        examples.extend(sec.get("examples", []) or [])
        exam_points.extend(sec.get("exam_points", []) or [])
        confusions.extend(sec.get("common_confusions", []) or [])
        formulas.extend(sec.get("formulas", []) or [])

    return {
        "concepts": _clean_items(concepts, 10),
        "definitions": definitions,
        "steps": _clean_items(steps, 8),
        "examples": _clean_items(examples, 8),
        "exam_points": _clean_items(exam_points, 8),
        "common_confusions": _clean_items(confusions, 6),
        "formulas": _clean_items(formulas, 8),
    }


def _build_grounded_quiz(cur: dict, visits: list, max_questions=5):
    """Build MCQs from concept definitions with same-lecture semantic distractors."""
    current_cards = _structured_concepts(cur, visits)
    if not current_cards:
        return []

    all_cards = []
    notes = st.session_state.get("lecture_notes")
    for sec in _lecture_note_sections(notes):
        for d in sec.get("definitions", []) or []:
            if not isinstance(d, dict):
                continue
            term = str(d.get("term", "")).strip()
            definition = str(d.get("definition", "")).strip()
            easy = str(d.get("easy_explanation", "")).strip()
            answer = definition or easy
            if term and answer:
                all_cards.append({
                    "term": term,
                    "answer_text": answer,
                    "source_slides": sec.get("source_slides", []),
                })

    # If definitions are sparse, use concise concept explanations from the
    # current/nearby sections, still entirely source-derived.
    if len(all_cards) < 4:
        for c in current_cards:
            if c["def"]:
                all_cards.append({
                    "term": c["term"],
                    "answer_text": c["def"],
                    "source_slides": [c["slide_id"]],
                })

    dedup = []
    seen = set()
    for c in all_cards:
        key = (c["term"].lower(), c["answer_text"].lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    all_cards = dedup

    questions = []
    current_terms = {c["term"].lower() for c in current_cards}
    for target in all_cards:
        if target["term"].lower() not in current_terms:
            continue
        distractors = [
            c for c in all_cards
            if c["term"].lower() != target["term"].lower()
            and c["answer_text"].lower() != target["answer_text"].lower()
        ][:3]
        if len(distractors) < 3:
            continue

        options = [target["answer_text"]] + [x["answer_text"] for x in distractors]
        shift = len(questions) % len(options)
        options = options[shift:] + options[:shift]
        answer_index = options.index(target["answer_text"])

        source_slides = target.get("source_slides") or [cur.get("slide_id")]
        source_slide = source_slides[0] if source_slides else cur.get("slide_id")
        questions.append({
            "question": f"Which statement best describes {target['term']} according to the lecture?",
            "options": options,
            "answer": answer_index,
            "term": target["term"],
            "slide_id": source_slide,
            "t": cur.get("start_time", 0),
        })
        if len(questions) >= max_questions:
            break
    return questions


# ============================================================
# AI STUDY-NOTES GENERATION
# ============================================================

_NOTES_MODEL = os.environ.get(
    "LECTURESENSE_NOTES_MODEL",
    "gemini-3.5-flash-lite",
)
_NOTES_FALLBACK_MODEL = os.environ.get(
    "LECTURESENSE_NOTES_FALLBACK_MODEL",
    "gemini-3.5-flash-lite",
)
_NOTES_MAX_CHARS = 18000
_NOTES_CONTEXT_CHARS = 5000
_NOTES_RETRIES = 3
_NOTES_RETRY_DELAYS = (1.5, 3.0, 6.0)

_NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "heading": {"type": "string", "description": "Short topic title."},
        "concept": {"type": "string", "description": "Simple student-friendly explanation."},
        "main_idea": {"type": "string", "description": "Central teaching point from this segment."},
        "step_by_step": {"type": "array", "items": {"type": "string"}},
        "definitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "easy_explanation": {"type": "string"},
                },
                "required": ["term", "definition", "easy_explanation"],
            },
        },
        "examples": {"type": "array", "items": {"type": "string"}},
        "exam_points": {"type": "array", "items": {"type": "string"}},
        "common_confusions": {"type": "array", "items": {"type": "string"}},
        "quick_revision": {"type": "array", "items": {"type": "string"}},
        "formulas": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "heading", "concept", "main_idea", "step_by_step", "definitions",
        "examples", "exam_points", "common_confusions", "quick_revision", "formulas",
    ],
}

_NOTES_SYSTEM = """
You are an expert university study-note writer.

Transform lecture evidence into TEACHING-ORIENTED, EXAM-READY study notes.
Do not merely summarize the speaker. Reorganize the material so a student can
understand, learn, and revise it.

SOURCE RULES:
1. Use ONLY information supported by the supplied lecture evidence.
2. The current aligned segment and its slide OCR are the primary evidence.
3. Previous/next context is only for continuity, references, terminology, or
   relationships between nearby concepts. Do not import unrelated facts.
4. Never invent facts, definitions, examples, formulas, steps, comparisons, or
   exam claims.
5. Rewrite in your own words; do not copy long speaker sentences.
6. Remove filler, greetings, logistics, repetition, and hesitation.
7. Leave a section empty when the source does not support it.
8. Include formulas/notation only when explicitly present in the source.
9. "What to remember for exam" means important source-supported facts or
   procedures, not predictions of actual exam questions.
10. Preserve the lecturer's terminology where it is clear from the source.

OUTPUT RULES:
Return ONLY one JSON object matching the supplied schema.

CONTENT FORMAT:
- concept: simple student-friendly explanation.
- main_idea: what the lecturer is actually trying to explain here.
- step_by_step: ordered learning sequence and relationships between ideas.
- definitions: proper definition plus a clearly labeled easy explanation when supported.
- examples: examples actually given or demonstrated in the evidence.
- exam_points: high-value source-supported facts, properties, distinctions,
  formulas, or procedures.
- common_confusions: only source-supported distinctions/confusions; otherwise empty.
- quick_revision: 3–5 short lines for fast revision.
- formulas: only source-supported formulas/notation.
"""


def _notes_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return None


def _clip_note_text(value, limit):
    value = str(value or "").strip()
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def _build_note_context(visit: dict, visits=None, position=None):
    visits = visits or []
    if position is None and visits:
        try:
            position = visits.index(visit)
        except ValueError:
            position = None

    parts = [
        f"CURRENT ALIGNED SEGMENT — Slide {visit.get('slide_id')} "
        f"({_clock(visit.get('start_time'))} - {_clock(visit.get('end_time'))})",
        "",
        "CURRENT TRANSCRIPT:",
        _clip_note_text(visit.get("transcript_text"), _NOTES_MAX_CHARS) or "(none)",
        "",
        "CURRENT SLIDE OCR:",
        _clip_note_text(visit.get("ocr_text"), 5000) or "(none)",
    ]

    if visits and position is not None:
        if position > 0:
            prev_visit = visits[position - 1]
            parts.extend([
                "",
                f"PREVIOUS CONTEXT — Slide {prev_visit.get('slide_id')} "
                f"({_clock(prev_visit.get('start_time'))} - {_clock(prev_visit.get('end_time'))})",
                "Transcript excerpt:",
                _clip_note_text(prev_visit.get("transcript_text"), _NOTES_CONTEXT_CHARS),
                "Slide OCR excerpt:",
                _clip_note_text(prev_visit.get("ocr_text"), 1800) or "(none)",
            ])

        if position + 1 < len(visits):
            next_visit = visits[position + 1]
            parts.extend([
                "",
                f"NEXT CONTEXT — Slide {next_visit.get('slide_id')} "
                f"({_clock(next_visit.get('start_time'))} - {_clock(next_visit.get('end_time'))})",
                "Transcript excerpt:",
                _clip_note_text(next_visit.get("transcript_text"), _NOTES_CONTEXT_CHARS),
                "Slide OCR excerpt:",
                _clip_note_text(next_visit.get("ocr_text"), 1800) or "(none)",
            ])

    return "\n".join(parts)


def _generate_ai_notes(visit: dict, visits=None, position=None):
    key = _notes_api_key()
    if not key:
        return None, (
            "Gemini API key not found. Add GEMINI_API_KEY "
            "to .streamlit/secrets.toml and restart the app."
        )

    if not (visit.get("transcript_text") or visit.get("ocr_text")):
        return None, "There is no transcript or slide text for this segment."

    context = _build_note_context(visit, visits=visits, position=position)
    prompt = (
        "Create teaching-oriented study notes for the CURRENT ALIGNED SEGMENT.\n\n"
        "The current segment is the primary source. Use previous/next context only "
        "when it directly clarifies continuity.\n\n" + context
    )

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None, (
            "The `google-genai` package is not installed. "
            "Run `pip install -U google-genai` and restart the app."
        )

    def _clean_json(raw: str):
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    models_to_try = []
    for model_name in (_NOTES_MODEL, _NOTES_FALLBACK_MODEL):
        if model_name and model_name not in models_to_try:
            models_to_try.append(model_name)

    last_error = None

    for model_name in models_to_try:
        for attempt in range(_NOTES_RETRIES):
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_NOTES_SYSTEM,
                        temperature=0.25,
                        max_output_tokens=2200,
                        response_mime_type="application/json",
                        response_schema=_NOTES_SCHEMA,
                    ),
                )
                raw = _clean_json(getattr(response, "text", "") or "")
                if not raw:
                    raise RuntimeError("Gemini returned an empty response.")
                data = json.loads(raw)

                def _str_list(value):
                    if not isinstance(value, list):
                        return []
                    return [str(x).strip() for x in value if str(x).strip()]

                definitions = []
                for item in data.get("definitions", []) if isinstance(data.get("definitions", []), list) else []:
                    if not isinstance(item, dict):
                        continue
                    term = str(item.get("term", "")).strip()
                    definition = str(item.get("definition", "")).strip()
                    easy = str(item.get("easy_explanation", "")).strip()
                    if term and (definition or easy):
                        definitions.append({
                            "term": term,
                            "definition": definition,
                            "easy_explanation": easy,
                        })

                return {
                    "heading": str(data.get("heading", "")).strip(),
                    "concept": str(data.get("concept", "")).strip(),
                    "main_idea": str(data.get("main_idea", "")).strip(),
                    "step_by_step": _str_list(data.get("step_by_step")),
                    "definitions": definitions,
                    "examples": _str_list(data.get("examples")),
                    "exam_points": _str_list(data.get("exam_points")),
                    "common_confusions": _str_list(data.get("common_confusions")),
                    "quick_revision": _str_list(data.get("quick_revision")),
                    "formulas": _str_list(data.get("formulas")),
                }, None

            except json.JSONDecodeError as exc:
                last_error = f"Invalid JSON from {model_name}: {exc}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                message = str(exc).lower()
                retryable = any(
                    token in message
                    for token in (
                        "503", "unavailable", "high demand", "429",
                        "rate limit", "resource exhausted", "500",
                        "internal", "timeout", "deadline",
                    )
                )
                if not retryable or attempt >= _NOTES_RETRIES - 1:
                    break
                time.sleep(_NOTES_RETRY_DELAYS[min(attempt, len(_NOTES_RETRY_DELAYS) - 1)])

    return None, (
        "Gemini notes generation failed after retrying. "
        f"Last error: {last_error or 'unknown error'}. "
        "You can override the model with LECTURESENSE_NOTES_MODEL."
    )



# ============================================================
# LECTURE-LEVEL CONCEPT-WISE STUDY NOTES
# ============================================================

_LECTURE_NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "source_slides": {"type": "array", "items": {"type": "string"}},
                    "basic_idea": {"type": "string"},
                    "what_we_know": {"type": "array", "items": {"type": "string"}},
                    "important_concepts": {"type": "array", "items": {"type": "string"}},
                    "definitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "definition": {"type": "string"},
                                "easy_explanation": {"type": "string"},
                            },
                            "required": ["term", "definition", "easy_explanation"],
                        },
                    },
                    "explanation": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "examples": {"type": "array", "items": {"type": "string"}},
                    "exam_points": {"type": "array", "items": {"type": "string"}},
                    "common_confusions": {"type": "array", "items": {"type": "string"}},
                    "formulas": {"type": "array", "items": {"type": "string"}},
                    "quick_revision": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "topic", "source_slides", "basic_idea", "what_we_know",
                    "important_concepts", "definitions", "explanation", "steps",
                    "examples", "exam_points", "common_confusions", "formulas",
                    "quick_revision",
                ],
            },
        },
        "final_revision": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "overview", "sections", "final_revision"],
}

_LECTURE_NOTES_SYSTEM = """
You are an expert university lecturer and study-note writer.

Convert the supplied COMPLETE LECTURE EVIDENCE into organized, student-ready
study material. The output must read like a student's well-made class notes,
not like a transcript summary.

PRIMARY GOAL:
Turn many slide-aligned transcript segments into FEWER, CONCEPT-WISE sections.
A section may cover several slides when they explain one connected concept.
Do not make one section per slide unless the topic genuinely changes.

SOURCE GROUNDING RULES:
1. Use ONLY facts supported by the supplied lecture transcript and slide OCR.
2. The evidence is authoritative. Do not add outside knowledge.
3. Never invent definitions, examples, formulas, proofs, steps, comparisons,
   applications, or exam questions.
4. Remove greetings, logistics, filler, hesitation, repeated wording and
   conversational noise.
5. Rewrite in clear student language; do not copy long speaker sentences.
6. Preserve technical terminology used by the lecturer when supported.
7. If a requested section is not supported by evidence, return an empty list
   or a concise statement that the source does not provide it.
8. Formulas and notation must come only from the supplied OCR/transcript.
9. "Exam points" means important source-supported facts/procedures, not guesses
   about what an examiner will ask.
10. Keep the distinction between what was explicitly taught and what is merely
    implied. Prefer explicit evidence.

STRUCTURE RULES:
- title: concise lecture/topic title.
- overview: 3–6 sentences explaining the overall learning flow.
- sections: 3–12 connected conceptual sections, ordered as taught.
- topic: meaningful concept heading, not "Slide 4".
- source_slides: slide IDs supporting the section.
- basic_idea: plain-language explanation of the section.
- what_we_know: prior/result points explicitly used by the lecturer.
- important_concepts: central terms or relationships.
- definitions: proper definition + easy explanation when supported.
- explanation: teaching-oriented explanation in coherent paragraphs.
- steps: ordered construction/procedure/reasoning when the lecture provides one.
- examples: actual examples or demonstrations from the lecture.
- exam_points: compact, high-value revision facts from the source.
- common_confusions: only distinctions actually supported by the evidence.
- formulas: only source-supported equations or notation.
- quick_revision: 3–5 very short recall lines.
- final_revision: 5–10 high-value lines covering the entire lecture.

WRITING STYLE:
Use headings, short paragraphs, arrows, compact notation and clear sequencing.
Make the student understand "why", "what", and "how" rather than merely what
was said. Avoid generic motivational language.

Return ONLY JSON matching the supplied schema.
"""


def _build_full_lecture_evidence(visits: list, max_chars: int = 48000) -> str:
    """Create compact, slide-labelled evidence for lecture-level synthesis."""
    parts = []
    used = 0
    for i, v in enumerate(visits, start=1):
        transcript = _clip_note_text(v.get("transcript_text"), 2600)
        ocr = _clip_note_text(v.get("ocr_text"), 900)
        block = (
            f"\n=== ALIGNED UNIT {i} | SLIDE {v.get('slide_id')} | "
            f"{_clock(v.get('start_time'))} - {_clock(v.get('end_time'))} ===\n"
            f"TRANSCRIPT:\n{transcript or '(none)'}\n"
            f"SLIDE OCR:\n{ocr or '(none)'}\n"
        )
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 400:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)
    return "".join(parts).strip()


def _normalize_lecture_notes(data: dict) -> dict:
    """Keep the model output predictable and safe for the UI."""
    if not isinstance(data, dict):
        raise ValueError("Lecture notes response is not an object.")

    def sl(v):
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]

    sections = []
    raw_sections = data.get("sections", [])
    if isinstance(raw_sections, list):
        for raw in raw_sections:
            if not isinstance(raw, dict):
                continue
            defs = []
            for d in raw.get("definitions", []) if isinstance(raw.get("definitions", []), list) else []:
                if not isinstance(d, dict):
                    continue
                term = str(d.get("term", "")).strip()
                definition = str(d.get("definition", "")).strip()
                easy = str(d.get("easy_explanation", "")).strip()
                if term and (definition or easy):
                    defs.append({
                        "term": term,
                        "definition": definition,
                        "easy_explanation": easy,
                    })
            sections.append({
                "topic": str(raw.get("topic", "Untitled concept")).strip() or "Untitled concept",
                "source_slides": sl(raw.get("source_slides")),
                "basic_idea": str(raw.get("basic_idea", "")).strip(),
                "what_we_know": sl(raw.get("what_we_know")),
                "important_concepts": sl(raw.get("important_concepts")),
                "definitions": defs,
                "explanation": str(raw.get("explanation", "")).strip(),
                "steps": sl(raw.get("steps")),
                "examples": sl(raw.get("examples")),
                "exam_points": sl(raw.get("exam_points")),
                "common_confusions": sl(raw.get("common_confusions")),
                "formulas": sl(raw.get("formulas")),
                "quick_revision": sl(raw.get("quick_revision")),
            })

    return {
        "title": str(data.get("title", "Lecture Study Notes")).strip() or "Lecture Study Notes",
        "overview": str(data.get("overview", "")).strip(),
        "sections": sections,
        "final_revision": sl(data.get("final_revision")),
    }


def _generate_lecture_notes(visits: list):
    key = _notes_api_key()
    if not key:
        return None, (
            "Lecture notes need GEMINI_API_KEY. Add it to "
            ".streamlit/secrets.toml and restart the app."
        )
    evidence = _build_full_lecture_evidence(visits)
    if not evidence:
        return None, "There is not enough transcript or slide evidence to build lecture notes."

    prompt = (
        "Create COMPLETE, CONCEPT-WISE STUDY NOTES for this lecture.\n\n"
        "Use the evidence below as the only source. Group related slide content "
        "into coherent concepts and preserve the order in which the lecture teaches them.\n\n"
        + evidence
    )

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None, "The google-genai package is not installed. Run: pip install -U google-genai"

    def clean_json(raw: str):
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    models = []
    for name in (_NOTES_MODEL, _NOTES_FALLBACK_MODEL):
        if name and name not in models:
            models.append(name)

    last_error = None
    for model_name in models:
        for attempt in range(_NOTES_RETRIES):
            try:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_LECTURE_NOTES_SYSTEM,
                        temperature=0.20,
                        max_output_tokens=7000,
                        response_mime_type="application/json",
                        response_schema=_LECTURE_NOTES_SCHEMA,
                    ),
                )
                raw = clean_json(getattr(response, "text", "") or "")
                if not raw:
                    raise RuntimeError("Gemini returned an empty response.")
                return _normalize_lecture_notes(json.loads(raw)), None
            except json.JSONDecodeError as exc:
                last_error = f"Invalid JSON: {exc}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                msg = str(exc).lower()
                retryable = any(x in msg for x in (
                    "429", "rate limit", "resource exhausted", "503", "unavailable",
                    "500", "internal", "timeout", "deadline", "high demand",
                ))
                if not retryable or attempt >= _NOTES_RETRIES - 1:
                    break
                time.sleep(_NOTES_RETRY_DELAYS[min(attempt, len(_NOTES_RETRY_DELAYS) - 1)])

    return None, (
        "Lecture notes generation failed. "
        f"Last error: {last_error or 'unknown error'}"
    )


def _render_lecture_notes(notes: dict):
    if not notes:
        return

    st.markdown(
        '<div class="ls-ai-badge">✦ Lecture notes · concept-wise · grounded in transcript + slide evidence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ls-lecture-title">{_esc(notes.get("title", "Lecture Study Notes"))}</div>',
        unsafe_allow_html=True,
    )
    if notes.get("overview"):
        st.markdown(
            f'<div class="ls-teach-card"><div class="ls-teach-lead">{_esc(notes["overview"])}</div></div>',
            unsafe_allow_html=True,
        )

    for i, sec in enumerate(notes.get("sections", []), 1):
        st.markdown(
            f'<div class="ls-lecture-section"><span class="n">{i}</span>'
            f'<span>{_esc(sec.get("topic", "Concept"))}</span></div>',
            unsafe_allow_html=True,
        )
        if sec.get("source_slides"):
            st.markdown(
                f'<p class="ls-hint">Source slides: {_esc(", ".join(sec["source_slides"]))}</p>',
                unsafe_allow_html=True,
            )
        if sec.get("basic_idea"):
            st.markdown('<div class="ls-sec">Basic idea</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ls-teach-card">{_esc(sec["basic_idea"])}</div>', unsafe_allow_html=True)
        if sec.get("what_we_know"):
            st.markdown('<div class="ls-sec">What we already know</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-teach-bullet">✓ {_esc(x)}</div>' for x in sec["what_we_know"]),
                unsafe_allow_html=True,
            )
        if sec.get("important_concepts"):
            st.markdown('<div class="ls-sec">Important concept</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<span class="ls-chip">{_esc(x)}</span>' for x in sec["important_concepts"]),
                unsafe_allow_html=True,
            )
        if sec.get("definitions"):
            st.markdown('<div class="ls-sec">Definitions</div>', unsafe_allow_html=True)
            for d in sec["definitions"]:
                st.markdown(
                    f'<div class="ls-teach-def"><div class="term">{_esc(d["term"])}</div>'
                    f'<div class="definition">{_esc(d["definition"])}</div>'
                    + (f'<div class="easy"><b>Easy explanation:</b> {_esc(d["easy_explanation"])}</div>' if d.get("easy_explanation") else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )
        if sec.get("explanation"):
            st.markdown('<div class="ls-sec">Explanation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ls-teach-card">{_esc(sec["explanation"])}</div>', unsafe_allow_html=True)
        if sec.get("steps"):
            st.markdown('<div class="ls-sec">Construction / steps</div>', unsafe_allow_html=True)
            for j, step in enumerate(sec["steps"], 1):
                st.markdown(
                    f'<div class="ls-teach-step"><span class="num">{j}</span><span class="text">{_esc(step)}</span></div>',
                    unsafe_allow_html=True,
                )
        if sec.get("examples"):
            st.markdown('<div class="ls-sec">Example</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-teach-bullet">• {_esc(x)}</div>' for x in sec["examples"]),
                unsafe_allow_html=True,
            )
        if sec.get("exam_points"):
            st.markdown('<div class="ls-sec">Exam points</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-teach-bullet"><b>{j}.</b> {_esc(x)}</div>' for j, x in enumerate(sec["exam_points"], 1)),
                unsafe_allow_html=True,
            )
        if sec.get("common_confusions"):
            st.markdown('<div class="ls-sec">Common confusion</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-teach-bullet">• {_esc(x)}</div>' for x in sec["common_confusions"]),
                unsafe_allow_html=True,
            )
        if sec.get("formulas"):
            st.markdown('<div class="ls-sec">Formula / notation</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-formula">{_esc(x)}</div>' for x in sec["formulas"]),
                unsafe_allow_html=True,
            )
        if sec.get("quick_revision"):
            st.markdown('<div class="ls-sec">Quick revision</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<div class="ls-teach-bullet">→ {_esc(x)}</div>' for x in sec["quick_revision"]),
                unsafe_allow_html=True,
            )

    if notes.get("final_revision"):
        st.markdown('<div class="ls-lecture-section"><span class="n">✓</span><span>Final quick revision</span></div>', unsafe_allow_html=True)
        st.markdown(
            ''.join(f'<div class="ls-teach-bullet">→ {_esc(x)}</div>' for x in notes["final_revision"]),
            unsafe_allow_html=True,
        )


def _lecture_notes_text(notes: dict) -> str:
    lines = [notes.get("title", "Lecture Study Notes"), "=" * len(notes.get("title", "Lecture Study Notes")), ""]
    if notes.get("overview"):
        lines += ["OVERVIEW", notes["overview"], ""]
    for i, sec in enumerate(notes.get("sections", []), 1):
        lines += [f"{i}. {sec.get('topic', 'Concept')}", "-" * 50]
        if sec.get("source_slides"):
            lines.append("Source slides: " + ", ".join(sec["source_slides"]))
        for label, key in [
            ("Basic idea", "basic_idea"),
            ("Explanation", "explanation"),
        ]:
            if sec.get(key):
                lines += [f"\n{label}", sec[key]]
        for label, key in [
            ("What we already know", "what_we_know"),
            ("Important concepts", "important_concepts"),
            ("Steps", "steps"),
            ("Examples", "examples"),
            ("Exam points", "exam_points"),
            ("Common confusion", "common_confusions"),
            ("Formula / notation", "formulas"),
            ("Quick revision", "quick_revision"),
        ]:
            values = sec.get(key, [])
            if values:
                lines.append(f"\n{label}")
                lines += [f"- {x}" for x in values]
        if sec.get("definitions"):
            lines.append("\nDefinitions")
            for d in sec["definitions"]:
                lines.append(f"{d['term']}: {d['definition']}")
                if d.get("easy_explanation"):
                    lines.append(f"Easy explanation: {d['easy_explanation']}")
        lines.append("")
    if notes.get("final_revision"):
        lines += ["FINAL QUICK REVISION", "-" * 50]
        lines += [f"- {x}" for x in notes["final_revision"]]
    return "\n".join(lines)

def _render_ai_notes(n: dict):
    out = ['<div class="ls-ai-badge">✦ AI teaching notes · grounded in lecture evidence</div>']

    if n.get("heading"):
        out.append(
            f'<div class="ls-term" style="font-size:1.25rem;margin:.55rem 0 .8rem">'
            f'{_esc(n["heading"])}</div>'
        )

    sections = [
        ("1. What is the concept?", "concept"),
        ("2. Main idea", "main_idea"),
    ]
    for title, key in sections:
        if n.get(key):
            out.append(f'<div class="ls-teach-section">{title}</div>')
            out.append(f'<div class="ls-teach-card"><div class="ls-teach-lead">{_esc(n[key])}</div></div>')

    if n.get("step_by_step"):
        out.append('<div class="ls-teach-section">3. Step-by-step understanding</div><div class="ls-teach-card">')
        for i, item in enumerate(n["step_by_step"], 1):
            out.append(
                f'<div class="ls-teach-step"><span class="num">{i}</span>'
                f'<span class="text">{_esc(item)}</span></div>'
            )
        out.append('</div>')

    if n.get("definitions"):
        out.append('<div class="ls-teach-section">4. Important definitions</div><div class="ls-teach-card">')
        for item in n["definitions"]:
            out.append(
                '<div class="ls-teach-def">'
                f'<div class="term">{_esc(item.get("term", ""))}</div>'
                f'<div class="definition">{_esc(item.get("definition", ""))}</div>'
            )
            if item.get("easy_explanation"):
                out.append(
                    f'<div class="easy"><b>Easy explanation:</b> '
                    f'{_esc(item["easy_explanation"])}</div>'
                )
            out.append('</div>')
        out.append('</div>')

    if n.get("examples"):
        out.append('<div class="ls-teach-section">5. Example</div><div class="ls-teach-card">')
        for item in n["examples"]:
            out.append(f'<div class="ls-teach-bullet">• {_esc(item)}</div>')
        out.append('</div>')

    if n.get("exam_points"):
        out.append('<div class="ls-teach-section">6. What to remember for exam</div><div class="ls-teach-card">')
        for i, item in enumerate(n["exam_points"], 1):
            out.append(f'<div class="ls-teach-bullet"><b>{i}.</b> {_esc(item)}</div>')
        out.append('</div>')

    if n.get("common_confusions"):
        out.append('<div class="ls-teach-section">7. Common confusion</div><div class="ls-teach-card">')
        for item in n["common_confusions"]:
            out.append(f'<div class="ls-teach-bullet">• {_esc(item)}</div>')
        out.append('</div>')

    if n.get("quick_revision"):
        out.append('<div class="ls-teach-section">8. Quick revision</div><div class="ls-teach-revision">')
        for item in n["quick_revision"]:
            out.append(f'<div class="row">• {_esc(item)}</div>')
        out.append('</div>')

    if n.get("formulas"):
        out.append('<div class="ls-teach-section">Formulas &amp; notation from the lecture</div><div class="ls-teach-card">')
        for item in n["formulas"]:
            out.append(f'<div class="ls-formula">{_esc(item)}</div>')
        out.append('</div>')

    st.markdown("".join(out), unsafe_allow_html=True)


# ============================================================
# PIPELINE: faster-whisper  ->  V4 inference
# ============================================================

@st.cache_resource(show_spinner=False)

def _load_whisper_pipeline():
    """
    Load faster-whisper only once and reuse it across Streamlit reruns.
    GPU FP16 is preferred; CPU INT8 is the fallback.
    """
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    last_error = None

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------
    if _WHISPER_DEVICE in ("auto", "cuda"):
        try:
            model = WhisperModel(
                _WHISPER_MODEL_SIZE,
                device="cuda",
                compute_type=_WHISPER_GPU_COMPUTE_TYPE,
            )

            pipeline = BatchedInferencePipeline(model=model)

            return pipeline, "cuda"

        except Exception as exc:
            last_error = exc

    # --------------------------------------------------------
    # CPU fallback
    # --------------------------------------------------------
    try:
        model = WhisperModel(
            _WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type=_WHISPER_CPU_COMPUTE_TYPE,
        )

        pipeline = BatchedInferencePipeline(model=model)

        return pipeline, "cpu"

    except Exception:
        if last_error is not None:
            raise last_error
        raise


def _transcribe_video(video_path: Path, out_srt_path: Path, on_progress=None):
    """Generate MaViLS-compatible word-level transcript."""
    try:
        pipeline, _device = _load_whisper_pipeline()
    except Exception as exc:
        st.session_state.error_details = f"Whisper load error: {exc!r}"
        return False, "unavailable"

    try:
        segments, info = pipeline.transcribe(
            str(video_path),
            batch_size=_WHISPER_BATCH_SIZE,
            beam_size=_WHISPER_BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": _WHISPER_VAD_MIN_SILENCE_MS
            },
            condition_on_previous_text=False,
        )

        total = float(getattr(info, "duration", 0) or 0)
        out_srt_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        word_count = 0

        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                word_text = (getattr(word, "word", "") or "").strip()
                start = getattr(word, "start", None)
                end = getattr(word, "end", None)

                if not word_text or start is None or end is None:
                    continue

                lines.append(
                    f"[{_format_mavils_timestamp(start)} --> "
                    f"{_format_mavils_timestamp(end)}] {word_text}"
                )
                word_count += 1

            if on_progress:
                done = float(getattr(segment, "end", 0) or 0)
                fraction = (
                    min(done / total, 1.0)
                    if total > 0
                    else None
                )
                on_progress(fraction, done, total)

        if word_count == 0:
            return False, "empty"

        out_srt_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        return True, None

    except Exception as exc:
        st.session_state.error_details = f"Transcription error: {exc!r}"
        return False, "failed"


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""



def _run_inference(
    transcript_path: Path,
    slides_path: Path,
    video_path: Path,
    out_path: Path,
    render_dir: Path,
    on_tick=None,
):
    """Run V11 inference and generate the slide timeline."""

    if not _INFERENCE_SCRIPT.exists():
        return False, "unavailable"

    if not transcript_path.exists():
        return False, "failed"

    if not slides_path.exists() or not video_path.exists():
        return False, "failed"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_INFERENCE_SCRIPT),
            "--transcript", str(transcript_path),
            "--slides", str(slides_path),
            "--video", str(video_path),
            "--out", str(out_path),
            "--render-dir", str(render_dir),
        ]

        started = time.monotonic()

        with _LOG_PATH.open(
            "w", encoding="utf-8", errors="replace"
        ) as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(_BASE_DIR),
            )

            while process.poll() is None:
                elapsed = time.monotonic() - started

                if on_tick is not None:
                    on_tick(elapsed)

                if elapsed >= _INFERENCE_TIMEOUT_S:
                    process.kill()
                    process.wait()
                    st.session_state.error_details = (
                        "V11 inference timed out.\n" + _tail(_LOG_PATH, 60)
                    )
                    return False, "timeout"

                time.sleep(1)

            return_code = process.returncode

        if return_code != 0:
            details = _tail(_LOG_PATH, 80)
            st.session_state.error_details = (
                "V11 inference failed.\n" + details
            )
            print(st.session_state.error_details)
            return False, "failed"

        if not out_path.exists():
            st.session_state.error_details = (
                "Inference finished without generating timeline.json.\n"
                + _tail(_LOG_PATH, 60)
            )
            return False, "failed"

        return True, None

    except subprocess.TimeoutExpired:
        return False, "timeout"

    except Exception as exc:
        st.session_state.error_details = (
            f"V11 inference error: {exc!r}\n" + _tail(_LOG_PATH, 40)
        )
        print(st.session_state.error_details)
        return False, "failed"



# ============================================================
# PROCESSING UI
# ============================================================

_STEPS = [
    ("Saving your files", "Preparing the slide deck and recording"),
    ("Transcribing the recording", "Word-level timestamps with faster-whisper"),
    ("Aligning transcript to slides", "V4.2 validation-tuned fusion + boundary-aware alignment"),
    ("Building your workspace", "Notes, timeline and slide previews"),
]


def _render_processing(ph, active: int, meta: str = "", fraction=None):
    rows = []
    for i, (title, _sub) in enumerate(_STEPS):
        cls = "done" if i < active else ("active" if i == active else "")
        icon = "✓" if i < active else str(i + 1)
        m = f'<span class="meta">{_esc(meta)}</span>' if (i == active and meta) else ""
        rows.append(f'<div class="ls-pstep {cls}"><span class="ic">{icon}</span>{_esc(title)}{m}</div>')
    if fraction is None:
        bar = '<div class="ls-bar indet"><div></div></div>'
    else:
        bar = f'<div class="ls-bar"><div style="width:{int(fraction * 100)}%"></div></div>'
    ph.markdown(
        f'<div class="ls-proc"><h3>Building your study workspace</h3>'
        f'<div class="sub">{_esc(_STEPS[min(active, len(_STEPS) - 1)][1])}. '
        f'Keep this tab open — long lectures can take several minutes.</div>'
        f'{"".join(rows)}{bar}</div>',
        unsafe_allow_html=True,
    )


def _process_upload(pdf_file, video_file):
    """Full pipeline for a fresh upload. Sets session state; caller reruns."""
    st.session_state.error = None
    st.session_state.error_details = None

    pdf_bytes = bytes(pdf_file.getbuffer())
    video_bytes = bytes(video_file.getbuffer())

    if not pdf_bytes.startswith(b"%PDF"):
        st.session_state.error = "That slide deck doesn't look like a valid PDF. Please export your slides as a PDF and try again."
        return
    if len(video_bytes) < 1024:
        st.session_state.error = "The lecture recording looks empty or corrupted. Please choose a different MP4 file."
        return

    fingerprint = _pair_fingerprint(pdf_bytes, video_bytes)
    fp_path = _UPLOAD_DIR / "fingerprint.txt"
    video_path = _UPLOAD_DIR / _VIDEO_NAME

    if (fp_path.exists() and fp_path.read_text(encoding="utf-8").strip() == fingerprint
            and _TIMELINE_PATH.exists() and video_path.exists()):
        data, err = _load_timeline_json(_TIMELINE_PATH)
        if err:
            st.session_state.error = err
        else:
            st.session_state.update(result=data, slides_dir=_SLIDES_DIR, video_path=video_path,
                                    pos=0, video_start=0, from_example=False)
        return

    ph = st.empty()
    _render_processing(ph, 0)
    try:
        if _RUN_DIR.exists():
            shutil.rmtree(_RUN_DIR, ignore_errors=True)
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = _UPLOAD_DIR / "slides.pdf"
        pdf_path.write_bytes(pdf_bytes)
        video_path.write_bytes(video_bytes)
    except Exception as exc:
        st.session_state.error = "We couldn't save your files. Please check free disk space and try again."
        st.session_state.error_details = repr(exc)
        ph.empty()
        return

    # ---- Step 2: transcription ----
    _render_processing(ph, 1, "starting…")
    t0 = time.time()

    def _on_tp(fraction, done, total):
        meta = f"{_clock(done)} / {_clock(total)}" if total else _clock(done)
        _render_processing(ph, 1, meta, fraction)

    srt_path = _UPLOAD_DIR / "transcript.srt"
    ok, reason = _transcribe_video(video_path, srt_path, on_progress=_on_tp)
    if not ok:
        st.session_state.error = {
            "unavailable": "Transcription isn't available right now. Please check that faster-whisper is installed and try again.",
            "empty": "We couldn't detect any speech in this recording. Please check the file and try again.",
            "failed": "We couldn't transcribe this recording. Check that it's a valid MP4 and try again.",
        }.get(reason)
        ph.empty()
        return

    # ---- Step 3: alignment ----
    def _on_tick(elapsed):
        _render_processing(ph, 2, f"{_clock(elapsed)} elapsed")

    _render_processing(ph, 2, "starting…")
    ok, reason = _run_inference(srt_path, pdf_path, video_path, _TIMELINE_PATH, _SLIDES_DIR, on_tick=_on_tick)
    if not ok:
        st.session_state.error = {
            "unavailable": "The alignment model isn't available right now. Please check the installation and try again.",
            "timeout": "This lecture is taking longer than expected. Try again with a shorter recording.",
            "failed": "We couldn't match this recording to the slides. Check that both files are from the same lecture and try again.",
        }.get(reason)
        ph.empty()
        return

    # ---- Step 4: load results ----
    _render_processing(ph, 3)
    data, err = _load_timeline_json(_TIMELINE_PATH)
    if err:
        st.session_state.error = err
        ph.empty()
        return
    try:
        fp_path.write_text(fingerprint, encoding="utf-8")
    except Exception:
        pass
    st.session_state.update(result=data, slides_dir=_SLIDES_DIR, video_path=video_path,
                            pos=0, video_start=0, from_example=False)
    ph.empty()


# ============================================================
# BRAND BAR
# ============================================================

st.markdown(
    '<div class="ls-brand"><div class="ls-logo"><span class="glyph">◆</span>LECTURESENSE</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# UPLOAD VIEW
# ============================================================

if st.session_state.result is None:
    st.markdown(
        """
        <div class="ls-hero">
            <h1>Turn any lecture into<br><span>study-ready notes.</span></h1>
            <p>Upload your slides and the lecture recording. LectureSense aligns what was said to the
            slide it belongs to, so you can jump straight to any moment, read the notes, and review
            the transcript — slide by slide.</p>
        </div>
        <div class="ls-features">
            <span class="ls-feature">▶ Synced video player</span>
            <span class="ls-feature">⏱ Clickable slide timeline</span>
            <span class="ls-feature">✎ Notes &amp; key concepts</span>
            <span class="ls-feature">⌕ Searchable transcript</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        pdf_done = st.session_state.get("pdf_upload") is not None
        st.markdown(
            f'<div class="ls-step{" done" if pdf_done else ""}"><span class="num">{"✓" if pdf_done else "1"}</span>Slide deck</div>'
            '<div class="ls-step-sub">The PDF export of your lecture slides</div>',
            unsafe_allow_html=True,
        )
        pdf_file = st.file_uploader("Slide deck", type=["pdf"], key="pdf_upload", label_visibility="collapsed")
        if pdf_file:
            st.markdown(f'<div class="ls-filemeta">{_esc(pdf_file.name)} · {_human_size(pdf_file.size)}</div>', unsafe_allow_html=True)
    with col2:
        video_done = st.session_state.get("video_upload") is not None
        st.markdown(
            f'<div class="ls-step{" done" if video_done else ""}"><span class="num">{"✓" if video_done else "2"}</span>Lecture recording</div>'
            '<div class="ls-step-sub">The MP4 video — a timestamped transcript is generated automatically</div>',
            unsafe_allow_html=True,
        )
        video_file = st.file_uploader("Lecture recording", type=["mp4"], key="video_upload", label_visibility="collapsed")
        if video_file:
            st.markdown(f'<div class="ls-filemeta">{_esc(video_file.name)} · {_human_size(video_file.size)}</div>', unsafe_allow_html=True)

    both_ready = bool(pdf_file and video_file)
    example_available = _EXAMPLE_TIMELINE_PATH.exists()

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.4, 1.1, 2.5], gap="small")
    with c1:
        generate_clicked = st.button(
            "Generate study workspace" if both_ready else "Add both files to continue",
            type="primary", disabled=not both_ready, **_FULL,
        )
    with c2:
        example_clicked = st.button("Open saved example", **_FULL) if example_available else False
    with c3:
        st.markdown(
            '<p class="ls-hint">Both files are in. Transcription runs first, then slide alignment.</p>'
            if both_ready else
            '<p class="ls-hint">Upload a PDF and an MP4 from the same lecture.</p>',
            unsafe_allow_html=True,
        )

    if generate_clicked and both_ready:
        _process_upload(pdf_file, video_file)
        st.rerun()

    if example_clicked:
        st.session_state.error = None
        data, err = _load_timeline_json(_EXAMPLE_TIMELINE_PATH)
        if err:
            st.session_state.error = "The saved example isn't available right now."
        else:
            st.session_state.update(result=data, slides_dir=_EXAMPLE_SLIDES_DIR,
                                    video_path=_find_example_video(), pos=0,
                                    video_start=0, from_example=True)
        st.rerun()

    if st.session_state.error:
        st.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)
        st.error(st.session_state.error)
        if st.session_state.error_details:
            with st.expander("Technical details"):
                st.code(st.session_state.error_details, language="text")

# ============================================================
# STUDY WORKSPACE
# ============================================================

else:
    data = st.session_state.result
    slides_dir = st.session_state.slides_dir
    video_path = st.session_state.video_path
    visits = _build_visits(data.get("timeline", []))

    head_l, head_r = st.columns([5, 1.3])
    with head_l:
        st.markdown('<span class="ls-pill"><span class="dot"></span>Study workspace ready</span>', unsafe_allow_html=True)
    with head_r:
        if st.button("New lecture", **_FULL):
            _reset_to_upload()
            st.rerun()

    if not visits:
        st.info("Nothing to display for this lecture yet.")
        st.stop()

    total = len(visits)
    pos = max(0, min(int(st.session_state.pos), total - 1))
    st.session_state.pos = pos
    cur = visits[pos]
    lecture_end = max(v["end_time"] for v in visits)
    unique_slides = len({v["slide_id"] for v in visits})

    # ---- Stats ----
    st.markdown(
        f'<div class="ls-stats">'
        f'<div class="ls-stat"><b>{unique_slides}</b><span>Slides matched</span></div>'
        f'<div class="ls-stat"><b>{total}</b><span>Timeline segments</span></div>'
        f'<div class="ls-stat"><b>{_clock(lecture_end)}</b><span>Lecture length</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Proportional timeline strip ----
    span = max(lecture_end, 1e-6)
    cells = []
    for i, v in enumerate(visits):
        width = max((v["end_time"] - v["start_time"]) / span * 100, 0.4)
        cls = "on" if i == pos else ("seen" if i < pos else "")
        cells.append(f'<div class="{cls}" style="flex:{width:.3f}" title="Slide {_esc(v["slide_id"])} · {_clock(v["start_time"])}"></div>')
    st.markdown(
        f'<div class="ls-strip">{"".join(cells)}</div>'
        f'<div class="ls-strip-label"><span>Segment {pos + 1} of {total} · Slide {_esc(cur["slide_id"])}</span>'
        f'<span>{_clock(cur["start_time"])} / {_clock(lecture_end)}</span></div>',
        unsafe_allow_html=True,
    )

    # ---- Row 1: video + slide preview ----
    vid_col, slide_col = st.columns(2, gap="medium")

    with vid_col:
        st.markdown('<div class="ls-label">Lecture video</div>', unsafe_allow_html=True)
        if video_path and Path(video_path).exists():
            start = int(st.session_state.video_start)
            try:
                st.video(str(video_path), start_time=start)
            except TypeError:
                st.video(str(video_path))
            st.markdown(
                f'<p class="ls-hint">Starts at {_clock(start)}. Pick a slide or timestamp below to jump the video there.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ls-empty">No video is attached to this workspace.</div>', unsafe_allow_html=True)

    with slide_col:
        st.markdown('<div class="ls-label">Slide preview</div>', unsafe_allow_html=True)
        image_path = _find_slide_image(slides_dir, cur["slide_index"], cur["slide_title"])
        if image_path and image_path.exists():
            st.markdown('<div class="ls-slide-frame">', unsafe_allow_html=True)
            st.image(str(image_path), **_FULL)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="ls-empty">No slide preview available for this segment.</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ls-slide-title">Slide {_esc(cur["slide_id"])}</div>'
            f'<div class="ls-slide-time">{_clock(cur["start_time"])} – {_clock(cur["end_time"])}</div>',
            unsafe_allow_html=True,
        )
        n1, n2 = st.columns(2)
        n1.button("← Previous", disabled=pos == 0, **_FULL,
                  on_click=_select_visit, args=(pos - 1,), kwargs={"visits": visits}, key="nav_prev")
        n2.button("Next →", disabled=pos == total - 1, **_FULL,
                  on_click=_select_visit, args=(pos + 1,), kwargs={"visits": visits}, key="nav_next")

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

    # ---- Row 2: timeline rail + study tabs ----
    rail_col, main_col = st.columns([1, 2.4], gap="medium")

    with rail_col:
        st.markdown('<div class="ls-label">Slide timeline</div>', unsafe_allow_html=True)

        jump = st.text_input("Jump to time", placeholder="Jump to time (mm:ss or hh:mm:ss)", key="jump_time")
        if jump.strip():
            target = _parse_clock(jump)
            if target is None:
                st.markdown('<p class="ls-hint">Use mm:ss, e.g. 12:30</p>', unsafe_allow_html=True)
            else:
                idx = next((i for i, v in enumerate(visits) if v["start_time"] <= target <= v["end_time"]), None)
                if idx is None:
                    idx = min(range(total), key=lambda i: abs(visits[i]["start_time"] - target))
                st.button(f"Go to {_clock(target)} → Slide {visits[idx]['slide_id']}", **_FULL,
                          on_click=_select_visit, args=(idx,), kwargs={"seek_to": target}, key="jump_go")

        with st.container(height=520):
            for i, v in enumerate(visits):
                active = i == pos
                st.button(
                    f"{'▶ ' if active else ''}{_clock(v['start_time'])}  ·  Slide {v['slide_id']}",
                    key=f"rail_{i}", **_FULL,
                    type="primary" if active else "secondary",
                    on_click=_select_visit, args=(i,), kwargs={"visits": visits},
                )

    with main_col:
        search_query = st.text_input("Search", placeholder="Search notes, keywords, or transcript…", key="search_q")
        q = search_query.strip().lower()
        if q:
            hits = [i for i, v in enumerate(visits)
                    if q in " ".join([v["transcript_text"], v["ocr_text"], " ".join(v["keywords"])]).lower()]
            if hits:
                st.markdown(f'<p class="ls-hint">{len(hits)} segment(s) match “{_esc(search_query.strip())}”</p>', unsafe_allow_html=True)
                for i in hits[:15]:
                    v = visits[i]
                    source = v["transcript_text"] or v["ocr_text"]
                    k = source.lower().find(q)
                    if k >= 0:
                        a = max(0, k - 50)
                        snippet = ("…" if a else "") + source[a:k + 70] + "…"
                    else:
                        snippet = source[:110]
                    st.button(f"{_clock(v['start_time'])} · Slide {v['slide_id']} — {snippet}",
                              key=f"hit_{i}", **_FULL,
                              on_click=_select_visit, args=(i,), kwargs={"visits": visits})
                if len(hits) > 15:
                    st.markdown(f'<p class="ls-hint">Showing the first 15 of {len(hits)}.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="ls-hint">No matches for “{_esc(search_query.strip())}”.</p>', unsafe_allow_html=True)

        # Per-lecture evidence index + conversation state (reset when a different lecture is loaded)
        units = _build_units(visits)
        sig = hashlib.md5(
            f"{total}|{lecture_end:.2f}|{visits[0]['transcript_text'][:200]}".encode("utf-8")
        ).hexdigest()
        if st.session_state.get("study_sig") != sig:
            st.session_state.update(
                study_sig=sig,
                lecture_notes=None,
                qa_history=[],
                chat_history=[],
                ai_notes={},
                reviewed_visits={0},
                quiz_items=[],
                quiz_answers={},
                quiz_score=0,
                quiz_submitted=False,
                flash_index=0,
            )

        # --------------------------------------------------------
        # KEY CONCEPTS + IMPORTANT POINTS (always visible)
        # --------------------------------------------------------
        # Lecture Notes remain the single structured source for all downstream
        # study tools. The alignment pipeline and workspace architecture remain
        # unchanged.
        _ensure_lecture_notes_available(visits)
        concepts_now = _concept_cards(cur, visits)
        kp_col, pc_col = st.columns([1.35, 1], gap="medium")
        with kp_col:
            st.markdown('<div class="ls-label">Key Concepts</div>', unsafe_allow_html=True)
            chips = concepts_now[:6]
            if chips:
                html_parts = []
                for c in chips:
                    html_parts.append(
                        f'<div class="ls-concept" style="margin-bottom:.45rem">'
                        f'<div class="ls-term">{_esc(c["term"])}</div>'
                        f'<div class="ls-def">{_esc(c["def"]) or "Mentioned in this lecture segment."}</div>'
                        f'</div>'
                    )
                st.markdown("".join(html_parts), unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">Generate Lecture Notes to populate concept-grounded study tools.</p>', unsafe_allow_html=True)
        with pc_col:
            st.markdown('<div class="ls-label">Important Points</div>', unsafe_allow_html=True)
            if cur["key_points"]:
                st.markdown(
                    "".join(
                        f'<div class="ls-take"><span class="n">{i}</span><span class="tx">{_esc(p)}</span>'
                        f'<span class="tm">{_clock(_point_time(cur, p))}</span></div>'
                        for i, p in enumerate(cur["key_points"][:5], 1)
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<p class="ls-hint">No distinct key points were found here.</p>', unsafe_allow_html=True)

        st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)

        tabs = st.tabs([
            "Lecture Notes", "Study Notes", "Quick Revision", "Flashcards", "Quiz Me",
            "Exam Prep", "AI Tutor", "Weak Topics", "Progress"
        ])

        # ---- Lecture Notes: concept-wise primary notes ----
        with tabs[0]:
            st.markdown('<div class="ls-label">Concept-wise study material</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="ls-hint">Related slides are merged into concepts. The content is rewritten from the aligned transcript and slide OCR; unsupported facts are left out.</p>',
                unsafe_allow_html=True,
            )

            lecture_notes = st.session_state.get("lecture_notes")
            has_key = bool(_notes_api_key())
            b1, b2 = st.columns([1.7, 1], gap="small")
            with b1:
                generate_lecture = st.button(
                    "Regenerate complete lecture notes" if lecture_notes else "Generate complete lecture notes",
                    key="generate_lecture_notes", type="primary", disabled=not has_key, **_FULL
                )
            with b2:
                if lecture_notes:
                    st.download_button(
                        "Download study notes (.txt)",
                        data=_lecture_notes_text(lecture_notes),
                        file_name="LectureSense_Study_Notes.txt",
                        mime="text/plain",
                        **_FULL,
                    )

            if lecture_notes is None and has_key:
                with st.spinner("Understanding the lecture and organizing concepts…"):
                    generated, err = _generate_lecture_notes(visits)
                if generated:
                    st.session_state.lecture_notes = generated
                    lecture_notes = generated
                else:
                    st.error(err)

            if generate_lecture:
                with st.spinner("Regenerating concept-wise study notes…"):
                    generated, err = _generate_lecture_notes(visits)
                if generated:
                    st.session_state.lecture_notes = generated
                    lecture_notes = generated
                else:
                    st.error(err)

            if not has_key:
                st.markdown(
                    '<p class="ls-hint">Complete lecture notes need GEMINI_API_KEY in Streamlit secrets.</p>',
                    unsafe_allow_html=True,
                )

            if lecture_notes:
                _render_lecture_notes(lecture_notes)
            else:
                st.markdown(
                    '<div class="ls-teach-card"><b>What this will produce</b><br>'
                    'Topic → Basic idea → What we know → Important concepts → Definitions → '
                    'Explanation → Steps / proof → Examples → Exam points → Quick revision.</div>',
                    unsafe_allow_html=True,
                )

        # ---- Slide Evidence / Study Notes ----
        with tabs[1]:
            st.markdown(
                f'<div class="ls-note-head"><h3>Slide {_esc(cur["slide_id"])}</h3>'
                f'<span>{_clock(cur["start_time"])} – {_clock(cur["end_time"])} · '
                f'{_clock(cur["end_time"] - cur["start_time"])} discussed</span></div>',
                unsafe_allow_html=True,
            )
            sc1, sc2 = st.columns([3, 1], gap="small")
            with sc1:
                st.markdown(
                    f'<div class="ls-source-card"><b>Source: Slide {_esc(cur["slide_id"])}</b> · '
                    f'Timestamp: {_clock(cur["start_time"])} · Evidence from the aligned lecture segment.</div>',
                    unsafe_allow_html=True,
                )
            with sc2:
                st.button("▶ Watch explanation", key="notes_watch", **_FULL,
                          on_click=_select_visit, args=(pos,), kwargs={"seek_to": cur["start_time"], "visits": visits})
            ai_store = st.session_state.setdefault("ai_notes", {})
            has_key = bool(_notes_api_key())
            g1, g2 = st.columns(2, gap="small")
            gen_one = g1.button("✦ Regenerate teaching notes" if pos in ai_store else "✦ Generate teaching notes for this slide",
                                key="ai_gen_one", type="primary", disabled=not has_key, **_FULL)
            gen_all = g2.button("Generate notes for all slides", key="ai_gen_all", disabled=not has_key, **_FULL)

            if gen_one:
                with st.spinner("Writing study notes…"):
                    notes, err = _generate_ai_notes(cur, visits=visits, position=pos)
                if notes:
                    ai_store[pos] = notes
                else:
                    st.error(err)
            if gen_all:
                bar = st.progress(0.0, text="Writing study notes…")
                failed = 0
                for i, v in enumerate(visits):
                    if i not in ai_store:
                        notes, err = _generate_ai_notes(v, visits=visits, position=i)
                        if notes:
                            ai_store[i] = notes
                        else:
                            failed += 1
                    bar.progress(
                        (i + 1) / max(total, 1),
                        text=f"Writing study notes… {i + 1}/{total}",
                    )
                    if i < total - 1:
                        time.sleep(0.35)
                bar.empty()
                if failed:
                    st.warning(
                        f"{failed} slide(s) couldn't be generated. "
                        "Try generating the missing slide(s) again after a short wait."
                    )
            if not has_key:
                st.markdown(
                '<p class="ls-hint">AI notes need a Gemini API key. '
                'Set <code>GEMINI_API_KEY</code> in Streamlit secrets and restart the app.</p>',
        unsafe_allow_html=True,
    )

            if pos in ai_store:
                _render_ai_notes(ai_store[pos])

            def _extracted_notes():
                if cur["key_points"]:
                    st.markdown(
                        "".join(
                            f'<div class="ls-take"><span class="n">{n}</span><span class="tx">{_esc(pt)}</span>'
                            f'<span class="tm">{_clock(_point_time(cur, pt))}</span></div>'
                            for n, pt in enumerate(cur["key_points"], start=1)
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<p class="ls-hint">No distinct key points were found here — see the transcript.</p>',
                                unsafe_allow_html=True)
                if cur["ocr_text"]:
                    st.markdown('<div class="ls-sec">Text on the slide</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="ls-block">{_esc(cur["ocr_text"])}</div>', unsafe_allow_html=True)
                grounded_terms = _grounded_terms(cur, visits)
                if grounded_terms:
                    st.markdown('<div class="ls-sec">Key terms</div>', unsafe_allow_html=True)
                    st.markdown("".join(f'<span class="ls-chip">{_esc(k)}</span>' for k in grounded_terms),
                                unsafe_allow_html=True)

            if pos in ai_store:
                with st.expander("Speaker's key sentences (extracted, verbatim)"):
                    _extracted_notes()
            else:
                st.markdown('<div class="ls-sec">Transcript Evidence</div>', unsafe_allow_html=True)
                _extracted_notes()

        # --------------------------------------------------------
        # Quick Revision — generated from structured lecture concepts
        # --------------------------------------------------------
        with tabs[2]:
            st.markdown('<div class="ls-label">60-second revision</div>', unsafe_allow_html=True)
            summary_points = _grounded_revision_points(cur, visits, limit=5)
            if summary_points:
                st.markdown(
                    "".join(
                        f'<div class="ls-take"><span class="n">{i}</span>'
                        f'<span class="tx">{_esc(p)}</span></div>'
                        for i, p in enumerate(summary_points, 1)
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="ls-hint">No source-supported quick-revision points were generated for this concept.</p>',
                    unsafe_allow_html=True,
                )

            grounded_terms = _grounded_terms(cur, visits)
            if grounded_terms:
                st.markdown('<div class="ls-sec">Key terms</div>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="ls-chip">{_esc(k)}</span>' for k in grounded_terms),
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="ls-sec">Source</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ls-source-card"><b>Slide { _esc(cur["slide_id"]) }</b>'
                f' · {_clock(cur["start_time"])} · lecture evidence from this aligned segment.</div>',
                unsafe_allow_html=True,
            )
            st.button("▶ Watch explanation", key="quick_watch", **_FULL,
                      on_click=_select_visit, args=(pos,), kwargs={"seek_to": cur["start_time"], "visits": visits})

        # --------------------------------------------------------
        # Flashcards — concept-grounded from Lecture Notes
        # --------------------------------------------------------
        with tabs[3]:
            cards = _concept_cards(cur, visits)
            cards = [c for c in cards if c.get("def") or c.get("definition") or c.get("easy_explanation")]
            if not cards:
                st.markdown('<p class="ls-hint">No flashcard-ready source-supported concepts were generated for this segment.</p>',
                            unsafe_allow_html=True)
            else:
                fi = min(max(int(st.session_state.get("flash_index", 0)), 0), len(cards) - 1)
                card = cards[fi]
                answer = card.get("easy_explanation") or card.get("definition") or card.get("def")
                st.markdown(
                    f'<div class="ls-card"><div class="ls-ai-badge">Flashcard {fi + 1} / {len(cards)}</div>'
                    f'<div class="ls-term" style="font-size:1.2rem">What is <b>{_esc(card["term"])}</b> in this lecture?</div>'
                    f'<div class="ls-source-card">Source: Slide {_esc(card["slide_id"])} · {_clock(card["t"])}</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Reveal answer"):
                    st.markdown(f'<div class="ls-ai-summary">{_esc(answer or "The lecture names this concept but does not provide a fuller source-supported definition.")}</div>',
                                unsafe_allow_html=True)
                    st.button("▶ Watch explanation", key=f"flash_watch_{fi}", **_FULL,
                              on_click=_select_visit, args=(card.get("visit", pos),), kwargs={"seek_to": card["t"], "visits": visits})
                b1, b2 = st.columns(2)
                if b1.button("← Previous card", key="flash_prev", disabled=fi == 0, **_FULL):
                    st.session_state.flash_index = fi - 1
                    st.rerun()
                if b2.button("Next card →", key="flash_next", disabled=fi == len(cards) - 1, **_FULL):
                    st.session_state.flash_index = fi + 1
                    st.rerun()

        # --------------------------------------------------------
        # Quiz Me — concept-grounded multiple choice
        # --------------------------------------------------------
        with tabs[4]:
            if not st.session_state.quiz_items:
                st.session_state.quiz_items = _build_grounded_quiz(cur, visits, max_questions=5)
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False

            quiz_items = st.session_state.quiz_items
            if not quiz_items:
                st.markdown(
                    '<p class="ls-hint">Not enough distinct source-supported concepts with definitions to build a grounded quiz for this segment.</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p class="ls-hint">Questions and distractors are built from semantically related concepts in the generated lecture notes, not unrelated transcript sentences.</p>',
                    unsafe_allow_html=True,
                )
                for qi, item in enumerate(quiz_items):
                    choice = st.radio(
                        f"{qi + 1}. {item['question']}",
                        item["options"],
                        key=f"quiz_choice_{qi}",
                        index=None if qi not in st.session_state.quiz_answers else st.session_state.quiz_answers[qi],
                    )
                    if choice is not None:
                        st.session_state.quiz_answers[qi] = item["options"].index(choice)

                if st.button("Check my answers", type="primary", key="quiz_check", **_FULL):
                    score = 0
                    for qi, item in enumerate(quiz_items):
                        if st.session_state.quiz_answers.get(qi) == item["answer"]:
                            score += 1
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True
                    st.session_state.reviewed_visits.add(pos)
                    st.rerun()

                if st.session_state.quiz_submitted:
                    score = st.session_state.quiz_score
                    pct_q = round(100 * score / max(1, len(quiz_items)))
                    st.markdown(
                        f'<div class="ls-ai-summary"><b>Score: {score}/{len(quiz_items)} ({pct_q}%)</b></div>',
                        unsafe_allow_html=True,
                    )
                    for qi, item in enumerate(quiz_items):
                        selected = st.session_state.quiz_answers.get(qi)
                        if selected == item["answer"]:
                            st.success(f"Q{qi + 1}: Correct — source: Slide {item['slide_id']} · {_clock(item['t'])}")
                        else:
                            st.warning(f"Q{qi + 1}: Review “{item['term']}” — source: Slide {item['slide_id']} · {_clock(item['t'])}")

                if st.button("Reset quiz", key="quiz_reset", **_FULL):
                    st.session_state.quiz_items = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_submitted = False
                    st.rerun()

        # --------------------------------------------------------
        # Exam Prep — generated from structured concepts/evidence
        # --------------------------------------------------------
        with tabs[5]:
            st.markdown('<div class="ls-label">Exam-focused revision</div>', unsafe_allow_html=True)
            material = _grounded_exam_material(cur, visits)

            if material["exam_points"]:
                st.markdown('<div class="ls-sec">Must revise</div>', unsafe_allow_html=True)
                for p in material["exam_points"]:
                    st.markdown(f'<div class="ls-take"><span class="n">•</span><span class="tx">{_esc(p)}</span></div>',
                                unsafe_allow_html=True)

            if material["definitions"]:
                st.markdown('<div class="ls-sec">Definitions</div>', unsafe_allow_html=True)
                for d in material["definitions"]:
                    if not isinstance(d, dict):
                        continue
                    term = _esc(d.get("term", ""))
                    definition = _esc(d.get("definition", ""))
                    easy = _esc(d.get("easy_explanation", ""))
                    if term and (definition or easy):
                        st.markdown(
                            f'<div class="ls-teach-def"><div class="term">{term}</div>'
                            f'<div class="definition">{definition}</div>'
                            + (f'<div class="easy"><b>Easy explanation:</b> {easy}</div>' if easy else '')
                            + '</div>',
                            unsafe_allow_html=True,
                        )

            if material["steps"]:
                st.markdown('<div class="ls-sec">Steps / procedure</div>', unsafe_allow_html=True)
                for i, step in enumerate(material["steps"], 1):
                    st.markdown(
                        f'<div class="ls-teach-step"><span class="num">{i}</span><span class="text">{_esc(step)}</span></div>',
                        unsafe_allow_html=True,
                    )

            if material["examples"]:
                st.markdown('<div class="ls-sec">Examples from the lecture</div>', unsafe_allow_html=True)
                for example in material["examples"]:
                    st.markdown(f'<div class="ls-teach-bullet">• {_esc(example)}</div>', unsafe_allow_html=True)

            if material["common_confusions"]:
                st.markdown('<div class="ls-sec">Common confusions</div>', unsafe_allow_html=True)
                for item in material["common_confusions"]:
                    st.markdown(f'<div class="ls-block">{_esc(item)}</div>', unsafe_allow_html=True)

            if material["formulas"]:
                st.markdown('<div class="ls-sec">Formula / notation evidence</div>', unsafe_allow_html=True)
                for formula in material["formulas"]:
                    st.markdown(f'<div class="ls-formula">{_esc(formula)}</div>', unsafe_allow_html=True)

            grounded_terms = _grounded_terms(cur, visits, limit=6)
            if grounded_terms:
                st.markdown('<div class="ls-sec">Practice prompts from this lecture</div>', unsafe_allow_html=True)
                for i, term in enumerate(grounded_terms, 1):
                    st.markdown(
                        f'<div class="ls-block"><b>{i}.</b> Explain {_esc(term)} using only the explanation and evidence provided in the lecture.</div>',
                        unsafe_allow_html=True,
                    )

            if not any(material.values()):
                st.markdown('<p class="ls-hint">No structured exam material was generated for this concept.</p>', unsafe_allow_html=True)

            st.markdown('<div class="ls-sec">Fast plan</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ls-block">1. Review the concept in Lecture Notes.<br>'
                '2. Learn the source-supported definitions / properties.<br>'
                '3. Review the listed steps, examples, and formulas.<br>'
                '4. Use the flashcards once.<br>'
                '5. Take Quiz Me and revisit only missed evidence.</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # AI Tutor — retrieval strictly grounded in the lecture
        # --------------------------------------------------------
        with tabs[6]:
            st.markdown(
                f'<p class="ls-hint">Ask about the lecture — currently on <b>Slide {_esc(cur["slide_id"])}</b>. '
                'Answers are retrieved from the transcript, slide text and generated notes only.</p>',
                unsafe_allow_html=True,
            )
            q1, q2, q3 = st.columns(3, gap="small")
            q1.button("Summarize this slide", key="tutor_sum", **_FULL, on_click=_ask,
                      args=("Summarize this slide", "chat_history", visits, units, pos))
            q2.button("Explain this topic simply", key="tutor_exp", **_FULL, on_click=_ask,
                      args=("Explain this topic simply", "chat_history", visits, units, pos))
            grounded_terms = _grounded_terms(cur, visits, limit=1)
            if grounded_terms:
                q3.button(f"Where was “{grounded_terms[0]}” discussed?", key="tutor_where", **_FULL, on_click=_ask,
                          args=(f"Where was {grounded_terms[0]} discussed?", "chat_history", visits, units, pos))
            _render_history("chat_history", assistant=True, newest_first=False)
            with st.form("tutor_form", clear_on_submit=True):
                st.text_input("Ask the lecture tutor", placeholder="Ask a doubt…", key="tutor_text")
                st.form_submit_button("Ask", type="primary", on_click=_submit_question,
                                      args=("tutor_text", "chat_history", visits, units, pos))
            if st.session_state.chat_history:
                st.button("Clear tutor chat", key="tutor_clear", on_click=_clear_history, args=("chat_history",))

        # --------------------------------------------------------
        # Weak Topics — derived from concept-level quiz misses
        # --------------------------------------------------------
        with tabs[7]:
            weak = []
            quiz_items = st.session_state.get("quiz_items", [])
            if st.session_state.get("quiz_submitted"):
                seen = set()
                for qi, item in enumerate(quiz_items):
                    if st.session_state.quiz_answers.get(qi) != item["answer"]:
                        term = item.get("term", "Concept")
                        key = term.lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        weak.append({
                            "term": term,
                            "t": item.get("t", cur.get("start_time", 0)),
                            "slide_id": item.get("slide_id", cur.get("slide_id")),
                        })

            if weak:
                st.markdown(
                    '<p class="ls-hint">These concept-level topics were missed in the current quiz. Review the linked evidence and retake the quiz.</p>',
                    unsafe_allow_html=True,
                )
                for i, w in enumerate(weak, 1):
                    st.markdown(
                        f'<div class="ls-concept"><div class="ls-term">{i}. {_esc(w["term"])}</div>'
                        f'<div class="ls-def">Source: Slide {_esc(w["slide_id"])} · {_clock(w["t"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"▶ Review {_clock(w['t'])}",
                        key=f"weak_{i}", **_FULL,
                        on_click=_select_visit,
                        args=(pos,),
                        kwargs={"seek_to": w["t"], "visits": visits},
                    )
            else:
                st.markdown(
                    '<div class="ls-empty">No weak concepts recorded yet. Take Quiz Me to create a concept-grounded weak-topic list.</div>',
                    unsafe_allow_html=True,
                )

        # --------------------------------------------------------
        # Progress — lecture review progress + quiz performance
        # --------------------------------------------------------
        with tabs[8]:
            reviewed = st.session_state.get("reviewed_visits", set())
            reviewed_count = len([i for i in reviewed if 0 <= i < total])
            review_pct = round(100 * reviewed_count / max(1, total))
            quiz_items = st.session_state.get("quiz_items", [])
            quiz_done = bool(st.session_state.get("quiz_submitted"))
            quiz_pct = round(100 * st.session_state.get("quiz_score", 0) / max(1, len(quiz_items))) if quiz_done else 0

            p1, p2, p3 = st.columns(3)
            p1.metric("Slides reviewed", f"{reviewed_count}/{total}")
            p2.metric("Lecture progress", f"{review_pct}%")
            p3.metric("Quiz score", f"{quiz_pct}%" if quiz_done else "—")

            st.progress(review_pct / 100, text=f"Lecture review coverage · {review_pct}%")
            st.markdown('<div class="ls-sec">Current source</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ls-source-card"><b>Slide { _esc(cur["slide_id"]) }</b> · '
                f'{_clock(cur["start_time"])} – {_clock(cur["end_time"])} · '
                'Use “Watch explanation” to revisit the aligned evidence.</div>',
                unsafe_allow_html=True,
            )
            st.button("▶ Watch current explanation", key="progress_watch", **_FULL,
                      on_click=_select_visit, args=(pos,), kwargs={"seek_to": cur["start_time"], "visits": visits})

            if quiz_done:
                st.markdown('<div class="ls-sec">Next action</div>', unsafe_allow_html=True)
                if quiz_pct < 70:
                    st.warning("Review the weak topics above and retake Quiz Me.")
                elif quiz_pct < 90:
                    st.info("Good progress. Rewatch missed timestamps and retry the quiz.")
                else:
                    st.success("Strong result for this lecture segment. Continue to the next slide.")

        # ---- Transcript ----
        # Transcript is intentionally kept directly under the AI Tutor/learning tabs
        # so every line remains tied to its original timestamp.
        with st.expander("Open slide transcript", expanded=False):
            lines = [s for s in cur["segments"] if (s.get("text") or "").strip()]
            if lines:
                st.markdown('<p class="ls-hint">Click a timestamp to play the video from that exact moment.</p>', unsafe_allow_html=True)
                for j, s in enumerate(lines):
                    t = float(s.get("start_time") or 0)
                    tc, tx = st.columns([1, 6], gap="small")
                    tc.button(f"▶ {_clock(t)}", key=f"ts_{pos}_{j}", **_FULL,
                              on_click=_select_visit, args=(pos,), kwargs={"seek_to": t, "visits": visits})
                    tx.markdown(f'<div class="ls-tline">{_esc(s.get("text"))}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">No transcript was captured for this segment.</p>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    dl, _ = st.columns([1.4, 3])
    with dl:
        st.download_button("Download all study notes", data=_compile_downloadable_notes(visits),
                           file_name="study_notes.txt", mime="text/plain", **_FULL)

    st.markdown('<p class="ls-hint" style="margin-top:.7rem">Every study feature above is grounded in the aligned transcript/slide evidence. AI notes require GEMINI_API_KEY; navigation and revision tools work without it.</p>', unsafe_allow_html=True)
