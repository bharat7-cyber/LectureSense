"""
LectureSense — Study Workspace
================================

PDF + MP4  ->  faster-whisper (word timestamps)
           ->  src/inference_temporal_v4.py  (checkpoints/temporal_v4_best_model.pt)
           ->  timeline.json + rendered slides
           ->  Study Workspace (video player, slide timeline, notes, transcript).

The V4 model, decoder and every inference flag are untouched; this file only
concerns the interface layer and the glue around the existing pipeline.

Run with:
    streamlit run app.py
"""

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

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
_INFERENCE_SCRIPT = _BASE_DIR / "src" / "inference_temporal_v4.py"
_CHECKPOINT_PATH = _BASE_DIR / "checkpoints" / "temporal_v4_best_model.pt"

_RUN_DIR = _BASE_DIR / "runs" / "session"
_TIMELINE_PATH = _RUN_DIR / "timeline.json"
_SLIDES_DIR = _RUN_DIR / "rendered_slides"
_UPLOAD_DIR = _RUN_DIR / "uploads"
_LOG_PATH = _RUN_DIR / "pipeline.log"
_VIDEO_NAME = "lecture.mp4"

_EXAMPLE_DIR = _BASE_DIR / "demo_v4"
_EXAMPLE_TIMELINE_PATH = _EXAMPLE_DIR / "timeline.json"
_EXAMPLE_SLIDES_DIR = _EXAMPLE_DIR / "rendered_slides"

_INFERENCE_TIMEOUT_S = 1800
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# faster-whisper config (affects transcript generation only)
_WHISPER_MODEL_SIZE = "distil-small.en"
_WHISPER_DEVICE = "cpu"
_WHISPER_GPU_COMPUTE_TYPE = "float16"
_WHISPER_CPU_COMPUTE_TYPE = "int8"
_WHISPER_BATCH_SIZE = 8
_WHISPER_BEAM_SIZE = 1
_WHISPER_VAD_MIN_SILENCE_MS = 500
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
    """Seconds -> MaViLS timestamp format HH:MM:SS.mmm."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1_000)
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
    h = hashlib.sha256()
    h.update(len(pdf_bytes).to_bytes(8, "little"))
    h.update(pdf_bytes[:1_000_000])
    h.update(len(video_bytes).to_bytes(8, "little"))
    h.update(video_bytes[:1_000_000])
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
        error_details=None, pos=0, video_start=0, from_example=False,
    )


def _select_visit(index: int, seek_to=None, visits=None):
    """Callback: select a slide visit and seek the video to it."""
    st.session_state.pos = index
    if seek_to is not None:
        st.session_state.video_start = int(seek_to)
    elif visits is not None:
        st.session_state.video_start = int(visits[index]["start_time"])


# ============================================================
# PIPELINE: faster-whisper  ->  V4 inference
# ============================================================

@st.cache_resource(show_spinner=False)
def _load_whisper_pipeline():
    """Load and cache faster-whisper with GPU-first execution."""
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    last_error = None
    if _WHISPER_DEVICE in ("auto", "cuda"):
        try:
            model = WhisperModel(_WHISPER_MODEL_SIZE, device="cuda", compute_type=_WHISPER_GPU_COMPUTE_TYPE)
            return BatchedInferencePipeline(model=model), "cuda"
        except Exception as exc:
            last_error = exc
    try:
        model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu", compute_type=_WHISPER_CPU_COMPUTE_TYPE)
        return BatchedInferencePipeline(model=model), "cpu"
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def _transcribe_video(video_path: Path, out_srt_path: Path, on_progress=None):
    """Write a MaViLS-compatible word-level transcript:
        [00:00:01.100 --> 00:00:01.850] word
    on_progress(fraction_or_None, seconds_done, total_seconds) is called as
    real segments arrive. Returns (ok, reason)."""
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
            vad_parameters={"min_silence_duration_ms": _WHISPER_VAD_MIN_SILENCE_MS},
            condition_on_previous_text=False,
        )
        total = float(getattr(info, "duration", 0) or 0)
        out_srt_path.parent.mkdir(parents=True, exist_ok=True)

        lines, word_count = [], 0
        for segment in segments:
            for word in getattr(segment, "words", None) or []:
                text = (getattr(word, "word", "") or "").strip()
                start, end = getattr(word, "start", None), getattr(word, "end", None)
                if not text or start is None or end is None:
                    continue
                lines.append(f"[{_format_mavils_timestamp(start)} --> {_format_mavils_timestamp(end)}] {text}")
                word_count += 1
            if on_progress:
                done = float(getattr(segment, "end", 0) or 0)
                on_progress((min(done / total, 1.0) if total > 0 else None), done, total)

        if word_count == 0:
            return False, "empty"
        out_srt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, None
    except Exception as exc:
        st.session_state.error_details = f"Transcription error: {exc!r}"
        return False, "failed"


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def _run_inference(transcript_path, slides_path, video_path, out_path, render_dir, on_tick=None):
    """Run the V4 temporal multimodal alignment pipeline (flags unchanged).
    Output is captured to a log file so failures can be diagnosed."""
    missing = [p.name for p in (_INFERENCE_SCRIPT, _CHECKPOINT_PATH) if not p.exists()]
    if missing:
        st.session_state.error_details = "Missing required pipeline file(s): " + ", ".join(missing)
        return False, "unavailable"

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(_INFERENCE_SCRIPT),
            "--transcript", str(transcript_path),
            "--slides", str(slides_path),
            "--video", str(video_path),
            "--checkpoint", str(_CHECKPOINT_PATH),
            "--out", str(out_path),
            "--render-dir", str(render_dir),
            "--jump-penalty", "0.05",
            "--max-duration", "120",
            "--duration-penalty", "0.003",
            "--video-weight", "0.4",
            "--video-sample-seconds", "1.0",
        ]

        started = time.time()
        with open(_LOG_PATH, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, cwd=str(_BASE_DIR))
            while proc.poll() is None:
                elapsed = time.time() - started
                if elapsed > _INFERENCE_TIMEOUT_S:
                    proc.kill()
                    proc.wait()
                    return False, "timeout"
                if on_tick:
                    on_tick(elapsed)
                time.sleep(0.5)

        if proc.returncode != 0 or not out_path.exists():
            st.session_state.error_details = _tail(_LOG_PATH) or f"Exit code {proc.returncode}"
            return False, "failed"
        return True, None
    except Exception as exc:
        st.session_state.error_details = f"Pipeline error: {exc!r}"
        return False, "failed"


# ============================================================
# PROCESSING UI
# ============================================================

_STEPS = [
    ("Saving your files", "Preparing the slide deck and recording"),
    ("Transcribing the recording", "Word-level timestamps with faster-whisper"),
    ("Aligning transcript to slides", "V4 temporal multimodal model + visual alignment"),
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

        tabs = st.tabs(["Study notes", "Key concepts", "Slide content", "Transcript"])

        with tabs[0]:
            if cur["key_points"]:
                st.markdown('<ul class="ls-keypoints">' + "".join(f"<li>{_esc(p)}</li>" for p in cur["key_points"]) + "</ul>",
                            unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">No distinct key points were found here — see the transcript.</p>', unsafe_allow_html=True)

        with tabs[1]:
            if cur["keywords"]:
                st.markdown("".join(f'<span class="ls-chip">{_esc(k)}</span>' for k in cur["keywords"]), unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">No standout keywords were detected for this segment.</p>', unsafe_allow_html=True)

        with tabs[2]:
            if cur["ocr_text"]:
                st.markdown(f'<div class="ls-block">{_esc(cur["ocr_text"])}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">No text was detected on this slide.</p>', unsafe_allow_html=True)

        with tabs[3]:
            lines = [s for s in cur["segments"] if (s.get("text") or "").strip()]
            if lines:
                st.markdown('<p class="ls-hint">Click a timestamp to play the video from that moment.</p>', unsafe_allow_html=True)
                for j, s in enumerate(lines):
                    t = float(s.get("start_time") or 0)
                    tc, tx = st.columns([1, 6], gap="small")
                    tc.button(f"▶ {_clock(t)}", key=f"ts_{pos}_{j}", **_FULL,
                              on_click=_select_visit, args=(pos,), kwargs={"seek_to": t})
                    tx.markdown(f'<div class="ls-tline">{_esc(s.get("text"))}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="ls-hint">No transcript was captured for this segment.</p>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    dl, _ = st.columns([1.4, 3])
    with dl:
        st.download_button("Download all study notes", data=_compile_downloadable_notes(visits),
                           file_name="study_notes.txt", mime="text/plain", **_FULL)
