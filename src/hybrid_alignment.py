import re
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import fitz
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import easyocr
except Exception:
    easyocr = None


def parse_srt_timestamp(x: str) -> float:
    x = x.strip().replace(",", ".")
    hh, mm, rest = x.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def read_srt(path: Path) -> List[Dict]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for block in blocks:
        lines = [x.strip() for x in block.splitlines() if x.strip()]
        time_idx = next((i for i, x in enumerate(lines) if "-->" in x), None)
        if time_idx is None:
            continue
        try:
            a, b = [x.strip() for x in lines[time_idx].split("-->")[:2]]
            start = parse_srt_timestamp(a)
            end = parse_srt_timestamp(b)
        except Exception:
            continue
        body = " ".join(lines[time_idx + 1:]).strip()
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body)
        if body:
            out.append({"start": start, "end": end, "text": body})
    return out


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str):
    return set(normalize_text(text).split())


def lexical_similarity(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    jaccard = inter / max(union, 1)
    recall = inter / max(len(ta), 1)
    precision = inter / max(len(tb), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(0.55 * f1 + 0.45 * jaccard)


def crop_slide_region(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = int(0.02 * w), int(0.02 * h)
    x2, y2 = int(0.98 * w), int(0.98 * h)
    crop = frame[y1:y2, x1:x2].copy()

    # Suppress a common presenter/webcam region without assuming the
    # lecture uses it. Only a small upper-right patch is masked.
    ch, cw = crop.shape[:2]
    mx1, my1 = int(0.77 * cw), int(0.02 * ch)
    mx2, my2 = int(0.98 * cw), int(0.25 * ch)
    if mx2 > mx1 and my2 > my1:
        patch = crop[max(0, my1-4):min(ch, my2+4), max(0, mx1-4):min(cw, mx2+4)]
        if patch.size:
            med = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
            crop[my1:my2, mx1:mx2] = med
    return crop


def visual_descriptor(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    small = cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).reshape(-1)

    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140)
    edges = cv2.resize(edges.astype(np.float32) / 255.0, (16, 9), interpolation=cv2.INTER_AREA).reshape(-1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hhist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    shist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    vhist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    d = np.concatenate([small, edges, hhist, shist, vhist]).astype(np.float32)
    d -= d.mean()
    d /= (np.linalg.norm(d) + 1e-8)
    return d


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return an @ bn.T


def render_slides(pdf_path: Path, out_dir: Path, zoom: float = 1.3) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    mat = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        p = out_dir / f"slide_{i+1:03d}.png"
        if not p.exists():
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(p))
        paths.append(p)
    doc.close()
    return paths


class OCRBackend:
    def __init__(self, use_gpu=True):
        self.reader = None
        self.backend = None
        if easyocr is not None:
            gpu = bool(use_gpu and torch is not None and torch.cuda.is_available())
            self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
            self.backend = "easyocr"

    def read(self, img_bgr: np.ndarray) -> str:
        if self.reader is None:
            return ""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.reader.readtext(rgb, detail=0, paragraph=True, text_threshold=0.55, low_text=0.35)
        text = " ".join(str(x) for x in result)
        return re.sub(r"\s+", " ", text).strip()


def extract_slide_ocr(slide_paths: List[Path], ocr: OCRBackend) -> List[str]:
    texts = []
    for i, p in enumerate(slide_paths, 1):
        img = cv2.imread(str(p))
        txt = ocr.read(img) if img is not None else ""
        texts.append(txt)
        print(f"    Slide OCR: {i}/{len(slide_paths)}")
    return texts


def extract_video_frames(video_path: Path, sample_seconds: float = 1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0) / max(fps, 1e-6)
    times = np.arange(0.0, max(duration, 0.0), sample_seconds, dtype=np.float64)
    if len(times) == 0 or times[-1] < duration:
        times = np.append(times, duration)

    descs = []
    actual_times = []
    prev = None
    changes = []

    for idx, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = crop_slide_region(frame)
        d = visual_descriptor(crop)
        descs.append(d)
        actual_times.append(float(t))
        if prev is None:
            changes.append(0.0)
        else:
            changes.append(float(1.0 - np.dot(prev, d)))
        prev = d
        if (idx + 1) % 128 == 0 or idx + 1 == len(times):
            print(f"    Visual frames: {idx+1}/{len(times)}")

    cap.release()
    return np.asarray(actual_times), np.asarray(descs, dtype=np.float32), np.asarray(changes, dtype=np.float32), duration


def select_ocr_times(times: np.ndarray, changes: np.ndarray, interval: float = 6.0, max_extra: int = 300):
    if len(times) == 0:
        return np.array([])
    regular = np.arange(times[0], times[-1] + 1e-6, interval)
    # Peak selection with non-maximum suppression.
    order = np.argsort(changes)[::-1]
    selected = list(regular.tolist())
    min_gap = max(1.5, interval * 0.35)
    for idx in order:
        t = float(times[idx])
        if all(abs(t - s) >= min_gap for s in selected):
            selected.append(t)
            if len(selected) >= len(regular) + max_extra:
                break
    selected = np.array(sorted(set(round(x, 3) for x in selected)), dtype=np.float64)
    return selected


def nearest_indices(query_times: np.ndarray, times: np.ndarray):
    pos = np.searchsorted(times, query_times)
    pos = np.clip(pos, 0, len(times) - 1)
    left = np.clip(pos - 1, 0, len(times) - 1)
    use_left = np.abs(times[left] - query_times) < np.abs(times[pos] - query_times)
    out = np.where(use_left, left, pos)
    return out.astype(int)


def extract_video_ocr(video_path: Path, ocr: OCRBackend, times: np.ndarray, ocr_times: np.ndarray):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    texts = []
    for i, t in enumerate(ocr_times, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        txt = ""
        if ok:
            txt = ocr.read(crop_slide_region(frame))
        texts.append(txt)
        if i % 20 == 0 or i == len(ocr_times):
            print(f"    Video OCR: {i}/{len(ocr_times)}")
    cap.release()
    return texts


def build_transcript_embeddings(segments: List[Dict], model):
    texts = [s["text"] for s in segments]
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def slide_text_similarity_matrix(frame_ocr_texts: List[str], slide_texts: List[str]) -> np.ndarray:
    out = np.zeros((len(frame_ocr_texts), len(slide_texts)), dtype=np.float32)
    for i, a in enumerate(frame_ocr_texts):
        for j, b in enumerate(slide_texts):
            out[i, j] = lexical_similarity(a, b)
    return out


def robust_row_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = np.percentile(x, 10, axis=1, keepdims=True)
    hi = np.percentile(x, 90, axis=1, keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def transcript_scores_for_times(times, segments, seg_emb, slide_emb):
    if len(times) == 0 or len(slide_emb) == 0 or len(seg_emb) == 0:
        return np.zeros((len(times), len(slide_emb)), dtype=np.float32)
    seg_starts = np.array([s["start"] for s in segments], dtype=np.float64)
    idx = np.searchsorted(seg_starts, times, side="right") - 1
    idx = np.clip(idx, 0, len(segments) - 1)
    scores = seg_emb[idx] @ slide_emb.T
    return robust_row_norm(scores)


def ocr_scores_for_times(times, ocr_times, ocr_matrix):
    if len(times) == 0 or len(ocr_times) == 0:
        return np.zeros((len(times), ocr_matrix.shape[1]), dtype=np.float32)
    idx = nearest_indices(times, ocr_times)
    return robust_row_norm(ocr_matrix[idx])


def local_change_scores(times, changes):
    if len(times) == 0:
        return np.zeros(0, dtype=np.float32)
    c = changes.astype(np.float32)
    # emphasize actual transition peaks rather than global motion
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med)) + 1e-6)
    z = (c - med) / (3.0 * mad)
    return 1.0 / (1.0 + np.exp(-z))


def hybrid_dp(scores: np.ndarray, boundary: np.ndarray,
              jump_penalty: float = 0.08,
              boundary_reward: float = 1.8,
              stay_boundary_penalty: float = 0.6,
              max_forward_jump: int = 3):
    n, s = scores.shape
    dp = np.full((n, s), -1e30, dtype=np.float64)
    prev = np.full((n, s), -1, dtype=np.int16)
    dp[0, 0] = float(scores[0, 0])

    for t in range(1, n):
        b = float(boundary[t])
        for j in range(s):
            best_val = -1e30
            best_k = -1
            lo = max(0, j - max_forward_jump)
            for k in range(lo, j + 1):
                if dp[t-1, k] <= -1e20:
                    continue
                jump = j - k
                if jump == 0:
                    trans = -stay_boundary_penalty * b
                else:
                    trans = boundary_reward * b - jump_penalty * jump
                val = dp[t-1, k] + trans
                if val > best_val:
                    best_val = val
                    best_k = k
            if best_k >= 0:
                dp[t, j] = best_val + float(scores[t, j])
                prev[t, j] = best_k

    j = int(np.argmax(dp[-1]))
    path = np.empty(n, dtype=np.int32)
    path[-1] = j
    for t in range(n - 1, 0, -1):
        j = int(prev[t, j]) if prev[t, j] >= 0 else j
        path[t-1] = j
    return path


def build_timeline(times: np.ndarray, path: np.ndarray):
    if len(times) == 0:
        return []
    entries = []
    start = 0
    for i in range(1, len(path)):
        if int(path[i]) != int(path[i-1]):
            entries.append({
                "start_time": float(times[start]),
                "end_time": float(times[i-1]),
                "slide": int(path[i-1]) + 1
            })
            start = i
    entries.append({
        "start_time": float(times[start]),
        "end_time": float(times[-1]),
        "slide": int(path[-1]) + 1
    })
    return entries


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next((c for c in ["Key", "Time", "Timestamp", "timestamp"] if c in df.columns), None)
    if time_col is None or "Slidenumber" not in df.columns:
        raise ValueError(f"Invalid ground truth columns in {path}")
    out = pd.DataFrame({
        "timestamp": pd.to_numeric(df[time_col], errors="coerce"),
        "slide": pd.to_numeric(df["Slidenumber"], errors="coerce")
    })
    # MaViLS ground truth contains transcript timestamps with Slidenumber=-1
    # for moments where no lecture slide is active. The verified evaluator
    # scores only rows with a valid positive slide number.
    out = out.dropna().sort_values("timestamp").reset_index(drop=True)
    out["slide"] = out["slide"].astype(int)
    out = out[out["slide"] > 0].reset_index(drop=True)
    return out


def score_predictions(times, path, gt):
    idx = np.searchsorted(times, gt["timestamp"].to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(times)-1)
    pred = path[idx] + 1
    true = gt["slide"].to_numpy()
    correct = int((pred == true).sum())
    return correct, len(true)


def prepare_lecture_cache(pdf_path: Path, srt_path: Path, video_path: Path,
                          cache_dir: Path, sample_seconds: float = 1.0,
                          ocr_interval: float = 6.0):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            required = ["times.npy", "frame_desc.npy", "changes.npy", "slide_desc.npy",
                        "slide_texts.json", "ocr_times.npy", "ocr_matrix.npy",
                        "transcript_emb.npy", "segments.json"]
            if all((cache_dir / x).exists() for x in required):
                print(f"  Cache found: {cache_dir}")
                return
        except Exception:
            pass

    print("  Rendering PDF slides...")
    slide_dir = cache_dir / "slides"
    slide_paths = render_slides(pdf_path, slide_dir)
    ocr = OCRBackend(use_gpu=True)
    if ocr.reader is None:
        raise RuntimeError("EasyOCR is not available in this environment. Install with: pip install easyocr")

    print(f"  Slide OCR backend: {ocr.backend}")
    slide_texts = extract_slide_ocr(slide_paths, ocr)
    slide_desc = np.stack([visual_descriptor(cv2.imread(str(p))) for p in slide_paths], axis=0)

    print("  Sampling video and detecting boundaries...")
    times, frame_desc, changes, duration = extract_video_frames(video_path, sample_seconds)

    print("  Selecting OCR keyframes...")
    ocr_times = select_ocr_times(times, changes, interval=ocr_interval, max_extra=min(350, max(80, len(times)//12)))
    print(f"  OCR keyframes: {len(ocr_times)}")
    video_ocr = extract_video_ocr(video_path, ocr, times, ocr_times)
    ocr_matrix = slide_text_similarity_matrix(video_ocr, slide_texts)

    segments = read_srt(srt_path)
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not available. Install with: pip install sentence-transformers")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Encoding transcript and slide text...")
    seg_emb = build_transcript_embeddings(segments, model)
    slide_emb = model.encode(slide_texts, normalize_embeddings=True, show_progress_bar=False)
    slide_emb = np.asarray(slide_emb, dtype=np.float32)

    np.save(cache_dir / "times.npy", times)
    np.save(cache_dir / "frame_desc.npy", frame_desc)
    np.save(cache_dir / "changes.npy", changes)
    np.save(cache_dir / "slide_desc.npy", slide_desc)
    np.save(cache_dir / "ocr_times.npy", ocr_times)
    np.save(cache_dir / "ocr_matrix.npy", ocr_matrix)
    np.save(cache_dir / "transcript_emb.npy", seg_emb)
    np.save(cache_dir / "slide_emb.npy", slide_emb)
    (cache_dir / "slide_texts.json").write_text(json.dumps(slide_texts, ensure_ascii=False, indent=2), encoding="utf-8")
    (cache_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    (cache_dir / "meta.json").write_text(json.dumps({
        "pdf": str(pdf_path), "srt": str(srt_path), "video": str(video_path),
        "sample_seconds": sample_seconds, "ocr_interval": ocr_interval,
        "duration": duration, "slides": len(slide_paths), "ocr_backend": ocr.backend
    }, indent=2), encoding="utf-8")


def run_from_cache(cache_dir: Path, weights: Dict[str, float], dp_cfg: Dict):
    times = np.load(cache_dir / "times.npy")
    frame_desc = np.load(cache_dir / "frame_desc.npy")
    changes = np.load(cache_dir / "changes.npy")
    slide_desc = np.load(cache_dir / "slide_desc.npy")
    ocr_times = np.load(cache_dir / "ocr_times.npy")
    ocr_matrix = np.load(cache_dir / "ocr_matrix.npy")
    seg_emb = np.load(cache_dir / "transcript_emb.npy")
    slide_emb = np.load(cache_dir / "slide_emb.npy")
    segments = json.loads((cache_dir / "segments.json").read_text(encoding="utf-8"))

    visual = robust_row_norm(cosine_matrix(frame_desc, slide_desc))
    semantic = transcript_scores_for_times(times, segments, seg_emb, slide_emb)
    ocr = ocr_scores_for_times(times, ocr_times, ocr_matrix)
    boundary = local_change_scores(times, changes)

    score = (
        weights["visual"] * visual +
        weights["ocr"] * ocr +
        weights["semantic"] * semantic
    )

    # Transition-focused evidence: next-slide candidate receives extra reward
    # only around genuine frame changes. This does not force a change.
    b = boundary.reshape(-1, 1)
    score = score + weights["boundary_to_score"] * b

    path = hybrid_dp(
        score,
        boundary,
        jump_penalty=dp_cfg.get("jump_penalty", 0.08),
        boundary_reward=dp_cfg.get("boundary_reward", 1.8),
        stay_boundary_penalty=dp_cfg.get("stay_boundary_penalty", 0.6),
        max_forward_jump=dp_cfg.get("max_forward_jump", 3),
    )
    return times, path, {"visual": visual, "ocr": ocr, "semantic": semantic, "boundary": boundary}


DEFAULT_WEIGHTS = {
    "visual": 0.25,
    "ocr": 0.45,
    "semantic": 0.30,
    "boundary_to_score": 0.05,
}

DEFAULT_DP = {
    "jump_penalty": 0.08,
    "boundary_reward": 1.8,
    "stay_boundary_penalty": 0.6,
    "max_forward_jump": 3,
}
