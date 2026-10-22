from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import cv2
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import numpy as np
import openpyxl


DEFAULT_WIDTH = 48
DEFAULT_HEIGHT = 27


def parse_num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower().endswith("s"):
        text = text[:-1].strip()
    if ":" in text:
        parts = text.split(":")
        try:
            nums = [float(x) for x in parts]
            if len(nums) == 3:
                return nums[0] * 3600 + nums[1] * 60 + nums[2]
            if len(nums) == 2:
                return nums[0] * 60 + nums[1]
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        return None


def load_ground_truth(path: Path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(x).strip().lower() if x is not None else "" for x in rows[0]]

    def find_col(*names):
        for name in names:
            name = name.lower()
            if name in headers:
                return headers.index(name)
        return None

    tcol = find_col("time", "timestamp", "start", "key")
    scol = find_col("slidenumber", "slide_number", "slide", "slideid")
    if tcol is None or scol is None:
        raise ValueError(f"Could not find GT columns in {path}: {headers}")

    result = []
    for row in rows[1:]:
        if max(tcol, scol) >= len(row):
            continue
        t = parse_num(row[tcol])
        s = parse_num(row[scol])
        if t is None or s is None or s < 0:
            continue
        result.append((float(t), int(s)))
    wb.close()
    return result


def find_asset(root: Path, stem: str, suffixes: Iterable[str]) -> Path | None:
    root = Path(root)
    norm = "".join(ch.lower() for ch in stem if ch.isalnum())
    candidates = []
    for suffix in suffixes:
        candidates.extend(root.rglob(f"*{suffix}"))
    scored = []
    for p in candidates:
        pnorm = "".join(ch.lower() for ch in p.stem if ch.isalnum())
        score = 0
        if pnorm == norm:
            score = 1000
        elif norm in pnorm or pnorm in norm:
            score = 800 - abs(len(pnorm) - len(norm))
        else:
            common = sum(a == b for a, b in zip(pnorm, norm))
            score = common
        scored.append((score, p))
    return max(scored, key=lambda x: x[0])[1] if scored else None


def render_slides(pdf_path: Path, scale: float = 1.5):
    doc = fitz.open(str(pdf_path))
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        images.append(arr)
    doc.close()
    return images


def sample_video(video_path: Path, sample_every: float = 1.0):
    """Compatibility sampler. Prefer stream_video_batches for long videos."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    frames = []
    times = []
    next_t = 0.0
    idx = 0
    step = max(0.25, float(sample_every))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if t + 1e-6 >= next_t:
            frames.append(frame)
            times.append(float(t))
            next_t += step
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("No sampled video frames")
    return frames, np.asarray(times, dtype=np.float32), duration


def stream_video_batches(video_path: Path, sample_every: float = 1.0, batch_size: int = 128):
    """Yield (frames, timestamps) without keeping the complete lecture in RAM."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    step = max(0.25, float(sample_every))
    next_t = 0.0
    idx = 0
    batch = []
    times = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps if fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if t + 1e-6 >= next_t:
                batch.append(frame)
                times.append(float(t))
                next_t += step
                if len(batch) >= batch_size:
                    yield batch, np.asarray(times, dtype=np.float32), duration
                    batch, times = [], []
            idx += 1
        if batch:
            yield batch, np.asarray(times, dtype=np.float32), duration
    finally:
        cap.release()


def _mask_overlay(img: np.ndarray, top_right_fraction: float = 0.20, top_fraction: float = 0.22):
    out = img.copy()
    h, w = out.shape[:2]
    x0 = int(w * (1.0 - top_right_fraction))
    y1 = int(h * top_fraction)
    # Replace webcam region by local median instead of black, so global
    # structure is not changed too aggressively.
    safe = out.copy()
    patch = safe[:, :max(x0, 1)]
    median = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    out[:y1, x0:] = median
    return out


def _normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-8)


def signature(image: np.ndarray, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
    image = _mask_overlay(image)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    g = _normalize((gray - gray.mean()).reshape(-1))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge = np.log1p(5.0 * edge)
    e = _normalize((edge - edge.mean()).reshape(-1))

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    # Low-frequency DCT catches layout/text blocks without overreacting to
    # the lecturer camera overlay.
    dct = cv2.dct(gray)
    d = _normalize(dct[:8, :8].reshape(-1))

    return {
        "gray": g,
        "edge": e,
        "hist": hist.astype(np.float32),
        "dct": d,
    }


def signature_matrix(images, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    sigs = [signature(x, width, height) for x in images]
    return {
        k: np.stack([s[k] for s in sigs]).astype(np.float32)
        for k in sigs[0]
    }


def visual_scores(frame_sigs, slide_sigs):
    gray = frame_sigs["gray"] @ slide_sigs["gray"].T
    edge = frame_sigs["edge"] @ slide_sigs["edge"].T
    dct = frame_sigs["dct"] @ slide_sigs["dct"].T

    # Histogram similarity using a symmetric intersection score.
    fh = frame_sigs["hist"]
    sh = slide_sigs["hist"]
    hist = np.minimum(fh[:, None, :], sh[None, :, :]).sum(axis=2)

    score = (
        0.48 * gray
        + 0.27 * edge
        + 0.10 * dct
        + 0.15 * hist
    )
    return score.astype(np.float32)


def viterbi_monotonic(score: np.ndarray, max_jump: int = 3, jump_penalty: float = 0.015):
    n_frames, n_slides = score.shape
    dp = np.full((n_frames, n_slides), -np.inf, dtype=np.float32)
    back = np.full((n_frames, n_slides), -1, dtype=np.int16)

    first_limit = min(n_slides, max_jump + 1)
    dp[0, :first_limit] = score[0, :first_limit]

    for t in range(1, n_frames):
        for j in range(n_slides):
            lo = max(0, j - max_jump)
            best_val = -np.inf
            best_prev = -1
            for p in range(lo, j + 1):
                if not np.isfinite(dp[t - 1, p]):
                    continue
                trans = 0.0 if p == j else -jump_penalty * (j - p)
                val = float(dp[t - 1, p]) + trans
                if val > best_val:
                    best_val = val
                    best_prev = p
            dp[t, j] = best_val + score[t, j]
            back[t, j] = best_prev

    j = int(np.nanargmax(dp[-1]))
    path = np.empty(n_frames, dtype=np.int16)
    path[-1] = j
    for t in range(n_frames - 1, 0, -1):
        j = int(back[t, j])
        if j < 0:
            j = int(path[t])
        path[t - 1] = j
    return path, dp


def build_timeline(path: np.ndarray, timestamps: np.ndarray, slide_text=None):
    entries = []
    start = 0
    for i in range(1, len(path)):
        if int(path[i]) != int(path[i - 1]):
            entries.append((start, i - 1, int(path[start]) + 1))
            start = i
    entries.append((start, len(path) - 1, int(path[start]) + 1))

    out = []
    for a, b, sid in entries:
        start_t = float(timestamps[a])
        end_t = float(timestamps[b])
        if b + 1 < len(timestamps):
            end_t = max(end_t, float(timestamps[b + 1]) - 1e-3)
        item = {"start_time": start_t, "end_time": end_t, "slide_id": sid}
        if slide_text and 1 <= sid <= len(slide_text):
            item["slide_text"] = slide_text[sid - 1]
        out.append(item)
    return out


def evaluate_timeline(timeline, gt):
    correct = 0
    errors = []
    for t, gs in gt:
        pred = None
        for e in timeline:
            if e["start_time"] <= t <= e["end_time"]:
                pred = int(e["slide_id"])
                break
        if pred == gs:
            correct += 1
        else:
            errors.append((t, gs, pred))
    total = len(gt)
    return correct, total, correct / total if total else 0.0, errors


def run_visual_first(video_path: Path, pdf_path: Path, sample_every=1.0, max_jump=3, jump_penalty=0.015, progress=True):
    slides = render_slides(pdf_path)
    slide_sigs = signature_matrix(slides)

    chunks = []
    all_times = []
    total_samples = 0
    duration = 0.0
    for frames, times, duration in stream_video_batches(video_path, sample_every=sample_every, batch_size=128):
        fs = signature_matrix(frames)
        chunks.append(visual_scores(fs, slide_sigs))
        all_times.append(times)
        total_samples += len(frames)
        if progress:
            print(f"Visual frames: {total_samples}", flush=True)

    if not chunks:
        raise RuntimeError("No sampled video frames")
    scores = np.concatenate(chunks, axis=0)
    times = np.concatenate(all_times, axis=0).astype(np.float32)
    path, _ = viterbi_monotonic(scores, max_jump=max_jump, jump_penalty=jump_penalty)
    timeline = build_timeline(path, times)
    return timeline, scores, times, duration


def save_timeline(path: Path, timeline, metadata=None):
    payload = {"timeline": timeline}
    if metadata:
        payload["metadata"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
