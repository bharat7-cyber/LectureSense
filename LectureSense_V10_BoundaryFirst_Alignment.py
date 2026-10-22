import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

V8_PATH = ROOT / "LectureSense_V8_TrainValidateTest.py"
V7_PATH = ROOT / "LectureSense_V7_FullDevelopment_GatedFusion.py"
if not V8_PATH.exists():
    raise FileNotFoundError(
        f"Required backbone file not found: {V8_PATH}\n"
        "Keep LectureSense_V8_TrainValidateTest.py beside this script."
    )
if not V7_PATH.exists():
    raise FileNotFoundError(
        f"Required asset/helper file not found: {V7_PATH}\n"
        "Keep LectureSense_V7_FullDevelopment_GatedFusion.py beside this script."
    )

import importlib.util

spec7 = importlib.util.spec_from_file_location("lecturesense_v7_v10", V7_PATH)
v7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(v7)

spec8 = importlib.util.spec_from_file_location("lecturesense_v8_v10", V8_PATH)
v8 = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(v8)


OUT = ROOT / "runs" / "v10_boundary_first"
CONFIG_PATH = ROOT / "checkpoints" / "v10_boundary_first_config.json"


@dataclass
class BoundaryConfig:
    min_gap_seconds: float = 4.0
    visual_percentile: float = 88.0
    v4_percentile: float = 78.0
    boundary_percentile: float = 82.0
    fusion_visual_weight: float = 0.45
    fusion_v4_boundary_weight: float = 0.35
    fusion_v4_peak_weight: float = 0.20
    segment_score_visual_weight: float = 0.35
    segment_score_v4_weight: float = 0.65
    max_slide_jump: int = 2
    jump_penalty: float = 0.035
    skip_boundary_seconds: float = 1.5
    min_segment_seconds: float = 3.0
    max_segment_seconds: float = 180.0


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def robust_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = np.percentile(x, 5.0)
    hi = np.percentile(x, 95.0)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def smooth_1d(x: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(x) == 0:
        return x
    k = 2 * radius + 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def merge_times(times: List[float], min_gap: float) -> List[float]:
    if not times:
        return []
    vals = sorted(float(x) for x in times)
    out = [vals[0]]
    for t in vals[1:]:
        if t - out[-1] >= min_gap:
            out.append(t)
        else:
            # Keep the stronger/later event by averaging nearby detections.
            out[-1] = (out[-1] + t) / 2.0
    return out


def peak_times(times: np.ndarray, scores: np.ndarray, percentile: float, min_gap: float) -> List[float]:
    if len(times) == 0:
        return []
    s = smooth_1d(robust_norm(scores), radius=2)
    threshold = float(np.percentile(s, percentile))
    candidates = []
    for i in range(1, len(s) - 1):
        if s[i] >= threshold and s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            candidates.append((float(s[i]), float(times[i])))
    candidates.sort(reverse=True)
    selected = []
    for score, t in candidates:
        if all(abs(t - q) >= min_gap for q in selected):
            selected.append(t)
    return sorted(selected)


def clip_bounds(t: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, t)))


def load_gt(path: Path) -> pd.DataFrame:
    return v7.load_gt(path)


# -----------------------------------------------------------------------------
# Video boundary detection
# -----------------------------------------------------------------------------


def frame_signature(frame: np.ndarray) -> np.ndarray:
    # Lightweight CPU signature: grayscale thumbnail + HSV histograms.
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [12], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [12], [0, 256]).flatten()
    h /= (h.sum() + 1e-8)
    s /= (s.sum() + 1e-8)
    v /= (v.sum() + 1e-8)
    return np.concatenate([gray.flatten(), h, s, v]).astype(np.float32)


def detect_video_boundaries(video: Path, sample_seconds: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

    times = []
    signatures = []
    next_t = 0.0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, next_t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        times.append(float(next_t))
        signatures.append(frame_signature(frame))
        next_t += float(sample_seconds)
        if duration > 0 and next_t > duration:
            break
    cap.release()

    if len(signatures) < 2:
        return np.asarray(times, dtype=np.float32), np.zeros(len(times), dtype=np.float32), duration

    sig = np.stack(signatures).astype(np.float32)
    d = np.mean(np.abs(sig[1:] - sig[:-1]), axis=1)
    # Match each diff to the later sampled timestamp.
    diff_times = np.asarray(times[1:], dtype=np.float32)
    diff = smooth_1d(d, radius=1)
    return diff_times, diff.astype(np.float32), duration


# -----------------------------------------------------------------------------
# V4 + transcript preparation
# -----------------------------------------------------------------------------


def ensure_srt(video: Path, srt: Optional[Path], whisper_model: str) -> Path:
    if srt is not None and srt.exists():
        return srt

    generated = OUT / "generated_srt" / f"{video.stem}.srt"
    if generated.exists():
        return generated

    generated.parent.mkdir(parents=True, exist_ok=True)

    # Prefer faster-whisper when installed. The output is plain SRT.
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "No SRT was supplied and faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        ) from exc

    compute_type = "int8_float16" if v8.DEVICE.type == "cuda" else "int8"
    print(f"[ASR] Loading Whisper model: {whisper_model}")
    model = WhisperModel(whisper_model, device="cuda" if v8.DEVICE.type == "cuda" else "cpu", compute_type=compute_type)
    segments, _ = model.transcribe(str(video), vad_filter=True)

    def ts(sec: float) -> str:
        sec = max(0.0, float(sec))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - math.floor(sec)) * 1000))
        if ms >= 1000:
            ms = 0
            s += 1
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    idx = 1
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.extend([str(idx), f"{ts(seg.start)} --> {ts(seg.end)}", text, ""])
        idx += 1
    generated.write_text("\n".join(lines), encoding="utf-8")
    print("[ASR] Saved:", generated)
    return generated


def ensure_v4_cache(srt: Path, pdf: Path, video: Path, cache_dir: Path) -> Path:
    score_path = cache_dir / "v4_final_score_matrix.npy"
    if score_path.exists():
        return cache_dir

    ck = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
    if not ck.exists():
        raise FileNotFoundError(
            f"Missing V4 checkpoint: {ck}\n"
            "Copy temporal_v4_best_model.pt into checkpoints/."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "src" / "inference_temporal_v4.py"),
        "--transcript", str(srt),
        "--slides", str(pdf),
        "--video", str(video),
        "--checkpoint", str(ck),
        "--out", str(cache_dir / "timeline_v4.json"),
        "--render-dir", str(cache_dir / "rendered_slides"),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", "0.40",
        "--video-sample-seconds", "1.0",
    ]
    print("[V4] Building V4 alignment cache...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return cache_dir


def read_transcript_mids(srt: Path) -> np.ndarray:
    segs = v7.get_segments(srt)
    return v7.segment_mids(segs)


# -----------------------------------------------------------------------------
# Boundary-first segmentation
# -----------------------------------------------------------------------------


def build_boundary_signal(
    mids: np.ndarray,
    v4_boundary: np.ndarray,
    v4_scores: np.ndarray,
    video_times: np.ndarray,
    video_diff: np.ndarray,
) -> np.ndarray:
    # Map video scene-change strength to nearest transcript midpoint.
    visual_on_mids = np.zeros(len(mids), dtype=np.float32)
    if len(video_times):
        for i, t in enumerate(mids):
            j = int(np.argmin(np.abs(video_times - float(t))))
            visual_on_mids[i] = float(video_diff[j])

    # V4 score instability: large change in best-slide score often precedes a slide change.
    top = np.max(v4_scores, axis=1)
    peak_shift = np.zeros_like(top, dtype=np.float32)
    if len(top) > 1:
        peak_shift[1:] = np.abs(top[1:] - top[:-1])

    a = robust_norm(visual_on_mids)
    b = robust_norm(v4_boundary)
    c = robust_norm(peak_shift)

    signal = 0.45 * a + 0.35 * b + 0.20 * c
    return smooth_1d(signal, radius=2).astype(np.float32)


def choose_boundaries(mids: np.ndarray, boundary_signal: np.ndarray, cfg: BoundaryConfig) -> List[float]:
    if len(mids) == 0:
        return []
    # Avoid a boundary too close to transcript start/end.
    candidates = peak_times(
        mids,
        boundary_signal,
        percentile=cfg.boundary_percentile,
        min_gap=cfg.min_gap_seconds,
    )
    if not candidates:
        return []

    out = []
    start = float(mids[0])
    end = float(mids[-1])
    for t in candidates:
        if t - start < cfg.min_segment_seconds:
            continue
        if end - t < cfg.min_segment_seconds:
            continue
        out.append(t)
    return merge_times(out, cfg.min_gap_seconds)


def build_segments(start_time: float, end_time: float, boundaries: List[float], cfg: BoundaryConfig) -> List[Tuple[float, float]]:
    cuts = [float(start_time)] + [float(x) for x in boundaries if start_time < x < end_time] + [float(end_time)]
    cuts = sorted(cuts)
    raw = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    merged = []
    for a, b in raw:
        dur = b - a
        if dur < cfg.min_segment_seconds and merged:
            pa, _ = merged[-1]
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))

    # Split overlong segments uniformly. This is a safeguard for missed boundaries.
    final = []
    for a, b in merged:
        dur = b - a
        if dur <= cfg.max_segment_seconds:
            final.append((a, b))
            continue
        n = int(math.ceil(dur / cfg.max_segment_seconds))
        step = dur / n
        for k in range(n):
            final.append((a + k * step, a + (k + 1) * step))
    return final


# -----------------------------------------------------------------------------
# Segment-level slide matching
# -----------------------------------------------------------------------------


def nearest_indices(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if len(times) == 0:
        return np.zeros((0,), dtype=np.int64)
    a = int(np.searchsorted(times, lo, side="left"))
    b = int(np.searchsorted(times, hi, side="right"))
    if b <= a:
        center = (lo + hi) / 2.0
        j = int(np.argmin(np.abs(times - center)))
        return np.asarray([j], dtype=np.int64)
    return np.arange(a, b, dtype=np.int64)


def v4_segment_emission(v4: np.ndarray, mids: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = nearest_indices(mids, lo, hi)
    if len(idx) == 0:
        return np.mean(v4, axis=0)
    # Mean + a light max term prevents one highly informative transcript segment from being washed out.
    return 0.70 * np.mean(v4[idx], axis=0) + 0.30 * np.max(v4[idx], axis=0)


def visual_segment_emission(visual_times: np.ndarray, visual_scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = nearest_indices(visual_times, lo, hi)
    if len(idx) == 0:
        j = int(np.argmin(np.abs(visual_times - ((lo + hi) / 2.0))))
        return visual_scores[j]
    return 0.70 * np.mean(visual_scores[idx], axis=0) + 0.30 * np.max(visual_scores[idx], axis=0)


def row_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(1e-4, temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / (np.sum(p, axis=1, keepdims=True) + 1e-8)


def rank_norm(row: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(row))
    if len(row) <= 1:
        return np.ones_like(row, dtype=np.float32)
    return 1.0 - (order.astype(np.float32) / float(len(row) - 1))


def normalize_score_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32)
    z = row - np.mean(row)
    sd = np.std(z)
    if sd < 1e-6:
        return np.zeros_like(z)
    return z / sd


def make_segment_emissions(
    segments: List[Tuple[float, float]],
    mids: np.ndarray,
    v4: np.ndarray,
    visual_times: np.ndarray,
    visual_scores: np.ndarray,
    cfg: BoundaryConfig,
) -> np.ndarray:
    rows = []
    for lo, hi in segments:
        a = normalize_score_row(v4_segment_emission(v4, mids, lo, hi))
        b = normalize_score_row(visual_segment_emission(visual_times, visual_scores, lo, hi))
        # Combine raw evidence and ranking evidence. V4 remains the anchor.
        ap = row_softmax(a[None, :])[0]
        bp = row_softmax(b[None, :])[0]
        ar = rank_norm(a)
        br = rank_norm(b)
        score = (
            cfg.segment_score_v4_weight * (0.55 * a + 0.30 * ap + 0.15 * ar)
            + cfg.segment_score_visual_weight * (0.55 * b + 0.30 * bp + 0.15 * br)
        )
        rows.append(score.astype(np.float32))
    return np.stack(rows).astype(np.float32)


def monotonic_segment_dp(
    emissions: np.ndarray,
    segments: List[Tuple[float, float]],
    max_jump: int,
    jump_penalty: float,
) -> np.ndarray:
    n, s = emissions.shape
    dp = np.full((n, s), -np.inf, dtype=np.float32)
    back = np.full((n, s), -1, dtype=np.int16)

    first = min(s, max_jump + 1)
    dp[0, :first] = emissions[0, :first]

    for t in range(1, n):
        a0, b0 = segments[t - 1]
        a1, b1 = segments[t]
        gap = max(0.0, a1 - b0)
        for j in range(s):
            lo = max(0, j - max_jump)
            best = -np.inf
            best_p = -1
            for p in range(lo, j + 1):
                if not np.isfinite(dp[t - 1, p]):
                    continue
                jump = j - p
                transition = -jump_penalty * float(jump * jump)
                # Staying on the same slide is preferred when there is no meaningful temporal gap.
                if jump == 0:
                    transition += 0.004 * min(30.0, b1 - a1)
                score = dp[t - 1, p] + transition
                if score > best:
                    best = score
                    best_p = p
            if best_p >= 0:
                dp[t, j] = best + emissions[t, j]
                back[t, j] = best_p

    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path


# -----------------------------------------------------------------------------
# End-to-end alignment
# -----------------------------------------------------------------------------


def run_alignment(
    srt: Path,
    pdf: Path,
    video: Path,
    cfg: BoundaryConfig,
    lecture_name: str,
) -> dict:
    cache = OUT / "v4_cache" / lecture_name
    cache = ensure_v4_cache(srt, pdf, video, cache)

    v4 = np.load(cache / "v4_final_score_matrix.npy").astype(np.float32)
    boundary_path = cache / "v4_boundary_probability.npy"
    v4_boundary = np.load(boundary_path).astype(np.float32) if boundary_path.exists() else np.zeros(v4.shape[0], dtype=np.float32)

    mids = read_transcript_mids(srt)
    visual_npz = cache / "visual_scores_v10.npz"
    if visual_npz.exists():
        z = np.load(visual_npz)
        visual_times = np.asarray(z["times"], dtype=np.float32)
        visual_scores = np.asarray(z["scores"], dtype=np.float32)
    else:
        try:
            visual_times, visual_scores = v7.get_visual_scores(lecture_name, pdf, video)
        except Exception:
            # Fall back to V4 rows if Visual-First cache is unavailable.
            visual_times = mids.copy()
            visual_scores = v4.copy()
        np.savez_compressed(visual_npz, times=visual_times, scores=visual_scores)

    video_times, video_diff, duration = detect_video_boundaries(video, sample_seconds=1.0)
    if duration <= 0:
        duration = max(float(mids[-1]), 1.0)

    signal = build_boundary_signal(mids, v4_boundary, v4, video_times, video_diff)
    boundaries = choose_boundaries(mids, signal, cfg)
    segments = build_segments(float(mids[0]), max(float(mids[-1]), duration), boundaries, cfg)
    emissions = make_segment_emissions(segments, mids, v4, visual_times, visual_scores, cfg)
    path = monotonic_segment_dp(emissions, segments, cfg.max_slide_jump, cfg.jump_penalty)

    rows = []
    for idx, ((lo, hi), slide_idx) in enumerate(zip(segments, path)):
        slide_num = int(slide_idx) + 1
        score = float(emissions[idx, slide_idx])
        rows.append({
            "segment_id": idx + 1,
            "start": float(lo),
            "end": float(hi),
            "duration": float(hi - lo),
            "slide": slide_num,
            "score": score,
        })

    return {
        "lecture": lecture_name,
        "srt": str(srt),
        "pdf": str(pdf),
        "video": str(video),
        "boundaries": boundaries,
        "boundary_count": len(boundaries),
        "segments": rows,
        "duration": duration,
        "v4_cache": str(cache),
    }


def save_timeline(result: dict, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["lecture"]
    json_path = out_dir / f"{name}_timeline.json"
    csv_path = out_dir / f"{name}_timeline.csv"
    html_path = out_dir / f"{name}_timeline.html"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["segments"]).to_csv(csv_path, index=False)

    html_rows = []
    for r in result["segments"]:
        html_rows.append(
            f"<tr><td>{r['segment_id']}</td><td>{r['start']:.2f}s</td><td>{r['end']:.2f}s</td><td>{r['slide']}</td><td>{r['score']:.4f}</td></tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>LectureSense V10 Timeline</title>
<style>body{{font-family:Arial,sans-serif;margin:30px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>LectureSense V10 — Boundary-First Timeline</h1>
<p><b>Lecture:</b> {name}</p>
<p><b>Detected boundaries:</b> {result['boundary_count']}</p>
<table><thead><tr><th>Segment</th><th>Start</th><th>End</th><th>Slide</th><th>Score</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return json_path, csv_path


# -----------------------------------------------------------------------------
# Benchmark evaluation
# -----------------------------------------------------------------------------


def accuracy_against_gt(result: dict, gt_path: Path) -> Tuple[float, int, int]:
    gt = load_gt(gt_path)
    correct = 0
    total = len(gt)
    segments = result["segments"]
    if not segments:
        return 0.0, 0, 0
    for _, row in gt.iterrows():
        t = float(row["time"])
        g = int(row["gt"])
        pred = None
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                pred = int(seg["slide"])
                break
        if pred is None:
            if t >= segments[-1]["end"]:
                pred = int(segments[-1]["slide"])
            else:
                pred = int(segments[0]["slide"])
        correct += int(pred == g)
    return correct / max(1, total), correct, total


def validate(lectures: List[str], cfg: BoundaryConfig) -> pd.DataFrame:
    rows = []
    for lecture in lectures:
        try:
            srt, pdf, gt, video = v7.asset_paths(lecture)
            if any(x is None for x in (srt, pdf, gt, video)):
                print(f"[VAL SKIP] {lecture}: missing exact asset")
                continue
            result = run_alignment(srt, pdf, video, cfg, lecture)
            acc, correct, total = accuracy_against_gt(result, gt)
            rows.append({
                "lecture": lecture,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "boundaries": result["boundary_count"],
                "segments": len(result["segments"]),
            })
            save_timeline(result, OUT / "validation")
            print(f"[VAL] {lecture}: {correct}/{total} = {acc*100:.2f}% | boundaries={result['boundary_count']} segments={len(result['segments'])}")
        except Exception as exc:
            print(f"[VAL SKIP] {lecture}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows)


def test(lectures: List[str], cfg: BoundaryConfig) -> pd.DataFrame:
    rows = []
    for lecture in lectures:
        try:
            srt, pdf, gt, video = v7.asset_paths(lecture)
            if any(x is None for x in (srt, pdf, gt, video)):
                print(f"[TEST SKIP] {lecture}: missing exact asset")
                continue
            result = run_alignment(srt, pdf, video, cfg, lecture)
            acc, correct, total = accuracy_against_gt(result, gt)
            rows.append({
                "lecture": lecture,
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "boundaries": result["boundary_count"],
                "segments": len(result["segments"]),
            })
            save_timeline(result, OUT / "test")
            print(f"[TEST] {lecture}: {correct}/{total} = {acc*100:.2f}% | boundaries={result['boundary_count']} segments={len(result['segments'])}")
        except Exception as exc:
            print(f"[TEST SKIP] {lecture}: {type(exc).__name__}: {exc}")
    return pd.DataFrame(rows)


def tune_validation() -> BoundaryConfig:
    candidates = []
    base = BoundaryConfig()
    for bp in (76, 80, 84, 88, 92):
        for gap in (3.0, 4.0, 5.0, 6.0):
            for mj in (1, 2, 3):
                cfg = BoundaryConfig(
                    boundary_percentile=bp,
                    min_gap_seconds=gap,
                    max_slide_jump=mj,
                    segment_score_v4_weight=0.65,
                    segment_score_visual_weight=0.35,
                    jump_penalty=base.jump_penalty,
                )
                df = validate(v8.VAL, cfg)
                if df.empty:
                    continue
                macro = float(df["accuracy"].mean())
                weighted = float(df["correct"].sum() / max(1, df["total"].sum()))
                candidates.append((macro, weighted, cfg))
                print(f"[SWEEP] boundary_pct={bp} gap={gap} max_jump={mj} macro={macro*100:.2f}% weighted={weighted*100:.2f}%")

    if not candidates:
        raise RuntimeError("Validation sweep produced no candidate.")
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, best = candidates[0]
    return best


def train_mode(args):
    # V10 is intentionally a decoder/segmentation upgrade over the trained V4 model.
    # There is no gradient training here; the existing V4 checkpoint remains unchanged.
    print("=" * 78)
    print("LECTURESENSE V10 — BOUNDARY-FIRST ALIGNMENT")
    print("=" * 78)
    print("Device:", v8.DEVICE)
    if getattr(v8, "DEVICE", None) is not None and getattr(v8.DEVICE, "type", "cpu") == "cuda":
        print("GPU:", __import__("torch").cuda.get_device_name(0))
    print("Test labels used for tuning: NO")

    if args.tune_validation:
        cfg = tune_validation()
    else:
        cfg = BoundaryConfig()

    payload = {
        "version": "LectureSense-V10-Boundary-First",
        "config": cfg.__dict__,
        "validation_lectures": list(v8.VAL),
        "test_lectures": list(v8.TEST),
        "test_used_for_model_selection": False,
        "backbone": "temporal_v4_best_model.pt",
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved config:", CONFIG_PATH)

    df = validate(v8.VAL, cfg)
    if not df.empty:
        macro = float(df["accuracy"].mean())
        print("\nValidation macro:", f"{macro*100:.2f}%")
        print(df.to_string(index=False))


def load_cfg() -> BoundaryConfig:
    if not CONFIG_PATH.exists():
        print("[CONFIG] No saved V10 config. Using defaults.")
        return BoundaryConfig()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return BoundaryConfig(**payload.get("config", {}))


def test_mode():
    cfg = load_cfg()
    print("=" * 78)
    print("LECTURESENSE V10 — HELD-OUT TEST")
    print("=" * 78)
    print("Test labels used for model selection: NO")
    df = test(v8.TEST, cfg)
    if df.empty:
        raise RuntimeError("No test lecture evaluated.")
    macro = float(df["accuracy"].mean())
    print("\nV10 held-out test macro:", f"{macro*100:.2f}%")
    print(df.to_string(index=False))


def demo_mode(args):
    video = Path(args.video).resolve()
    pdf = Path(args.pdf).resolve()
    srt = Path(args.srt).resolve() if args.srt else None
    if not video.exists():
        raise FileNotFoundError(video)
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    name = args.name or video.stem
    srt = ensure_srt(video, srt, args.whisper_model)
    cfg = load_cfg()
    result = run_alignment(srt, pdf, video, cfg, name)
    json_path, csv_path = save_timeline(result, OUT / "demo")

    print("\n" + "=" * 78)
    print("LECTURESENSE V10 DEMO RESULT")
    print("=" * 78)
    print("Video:", video)
    print("PDF:", pdf)
    print("SRT:", srt)
    print("Detected boundaries:", result["boundary_count"])
    print("Generated segments:", len(result["segments"]))
    print("JSON:", json_path)
    print("CSV :", csv_path)
    print("HTML:", OUT / "demo" / f"{name}_timeline.html")
    print("\nFirst timeline rows:")
    print(pd.DataFrame(result["segments"]).head(30).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="LectureSense V10 boundary-first slide alignment")
    ap.add_argument("--mode", choices=["train", "validate", "test", "demo"], default="validate")
    ap.add_argument("--tune-validation", action="store_true", help="Tune only on the official validation split and save the V10 config")
    ap.add_argument("--video", type=str, help="Input lecture video for demo mode")
    ap.add_argument("--pdf", type=str, help="Input slide PDF for demo mode")
    ap.add_argument("--srt", type=str, default=None, help="Optional transcript SRT; generated with faster-whisper when omitted")
    ap.add_argument("--name", type=str, default=None, help="Lecture/demo name")
    ap.add_argument("--whisper-model", type=str, default="base", help="faster-whisper model name")
    args = ap.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "validate":
        cfg = load_cfg()
        df = validate(v8.VAL, cfg)
        if df.empty:
            raise RuntimeError("No validation lecture evaluated.")
        print("\nValidation macro:", f"{float(df['accuracy'].mean())*100:.2f}%")
        print(df.to_string(index=False))
    elif args.mode == "test":
        test_mode()
    else:
        if not args.video or not args.pdf:
            raise ValueError("Demo mode requires --video and --pdf")
        demo_mode(args)


if __name__ == "__main__":
    main()
