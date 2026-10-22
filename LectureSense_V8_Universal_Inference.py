import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_v7_module():
    path = ROOT / "LectureSense_V7_FullDevelopment_GatedFusion.py"
    if not path.exists():
        raise FileNotFoundError(
            f"V7 script not found: {path}. Put this V8 file beside the V7 file."
        )
    spec = importlib.util.spec_from_file_location("lecturesense_v7", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if ms >= 1000:
        whole += 1
        ms = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def transcribe_with_faster_whisper(video: Path, model_name: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None

    fw_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "int8_float16" if fw_device == "cuda" else "int8"
    try:
        model = WhisperModel(
            model_name,
            device=fw_device,
            compute_type=compute_type,
        )
        segments, info = model.transcribe(
            str(video),
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
        )
        out = []
        for seg in segments:
            text = clean_text(seg.text)
            if text:
                out.append((float(seg.start), float(seg.end), text))
        return out
    except Exception as exc:
        print(f"[Whisper faster-whisper fallback] {type(exc).__name__}: {exc}")
        return None


def transcribe_with_openai_whisper(video: Path, model_name: str, device: str):
    try:
        import whisper
    except Exception:
        return None

    try:
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(
            str(video),
            fp16=(device == "cuda"),
            verbose=False,
        )
        out = []
        for seg in result.get("segments", []):
            text = clean_text(seg.get("text", ""))
            if text:
                out.append((float(seg["start"]), float(seg["end"]), text))
        return out
    except Exception as exc:
        print(f"[Whisper openai fallback] {type(exc).__name__}: {exc}")
        return None


def write_srt(segments: List[Tuple[float, float, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(a)} --> {format_srt_time(b)}\n")
            f.write(f"{text}\n\n")


def ensure_transcript(video: Path, work_dir: Path, whisper_model: str):
    srt = work_dir / "auto_transcript.srt"
    if srt.exists() and srt.stat().st_size > 50:
        return srt

    print("\n[1/7] Transcribing lecture audio with Whisper...")
    segments = transcribe_with_faster_whisper(video, whisper_model, DEVICE.type)
    backend = "faster-whisper"
    if segments is None:
        segments = transcribe_with_openai_whisper(video, whisper_model, DEVICE.type)
        backend = "openai-whisper"

    if not segments:
        raise RuntimeError(
            "Whisper transcription failed. Install faster-whisper or openai-whisper."
        )

    write_srt(segments, srt)
    print(f"Transcript backend: {backend}")
    print(f"Transcript segments: {len(segments)}")
    print(f"Transcript saved: {srt}")
    return srt


def extract_notes_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".text"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        import fitz
        doc = fitz.open(str(path))
        return "\n".join(page.get_text("text") for page in doc)

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Supported notes formats: .txt, .md, .pdf, .docx")


def extract_pdf_texts(pdf: Path) -> List[str]:
    import fitz
    doc = fitz.open(str(pdf))
    texts = []
    for page in doc:
        texts.append(clean_text(page.get_text("text")))
    return texts


def add_notes_prior(emissions: np.ndarray, pdf: Path, notes: Path | None):
    if notes is None or not notes.exists():
        return emissions

    note_text = clean_text(extract_notes_text(notes))
    if len(note_text) < 20:
        return emissions

    slide_texts = extract_pdf_texts(pdf)
    if not any(slide_texts):
        return emissions

    try:
        from sentence_transformers import SentenceTransformer, util
        enc = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE.type)
        # Notes are global context, so the contribution is intentionally small.
        note_emb = enc.encode([note_text], convert_to_tensor=True, normalize_embeddings=True)
        slide_emb = enc.encode(slide_texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(note_emb, slide_emb).detach().cpu().numpy()[0].astype(np.float32)
    except Exception as exc:
        print(f"[Notes prior skipped] {type(exc).__name__}: {exc}")
        return emissions

    s = emissions.shape[1]
    if len(sims) < s:
        sims = np.pad(sims, (0, s - len(sims)), mode="edge")
    elif len(sims) > s:
        sims = sims[:s]

    sims = (sims - sims.mean()) / (sims.std() + 1e-6)
    sims = np.tanh(sims)
    return emissions + 0.08 * sims[None, :]


def load_v7_ensemble(v7):
    model_path = ROOT / "checkpoints" / "v7_full_development_gated_ensemble.pt"
    if not model_path.exists():
        return None, None

    ck = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )
    models = []
    for state in ck["models"]:
        model = v7.GatedCandidateTransformer(
            feature_dim=int(ck.get("feature_dim", 28))
        ).to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    decoder = ck.get(
        "decoder",
        {"max_jump": 3, "jump_penalty": 0.02, "boundary_reward": 0.12},
    )
    return models, decoder


def run_v4(video: Path, pdf: Path, srt: Path, work_dir: Path):
    ck = ROOT / "checkpoints" / "temporal_v4_best_model.pt"
    script = ROOT / "src" / "inference_temporal_v4.py"
    if not ck.exists():
        raise FileNotFoundError(f"V4 checkpoint missing: {ck}")
    if not script.exists():
        raise FileNotFoundError(f"V4 inference script missing: {script}")

    out = work_dir / "v4_timeline.json"
    rendered = work_dir / "rendered_slides"
    score = work_dir / "v4_final_score_matrix.npy"
    boundary = work_dir / "v4_boundary_probability.npy"

    if score.exists() and out.exists():
        return out, score, boundary

    print("\n[2/7] Running V4 multimodal temporal alignment...")
    cmd = [
        sys.executable,
        str(script),
        "--transcript", str(srt),
        "--slides", str(pdf),
        "--video", str(video),
        "--checkpoint", str(ck),
        "--out", str(out),
        "--render-dir", str(rendered),
        "--jump-penalty", "0.05",
        "--max-duration", "120",
        "--duration-penalty", "0.003",
        "--video-weight", "0.40",
        "--video-sample-seconds", "1.0",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not score.exists():
        raise RuntimeError(f"V4 did not create score matrix: {score}")
    return out, score, boundary


def visual_scores(v7, lecture_key: str, pdf: Path, video: Path, work_dir: Path):
    cache = work_dir / "visual_scores.npz"
    if cache.exists():
        z = np.load(cache)
        return np.asarray(z["times"], np.float32), np.asarray(z["scores"], np.float32)

    print("\n[3/7] Extracting video-frame visual evidence...")
    from visual_first_alignment import run_visual_first
    _, scores, times, duration = run_visual_first(
        Path(video),
        Path(pdf),
        sample_every=2.0,
        max_jump=2,
        jump_penalty=0.05,
        progress=True,
    )
    np.savez_compressed(
        cache,
        times=np.asarray(times, np.float32),
        scores=np.asarray(scores, np.float32),
        duration=np.float32(duration),
    )
    return np.asarray(times, np.float32), np.asarray(scores, np.float32)


def load_boundary(v4_boundary_path: Path, n: int):
    if v4_boundary_path.exists():
        b = np.load(v4_boundary_path).astype(np.float32)
        return b[:n] if len(b) >= n else np.pad(b, (0, n - len(b)))
    return np.zeros(n, dtype=np.float32)


def save_report(
    out_dir: Path,
    video: Path,
    pdf: Path,
    timeline_rows: List[dict],
    slide_count: int,
):
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<title>LectureSense V8 Timeline</title>")
    html.append("<style>body{font-family:Arial,sans-serif;margin:32px;background:#f5f5f5;color:#222}h1{margin-bottom:4px}.meta{color:#666;margin-bottom:20px}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:10px;border:1px solid #ddd;vertical-align:top}th{background:#eee}.thumb{max-width:220px;max-height:130px}.time{font-family:monospace;white-space:nowrap}</style>")
    html.append("</head><body>")
    html.append(f"<h1>LectureSense V8 — Lecture Timeline</h1>")
    html.append(f"<div class='meta'><b>Video:</b> {video.name}<br><b>PDF:</b> {pdf.name}<br><b>Slides:</b> {slide_count}</div>")
    html.append("<table><tr><th>Time</th><th>Slide</th><th>Confidence</th><th>Transcript</th><th>Preview</th></tr>")

    for r in timeline_rows:
        preview = ""
        img = r.get("preview")
        if img:
            preview = f"<img class='thumb' src='{img}'>"
        html.append(
            "<tr>"
            f"<td class='time'>{r['start']:.2f}s → {r['end']:.2f}s</td>"
            f"<td><b>{r['slide']}</b></td>"
            f"<td>{r['confidence']:.3f}</td>"
            f"<td>{r['text']}</td>"
            f"<td>{preview}</td>"
            "</tr>"
        )

    html.append("</table></body></html>")
    (out_dir / "timeline.html").write_text("\n".join(html), encoding="utf-8")


def align_mode(args):
    video = Path(args.video).expanduser().resolve()
    pdf = Path(args.pdf).expanduser().resolve()
    notes = Path(args.notes).expanduser().resolve() if args.notes else None

    if not video.exists():
        raise FileNotFoundError(video)
    if not pdf.exists():
        raise FileNotFoundError(pdf)
    if notes is not None and not notes.exists():
        raise FileNotFoundError(notes)

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LECTURESENSE V8 — UNIVERSAL VIDEO + PDF ALIGNMENT")
    print("=" * 78)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Video:", video)
    print("PDF  :", pdf)
    print("Notes:", notes if notes else "None")

    v7 = load_v7_module()
    srt = ensure_transcript(video, work, args.whisper_model)

    print("\n[4/7] Loading V4 score matrix and visual features...")
    v4_json, v4_path, b_path = run_v4(video, pdf, srt, work)
    v4 = np.load(v4_path).astype(np.float32)
    boundary = load_boundary(b_path, len(v4))

    segments = v7.get_segments(srt)
    mids = v7.segment_mids(segments)
    if len(mids) != len(v4):
        raise RuntimeError(
            f"Transcript/V4 mismatch: transcript={len(mids)}, v4={len(v4)}"
        )

    vis_times, vis_scores = visual_scores(
        v7,
        "custom_lecture",
        pdf,
        video,
        work,
    )

    print("\n[5/7] Building candidate features...")
    X, _ = v7.build_features_from_mids(
        v4=v4,
        boundary=boundary,
        mids=mids,
        visual_times=vis_times,
        visual_scores=vis_scores,
        times=mids,
        gt_slides=None,
    )
    X = v7.normalize_rows(X)

    models, decoder = load_v7_ensemble(v7)

    print("\n[6/7] Predicting slide sequence...")
    if models:
        emissions = np.mean([v7.predict(m, X) for m in models], axis=0)
        model_name = "V7 ensemble + V4 + visual"
    else:
        print("[V7 checkpoint not found] Falling back to V4 rank + visual rank.")
        model_name = "V4 + visual fallback"
        v4_rank = v7.row_rank(v4)
        # Nearest visual row for each transcript midpoint.
        vr = v7.row_rank(vis_scores)
        visual_rows = []
        for t in mids:
            idx = int(np.argmin(np.abs(vis_times - float(t))))
            visual_rows.append(vr[idx])
        visual_rank = np.asarray(visual_rows, np.float32)
        emissions = 0.80 * v4_rank + 0.20 * visual_rank
        decoder = {"max_jump": 3, "jump_penalty": 0.02, "boundary_reward": 0.12}

    emissions = add_notes_prior(emissions, pdf, notes)
    path = v7.decode_monotonic(
        emissions,
        boundaries=boundary,
        max_jump=int(decoder.get("max_jump", 3)),
        jump_penalty=float(decoder.get("jump_penalty", 0.02)),
        boundary_reward=float(decoder.get("boundary_reward", 0.12)),
    )

    probs = v7.row_softmax(emissions)
    conf = probs[np.arange(len(path)), path]

    # Rendered slides are produced by V4; use them in the report when possible.
    rendered_dir = work / "rendered_slides"
    slide_images = {}
    for p in rendered_dir.glob("*.png"):
        m = re.search(r"(\d+)", p.stem)
        if m:
            slide_images[int(m.group(1))] = p

    # Group consecutive identical slide predictions into a compact timeline.
    rows = []
    current_slide = int(path[0]) + 1 if len(path) else 1
    current_start = float(mids[0]) if len(mids) else 0.0
    current_end = current_start
    current_text = []
    current_conf = []

    def flush():
        if not current_text:
            return
        idx = min(len(current_conf), len(current_text))
        slide = current_slide
        img = slide_images.get(slide)
        preview = str(img.relative_to(out_dir)).replace("\\", "/") if img and img.exists() else ""
        rows.append({
            "start": current_start,
            "end": current_end,
            "slide": slide,
            "confidence": float(np.mean(current_conf)) if current_conf else 0.0,
            "text": clean_text(" ".join(current_text))[:800],
            "preview": preview,
        })

    for i, (p, c) in enumerate(zip(path, conf)):
        slide = int(p) + 1
        start = float(mids[i])
        end = float(mids[i + 1]) if i + 1 < len(mids) else start + 2.0
        text = ""
        seg = segments[i]
        if isinstance(seg, dict):
            text = clean_text(seg.get("text", seg.get("content", "")))
        else:
            text = clean_text(getattr(seg, "text", ""))

        if slide != current_slide:
            flush()
            current_slide = slide
            current_start = start
            current_text = []
            current_conf = []

        current_end = end
        current_text.append(text)
        current_conf.append(float(c))

    if current_text:
        flush()

    timeline_json = out_dir / "timeline.json"
    payload = {
        "version": "LectureSense-V8-Universal",
        "video": str(video),
        "pdf": str(pdf),
        "notes": str(notes) if notes else None,
        "model": model_name,
        "transcript": str(srt),
        "segments": int(len(mids)),
        "slides": int(v4.shape[1]),
        "timeline": rows,
    }
    timeline_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(rows).to_csv(out_dir / "timeline.csv", index=False)
    save_report(out_dir, video, pdf, rows, int(v4.shape[1]))

    print("\n[7/7] Output complete")
    print("Timeline JSON:", timeline_json)
    print("Timeline CSV :", out_dir / "timeline.csv")
    print("HTML report  :", out_dir / "timeline.html")
    print("Transcript   :", srt)
    print("Model        :", model_name)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="LectureSense V8 Universal: video + PDF -> timestamped slide timeline"
    )
    ap.add_argument("--mode", choices=["align"], default="align")
    ap.add_argument("--video", required=True, help="Lecture MP4/MKV/etc.")
    ap.add_argument("--pdf", required=True, help="Lecture slide PDF")
    ap.add_argument("--notes", default=None, help="Optional .txt/.md/.pdf/.docx notes")
    ap.add_argument("--output", default="runs/v8_universal", help="Output directory")
    ap.add_argument("--whisper-model", default="base", help="Whisper model: tiny/base/small/medium")
    args = ap.parse_args()
    align_mode(args)


if __name__ == "__main__":
    main()
