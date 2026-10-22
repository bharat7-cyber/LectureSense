
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

VAL = {
    "physics_intro_02": (
        "physics_intro_02.srt", "physics_intro_02.pdf",
        "physics.mp4"
    ),
    "psychology_MIT": (
        "psychology_MIT.srt", "psychology.pdf",
        "psychology_high_res.mp4"
    ),
    "reinforcement_learning_silver": (
        "reinforcement_learning_silver.srt", "reinforcement_learning.pdf",
        "reinforcement_learning_high_res.mp4"
    ),
    "visual_system_MIT": (
        "visual_system_MIT.srt", "sensory_systems.pdf",
        "sensory_systems_high_res.mp4"
    ),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--only", default=None)
    ap.add_argument("--video-weight", type=float, default=0.40)
    ap.add_argument("--video-sample-seconds", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outroot = root / "runs" / "v5_v4_export_fixed"
    outroot.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "src"))

    # Import the exact transcript and feature pipeline used by V4.
    import inference_temporal_v4 as v4
    from boundary_dp_v4 import boundary_aware_dp
    import torch

    ck = torch.load(
        root / "checkpoints" / "temporal_v4_best_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    decoder_cfg = ck.get("decoder_config", {
        "jump_penalty": 0.05,
        "max_forward_jump": 1,
        "max_duration_seconds": 120.0,
        "duration_penalty": 0.001,
        "max_run_seconds": 1200.0,
        "boundary_reward": 1.0,
        "boundary_stay_penalty": 0.50,
        "transition_logits_weight": 0.35,
        "premature_transition_penalty": 0.50,
        "boundary_threshold": 0.20,
    })

    names = [args.only] if args.only else list(VAL)

    for name in names:
        srt, pdf, video = VAL[name]
        od = outroot / name
        od.mkdir(parents=True, exist_ok=True)

        transcript_path = (
            root / "data" / "MaViLS" / "data" / "audioscripts" / srt
        )
        pdf_path = (
            root / "data" / "MaViLS" / "data" / "lectures" / pdf
        )
        video_path = (
            Path(r"C:\Users\Bharat\Downloads\archive\video") / video
        )

        print("\n" + "=" * 78)
        print("V4 FIXED EXPORT:", name)
        print("=" * 78)

        # Use V4's own transcript parser. This is the critical fix.
        transcript_segments = v4.base.read_mavils_srt(
        str(transcript_path)
        )

        print("V4 transcript segments:", len(transcript_segments))

        # Rather than executing V4 a second time, invoke the official V4
        # inference once so that all matrices are generated exactly as before.
        timeline = od / "timeline.json"
        cmd = [
            sys.executable,
            str(root / "src" / "inference_temporal_v4.py"),
            "--transcript", str(transcript_path),
            "--slides", str(pdf_path),
            "--video", str(video_path),
            "--checkpoint",
            str(root / "checkpoints" / "temporal_v4_best_model.pt"),
            "--out", str(timeline),
            "--render-dir", str(od / "rendered_slides"),
            "--video-weight", str(args.video_weight),
            "--video-sample-seconds", str(args.video_sample_seconds),
        ]

        rc = subprocess.run(cmd, cwd=root).returncode
        if rc != 0:
            raise RuntimeError(f"V4 inference failed for {name}")

        score = np.load(od / "v4_final_score_matrix.npy")
        boundary = np.load(od / "v4_boundary_probability.npy")
        next_logits = np.load(od / "v4_next_slide_logits.npy")

        if score.shape[0] != len(transcript_segments):
            raise RuntimeError(
                f"Still mismatched: score rows={score.shape[0]}, "
                f"V4 transcript segments={len(transcript_segments)}"
            )

        starts = [float(x["start"]) for x in transcript_segments]
        ends = [float(x["end"]) for x in transcript_segments]

        alignment = boundary_aware_dp(
            score,
            boundary,
            next_logits,
            starts,
            ends,
            **decoder_cfg,
        )

        if len(alignment) != len(transcript_segments):
            raise RuntimeError("V4 alignment length mismatch")

        rows = []

        for i, (seg, pred) in enumerate(
            zip(transcript_segments, alignment)
        ):
            vals = np.asarray(score[i])
            order = np.argsort(vals)[::-1]

            best_idx = int(order[0])
            best = float(vals[best_idx])
            second = float(vals[order[1]]) if len(order) > 1 else best

            # Softmax confidence over the available slide candidates.
            shifted = vals - np.max(vals)
            probs = np.exp(shifted)
            probs /= max(np.sum(probs), 1e-12)

            rows.append({
                "segment_index": i,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", ""),
                "v4_pred": int(pred) + 1,
                "v4_raw_top1": best_idx + 1,
                "v4_confidence": float(probs[best_idx]),
                "v4_margin": best - second,
                "boundary_probability": float(boundary[i]),
                "next_slide_pred": int(np.argmax(next_logits[i])) + 1,
            })

        df = pd.DataFrame(rows)
        csv_path = od / "v4_segment_predictions.csv"
        df.to_csv(csv_path, index=False)

        (od / "decoder_config.json").write_text(
            json.dumps(decoder_cfg, indent=2),
            encoding="utf-8",
        )

        print("Saved:", csv_path)
        print("Rows:", len(df))

    print("\n" + "=" * 78)
    print("V4 FIXED TIMESTAMP EXPORT COMPLETE")
    print("=" * 78)
    print("VALIDATION ONLY — TEST SET NOT USED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
