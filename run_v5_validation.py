import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

VAL_LECTURES = [
    "physics_intro_02",
    "psychology_MIT",
    "reinforcement_learning_silver",
]

FEATURES = [
    "v4_score", "v4_margin", "boundary_probability",
    "visual_score", "visual_margin",
    "score_gap_visual_v4", "margin_gap_visual_v4",
    "prediction_agree", "prediction_distance",
    "v4_pred_stable", "visual_pred_stable",
]

class Gate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    ckpt_path = root / "checkpoints" / "v5_gate" / "v5_gate_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("Train the gate first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(FEATURES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Gate(len(FEATURES)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    parts = []
    for lecture in VAL_LECTURES:
        p = root / "runs" / "v5_real_gate_analysis" / lecture / "timestamp_comparison.csv"
        if not p.exists():
            print("Missing:", p)
            continue
        df = pd.read_csv(p)
        df["lecture"] = lecture
        parts.append(df)

    if not parts:
        raise RuntimeError("Run the real V5 gate analysis first.")

    df = pd.concat(parts, ignore_index=True)
    X = scaler.transform(df[FEATURES].to_numpy(float)).astype("float32")
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X, device=device))).cpu().numpy()

    # Threshold is intentionally exposed for validation rather than fixed
    # from training. Sweep only on the validation set for reporting.
    best = None
    for th in np.linspace(0.05, 0.95, 19):
        pred = np.where(prob >= th, df["visual_pred"], df["v4_pred"])
        acc = float(np.mean(pred == df["gt_slide"]))
        if best is None or acc > best["accuracy"]:
            best = {"threshold": float(th), "accuracy": acc}

    df["gate_probability_visual"] = prob
    df["gated_pred"] = np.where(
        prob >= best["threshold"], df["visual_pred"], df["v4_pred"]
    )

    out = root / "runs" / "v5_gate_validation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "v5_validation_predictions.csv", index=False)

    summary = {
        "rows": len(df),
        "v4_accuracy": float(np.mean(df["v4_pred"] == df["gt_slide"])),
        "visual_accuracy": float(np.mean(df["visual_pred"] == df["gt_slide"])),
        "gated_accuracy": best["accuracy"],
        "selected_threshold": best["threshold"],
        "lecture_results": {},
    }

    for lecture, g in df.groupby("lecture"):
        summary["lecture_results"][lecture] = {
            "v4_accuracy": float(np.mean(g["v4_pred"] == g["gt_slide"])),
            "visual_accuracy": float(np.mean(g["visual_pred"] == g["gt_slide"])),
            "gated_accuracy": float(np.mean(g["gated_pred"] == g["gt_slide"])),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
