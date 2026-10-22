"""
Turn MaViLS ground-truth alignment into labeled (sentence, slide) pairs for
training the Fusion MLP.

Positive pairs: (transcript sentence, slide) where MaViLS ground truth says
that slide is what's being discussed at that point.

Negative pairs: the same sentence paired with a *different* slide from the
same lecture (non-matching slides only — negatives never cross lectures,
since that would make the task trivially easy and wouldn't reflect real
alignment difficulty).

Unlabeled points (ground truth == -1, meaning "no uniquely identifiable
slide") are excluded from the training pairs but kept in a separate table
for qualitative evaluation later, per the proposal.

Output: data/pairs.csv with columns
    lecture_id, sentence_idx, sentence_text, slide_id, label, split

Run:
    python src/build_pairs.py --neg-ratio 3
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAVILS_DIR = DATA_DIR / "MaViLS"
OUT_PATH = DATA_DIR / "pairs.csv"
UNLABELED_PATH = DATA_DIR / "unlabeled_points.csv"

# Lecture-wise split: 12 train / 4 val / 4 test, matching the proposal.
# Fill in with actual lecture IDs once download_data.py confirms the
# repo's naming convention — placeholders shown here follow the paper's
# stated source institutions (MIT OCW / Tübingen / DeepMind).
SPLIT_PLACEHOLDER = {
    "train": [f"lecture_{i:02d}" for i in range(1, 13)],
    "val": [f"lecture_{i:02d}" for i in range(13, 17)],
    "test": [f"lecture_{i:02d}" for i in range(17, 21)],
}


def load_ground_truth(lecture_id: str):
    """Load one lecture's ground-truth alignment.

    Expected shape per MaViLS: a list of records, each mapping a transcript
    sentence (with its index/timestamp) to a slide id, or -1 if unlabeled.
    This function is intentionally defensive: real field names must be
    confirmed against the actual downloaded files (run download_data.py's
    inventory() first and inspect one JSON/CSV by hand) before trusting
    this parses correctly. Adjust the key names below to match reality.
    """
    candidates = list((MAVILS_DIR).rglob(f"*{lecture_id}*gt*.json")) + \
                 list((MAVILS_DIR).rglob(f"*{lecture_id}*label*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No ground-truth file found for {lecture_id}. "
            f"Run download_data.py first and check the printed inventory "
            f"to find the real ground-truth filename pattern, then update "
            f"the glob patterns in load_ground_truth()."
        )
    with open(candidates[0]) as f:
        return json.load(f)


def build_pairs_for_lecture(lecture_id: str, neg_ratio: int, rng: random.Random):
    gt = load_ground_truth(lecture_id)

    # gt assumed to be: [{"sentence_idx": int, "text": str, "slide_id": int|-1}, ...]
    labeled = [r for r in gt if r.get("slide_id", -1) != -1]
    unlabeled = [r for r in gt if r.get("slide_id", -1) == -1]

    all_slide_ids = sorted({r["slide_id"] for r in labeled})

    rows = []
    for r in labeled:
        sent_idx = r["sentence_idx"]
        sent_text = r["text"]
        true_slide = r["slide_id"]

        # positive
        rows.append({
            "lecture_id": lecture_id, "sentence_idx": sent_idx,
            "sentence_text": sent_text, "slide_id": true_slide, "label": 1,
        })

        # negatives: sample non-matching slides from the SAME lecture only
        candidates = [s for s in all_slide_ids if s != true_slide]
        n_neg = min(neg_ratio, len(candidates))
        for neg_slide in rng.sample(candidates, n_neg):
            rows.append({
                "lecture_id": lecture_id, "sentence_idx": sent_idx,
                "sentence_text": sent_text, "slide_id": neg_slide, "label": 0,
            })

    unlabeled_rows = [
        {"lecture_id": lecture_id, "sentence_idx": r["sentence_idx"],
         "sentence_text": r["text"]}
        for r in unlabeled
    ]
    return rows, unlabeled_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=int, default=3,
                     help="negative pairs sampled per positive pair")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_lectures = (SPLIT_PLACEHOLDER["train"] + SPLIT_PLACEHOLDER["val"]
                     + SPLIT_PLACEHOLDER["test"])
    split_of = {}
    for split, ids in SPLIT_PLACEHOLDER.items():
        for lid in ids:
            split_of[lid] = split

    all_rows, all_unlabeled = [], []
    for lecture_id in all_lectures:
        try:
            rows, unlabeled = build_pairs_for_lecture(lecture_id, args.neg_ratio, rng)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            continue
        for r in rows:
            r["split"] = split_of[lecture_id]
        all_rows.extend(rows)
        all_unlabeled.extend(unlabeled)

    if not all_rows:
        print(
            "[error] No pairs built. This almost certainly means the "
            "ground-truth filename/field-name assumptions in "
            "load_ground_truth() don't match the real MaViLS repo layout. "
            "Run download_data.py, open one ground-truth file by hand, and "
            "fix the glob pattern + field names above."
        )
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    pd.DataFrame(all_unlabeled).to_csv(UNLABELED_PATH, index=False)

    print(f"\nWrote {len(df)} pairs to {OUT_PATH}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"\nWrote {len(all_unlabeled)} unlabeled points to {UNLABELED_PATH}")
    print(
        "\nSanity check: paper reports 12,830 ground-truth points total. "
        f"Positive pairs here = {len(df[df.label==1])}. If that's far off "
        "from 12,830, re-check load_ground_truth()'s parsing."
    )


if __name__ == "__main__":
    main()
