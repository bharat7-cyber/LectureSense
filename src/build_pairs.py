import argparse
import random
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAVILS_DIR = DATA_DIR / "MaViLS"
GT_DIR = MAVILS_DIR / "data" / "ground_truth_files"
OUT_PATH = DATA_DIR / "pairs.csv"
UNLABELED_PATH = DATA_DIR / "unlabeled_points.csv"

LECTURES = {
    "cities_and_decarbonization": ("cities_and_decarbonization", "train"),
    "climate_and_cities": ("climate_and_cities", "train"),
    "climate_science_policy_MIT2": ("climate_science_policy_MIT", "train"),
    "cognitive_robotics_MIT": ("cognitive_robotics_MIT", "train"),
    "computer_vision_2_2": ("computer_vision_2_2", "train"),
    "creating_breakthrough_products_MIT": ("creating_breakthrough_products", "train"),
    "cryptocurrency_MIT": ("cryptocurrency", "train"),
    "deeplearning": ("deeplearning_goodfellow", "train"),
    "image_processing": ("image_processing_high_res", "train"),
    "ML_for_health_MIT": ("ML_for_health_MIT", "train"),
    "numerics": ("numerics_hennig", "train"),
    "phonetics": ("phonetics", "train"),

    "physics": ("physics_intro_02", "val"),
    "psychology": ("psychology_MIT", "val"),
    "reinforcement_learning": ("reinforcement_learning_silver", "val"),
    "sensory_systems": ("visual_system_MIT", "val"),

    "short_range": ("short_range_mit", "test"),
    "solar_resource": ("solar_resource", "test"),
    "team_dynamics": ("team_dynamics_game_design_MIT", "test"),
    "theory_of_computation": ("theory_of_computation", "test"),
}

def load_gt(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    # MaViLS uses both "Key" and "Time" in different files.
    time_col = None
    for candidate in ["Key", "Time", "Timestamp", "timestamp"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        raise ValueError(
            f"No timestamp column found in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )

    if "Slidenumber" not in df.columns or "Value" not in df.columns:
        raise ValueError(
            f"Missing required columns in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={
        time_col: "timestamp",
        "Slidenumber": "slide_id",
        "Value": "sentence_text"
    })

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["slide_id"] = pd.to_numeric(df["slide_id"], errors="coerce")
    df["sentence_text"] = df["sentence_text"].fillna("").astype(str)

    df = df.dropna(subset=["timestamp", "slide_id"])
    df["sentence_idx"] = range(len(df))

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_rows = []
    unlabeled = []

    for key, (lecture_name, split) in LECTURES.items():

        candidates = list(GT_DIR.glob(f"ground_truth_{key}*.xlsx"))

        if not candidates:
            print(f"[WARN] No GT found for {key}")
            continue

        gt_file = candidates[0]
        print(f"[OK] {gt_file.name} -> {split}")

        df = load_gt(gt_file)

        labeled = df[df.slide_id >= 0]
        unknown = df[df.slide_id < 0]

        slides = sorted(
            labeled.slide_id.astype(int).unique().tolist()
        )

        for _, row in labeled.iterrows():

            true_slide = int(row.slide_id)

            all_rows.append({
                "lecture_id": lecture_name,
                "sentence_idx": int(row.sentence_idx),
                "sentence_text": row.sentence_text,
                "timestamp": float(row.timestamp),
                "slide_id": true_slide,
                "label": 1,
                "split": split
            })

            negatives = [s for s in slides if s != true_slide]
            n = min(args.neg_ratio, len(negatives))

            for neg in rng.sample(negatives, n):

                all_rows.append({
                    "lecture_id": lecture_name,
                    "sentence_idx": int(row.sentence_idx),
                    "sentence_text": row.sentence_text,
                    "timestamp": float(row.timestamp),
                    "slide_id": neg,
                    "label": 0,
                    "split": split
                })

        for _, row in unknown.iterrows():

            unlabeled.append({
                "lecture_id": lecture_name,
                "sentence_idx": int(row.sentence_idx),
                "sentence_text": row.sentence_text,
                "timestamp": float(row.timestamp)
            })

    if not all_rows:
        raise RuntimeError("No training pairs were created.")

    out = pd.DataFrame(all_rows)

    out.to_csv(OUT_PATH, index=False)

    pd.DataFrame(unlabeled).to_csv(
        UNLABELED_PATH,
        index=False
    )

    print()
    print(f"Created: {OUT_PATH}")
    print(f"Total pairs: {len(out)}")
    print()
    print(out.groupby(["split", "label"]).size())
    print()
    print(f"Unlabeled points: {len(unlabeled)}")


if __name__ == "__main__":
    main()
