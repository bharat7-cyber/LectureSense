from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OCR_INDEX = ROOT / "data" / "ocr_embedding_index.csv"
OCR_TEXT = ROOT / "data" / "ocr_text.csv"

# OCR folder name -> canonical feature lecture ID
LECTURE_MAP = {
    "ML_for_health_care_MIT": "ML_for_health_MIT",
    "RL_deepmind": "RL_deepmind",
    "cities&climate": "climate_and_cities",
    "cities&decarbonization": "cities_and_decarbonization",
    "climate_science_policy_MIT": "climate_science_policy_MIT2",
    "cognitive_robotics_MIT": "cognitive_robotics_MIT",
    "computer_vision_2_2_geiger": "computer_vision_2_2",
    "creating_breakthrough_products": "creating_breakthrough_products_MIT",
    "cryptocurrency": "cryptocurrency_MIT",
    "deeplearning_goodfellow": "deeplearning",
    "image_processing": "image_processing",
    "numerics": "numerics",
    "phonetics": "phonetics",
    "physics_intro_02": "physics",
    "psychology": "psychology",
    "reinforcement_learning": "reinforcement_learning",
    "sensory_systems": "sensory_systems",
    "short_range_mit": "short_range",
    "solar_resource": "solar_resource",
    "team_dynamics_game_design_mit": "team_dynamics_game_design_mit",
    "theory_of_computation": "theory_of_computation",
}


def normalize(df):
    original = df["lecture_id"].copy()

    df["lecture_id"] = df["lecture_id"].map(
        lambda x: LECTURE_MAP.get(x, x)
    )

    changed = (original != df["lecture_id"]).sum()

    print(f"Normalized {changed}/{len(df)} rows")

    return df


print("Fixing OCR embedding index...")

index_df = pd.read_csv(OCR_INDEX)
index_df = normalize(index_df)
index_df.to_csv(OCR_INDEX, index=False)

print("Saved:", OCR_INDEX)


print("\nFixing OCR text metadata...")

text_df = pd.read_csv(OCR_TEXT)
text_df = normalize(text_df)
text_df.to_csv(OCR_TEXT, index=False)

print("Saved:", OCR_TEXT)


print("\nCurrent lecture IDs:")

for lecture in sorted(index_df["lecture_id"].unique()):
    print(" ", lecture)

print("\nSUCCESS")