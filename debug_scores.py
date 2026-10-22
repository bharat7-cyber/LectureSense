import sys
import numpy as np

sys.path.insert(0, "src")

from inference import read_mavils_srt

transcript_path = r"data\MaViLS\data\audioscripts\computer_vision_2_2_high_res.srt"
score_path = r"runs\session\score_matrix.npy"
alignment_path = r"runs\session\alignment.npy"

segments = read_mavils_srt(transcript_path)
scores = np.load(score_path)
alignment = np.load(alignment_path)

print("TOTAL SEGMENTS:", len(segments))
print()
print("--- REGION 430s-610s ---")

for i, seg in enumerate(segments):
    start = seg["start"]
    end = seg["end"]

    if 430 <= start <= 610:
        ranked = np.argsort(scores[i])[-5:][::-1]

        top5 = []
        for j in ranked:
            top5.append(
                (int(j) + 1, round(float(scores[i, j]), 3))
            )

        print(
            f"{i:3d} | "
            f"{start:7.2f}-{end:7.2f}s | "
            f"Predicted Slide {int(alignment[i]) + 1:2d} | "
            f"TOP5: {top5}"
        )
