
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

LECTURES = [
    "physics_intro_02",
    "psychology_MIT",
    "reinforcement_learning_silver",
    "visual_system_MIT",
]

RESULT_RE = re.compile(r"Result:\s*(\d+)\s*/\s*(\d+)\s*=\s*([0-9.]+)%")
RUNTIME_RE = re.compile(r"Runtime:\s*([0-9.]+)s")
ERROR_RE = re.compile(r"^\s*([0-9.]+)s\s*\|\s*GT\s*(-?\d+)\s*\|\s*Pred\s*(-?\d+)", re.M)

def run_one(root, lecture, sample_seconds, max_jump):
    cmd = [
        sys.executable,
        str(root / "run_visual_first_validation.py"),
        "--root", str(root),
        "--only", lecture,
        "--sample-seconds", str(sample_seconds),
        "--max-jump", str(max_jump),
    ]
    p = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
    )
    output = p.stdout + ("\n" + p.stderr if p.stderr else "")
    m = RESULT_RE.search(output)
    runtime = RUNTIME_RE.search(output)
    errors = ERROR_RE.findall(output)

    row = {
        "lecture": lecture,
        "status": "OK" if p.returncode == 0 and m else "FAILED",
        "correct": int(m.group(1)) if m else "",
        "total": int(m.group(2)) if m else "",
        "accuracy_percent": float(m.group(3)) if m else "",
        "runtime_seconds": float(runtime.group(1)) if runtime else "",
        "first_error_count": len(errors),
        "return_code": p.returncode,
    }
    return row, output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample-seconds", type=float, default=1.0)
    ap.add_argument("--max-jump", type=int, default=3)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lectures = [args.only] if args.only else LECTURES
    outdir = root / "runs" / "v5_diagnostics_fixed"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LECTURESENSE V5 — FIXED VISUAL DIAGNOSTIC")
    print("=" * 78)
    print("This diagnostic uses the already-validated Visual-First evaluator.")
    print("It does NOT modify V4 and does NOT load test lectures.")
    print("-" * 78)

    rows = []
    for lecture in lectures:
        print(f"\n[{lecture}]")
        row, raw = run_one(root, lecture, args.sample_seconds, args.max_jump)
        rows.append(row)
        (outdir / f"{lecture}.log").write_text(raw, encoding="utf-8")

        if row["status"] == "OK":
            print(
                f"Visual accuracy: {row['correct']}/{row['total']} "
                f"= {row['accuracy_percent']:.4f}%"
            )
            print(f"Runtime: {row['runtime_seconds']:.1f}s")
        else:
            print("FAILED — inspect the saved log.")
            print(raw[-2000:])

    csv_path = outdir / "visual_validation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["lecture"])
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["status"] == "OK"]
    correct = sum(r["correct"] for r in valid)
    total = sum(r["total"] for r in valid)

    print("\n" + "=" * 78)
    print("V5 FIXED DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"Lectures evaluated: {len(valid)}/{len(rows)}")
    if total:
        print(f"Aggregate visual accuracy: {correct}/{total} = {100.0*correct/total:.4f}%")
    print(f"Saved: {csv_path}")
    print("TEST SET WAS NOT USED.")
    print("=" * 78)

if __name__ == "__main__":
    main()
