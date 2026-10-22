import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

V10_PATH = ROOT / 'LectureSense_V10_BoundaryFirst_Alignment.py'
if not V10_PATH.exists():
    raise FileNotFoundError(f'Missing {V10_PATH}')

import importlib.util
spec = importlib.util.spec_from_file_location('lecturesense_v10_base', V10_PATH)
v10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v10)

FAST_OUT = v10.OUT / 'fast_cached'
PREP_ROOT = FAST_OUT / 'prepared'
SWEEP_CSV = FAST_OUT / 'validation_sweep.csv'
BEST_JSON = FAST_OUT / 'best_config.json'
CONFIG_PATH = v10.CONFIG_PATH


def prepare_one(lecture: str):
    cache = PREP_ROOT / lecture
    cache.mkdir(parents=True, exist_ok=True)

    srt, pdf, gt, video = v10.v7.asset_paths(lecture)
    if any(x is None for x in (srt, pdf, gt, video)):
        print(f'[PREP SKIP] {lecture}: missing exact asset')
        return None

    v4_cache = v10.ensure_v4_cache(srt, pdf, video, cache / 'v4_cache')
    mids = v10.read_transcript_mids(srt)

    visual_npz = v4_cache / 'visual_scores_v10.npz'
    if visual_npz.exists():
        z = np.load(visual_npz)
        visual_times = np.asarray(z['times'], dtype=np.float32)
        visual_scores = np.asarray(z['scores'], dtype=np.float32)
    else:
        try:
            visual_times, visual_scores = v10.v7.get_visual_scores(lecture, pdf, video)
        except Exception:
            visual_times = mids.copy()
            visual_scores = np.load(v4_cache / 'v4_final_score_matrix.npy').astype(np.float32)
        np.savez_compressed(visual_npz, times=visual_times, scores=visual_scores)

    v4 = np.load(v4_cache / 'v4_final_score_matrix.npy').astype(np.float32)
    boundary_path = v4_cache / 'v4_boundary_probability.npy'
    v4_boundary = np.load(boundary_path).astype(np.float32) if boundary_path.exists() else np.zeros(v4.shape[0], dtype=np.float32)

    boundary_cache = cache / 'boundary_signal.npz'
    if boundary_cache.exists():
        z = np.load(boundary_cache)
        signal = np.asarray(z['signal'], dtype=np.float32)
        duration = float(z['duration'])
    else:
        print(f'[PREP] Reading video once for {lecture} boundary signal...')
        video_times, video_diff, duration = v10.detect_video_boundaries(video, sample_seconds=1.0)
        signal = v10.build_boundary_signal(mids, v4_boundary, v4, video_times, video_diff)
        np.savez_compressed(boundary_cache, signal=signal, duration=np.float32(duration))

    data = {
        'lecture': lecture,
        'srt': srt,
        'pdf': pdf,
        'gt': gt,
        'video': video,
        'v4': v4,
        'mids': mids,
        'visual_times': visual_times,
        'visual_scores': visual_scores,
        'signal': signal,
        'duration': duration,
    }
    print(f'[PREP DONE] {lecture}: transcript={len(mids)} slides={v4.shape[1]} visual_samples={len(visual_times)}')
    return data


def eval_one(data, cfg):
    segments = v10.build_segments(float(data['mids'][0]), max(float(data['mids'][-1]), data['duration']),
                                   v10.choose_boundaries(data['mids'], data['signal'], cfg), cfg)
    emissions = v10.make_segment_emissions(segments, data['mids'], data['v4'],
                                           data['visual_times'], data['visual_scores'], cfg)
    path = v10.monotonic_segment_dp(emissions, segments, cfg.max_slide_jump, cfg.jump_penalty)
    result = {'lecture': data['lecture'], 'segments': [
        {'start': float(a), 'end': float(b), 'slide': int(s)+1, 'duration': float(b-a), 'score': float(emissions[i, s])}
        for i, ((a,b),s) in enumerate(zip(segments, path))
    ]}
    acc, correct, total = v10.accuracy_against_gt(result, data['gt'])
    return acc, correct, total, len(segments)


def tune():
    FAST_OUT.mkdir(parents=True, exist_ok=True)
    lectures = list(v10.v8.VAL)
    prepared = []
    for lecture in lectures:
        d = prepare_one(lecture)
        if d is not None:
            prepared.append(d)

    if not prepared:
        raise RuntimeError('No validation lectures prepared.')

    completed = set()
    rows = []
    if SWEEP_CSV.exists():
        try:
            old = pd.read_csv(SWEEP_CSV)
            for _, r in old.iterrows():
                completed.add((int(r.boundary_percentile), float(r.min_gap_seconds), int(r.max_slide_jump)))
                rows.append(r.to_dict())
            print(f'[RESUME] Existing sweep rows: {len(rows)}')
        except Exception:
            pass

    configs = []
    for bp in (76, 80, 84, 88, 92):
        for gap in (3.0, 4.0, 5.0, 6.0):
            for mj in (1, 2, 3):
                configs.append((bp, gap, mj))

    total_cfg = len(configs)
    for idx, (bp, gap, mj) in enumerate(configs, 1):
        key = (bp, gap, mj)
        if key in completed:
            print(f'[SKIP CACHED] {idx}/{total_cfg}: bp={bp} gap={gap} max_jump={mj}')
            continue

        cfg = v10.BoundaryConfig(boundary_percentile=bp, min_gap_seconds=gap, max_slide_jump=mj,
                                 segment_score_v4_weight=0.65, segment_score_visual_weight=0.35,
                                 jump_penalty=0.035)
        accs = []
        correct_sum = 0
        total_sum = 0
        for d in prepared:
            acc, correct, total, segs = eval_one(d, cfg)
            accs.append(acc); correct_sum += correct; total_sum += total
        macro = float(np.mean(accs))
        weighted = float(correct_sum / max(1, total_sum))
        row = {'boundary_percentile': bp, 'min_gap_seconds': gap, 'max_slide_jump': mj,
               'macro_accuracy': macro, 'weighted_accuracy': weighted}
        rows.append(row)
        pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False)
        print(f'[FAST SWEEP {idx}/{total_cfg}] boundary_pct={bp} gap={gap} max_jump={mj} macro={macro*100:.2f}% weighted={weighted*100:.2f}%')

    df = pd.DataFrame(rows)
    df = df.sort_values(['macro_accuracy','weighted_accuracy'], ascending=False)
    best = df.iloc[0]
    cfg = v10.BoundaryConfig(boundary_percentile=int(best.boundary_percentile),
                             min_gap_seconds=float(best.min_gap_seconds),
                             max_slide_jump=int(best.max_slide_jump),
                             segment_score_v4_weight=0.65,
                             segment_score_visual_weight=0.35,
                             jump_penalty=0.035)

    payload = {
        'version': 'LectureSense-V10-Fast-Cached',
        'config': cfg.__dict__,
        'validation_lectures': lectures,
        'test_lectures': list(v10.v8.TEST),
        'test_used_for_model_selection': False,
        'backbone': 'temporal_v4_best_model.pt',
        'sweep_rows': int(len(df)),
        'sweep_csv': str(SWEEP_CSV),
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    BEST_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('\n' + '='*78)
    print('FAST V10 TUNING COMPLETE')
    print('='*78)
    print('Best config:', cfg.__dict__)
    print('Validation macro:', f"{float(best.macro_accuracy)*100:.2f}%")
    print('Validation weighted:', f"{float(best.weighted_accuracy)*100:.2f}%")
    print('Sweep CSV:', SWEEP_CSV)
    print('Config:', CONFIG_PATH)
    return cfg


def validate_saved():
    cfg = v10.load_cfg()
    df = v10.validate(list(v10.v8.VAL), cfg)
    if not df.empty:
        print('\nValidation macro:', f"{float(df.accuracy.mean())*100:.2f}%")
        print(df.to_string(index=False))


def test_saved():
    cfg = v10.load_cfg()
    df = v10.test(list(v10.v8.TEST), cfg)
    if not df.empty:
        print('\nHeld-out test macro:', f"{float(df.accuracy.mean())*100:.2f}%")
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['tune','validate','test'], default='tune')
    args = ap.parse_args()
    if args.mode == 'tune':
        tune()
    elif args.mode == 'validate':
        validate_saved()
    else:
        test_saved()

if __name__ == '__main__':
    main()
