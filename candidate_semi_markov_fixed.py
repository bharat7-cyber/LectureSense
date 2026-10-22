from __future__ import annotations

import numpy as np


def _duration_cost(duration_seconds: float, max_duration_seconds: float, duration_penalty: float) -> float:
    """Quadratic penalty for staying on a slide beyond the preferred duration."""
    excess = max(0.0, float(duration_seconds) - float(max_duration_seconds))
    if excess <= 0.0:
        return 0.0
    maximum = max(1e-6, float(max_duration_seconds))
    return float(duration_penalty) * (excess / maximum) ** 2 * maximum


def semi_markov_decode(
    emission,
    boundary_prob,
    movement_log_probs,
    segment_starts,
    segment_ends,
    max_forward_jump=1,
    jump_penalty=0.0,
    boundary_reward=0.25,
    boundary_stay_penalty=0.15,
    movement_weight=0.30,
    duration_penalty=0.001,
    max_duration_seconds=150.0,
    max_run_seconds=1200.0,
):
    """Decode a monotonic slide sequence with semi-Markov segment durations.

    Args:
        emission: [N, S] candidate slide scores; larger is better.
        boundary_prob: [N] probability that a new slide starts at each row.
        movement_log_probs: [N, 5] log probabilities for relative movement
            {-2, -1, 0, +1, +2}, as produced by log-softmax.
        segment_starts/segment_ends: aligned start/end timestamps in seconds.

    The path starts at slide 0 and may only advance, never move backwards.
    The held-out test labels are not an input to this decoder.
    """
    scores = np.asarray(emission, dtype=np.float64)
    boundary = np.asarray(boundary_prob, dtype=np.float64).reshape(-1)
    movement = np.asarray(movement_log_probs, dtype=np.float64)
    starts = np.asarray(segment_starts, dtype=np.float64).reshape(-1)
    ends = np.asarray(segment_ends, dtype=np.float64).reshape(-1)

    if scores.ndim != 2:
        raise ValueError("emission must be a 2D [segments, slides] matrix")
    n, s = scores.shape
    if n == 0 or s == 0:
        return []
    if len(boundary) != n or len(starts) != n or len(ends) != n:
        raise ValueError("timestamps and boundary probabilities must match emission rows")
    if movement.shape != (n, 5):
        raise ValueError(f"movement_log_probs must have shape ({n}, 5), got {movement.shape}")
    if np.any(np.diff(starts) < 0):
        raise ValueError("segment starts must be non-decreasing")
    if np.any(ends < starts):
        raise ValueError("segment end precedes start")

    max_forward_jump = max(1, min(int(max_forward_jump), s - 1 if s > 1 else 1))
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    max_run_seconds = max(max_duration_seconds, float(max_run_seconds))
    duration_penalty = max(0.0, float(duration_penalty))
    jump_penalty = max(0.0, float(jump_penalty))
    boundary_stay_penalty = max(0.0, float(boundary_stay_penalty))
    boundary = np.clip(boundary, 1e-5, 1.0 - 1e-5)

    # Prefix sums let each proposed slide run be scored in O(S).
    score_prefix = np.zeros((n + 1, s), dtype=np.float64)
    score_prefix[1:] = np.cumsum(scores, axis=0)
    boundary_prefix = np.zeros(n + 1, dtype=np.float64)
    boundary_prefix[1:] = np.cumsum(boundary)

    dp = np.full((n, s), -np.inf, dtype=np.float64)
    back_start = np.full((n, s), -1, dtype=np.int32)
    back_prev = np.full((n, s), -1, dtype=np.int32)

    target_slides = np.arange(s, dtype=np.int32)
    # The first legal start k for each end row i is found by timestamp, avoiding
    # scanning possible runs that would exceed max_run_seconds.
    for i in range(n):
        first_k = int(np.searchsorted(starts, ends[i] - max_run_seconds, side="left"))
        ks = np.arange(first_k, i + 1, dtype=np.int32)
        if len(ks) == 0:
            continue

        run_scores = score_prefix[i + 1][None, :] - score_prefix[ks]
        durations = np.maximum(0.0, ends[i] - starts[ks])
        duration_costs = np.asarray([
            _duration_cost(d, max_duration_seconds, duration_penalty) for d in durations
        ], dtype=np.float64)
        stay_costs = boundary_stay_penalty * (
            boundary_prefix[i + 1] - boundary_prefix[ks + 1]
        )
        candidates = run_scores - duration_costs[:, None] - stay_costs[:, None]
        prev_slides = np.full((len(ks), s), -1, dtype=np.int32)

        # A sequence must begin on slide 0; only slide 0 can use k=0.
        k_zero_rows = np.flatnonzero(ks == 0)
        if len(k_zero_rows):
            rz = int(k_zero_rows[0])
            candidates[rz, 1:] = -np.inf
            prev_slides[rz, :] = -1

        # For k>0, the preceding run must finish on a lower slide index.
        # Vectorize across candidate run starts and target slides.
        nonzero_rows = np.flatnonzero(ks > 0)
        if len(nonzero_rows):
            k_values = ks[nonzero_rows]
            best_prev = np.full((len(k_values), s), -np.inf, dtype=np.float64)
            best_prev_slide = np.full((len(k_values), s), -1, dtype=np.int32)
            boundary_term = float(boundary_reward) * (2.0 * boundary[k_values] - 1.0)
            for delta in range(1, max_forward_jump + 1):
                if delta >= s:
                    break
                destinations = np.arange(delta, s, dtype=np.int32)
                source_slides = destinations - delta
                move_idx = min(4, delta + 2)
                trans = (
                    boundary_term[:, None]
                    + float(movement_weight) * movement[k_values, move_idx][:, None]
                    - jump_penalty * max(0, delta - 1)
                )
                prior = dp[k_values[:, None] - 1, source_slides[None, :]]
                values = prior + trans
                current = best_prev[:, destinations]
                improved = values > current
                best_prev[:, destinations] = np.where(improved, values, current)
                prev_current = best_prev_slide[:, destinations]
                best_prev_slide[:, destinations] = np.where(improved, source_slides[None, :], prev_current)

            candidates[nonzero_rows, :] = (
                run_scores[nonzero_rows, :]
                - duration_costs[nonzero_rows, None]
                - stay_costs[nonzero_rows, None]
                + best_prev
            )
            prev_slides[nonzero_rows, :] = best_prev_slide

        best_k_local = np.argmax(candidates, axis=0)
        dp[i, :] = candidates[best_k_local, target_slides]
        back_start[i, :] = ks[best_k_local]
        back_prev[i, :] = prev_slides[best_k_local, target_slides]

    final_slide = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final_slide]):
        raise RuntimeError("No valid monotonic semi-Markov path found")

    path = np.zeros(n, dtype=np.int64)
    end_idx = n - 1
    slide = final_slide
    while end_idx >= 0:
        k = int(back_start[end_idx, slide])
        if k < 0:
            raise RuntimeError("Invalid backpointer while reconstructing semi-Markov path")
        path[k:end_idx + 1] = slide
        if k == 0:
            break
        slide = int(back_prev[end_idx, slide])
        if slide < 0 or slide >= s:
            raise RuntimeError("Invalid previous slide in semi-Markov backtracking")
        end_idx = k - 1

    if path[0] != 0 or np.any(np.diff(path) < 0):
        raise RuntimeError("Decoded path violates monotonic slide-order constraints")
    return path.tolist()
