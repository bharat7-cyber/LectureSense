from __future__ import annotations

import numpy as np


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _row_zscore(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-6)


def _normalize_signal(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x
    lo = float(np.percentile(x, 10.0))
    hi = float(np.percentile(x, 90.0))
    if hi - lo < 1e-8:
        lo = float(x.min())
        hi = float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _smooth_signal(x, width=3):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) <= 2 or width <= 1:
        return x.copy()
    width = int(width)
    if width % 2 == 0:
        width += 1
    pad = width // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(xp, kernel, mode="valid")


def _duration_cost(duration_seconds, max_duration_seconds, duration_penalty):
    excess = max(0.0, float(duration_seconds) - float(max_duration_seconds))
    if excess <= 0.0:
        return 0.0
    m = max(1e-6, float(max_duration_seconds))
    return float(duration_penalty * (excess / m) ** 2 * m)


def boundary_aware_dp_final(
    score_matrix,
    boundary_prob,
    next_logits,
    segment_starts,
    segment_ends,
    visual_change=None,
    jump_penalty=0.03,
    max_forward_jump=2,
    max_duration_seconds=120.0,
    duration_penalty=0.001,
    max_run_seconds=1200.0,
    boundary_reward=1.50,
    boundary_stay_penalty=0.25,
    transition_logits_weight=0.25,
    premature_transition_penalty=0.25,
    boundary_threshold=0.20,
    adjacent_margin_weight=1.00,
    lookahead_weight=0.60,
    visual_change_weight=1.50,
    stay_pressure_weight=1.20,
    stay_margin=0.05,
    stay_temperature=0.25,
    lookahead_segments=4,
):
    """Hybrid monotonic semi-Markov decoder.

    Compared with the original V4 decoder, this version explicitly penalizes
    staying on slide k when slide k+1 has accumulated stronger evidence, and
    combines learned boundaries with visual-change and adjacent-slide evidence.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    b = np.asarray(boundary_prob, dtype=np.float64).reshape(-1)
    nxt = np.asarray(next_logits, dtype=np.float64)

    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2-dimensional")
    n, s = scores.shape
    if n == 0 or s == 0:
        return []
    if len(b) != n:
        raise ValueError("boundary_prob length must match score_matrix rows")
    if nxt.shape != (n, s):
        raise ValueError("next_logits must have shape (num_segments, num_slides)")

    starts = np.asarray(segment_starts, dtype=np.float64)
    ends = np.asarray(segment_ends, dtype=np.float64)
    if len(starts) != n or len(ends) != n:
        raise ValueError("segment timestamps must match score_matrix rows")
    if np.any(ends < starts):
        raise ValueError("Transcript segment end precedes its start")
    if np.any(np.diff(starts) < 0):
        raise ValueError("Transcript segment starts must be non-decreasing")

    if visual_change is None:
        # Derive a score-distribution change signal when no raw video signal
        # was supplied. This keeps the decoder usable without a video file.
        ztmp = _row_zscore(scores)
        vc = np.zeros(n, dtype=np.float64)
        if n > 1:
            a = ztmp[:-1]
            c = ztmp[1:]
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
            c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
            vc[1:] = np.clip(1.0 - np.sum(a * c, axis=1), 0.0, 1.0)
    else:
        vc = np.asarray(visual_change, dtype=np.float64).reshape(-1)
        if len(vc) != n:
            raise ValueError("visual_change length must match score_matrix rows")
        vc = np.clip(vc, 0.0, None)

    vc = _normalize_signal(_smooth_signal(vc, width=3))
    z = _row_zscore(scores)

    # Learned boundary + observed temporal change.
    b = np.clip(b, 1e-4, 1.0 - 1e-4)
    b_logit = np.log(b / (1.0 - b))

    score_change = np.zeros(n, dtype=np.float64)
    if n > 1:
        a = z[:-1]
        c = z[1:]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        score_change[1:] = np.clip(1.0 - np.sum(a * c, axis=1), 0.0, 1.0)
    score_change = _normalize_signal(_smooth_signal(score_change, width=3))

    combined_boundary = _sigmoid(
        b_logit
        + visual_change_weight * (vc - 0.5)
        + 0.75 * (score_change - 0.5)
    )

    # Next-slide probabilities from the trained head.
    next_logprob = nxt - np.max(nxt, axis=1, keepdims=True)
    next_logprob = next_logprob - np.log(
        np.maximum(np.sum(np.exp(next_logprob), axis=1, keepdims=True), 1e-12)
    )

    # Evidence that slide k+1 is stronger than slide k. This specifically
    # attacks the long-slide sticking observed in the original V4 test.
    stay_pressure = np.zeros((n, s), dtype=np.float64)
    if s > 1:
        adjacent_gap = z[:, 1:] - z[:, :-1]
        stay_pressure[:, :-1] = _sigmoid(
            (adjacent_gap - float(stay_margin)) / max(1e-4, float(stay_temperature))
        )
        stay_pressure[:, :-1] *= (0.5 + 0.5 * vc[:, None])

    pressure_prefix = np.zeros((n + 1, s), dtype=np.float64)
    pressure_prefix[1:] = np.cumsum(stay_pressure, axis=0)

    # Precompute short look-ahead evidence for transitions.
    lookahead = np.zeros((n, s), dtype=np.float64)
    L = max(1, int(lookahead_segments))
    for slide in range(1, s):
        prev_slide = slide - 1
        for i in range(n):
            j = min(n, i + L)
            lookahead[i, slide] = float(
                np.mean(z[i:j, slide] - z[i:j, prev_slide])
            )

    jump_penalty = max(0.0, float(jump_penalty))
    max_forward_jump = max(1, int(max_forward_jump))
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    duration_penalty = max(0.0, float(duration_penalty))
    max_run_seconds = max(max_duration_seconds, float(max_run_seconds))

    score_prefix = np.zeros((n + 1, s), dtype=np.float64)
    score_prefix[1:] = np.cumsum(scores, axis=0)
    boundary_excess = np.maximum(combined_boundary - float(boundary_threshold), 0.0)
    boundary_prefix = np.zeros(n + 1, dtype=np.float64)
    boundary_prefix[1:] = np.cumsum(boundary_excess)

    dp = np.full((n, s), -np.inf, dtype=np.float64)
    back_start = np.full((n, s), -1, dtype=np.int32)
    back_prev = np.full((n, s), -1, dtype=np.int32)

    for i in range(n):
        for slide in range(s):
            best_value = -np.inf
            best_k = -1
            best_prev = -1

            k = i
            while k >= 0:
                duration = float(ends[i] - starts[k])
                if duration > max_run_seconds:
                    break

                run_score = score_prefix[i + 1, slide] - score_prefix[k, slide]
                duration_cost = _duration_cost(
                    duration,
                    max_duration_seconds,
                    duration_penalty,
                )

                if i > k:
                    boundary_inside = boundary_prefix[i + 1] - boundary_prefix[k + 1]
                    if slide < s - 1:
                        pressure_inside = (
                            pressure_prefix[i + 1, slide]
                            - pressure_prefix[k + 1, slide]
                        )
                    else:
                        pressure_inside = 0.0
                else:
                    boundary_inside = 0.0
                    pressure_inside = 0.0

                stay_cost = (
                    boundary_stay_penalty * boundary_inside
                    + stay_pressure_weight * pressure_inside
                )

                if k == 0:
                    if slide != 0:
                        k -= 1
                        continue
                    candidate = run_score - duration_cost - stay_cost
                    prev_slide = -1
                else:
                    previous_end = k - 1
                    first_prev = max(0, slide - max_forward_jump)
                    best_previous = -np.inf
                    best_previous_slide = -1
                    for ps in range(first_prev, slide):
                        value = dp[previous_end, ps]
                        if value > best_previous:
                            best_previous = value
                            best_previous_slide = ps

                    if not np.isfinite(best_previous):
                        k -= 1
                        continue

                    ps = best_previous_slide
                    p = float(combined_boundary[k])
                    boundary_term = boundary_reward * (2.0 * p - 1.0)
                    visual_term = visual_change_weight * float(vc[k])
                    next_term = transition_logits_weight * float(next_logprob[k, slide])
                    adj_term = 0.0
                    future_term = 0.0
                    if slide == ps + 1:
                        adj_term = adjacent_margin_weight * np.tanh(
                            0.5 * float(z[k, slide] - z[k, ps])
                        )
                        future_term = lookahead_weight * np.tanh(
                            0.5 * float(lookahead[k, slide])
                        )

                    premature = premature_transition_penalty * (1.0 - p)
                    jump_cost = jump_penalty * float(slide - ps)

                    candidate = (
                        best_previous
                        + run_score
                        - duration_cost
                        - stay_cost
                        - premature
                        - jump_cost
                        + boundary_term
                        + visual_term
                        + next_term
                        + adj_term
                        + future_term
                    )
                    prev_slide = ps

                if candidate > best_value:
                    best_value = candidate
                    best_k = k
                    best_prev = prev_slide

                k -= 1

            dp[i, slide] = best_value
            back_start[i, slide] = best_k
            back_prev[i, slide] = best_prev

    final_slide = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, final_slide]):
        raise RuntimeError("No valid alignment found")

    path = np.zeros(n, dtype=np.int32)
    current_end = n - 1
    current_slide = final_slide

    while current_end >= 0:
        k = int(back_start[current_end, current_slide])
        if k < 0:
            raise RuntimeError("Broken backpointer during backtracking")
        path[k: current_end + 1] = current_slide
        if k == 0:
            break
        previous_slide = int(back_prev[current_end, current_slide])
        if previous_slide < 0 or previous_slide >= current_slide:
            raise RuntimeError("Invalid previous slide during backtracking")
        current_end = k - 1
        current_slide = previous_slide

    if path[0] != 0:
        raise RuntimeError("Alignment did not start from Slide 1")
    if np.any(np.diff(path) < 0):
        raise RuntimeError("Alignment contains a backward transition")

    return path.tolist()
