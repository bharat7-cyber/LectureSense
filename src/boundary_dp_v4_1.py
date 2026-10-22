from __future__ import annotations

import numpy as np


def _duration_cost(duration_seconds, max_duration_seconds, duration_penalty):
    excess = max(0.0, float(duration_seconds) - float(max_duration_seconds))
    if excess <= 0.0:
        return 0.0
    m = max(1e-6, float(max_duration_seconds))
    return float(duration_penalty * (excess / m) ** 2 * m)


def boundary_aware_dp_v41(
    score_matrix,
    boundary_prob,
    next_logits,
    segment_starts,
    segment_ends,
    jump_penalty=0.03,
    max_forward_jump=2,
    max_duration_seconds=150.0,
    duration_penalty=0.001,
    max_run_seconds=1500.0,
    boundary_reward=0.75,
    boundary_stay_penalty=0.35,
    transition_logits_weight=0.30,
    premature_transition_penalty=0.35,
    boundary_threshold=0.20,
    jump_penalty_power=1.35,
):
    """
    V4.1 boundary-aware monotonic DP.

    Compared with V4:
      - forward jumps can exceed one slide, while paying a super-linear
        jump cost;
      - all transition terms remain validation-tunable;
      - no test labels are used by this function.
    """
    scores = np.asarray(score_matrix, dtype=np.float32)
    b = np.asarray(boundary_prob, dtype=np.float32).reshape(-1)
    nxt = np.asarray(next_logits, dtype=np.float32)

    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2-dimensional")

    n, s = scores.shape
    if n == 0 or s == 0:
        return []

    if len(b) != n:
        raise ValueError("boundary_prob length must match score_matrix rows")
    if nxt.shape != (n, s):
        raise ValueError("next_logits must have shape (num_segments, num_slides)")

    starts = np.asarray(segment_starts, dtype=np.float32)
    ends = np.asarray(segment_ends, dtype=np.float32)
    if len(starts) != n or len(ends) != n:
        raise ValueError("segment timestamps must match score_matrix rows")
    if np.any(ends < starts):
        raise ValueError("Transcript segment end precedes its start")
    if np.any(np.diff(starts) < 0):
        raise ValueError("Transcript segment starts must be non-decreasing")

    jump_penalty = max(0.0, float(jump_penalty))
    max_forward_jump = max(1, int(max_forward_jump))
    max_duration_seconds = max(0.1, float(max_duration_seconds))
    duration_penalty = max(0.0, float(duration_penalty))
    max_run_seconds = max(max_duration_seconds, float(max_run_seconds))
    boundary_reward = float(boundary_reward)
    boundary_stay_penalty = max(0.0, float(boundary_stay_penalty))
    transition_logits_weight = float(transition_logits_weight)
    premature_transition_penalty = max(0.0, float(premature_transition_penalty))
    boundary_threshold = float(boundary_threshold)
    jump_penalty_power = max(1.0, float(jump_penalty_power))

    b = np.clip(b, 1e-4, 1.0 - 1e-4)

    score_prefix = np.zeros((n + 1, s), dtype=np.float64)
    score_prefix[1:] = np.cumsum(scores.astype(np.float64), axis=0)

    boundary_excess = np.maximum(b - boundary_threshold, 0.0)
    boundary_prefix = np.zeros(n + 1, dtype=np.float64)
    boundary_prefix[1:] = np.cumsum(boundary_excess, dtype=np.float64)

    next_logprob = nxt.astype(np.float64)
    next_logprob = next_logprob - np.max(next_logprob, axis=1, keepdims=True)
    next_logprob = next_logprob - np.log(
        np.maximum(np.sum(np.exp(next_logprob), axis=1, keepdims=True), 1e-12)
    )

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
                    duration, max_duration_seconds, duration_penalty
                )

                if i > k:
                    inside_boundary_penalty = (
                        boundary_prefix[i + 1] - boundary_prefix[k + 1]
                    )
                else:
                    inside_boundary_penalty = 0.0

                stay_cost = boundary_stay_penalty * inside_boundary_penalty

                if k == 0:
                    if slide != 0:
                        k -= 1
                        continue
                    candidate = run_score - duration_cost - stay_cost
                    prev_slide = -1
                else:
                    first_prev = max(0, slide - max_forward_jump)
                    best_previous = -np.inf
                    best_previous_slide = -1

                    for ps in range(first_prev, slide):
                        value = dp[k - 1, ps]
                        if value > best_previous:
                            best_previous = value
                            best_previous_slide = ps

                    if not np.isfinite(best_previous):
                        k -= 1
                        continue

                    p = float(b[k])
                    boundary_logit = float(np.log(p / (1.0 - p)))

                    jump = slide - best_previous_slide
                    jump_cost = jump_penalty * float(max(0, jump - 1) ** jump_penalty_power)

                    transition_score = (
                        boundary_reward * np.tanh(0.5 * boundary_logit)
                        + transition_logits_weight * float(next_logprob[k, slide])
                        - premature_transition_penalty * (1.0 - p)
                    )

                    candidate = (
                        best_previous
                        + run_score
                        - duration_cost
                        - stay_cost
                        - jump_cost
                        + transition_score
                    )
                    prev_slide = best_previous_slide

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
        raise RuntimeError("No valid V4.1 alignment found")

    path = np.zeros(n, dtype=np.int32)
    current_end = n - 1
    current_slide = final_slide

    while current_end >= 0:
        k = int(back_start[current_end, current_slide])
        if k < 0:
            raise RuntimeError("Broken backpointer during V4.1 backtracking")

        path[k:current_end + 1] = current_slide

        if k == 0:
            break

        previous_slide = int(back_prev[current_end, current_slide])
        if previous_slide < 0 or previous_slide >= current_slide:
            raise RuntimeError("Invalid previous slide during V4.1 backtracking")

        current_end = k - 1
        current_slide = previous_slide

    if path[0] != 0:
        raise RuntimeError("Alignment did not start from Slide 1")
    if np.any(np.diff(path) < 0):
        raise RuntimeError("Alignment contains a backward transition")

    return path.tolist()
