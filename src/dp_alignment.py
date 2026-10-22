import numpy as np


# ============================================================
# LectureSense duration-aware monotonic alignment
# ============================================================
# This decoder uses the actual transcript timestamps so that slide
# duration is measured in seconds rather than estimated from a fixed
# number of transcript segments.
#
# It enforces:
#   - monotonic slide order
#   - no backward transitions
#   - at most one-slide forward transition
#   - explicit duration regularization
#
# The decoder searches for the best contiguous run assigned to each
# slide and therefore avoids the cumulative-state problem of the old
# decoder that could remain on one slide for several minutes.
# ============================================================

DEFAULT_MAX_DURATION_SECONDS = 120.0
DEFAULT_DURATION_PENALTY = 0.01
DEFAULT_MAX_RUN_SECONDS = 480.0


def _duration_cost(
    duration_seconds: float,
    max_duration_seconds: float,
    duration_penalty: float,
) -> float:
    """Quadratic penalty for time spent beyond the preferred duration."""

    excess = max(0.0, float(duration_seconds) - float(max_duration_seconds))

    if excess <= 0.0:
        return 0.0

    max_duration_seconds = max(1e-6, float(max_duration_seconds))

    return float(
        duration_penalty
        * (excess / max_duration_seconds) ** 2
        * max_duration_seconds
    )


def dp_align(
    score_matrix: np.ndarray,
    jump_penalty: float = 0.05,
    max_forward_jump: int = 1,
    segment_starts: list[float] | np.ndarray | None = None,
    segment_ends: list[float] | np.ndarray | None = None,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    duration_penalty: float = DEFAULT_DURATION_PENALTY,
    max_run_seconds: float = DEFAULT_MAX_RUN_SECONDS,
) -> list[int]:
    """
    Duration-aware monotonic transcript-to-slide alignment.

    Parameters
    ----------
    score_matrix:
        Shape (num_transcript_segments, num_slides). Higher is better.

    jump_penalty:
        Cost for moving forward by one slide.

    max_forward_jump:
        Maximum forward slide jump. This implementation supports 1 by
        default. A value above 1 is accepted, but the decoder still uses
        contiguous slide runs and is intended for normal lecture order.

    segment_starts / segment_ends:
        Actual transcript timestamps in seconds. Supplying these is
        strongly recommended because duration regularization should use
        real time rather than a fixed segment-length approximation.

    max_duration_seconds:
        Preferred maximum duration of a continuous slide assignment.

    duration_penalty:
        Strength of the quadratic penalty beyond max_duration_seconds.

    max_run_seconds:
        Maximum current-slide run duration considered when searching for
        a transition point. This bounds the computation while still
        allowing slides to remain visible substantially longer than the
        preferred duration.

    Returns
    -------
    list[int]
        Zero-based slide index for each transcript segment.
    """

    scores = np.asarray(score_matrix, dtype=np.float32)

    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2-dimensional")

    n_segments, n_slides = scores.shape

    if n_segments == 0 or n_slides == 0:
        return []

    max_forward_jump = max(1, int(max_forward_jump))
    jump_penalty = max(0.0, float(jump_penalty))
    max_duration_seconds = max(
        0.1,
        float(max_duration_seconds),
    )
    duration_penalty = max(0.0, float(duration_penalty))
    max_run_seconds = max(
        max_duration_seconds,
        float(max_run_seconds),
    )

    # ------------------------------------------------------------
    # Actual timestamps
    # ------------------------------------------------------------
    if segment_starts is None or segment_ends is None:
        # Backward-compatible fallback. This is only an approximation.
        starts = np.arange(n_segments, dtype=np.float32) * 8.0
        ends = starts + 8.0
    else:
        starts = np.asarray(segment_starts, dtype=np.float32)
        ends = np.asarray(segment_ends, dtype=np.float32)

        if len(starts) != n_segments or len(ends) != n_segments:
            raise ValueError(
                "segment_starts/segment_ends length must match "
                "score_matrix rows"
            )

        if np.any(ends < starts):
            raise ValueError("Transcript segment end precedes its start")

        if np.any(np.diff(starts) < 0):
            raise ValueError("Transcript segment starts must be non-decreasing")

    # ------------------------------------------------------------
    # Prefix sums for fast contiguous-run score calculation.
    # prefix[i+1, j] = sum(scores[0:i+1, j])
    # ------------------------------------------------------------
    prefix = np.zeros(
        (n_segments + 1, n_slides),
        dtype=np.float64,
    )
    prefix[1:] = np.cumsum(
        scores.astype(np.float64),
        axis=0,
    )

    # dp[i, slide] = best score for all transcript segments 0..i,
    # ending with a contiguous run on `slide`.
    dp = np.full(
        (n_segments, n_slides),
        -np.inf,
        dtype=np.float64,
    )

    # Backpointer stores the first transcript index of the current
    # slide's contiguous run.
    run_start_back = np.full(
        (n_segments, n_slides),
        -1,
        dtype=np.int32,
    )

    # ------------------------------------------------------------
    # DP over transcript endpoint and slide.
    # ------------------------------------------------------------
    for i in range(n_segments):

        for slide in range(n_slides):

            best_value = -np.inf
            best_start = -1

            # A run for this slide begins at k and ends at i.
            # Search only plausible durations.
            k = i

            while k >= 0:
                duration = float(ends[i] - starts[k])

                if duration > max_run_seconds:
                    break

                # Current slide score from k..i.
                current_run_score = (
                    prefix[i + 1, slide]
                    - prefix[k, slide]
                )

                duration_cost = _duration_cost(
                    duration,
                    max_duration_seconds,
                    duration_penalty,
                )

                if slide == 0:
                    # Slide 1 must be the first slide.
                    if k != 0:
                        k -= 1
                        continue

                    candidate = (
                        current_run_score
                        - duration_cost
                    )

                else:
                    # Normal lecture progression: move from a previous
                    # slide into this slide at transcript index k.
                    previous_end = k - 1

                    best_previous = -np.inf
                    previous_slide = slide - 1

                    # Normal case: previous slide ends immediately before k.
                    if previous_end >= 0:
                        best_previous = dp[
                            previous_end,
                            previous_slide,
                        ]

                    # If explicit forward jumps are requested, also allow
                    # preceding slide states within the allowed jump.
                    if max_forward_jump > 1 and previous_end >= 0:
                        first_previous_slide = max(
                            0,
                            slide - max_forward_jump,
                        )
                        for candidate_previous_slide in range(
                            first_previous_slide,
                            slide - 1,
                        ):
                            value = dp[
                                previous_end,
                                candidate_previous_slide,
                            ]
                            if value > best_previous:
                                best_previous = value

                    if not np.isfinite(best_previous):
                        k -= 1
                        continue

                    candidate = (
                        best_previous
                        + current_run_score
                        - duration_cost
                        - jump_penalty
                    )

                if candidate > best_value:
                    best_value = candidate
                    best_start = k

                k -= 1

            dp[i, slide] = best_value
            run_start_back[i, slide] = best_start

    # ------------------------------------------------------------
    # We do not force the lecture to end on the last PDF slide.
    # This matters because a lecture can legitimately skip slides or
    # have unannotated final pages.
    # ------------------------------------------------------------
    final_slide = int(np.argmax(dp[-1]))

    if not np.isfinite(dp[-1, final_slide]):
        raise RuntimeError(
            "No valid monotonic slide alignment was found."
        )

    # ------------------------------------------------------------
    # Backtracking by contiguous runs.
    # ------------------------------------------------------------
    path = np.zeros(
        n_segments,
        dtype=np.int32,
    )

    current_end = n_segments - 1
    current_slide = final_slide

    while current_end >= 0:

        run_start = int(
            run_start_back[current_end, current_slide]
        )

        if run_start < 0:
            raise RuntimeError(
                "Broken DP backpointer during alignment backtracking."
            )

        path[run_start:current_end + 1] = current_slide

        if run_start == 0:
            break

        # The previous contiguous run belongs to the previous slide.
        current_end = run_start - 1
        current_slide -= 1

        if current_slide < 0:
            raise RuntimeError(
                "Invalid slide order during DP backtracking."
            )

    # ------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------
    if path[0] != 0:
        raise RuntimeError(
            "DP alignment did not start from Slide 1."
        )

    if np.any(np.diff(path) < 0):
        raise RuntimeError(
            "DP alignment produced a backward slide transition."
        )

    return path.tolist()


def build_timeline(
    path: list[int],
    sentence_timestamps: list[float],
    slide_titles: list[str],
) -> list[dict]:
    """Convert a slide path into merged timeline entries."""

    if not path:
        return []

    if len(path) != len(sentence_timestamps):
        raise ValueError(
            "path and sentence_timestamps must have the same length"
        )

    timeline = []
    current_slide = int(path[0])
    start_time = float(sentence_timestamps[0])

    for i in range(1, len(path)):
        next_slide = int(path[i])

        if next_slide != current_slide:
            timeline.append(
                {
                    "slide_id": current_slide + 1,
                    "slide_index": current_slide,
                    "slide_title": (
                        slide_titles[current_slide]
                        if current_slide < len(slide_titles)
                        else f"slide_{current_slide + 1:04d}"
                    ),
                    "start_time": float(start_time),
                    "end_time": float(sentence_timestamps[i]),
                }
            )

            current_slide = next_slide
            start_time = float(sentence_timestamps[i])

    timeline.append(
        {
            "slide_id": current_slide + 1,
            "slide_index": current_slide,
            "slide_title": (
                slide_titles[current_slide]
                if current_slide < len(slide_titles)
                else f"slide_{current_slide + 1:04d}"
            ),
            "start_time": float(start_time),
            "end_time": float(sentence_timestamps[-1]),
        }
    )

    return timeline
