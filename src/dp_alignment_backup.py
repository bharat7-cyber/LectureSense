import numpy as np


def dp_align(
    score_matrix: np.ndarray,
    jump_penalty: float = 0.15,
) -> list[int]:
    """
    Temporally align transcript sentences to slides.

    score_matrix:
        Shape = (number_of_sentences, number_of_slides)

        Each value represents how strongly a sentence matches
        a particular slide.

    The DP prefers:
        - staying on the same slide
        - moving to the next slide
        - avoiding large jumps

    Returns:
        A list containing the predicted slide index for every
        transcript sentence.
    """

    if score_matrix.ndim != 2:
        raise ValueError(
            "score_matrix must be a 2D matrix."
        )

    n_sentences, n_slides = score_matrix.shape

    if n_sentences == 0 or n_slides == 0:
        return []

    # DP score.
    dp = np.full(
        (n_sentences, n_slides),
        -np.inf,
        dtype=np.float32,
    )

    # Previous slide for backtracking.
    backpointer = np.zeros(
        (n_sentences, n_slides),
        dtype=np.int32,
    )

    # First sentence can start on any slide.
    dp[0, :] = score_matrix[0, :]

    # --------------------------------------------------------
    # Dynamic programming
    # --------------------------------------------------------

    for i in range(1, n_sentences):

        for current_slide in range(n_slides):

            best_score = -np.inf
            best_previous = 0

            for previous_slide in range(n_slides):

                distance = abs(
                    current_slide - previous_slide
                )

                if distance == 0:
                    transition_penalty = 0.0

                elif distance == 1:
                    transition_penalty = 0.0

                else:
                    transition_penalty = (
                        jump_penalty * distance
                    )

                candidate = (
                    dp[i - 1, previous_slide]
                    - transition_penalty
                )

                if candidate > best_score:

                    best_score = candidate
                    best_previous = previous_slide

            dp[i, current_slide] = (
                best_score
                + score_matrix[i, current_slide]
            )

            backpointer[
                i,
                current_slide,
            ] = best_previous

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    path = np.zeros(
        n_sentences,
        dtype=np.int32,
    )

    path[-1] = int(
        np.argmax(dp[-1, :])
    )

    for i in range(
        n_sentences - 1,
        0,
        -1,
    ):

        path[i - 1] = backpointer[
            i,
            path[i],
        ]

    return path.tolist()


def build_timeline(
    path: list[int],
    sentence_timestamps: list[float],
    slide_titles: list[str],
) -> list[dict]:
    """
    Convert sentence-level predictions into contiguous
    slide timeline entries.
    """

    if not path:
        return []

    if len(path) != len(sentence_timestamps):
        raise ValueError(
            "path and sentence_timestamps "
            "must have the same length."
        )

    timeline = []

    current_slide = int(path[0])
    start_time = float(
        sentence_timestamps[0]
    )

    for i in range(1, len(path)):

        slide = int(path[i])

        if slide != current_slide:

            if current_slide < len(slide_titles):
                title = slide_titles[
                    current_slide
                ]
            else:
                title = (
                    f"Slide {current_slide + 1}"
                )

            timeline.append(
                {
                    "slide_id": current_slide + 1,
                    "slide_index": current_slide,
                    "title": title,
                    "start_time": start_time,
                    "end_time": float(
                        sentence_timestamps[i]
                    ),
                }
            )

            current_slide = slide
            start_time = float(
                sentence_timestamps[i]
            )

    # Final timeline entry.
    if current_slide < len(slide_titles):
        title = slide_titles[
            current_slide
        ]
    else:
        title = (
            f"Slide {current_slide + 1}"
        )

    timeline.append(
        {
            "slide_id": current_slide + 1,
            "slide_index": current_slide,
            "title": title,
            "start_time": start_time,
            "end_time": None,
        }
    )

    return timeline


def format_time(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.
    """

    seconds = max(
        0,
        int(seconds),
    )

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


if __name__ == "__main__":

    print(
        "Running DP alignment sanity test..."
    )

    rng = np.random.RandomState(42)

    n_sentences = 12
    n_slides = 5

    scores = rng.uniform(
        0.05,
        0.30,
        size=(
            n_sentences,
            n_slides,
        ),
    )

    intended_path = [
        0, 0, 0,
        1, 1, 1,
        2, 2,
        3, 3,
        4, 4,
    ]

    for i, slide in enumerate(
        intended_path
    ):

        scores[i, slide] = 0.90

    # Add one noisy incorrect spike.
    scores[6, 4] = 0.85

    greedy_path = list(
        np.argmax(
            scores,
            axis=1,
        )
    )

    dp_path = dp_align(
        scores,
        jump_penalty=0.20,
    )

    print(
        "Intended:",
        intended_path,
    )

    print(
        "Greedy:  ",
        greedy_path,
    )

    print(
        "DP:      ",
        dp_path,
    )

    timeline = build_timeline(
        dp_path,
        [
            float(x)
            for x in range(
                0,
                n_sentences * 30,
                30,
            )
        ],
        [
            f"Slide {i + 1}"
            for i in range(n_slides)
        ],
    )

    print("\nTimeline:")

    for entry in timeline:

        print(
            f"{format_time(entry['start_time'])} "
            f"-> {entry['title']}"
        )