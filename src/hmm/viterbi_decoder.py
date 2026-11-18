
import numpy as np


def decode_viterbi(
    candidate_groups,
    arc_start_nodes,
    arc_end_nodes,
    gps_distances=None,
    time_gaps=None,
    transition_penalty=3.0,
    distance_weight=0.01,
    gap_threshold=120.0
):
    """
    candidate_groups:
        List of DataFrames, one per GPS observation.
        Each must contain:
            arc_id
            probability
            distance_m

    arc_start_nodes, arc_end_nodes:
        Arrays indexed by arc_id.

    gps_distances:
        Distances between consecutive GPS observations.
        Length = number of observations - 1.

    time_gaps:
        Time differences between consecutive GPS observations.
        Length = number of observations - 1.

    Returns:
        One selected arc_id per GPS observation.
    """

    n = len(candidate_groups)

    if n == 0:
        return []

    if gps_distances is None:
        gps_distances = np.zeros(n - 1)

    if time_gaps is None:
        time_gaps = np.ones(n - 1)

    # Precompute candidate arrays.
    arc_ids = []
    emissions = []

    for group in candidate_groups:
        if len(group) == 0:
            raise ValueError(
                "Each observation must have at least one candidate"
            )

        ids = group["arc_id"].to_numpy(dtype=int)
        probabilities = np.clip(
            group["probability"].to_numpy(dtype=float),
            1e-8,
            1.0
        )

        arc_ids.append(ids)
        emissions.append(np.log(probabilities))

    scores = [emissions[0].copy()]
    backpointers = [None]

    for t in range(1, n):
        previous_ids = arc_ids[t - 1]
        current_ids = arc_ids[t]

        previous_scores = scores[t - 1]

        current_scores = np.full(
            len(current_ids),
            -np.inf
        )

        pointers = np.zeros(
            len(current_ids),
            dtype=int
        )

        gap = float(time_gaps[t - 1])

        for j, current_arc in enumerate(current_ids):
            best_score = -np.inf
            best_previous = 0

            current_start = arc_start_nodes[current_arc]

            for i, previous_arc in enumerate(previous_ids):
                previous_end = arc_end_nodes[previous_arc]

                # Direct connection or remaining on same arc.
                connected = (
                    previous_arc == current_arc
                    or previous_end == current_start
                )

                if connected:
                    penalty = 0.0
                else:
                    penalty = transition_penalty

                # Do not enforce strong continuity over long gaps.
                if gap > gap_threshold:
                    penalty = 0.0

                # Candidate snap-distance penalty.
                snap_distance = float(
                    candidate_groups[t].iloc[j]["distance_m"]
                )

                score = (
                    previous_scores[i]
                    - penalty
                    - distance_weight * snap_distance
                    + emissions[t][j]
                )

                if score > best_score:
                    best_score = score
                    best_previous = i

            current_scores[j] = best_score
            pointers[j] = best_previous

        scores.append(current_scores)
        backpointers.append(pointers)

    # Backtrack through the best path.
    selected_indices = [0] * n
    selected_indices[-1] = int(
        np.argmax(scores[-1])
    )

    for t in range(n - 1, 0, -1):
        selected_indices[t - 1] = int(
            backpointers[t][selected_indices[t]]
        )

    return [
        int(arc_ids[t][selected_indices[t]])
        for t in range(n)
    ]
