
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from viterbi_decoder import decode_viterbi

TRIP = "00000000"

ROOT = Path("Student folder") / "student_test" / TRIP

FEATURES = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]


def load_data():
    track = np.loadtxt(ROOT / f"{TRIP}.track")
    nodes = np.loadtxt(ROOT / f"{TRIP}.nodes")
    arcs = np.loadtxt(
        ROOT / f"{TRIP}.arcs",
        dtype=int
    )
    route = np.loadtxt(
        ROOT / f"{TRIP}.route",
        dtype=int,
        ndmin=1
    )

    return track, nodes, arcs, route


def build_candidates(track, nodes, arcs):
    # Local approximate conversion from degrees to metres.
    latitude = np.mean(nodes[:, 1])

    scale_x = 111320 * np.cos(np.radians(latitude))
    scale_y = 111132

    def project(points):
        return np.column_stack([
            points[:, 0] * scale_x,
            points[:, 1] * scale_y
        ])

    gps_xy = project(track[:, :2])
    node_xy = project(nodes)

    starts = node_xy[arcs[:, 0]]
    ends = node_xy[arcs[:, 1]]

    vectors = ends - starts
    lengths_sq = np.sum(vectors ** 2, axis=1)

    lengths = np.sqrt(lengths_sq)

    midpoints = (starts + ends) / 2
    tree = cKDTree(midpoints)

    rows = []

    for i, point in enumerate(gps_xy):
        k = min(500, len(arcs))

        _, candidate_ids = tree.query(point, k=k)

        candidate_ids = np.atleast_1d(candidate_ids)

        a = starts[candidate_ids]
        v = vectors[candidate_ids]
        denom = lengths_sq[candidate_ids]

        t = np.clip(
            np.sum((point - a) * v, axis=1)
            / np.maximum(denom, 1e-12),
            0,
            1
        )

        projections = a + t[:, None] * v

        distances = np.linalg.norm(
            projections - point,
            axis=1
        )

        closest = np.argsort(distances)[:10]

        if i == 0:
            speed = 0.0
            dt = 0.0
            gps_bearing = 0.0
        else:
            displacement = gps_xy[i] - gps_xy[i - 1]
            dt = max(
                0.0,
                float(track[i, 2] - track[i - 1, 2])
            )

            speed = (
                np.linalg.norm(displacement) / dt
                if dt > 0 else 0.0
            )

            gps_bearing = np.arctan2(
                displacement[0],
                displacement[1]
            )

        for index in closest:
            arc_id = int(candidate_ids[index])

            road_vector = vectors[arc_id]

            road_bearing = np.arctan2(
                road_vector[0],
                road_vector[1]
            )

            bearing_diff = abs(
                np.arctan2(
                    np.sin(gps_bearing - road_bearing),
                    np.cos(gps_bearing - road_bearing)
                )
            )

            rows.append({
                "observation_id": i,
                "arc_id": arc_id,
                "distance_m": float(distances[index]),
                "segment_length_m": float(lengths[arc_id]),
                "bearing_difference": float(bearing_diff),
                "speed_mps": float(speed),
                "time_delta_s": float(dt)
            })

    return pd.DataFrame(rows)


def direct_continuity(predictions, arcs):
    if len(predictions) < 2:
        return 1.0

    a = np.asarray(predictions[:-1])
    b = np.asarray(predictions[1:])

    connected = (
        (a == b)
        | (arcs[a, 1] == arcs[b, 0])
    )

    return float(np.mean(connected))


def main():
    start_time = time.time()

    track, nodes, arcs, route = load_data()

    print("STUDENT TEST TRIP:", TRIP)
    print("GPS observations:", len(track))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    candidates = build_candidates(track, nodes, arcs)

    model = joblib.load(
        "pathsnap_tuned_model.joblib"
    )

    candidates["probability"] = model.predict_proba(
        candidates[FEATURES].fillna(0)
    )[:, 1]

    groups = [
        group.reset_index(drop=True)
        for _, group in candidates.groupby(
            "observation_id",
            sort=True
        )
    ]

    classifier = [
        int(
            group.loc[
                group["probability"].idxmax(),
                "arc_id"
            ]
        )
        for group in groups
    ]

    time_gaps = np.diff(track[:, 2])

    gps_distances = np.linalg.norm(
        np.diff(
            np.column_stack([
                track[:, 0] * 111320
                * np.cos(np.radians(np.mean(nodes[:, 1]))),
                track[:, 1] * 111132
            ]),
            axis=0
        ),
        axis=1
    )

    viterbi = decode_viterbi(
        candidate_groups=groups,
        arc_start_nodes=arcs[:, 0],
        arc_end_nodes=arcs[:, 1],
        gps_distances=gps_distances,
        time_gaps=time_gaps,
        transition_penalty=3.0,
        distance_weight=0.01
    )

    print("\nClassifier-only")
    print(
        "Direct continuity:",
        round(direct_continuity(classifier, arcs), 4)
    )

    print("\nClassifier + Viterbi")
    print(
        "Direct continuity:",
        round(direct_continuity(viterbi, arcs), 4)
    )

    # Reference route is an ordered list of road arcs,
    # not one label per GPS observation.
    # This is only a route-membership diagnostic.
    route_set = set(map(int, route))

    print("\nReference route membership (diagnostic)")
    print(
        "Classifier:",
        round(np.mean([
            arc in route_set for arc in classifier
        ]), 4)
    )
    print(
        "Viterbi:",
        round(np.mean([
            arc in route_set for arc in viterbi
        ]), 4)
    )

    result = pd.DataFrame({
        "observation_id": np.arange(len(track)),
        "classifier_arc": classifier,
        "viterbi_arc": viterbi
    })

    result.to_csv(
        f"student_test_{TRIP}_predictions.csv",
        index=False
    )

    print(
        "\nTotal time:",
        round(time.time() - start_time, 2),
        "seconds"
    )

    print(
        f"Saved student_test_{TRIP}_predictions.csv"
    )


if __name__ == "__main__":
    main()
