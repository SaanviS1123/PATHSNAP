
from pathlib import Path
from functools import lru_cache
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from fastapi import FastAPI, HTTPException

from viterbi_decoder import decode_viterbi

app = FastAPI(
    title="PathSnap API",
    description="ML-based map matching and route reconstruction",
    version="1.0"
)

BASE = Path(__file__).resolve().parent
TEST_ROOT = BASE / "Student folder" / "student_test"

FEATURES = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

model = joblib.load(
    BASE / "pathsnap_tuned_model.joblib"
)


@lru_cache(maxsize=4)
def load_trip(trip_id):
    folder = TEST_ROOT / trip_id

    if not folder.is_dir():
        raise FileNotFoundError("Trip not found")

    track = np.loadtxt(
        folder / f"{trip_id}.track",
        ndmin=2
    )
    nodes = np.loadtxt(
        folder / f"{trip_id}.nodes",
        ndmin=2
    )
    arcs = np.loadtxt(
        folder / f"{trip_id}.arcs",
        dtype=int,
        ndmin=2
    )

    if track.shape[1] < 3 or nodes.shape[1] < 2 or arcs.shape[1] < 2:
        raise ValueError("Invalid trip file format")

    if (
        np.any(arcs[:, :2] < 0)
        or np.any(arcs[:, :2] >= len(nodes))
    ):
        raise ValueError("Invalid node IDs in arcs")

    return track, nodes, arcs


def make_candidates(track, nodes, arcs):
    latitude = float(np.mean(nodes[:, 1]))
    sx = 111320 * np.cos(np.radians(latitude))
    sy = 111132

    gps = np.column_stack([
        track[:, 0] * sx,
        track[:, 1] * sy
    ])

    xy = np.column_stack([
        nodes[:, 0] * sx,
        nodes[:, 1] * sy
    ])

    starts = xy[arcs[:, 0]]
    ends = xy[arcs[:, 1]]
    vectors = ends - starts

    lengths_sq = np.sum(vectors ** 2, axis=1)
    lengths = np.sqrt(lengths_sq)

    tree = cKDTree((starts + ends) / 2)

    rows = []

    for i, point in enumerate(gps):
        k = min(500, len(arcs))
        _, ids = tree.query(point, k=k)
        ids = np.atleast_1d(ids).astype(int)

        a = starts[ids]
        v = vectors[ids]

        fraction = np.clip(
            np.sum((point - a) * v, axis=1)
            / np.maximum(lengths_sq[ids], 1e-12),
            0,
            1
        )

        projected = a + fraction[:, None] * v
        distances = np.linalg.norm(
            projected - point,
            axis=1
        )

        closest = np.argsort(distances)[:10]

        if i == 0:
            dt = 0.0
            speed = 0.0
            gps_bearing = 0.0
        else:
            displacement = gps[i] - gps[i - 1]
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

        for j in closest:
            arc_id = int(ids[j])
            road = vectors[arc_id]

            road_bearing = np.arctan2(
                road[0],
                road[1]
            )

            difference = abs(
                np.arctan2(
                    np.sin(gps_bearing - road_bearing),
                    np.cos(gps_bearing - road_bearing)
                )
            )

            rows.append({
                "observation_id": i,
                "arc_id": arc_id,
                "distance_m": float(distances[j]),
                "segment_length_m": float(lengths[arc_id]),
                "bearing_difference": float(difference),
                "speed_mps": float(speed),
                "time_delta_s": float(dt)
            })

    return pd.DataFrame(rows), gps


def continuity(predictions, arcs):
    if len(predictions) < 2:
        return 1.0

    a = np.asarray(predictions[:-1])
    b = np.asarray(predictions[1:])

    return float(np.mean(
        (a == b) | (arcs[a, 1] == arcs[b, 0])
    ))


@app.get("/")
def home():
    return {
        "project": "PathSnap",
        "status": "running",
        "model": "Tuned HistGradientBoosting + Viterbi"
    }


@app.get("/trips")
def list_trips():
    return {
        "trips": sorted(
            folder.name
            for folder in TEST_ROOT.iterdir()
            if folder.is_dir()
            and folder.name.isdigit()
        )
    }


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if not (len(trip_id) == 8 and trip_id.isdigit()):
        raise HTTPException(
            status_code=400,
            detail="Trip ID must contain 8 digits"
        )

    started = time.perf_counter()

    try:
        track, nodes, arcs = load_trip(trip_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Trip not found"
        )
    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    candidates, gps = make_candidates(
        track, nodes, arcs
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

    decoded = decode_viterbi(
        candidate_groups=groups,
        arc_start_nodes=arcs[:, 0],
        arc_end_nodes=arcs[:, 1],
        gps_distances=np.linalg.norm(
            np.diff(gps, axis=0),
            axis=1
        ),
        time_gaps=np.diff(track[:, 2]),
        transition_penalty=3.0,
        distance_weight=0.01
    )

    # Each matched arc is represented by its two endpoints.
    # These are selected arc geometries, not a fully bridged route.
    matched_segments = [
        [
            nodes[arcs[arc_id, 0], :2].tolist(),
            nodes[arcs[arc_id, 1], :2].tolist()
        ]
        for arc_id in decoded
    ]

    elapsed = time.perf_counter() - started

    return {
        "trip_id": trip_id,
        "gps_count": int(len(track)),
        "processing_seconds": round(elapsed, 3),
        "classifier_continuity": round(
            continuity(classifier, arcs), 4
        ),
        "viterbi_continuity": round(
            continuity(decoded, arcs), 4
        ),
        "gps_coordinates": track[:, :2].tolist(),
        "classifier_arc_ids": classifier,
        "viterbi_arc_ids": decoded,
        "matched_segments": matched_segments,
        "coordinate_order": "longitude, latitude"
    }
