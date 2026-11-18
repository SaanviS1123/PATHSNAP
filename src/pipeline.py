
"""PATHSNAP prediction pipeline: classifier-only and Viterbi decoding.

Reference .route files are never loaded during inference.
"""

import json
import re
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from api.config import CONFIG_DIR, MODEL_DIR, STUDENT_TEST_DIR
from src.data_processing.loaders import load_trip
from src.data_processing.geo import project_trip
from src.features.build import build_features
from src.models.member1_harness import load_scorer
from src.hmm.viterbi_decoder import decode_viterbi


ALLOWED_MODELS = {"knn", "mlp"}
TRIP_ID_PATTERN = re.compile(r"^[0-9]{8}$")
MAX_NETWORK_ARCS = 10000


def predict_trip(trip_id: str, model_name: str = "mlp") -> dict:
    """Generate classifier-only and classifier + Viterbi predictions."""

    start_time = time.perf_counter()

    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise ValueError("Trip ID must contain exactly eight digits.")

    if model_name not in ALLOWED_MODELS:
        raise ValueError("Model must be either 'knn' or 'mlp'.")

    trip_dir = STUDENT_TEST_DIR / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip {trip_id} was not found in student_test."
        )

    # Inference-safe loading: never reads the reference .route file.
    trip = load_trip(trip_dir)
    metric = project_trip(trip)

    gps_geometry = [
        [float(lat), float(lon)]
        for lon, lat in trip.track[:, :2]
    ]

    with (CONFIG_DIR / "candidates.json").open(
        "r", encoding="utf-8"
    ) as file:
        candidate_config = json.load(file)

    feature_table = build_features(
        metric,
        radius_m=float(candidate_config["radius_m"]),
        k=int(candidate_config["k"]),
    )

    if feature_table.empty:
        raise RuntimeError("No candidate road arcs were generated.")

    scorer = load_scorer(MODEL_DIR / "member1" / model_name)

    probabilities = np.asarray(
        scorer.predict_proba(feature_table),
        dtype=float,
    ).reshape(-1)

    if (
        len(probabilities) != len(feature_table)
        or not np.isfinite(probabilities).all()
    ):
        raise RuntimeError("Invalid model prediction scores.")

    # Build one candidate group per GPS observation.
    scored = feature_table.copy()
    scored["probability"] = probabilities

    candidate_groups = [
        group.sort_values(
            ["distance_m", "arc_id"],
            kind="stable",
        ).reset_index(drop=True)
        for _, group in scored.groupby("obs_idx", sort=True)
    ]

    if len(candidate_groups) != len(trip.track):
        raise RuntimeError(
            "Candidate groups do not cover every GPS observation."
        )

    # Classifier-only: choose the highest probability candidate.
    classifier_arcs = [
        int(
            group.loc[
                group["probability"].idxmax(),
                "arc_id",
            ]
        )
        for group in candidate_groups
    ]

    # Viterbi: select the best connected sequence.
    time_gaps = np.diff(trip.track[:, 2]).astype(float)

    gps_distances = np.linalg.norm(
        np.diff(metric.xy, axis=0),
        axis=1,
    )

    viterbi_arcs = decode_viterbi(
        candidate_groups=candidate_groups,
        arc_start_nodes=trip.arcs[:, 0],
        arc_end_nodes=trip.arcs[:, 1],
        gps_distances=gps_distances,
        time_gaps=time_gaps,
        transition_penalty=3.0,
        distance_weight=0.01,
        gap_threshold=120.0,
    )

    # Convert road segments into map-ready WGS84 coordinates.
    inverse_transform = Transformer.from_crs(
        CRS.from_proj4(metric.crs),
        CRS.from_epsg(4326),
        always_xy=True,
    )

    def convert_arcs_to_geometry(selected_arc_ids):
        segments = []

        for arc_id in selected_arc_ids:
            start_node, end_node = trip.arcs[int(arc_id)]
            endpoints_xy = metric.node_xy[[start_node, end_node]]

            longitudes, latitudes = inverse_transform.transform(
                endpoints_xy[:, 0],
                endpoints_xy[:, 1],
            )

            segments.append([
                [float(latitudes[0]), float(longitudes[0])],
                [float(latitudes[1]), float(longitudes[1])],
            ])

        return segments

    classifier_geometry = convert_arcs_to_geometry(classifier_arcs)
    viterbi_geometry = convert_arcs_to_geometry(viterbi_arcs)

    candidate_arc_ids = np.unique(
        feature_table["arc_id"].to_numpy(dtype=int)
    )

    if len(candidate_arc_ids) > MAX_NETWORK_ARCS:
        indices = np.linspace(
            0,
            len(candidate_arc_ids) - 1,
            MAX_NETWORK_ARCS,
            dtype=int,
        )
        candidate_arc_ids = candidate_arc_ids[indices]

    road_geometry = convert_arcs_to_geometry(candidate_arc_ids)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "trip_id": trip_id,
        "model": model_name,
        "status": "success",
        "message": "Classifier-only and Viterbi predictions completed.",
        "n_observations": int(len(trip.track)),
        "n_predicted_observations": int(len(viterbi_arcs)),
        "n_road_segments": int(len(road_geometry)),
        "gps_geometry": gps_geometry,
        "road_geometry": road_geometry,
        "classifier_arcs": classifier_arcs,
        "classifier_geometry": classifier_geometry,
        "predicted_arcs": viterbi_arcs,
        "predicted_geometry": viterbi_geometry,
        "timing_ms": round(elapsed_ms, 2),
    }
