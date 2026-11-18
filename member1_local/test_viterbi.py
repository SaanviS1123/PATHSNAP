
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from viterbi_decoder import decode_viterbi

TRIP = "00000001"

FEATURES = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

# Load the original Trip 1 candidate table.
df = pd.read_csv("candidate_training_trip1.csv")

# -------------------------------------------------
# FIX: Find the road-network files recursively.
# -------------------------------------------------
train_dir = Path("Student folder") / "train"

arc_matches = sorted(
    train_dir.rglob(f"{TRIP}.arcs")
)

if not arc_matches:
    raise FileNotFoundError(
        f"Could not find {TRIP}.arcs inside {train_dir}. "
        "Check your dataset folder structure."
    )

arcs_path = arc_matches[0]

print("Found arcs:", arcs_path)

arcs = np.loadtxt(arcs_path, dtype=int)

# -------------------------------------------------
# Load tuned model and predict candidate probabilities.
# -------------------------------------------------
model = joblib.load("pathsnap_tuned_model.joblib")

df["probability"] = model.predict_proba(
    df[FEATURES].fillna(0)
)[:, 1]

groups = [
    group.sort_values("distance_m").reset_index(drop=True)
    for _, group in df.groupby(
        "observation_id",
        sort=True
    )
]

# -------------------------------------------------
# Classifier-only predictions.
# -------------------------------------------------
classifier_predictions = [
    int(
        group.loc[
            group["probability"].idxmax(),
            "arc_id"
        ]
    )
    for group in groups
]

# Road-network arc endpoints (0-based node IDs).
arc_start_nodes = arcs[:, 0]
arc_end_nodes = arcs[:, 1]

# -------------------------------------------------
# Viterbi predictions.
# -------------------------------------------------
viterbi_predictions = decode_viterbi(
    candidate_groups=groups,
    arc_start_nodes=arc_start_nodes,
    arc_end_nodes=arc_end_nodes,
    transition_penalty=3.0,
    distance_weight=0.01
)

# -------------------------------------------------
# Reference-derived provisional labels.
# -------------------------------------------------
reference = []

for group in groups:
    positives = group.loc[
        group["is_correct_arc"] == 1,
        "arc_id"
    ]

    if len(positives):
        reference.append(int(positives.iloc[0]))
    else:
        reference.append(None)


def top1_accuracy(predictions):
    valid = [
        i
        for i, ref in enumerate(reference)
        if ref is not None
    ]

    if not valid:
        return float("nan")

    return float(np.mean([
        predictions[i] == reference[i]
        for i in valid
    ]))


def direct_continuity(predictions):
    if len(predictions) < 2:
        return 1.0

    connected = 0

    for a, b in zip(
        predictions[:-1],
        predictions[1:]
    ):
        if (
            a == b
            or arc_end_nodes[a] == arc_start_nodes[b]
        ):
            connected += 1

    return connected / (len(predictions) - 1)


# -------------------------------------------------
# Results.
# -------------------------------------------------
print("\nVITERBI TEST — TRIP", TRIP)
print("GPS observations:", len(groups))

print(
    "Observations with provisional labels in candidates:",
    sum(ref is not None for ref in reference)
)

print("\nClassifier-only")
print(
    "Top-1 accuracy:",
    round(top1_accuracy(classifier_predictions), 4)
)
print(
    "Direct continuity:",
    round(direct_continuity(classifier_predictions), 4)
)

print("\nClassifier + Viterbi")
print(
    "Top-1 accuracy:",
    round(top1_accuracy(viterbi_predictions), 4)
)
print(
    "Direct continuity:",
    round(direct_continuity(viterbi_predictions), 4)
)

# -------------------------------------------------
# Save predictions.
# -------------------------------------------------
output = pd.DataFrame({
    "observation_id": [
        int(group["observation_id"].iloc[0])
        for group in groups
    ],
    "classifier_arc": classifier_predictions,
    "viterbi_arc": viterbi_predictions,
    "reference_arc": reference
})

output.to_csv(
    "viterbi_trip1_comparison.csv",
    index=False
)

print("\nSaved viterbi_trip1_comparison.csv")
