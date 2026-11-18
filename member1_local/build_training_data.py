
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
TRIP = BASE / "Student folder" / "train" / "00000001"

gps = np.loadtxt(TRIP / "00000001.track")
nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)

labels = pd.read_csv(BASE / "training_labels_trip1.csv")
features = pd.read_csv(BASE / "gps_features.csv")

lat0 = np.radians(nodes[:, 1].mean())
sx = 111320 * np.cos(lat0)
sy = 111132

def project(coords):
    return np.column_stack((
        coords[:, 0] * sx,
        coords[:, 1] * sy
    ))

gps_xy = project(gps[:, :2])
nodes_xy = project(nodes)

starts = nodes_xy[arcs[:, 0]]
ends = nodes_xy[arcs[:, 1]]
vectors = ends - starts

lengths = np.linalg.norm(vectors, axis=1)
road_bearings = (
    np.degrees(np.arctan2(vectors[:, 0], vectors[:, 1]))
    + 360
) % 360

tree = cKDTree((starts + ends) / 2)

rows = []
K = 500
TOP_N = 10

for i, point in enumerate(gps_xy):
    _, ids = tree.query(point, k=min(K, len(arcs)))
    ids = np.atleast_1d(ids)

    a = starts[ids]
    ab = vectors[ids]
    denom = np.sum(ab * ab, axis=1)

    t = np.divide(
        np.sum((point - a) * ab, axis=1),
        denom,
        out=np.zeros(len(ids)),
        where=denom > 0
    )

    t = np.clip(t, 0, 1)
    snapped = a + t[:, None] * ab
    distances = np.linalg.norm(point - snapped, axis=1)

    selected = np.argsort(distances)[:TOP_N]

    for j in selected:
        arc_id = int(ids[j])

        gps_bearing = features.iloc[i]["bearing_deg"]
        road_bearing = road_bearings[arc_id]

        angle_diff = abs(gps_bearing - road_bearing)
        angle_diff = min(angle_diff, 360 - angle_diff)

        rows.append({
            "trip_id": "00000001",
            "observation_id": i,
            "arc_id": arc_id,
            "distance_m": float(distances[j]),
            "segment_length_m": float(lengths[arc_id]),
            "bearing_difference": float(angle_diff),
            "speed_mps": float(features.iloc[i]["speed_mps"]),
            "time_delta_s": float(features.iloc[i]["time_delta_s"]),
            "is_correct_arc": int(
                arc_id == int(labels.iloc[i]["arc_id"])
            )
        })

df = pd.DataFrame(rows)
df.to_csv(BASE / "candidate_training_trip1.csv", index=False)

print("TRAINING DATASET CREATED")
print("Total candidate rows:", len(df))
print("Positive candidate rows:", df["is_correct_arc"].sum())
print("Negative candidate rows:",
      (df["is_correct_arc"] == 0).sum())
print("Unique GPS observations:", df["observation_id"].nunique())
print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))
print("\nSaved candidate_training_trip1.csv")
