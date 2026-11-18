
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
TRIP = BASE / "Student folder" / "train" / "00000001"

gps = np.loadtxt(TRIP / "00000001.track")
nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)
route = np.loadtxt(
    TRIP / "00000001.route", dtype=int
).reshape(-1)

lat0 = np.radians(np.mean(nodes[:, 1]))
sx = 111320 * np.cos(lat0)
sy = 111132

def project(coords):
    return np.column_stack((coords[:, 0] * sx, coords[:, 1] * sy))

gps_xy = project(gps[:, :2])
nodes_xy = project(nodes)

a = nodes_xy[arcs[route, 0]]
b = nodes_xy[arcs[route, 1]]

vectors = b - a
lengths = np.linalg.norm(vectors, axis=1)

# Cumulative distance along the reference route
cumulative = np.concatenate(([0], np.cumsum(lengths)))
total_length = cumulative[-1]

# Spatial index of reference-route midpoints
tree = cKDTree((a + b) / 2)

labels = []
positions = []
distances = []

previous_position = 0.0
K = min(30, len(route))

for point in gps_xy:
    _, candidate_indices = tree.query(point, k=K)
    candidate_indices = np.atleast_1d(candidate_indices)

    best = None
    best_score = float("inf")

    for j in candidate_indices:
        v = vectors[j]
        denominator = np.dot(v, v)

        t = (
            np.clip(np.dot(point - a[j], v) / denominator, 0, 1)
            if denominator > 0 else 0
        )

        snapped = a[j] + t * v
        distance = np.linalg.norm(point - snapped)
        position = cumulative[j] + t * lengths[j]

        # Penalize backwards movement along the route
        backward = max(0, previous_position - position)
        score = distance + 2.0 * backward

        if score < best_score:
            best_score = score
            best = (route[j], position, distance)

    arc_id, position, distance = best
    previous_position = max(previous_position, position)

    labels.append(arc_id)
    positions.append(position)
    distances.append(distance)

df = pd.DataFrame({
    "longitude": gps[:, 0],
    "latitude": gps[:, 1],
    "timestamp": gps[:, 2],
    "arc_id": labels,
    "route_position_m": positions,
    "snap_distance_m": distances
})

output = BASE / "training_labels_trip1.csv"
df.to_csv(output, index=False)

print("Generated labels:", len(df))
print("Unique matched arcs:", df["arc_id"].nunique())
print("Median snap distance:", round(df["snap_distance_m"].median(), 2))
print("95th percentile snap distance:",
      round(df["snap_distance_m"].quantile(0.95), 2))
print("Maximum snap distance:", round(df["snap_distance_m"].max(), 2))
print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))
print("\nSaved:", output.name)
