
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
TRIP = BASE / "Student folder" / "train" / "00000001"

gps = np.loadtxt(TRIP / "00000001.track")
nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)

labels = pd.read_csv(
    BASE / "training_labels_trip1.csv"
)["arc_id"].to_numpy(dtype=int)

lat0 = np.radians(np.mean(nodes[:, 1]))
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

tree = cKDTree((starts + ends) / 2)

def point_segment_distance(point, ids):
    a = starts[ids]
    ab = ends[ids] - a
    denom = np.sum(ab * ab, axis=1)

    t = np.divide(
        np.sum((point - a) * ab, axis=1),
        denom,
        out=np.zeros(len(ids)),
        where=denom > 0
    )

    t = np.clip(t, 0, 1)
    projected = a + t[:, None] * ab
    return np.linalg.norm(point - projected, axis=1)

print("Candidate coverage evaluation\n")

# for k in [10, 30, 50, 100]:
for k in [100, 200, 300, 500]:
    hits = 0

    for point, true_arc in zip(gps_xy, labels):
        _, ids = tree.query(point, k=k)
        ids = np.atleast_1d(ids)

        distances = point_segment_distance(point, ids)
        # top_ids = ids[np.argsort(distances)[:min(5, k)]]
        top_ids = ids[np.argsort(distances)[:min(10, k)]]

        if true_arc in top_ids:
            hits += 1

    coverage = 100 * hits / len(labels)

    print(
        f"Midpoint K={k:3d} | "
        # f"Top-5 candidate coverage: {coverage:.2f}% | "
        f"Top-10 candidate coverage: {coverage:.2f}% | "
        f"Miss rate: {100 - coverage:.2f}%"
    )
