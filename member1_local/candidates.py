
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
TRIP = BASE / "Student folder" / "train" / "00000001"

gps = np.loadtxt(TRIP / "00000001.track")
nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)
route = np.loadtxt(
    TRIP / "00000001.route", dtype=int
).reshape(-1)

# Convert longitude/latitude into approximate local metres.
lat0 = np.radians(np.mean(nodes[:, 1]))
scale_x = 111320 * np.cos(lat0)
scale_y = 111132

def project(coords):
    return np.column_stack((
        coords[:, 0] * scale_x,
        coords[:, 1] * scale_y
    ))

gps_xy = project(gps[:, :2])
nodes_xy = project(nodes)

starts = nodes_xy[arcs[:, 0]]
ends = nodes_xy[arcs[:, 1]]

# Spatial index of segment midpoints.
midpoints = (starts + ends) / 2
tree = cKDTree(midpoints)

# Exact point-to-segment distance for candidate arcs.
def segment_distances(point, ids):
    a = starts[ids]
    b = ends[ids]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)

    t = np.divide(
        np.sum((point - a) * ab, axis=1),
        denom,
        out=np.zeros(len(ids)),
        where=denom > 0
    )
    t = np.clip(t, 0, 1)
    nearest = a + t[:, None] * ab
    return np.linalg.norm(point - nearest, axis=1)

K = 30
correct = 0
total = 0

print("First 5 GPS observations and nearby roads:\n")

for i, point in enumerate(gps_xy[:5]):
    _, ids = tree.query(point, k=K)
    ids = np.atleast_1d(ids)
    distances = segment_distances(point, ids)
    order = np.argsort(distances)
    best_ids = ids[order[:5]]

    print(f"GPS observation {i}:")
    for arc_id in best_ids:
        d = segment_distances(point, np.array([arc_id]))[0]
        print(f"  Arc {arc_id}: {d:.2f} metres")
    print()

print("Candidate generation completed.")
