
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
TRIP = BASE / "Student folder" / "train" / "00000001"

nodes = np.loadtxt(TRIP / "00000001.nodes")
arcs = np.loadtxt(TRIP / "00000001.arcs", dtype=int)
route = np.loadtxt(
    TRIP / "00000001.route", dtype=int
).reshape(-1)

print("Total nodes:", len(nodes))
print("Total arcs:", len(arcs))
print("Route length:", len(route))

valid = (route >= 0) & (route < len(arcs))
print("Valid route IDs:", int(valid.sum()))
print("Invalid route IDs:", int((~valid).sum()))

if not valid.all():
    raise ValueError("Route contains invalid arc IDs")

segments = arcs[route]

# Directed connectivity: end of arc i equals start of arc i+1
directed = segments[:-1, 1] == segments[1:, 0]

# Shared endpoints, regardless of direction
shared = (
    (segments[:-1, 0] == segments[1:, 0])
    | (segments[:-1, 0] == segments[1:, 1])
    | (segments[:-1, 1] == segments[1:, 0])
    | (segments[:-1, 1] == segments[1:, 1])
)

print("\nDirected connected pairs:", int(directed.sum()))
print("Pairs sharing an endpoint:", int(shared.sum()))
print("Total consecutive pairs:", len(route) - 1)

print("\nFirst 10 reference arcs:")
for arc_id, (start, end) in zip(route[:10], segments[:10]):
    print(f"Arc {arc_id}: node {start} -> node {end}")
