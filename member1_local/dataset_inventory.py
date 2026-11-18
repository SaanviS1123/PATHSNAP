
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "Student folder"

rows = []

for partition in ["train", "student_test"]:
    folder = DATA / partition

    for trip in sorted(folder.iterdir()):
        if not trip.is_dir():
            continue

        files = {
            ext: trip / f"{trip.name}.{ext}"
            for ext in ["track", "nodes", "arcs", "route"]
        }

        if not all(p.exists() for p in files.values()):
            print("Missing files:", trip.name)
            continue

        track = np.loadtxt(files["track"], ndmin=2)
        nodes = np.loadtxt(files["nodes"], ndmin=2)
        arcs = np.loadtxt(files["arcs"], ndmin=2)
        route = np.loadtxt(files["route"], ndmin=2)

        rows.append({
            "partition": partition,
            "trip_id": trip.name,
            "gps_points": len(track),
            "nodes": len(nodes),
            "arcs": len(arcs),
            "route_segments": len(route),
            "max_time_gap": (
                np.diff(track[:, 2]).max()
                if len(track) > 1 else 0
            )
        })

df = pd.DataFrame(rows)
df.to_csv(BASE / "dataset_inventory.csv", index=False)

print("\nDATASET INVENTORY")
print(df.groupby("partition").agg(
    trips=("trip_id", "count"),
    gps_points=("gps_points", "sum"),
    nodes=("nodes", "sum"),
    arcs=("arcs", "sum"),
    route_segments=("route_segments", "sum"),
    largest_gap=("max_time_gap", "max")
).to_string())

print("\nTRAINING TRIP SIZE STATISTICS")
print(
    df[df["partition"] == "train"]["gps_points"]
    .describe()
    .round(1)
    .to_string()
)

print("\nSaved dataset_inventory.csv")
