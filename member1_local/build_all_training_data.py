
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(__file__).resolve().parent
TRAIN = BASE / "Student folder" / "train"

K = 500
TOP_N = 10
MAX_POINTS = 500

all_rows = []
summary = []

start_time = time.time()

for trip_dir in sorted(TRAIN.iterdir()):
    if not trip_dir.is_dir():
        continue

    trip_id = trip_dir.name

    try:
        gps = np.loadtxt(trip_dir / f"{trip_id}.track", ndmin=2)
        nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", ndmin=2)
        arcs = np.loadtxt(
            trip_dir / f"{trip_id}.arcs", dtype=int, ndmin=2
        )
        route = np.loadtxt(
            trip_dir / f"{trip_id}.route", dtype=int
        ).reshape(-1)

        lat0 = np.radians(nodes[:, 1].mean())
        scale = np.array([
            111320 * np.cos(lat0),
            111132
        ])

        nodes_xy = nodes[:, :2] * scale
        gps_xy = gps[:, :2] * scale

        starts = nodes_xy[arcs[:, 0]]
        ends = nodes_xy[arcs[:, 1]]
        vectors = ends - starts

        lengths = np.linalg.norm(vectors, axis=1)
        midpoints = (starts + ends) / 2

        tree = cKDTree(midpoints)

        route_ids = route[
            (route >= 0) & (route < len(arcs))
        ]

        if len(route_ids) == 0:
            print("Skipping", trip_id, "- empty valid route")
            continue

        route_tree = cKDTree(midpoints[route_ids])

        # Sample GPS observations uniformly along each trip.
        sample_ids = np.unique(
            np.linspace(
                0, len(gps) - 1,
                min(MAX_POINTS, len(gps)),
                dtype=int
            )
        )

        trip_rows = []
        missed = 0

        for i in sample_ids:
            point = gps_xy[i]

            _, candidate_ids = tree.query(
                point, k=min(K, len(arcs))
            )
            candidate_ids = np.atleast_1d(candidate_ids)

            a = starts[candidate_ids]
            ab = vectors[candidate_ids]
            denom = np.sum(ab * ab, axis=1)

            t = np.divide(
                np.sum((point - a) * ab, axis=1),
                denom,
                out=np.zeros(len(candidate_ids)),
                where=denom > 0
            )
            t = np.clip(t, 0, 1)

            snapped = a + t[:, None] * ab
            distances = np.linalg.norm(point - snapped, axis=1)

            selected = np.argsort(distances)[:TOP_N]
            chosen_ids = candidate_ids[selected]

            # Provisional route-constrained pseudo-label.
            _, ref_indices = route_tree.query(
                point, k=min(30, len(route_ids))
            )
            ref_indices = np.atleast_1d(ref_indices)
            ref_ids = route_ids[ref_indices]

            ra = starts[ref_ids]
            rab = vectors[ref_ids]
            rdenom = np.sum(rab * rab, axis=1)

            rt = np.divide(
                np.sum((point - ra) * rab, axis=1),
                rdenom,
                out=np.zeros(len(ref_ids)),
                where=rdenom > 0
            )
            rt = np.clip(rt, 0, 1)

            ref_snap = ra + rt[:, None] * rab
            ref_dist = np.linalg.norm(point - ref_snap, axis=1)

            label_arc = int(ref_ids[np.argmin(ref_dist)])

            if label_arc not in chosen_ids:
                missed += 1

            previous = max(0, i - 1)
            dt = float(gps[i, 2] - gps[previous, 2])
            dt = max(0, dt)

            delta = gps_xy[i] - gps_xy[previous]
            speed = (
                float(np.linalg.norm(delta) / dt)
                if dt > 0 else 0.0
            )

            gps_bearing = (
                np.degrees(np.arctan2(delta[0], delta[1]))
                + 360
            ) % 360

            for j in selected:
                arc_id = int(candidate_ids[j])
                v = vectors[arc_id]

                road_bearing = (
                    np.degrees(np.arctan2(v[0], v[1]))
                    + 360
                ) % 360

                angle = abs(gps_bearing - road_bearing)
                angle = min(angle, 360 - angle)

                trip_rows.append({
                    "trip_id": trip_id,
                    "observation_id": int(i),
                    "arc_id": arc_id,
                    "distance_m": float(distances[j]),
                    "segment_length_m": float(lengths[arc_id]),
                    "bearing_difference": float(angle),
                    "speed_mps": speed,
                    "time_delta_s": dt,
                    "is_correct_arc": int(arc_id == label_arc)
                })

        all_rows.extend(trip_rows)

        summary.append({
            "trip_id": trip_id,
            "sampled_points": len(sample_ids),
            "candidate_misses": missed,
            "coverage_percent": round(
                100 * (1 - missed / len(sample_ids)), 2
            )
        })

        print(
            f"Trip {trip_id}: "
            f"{len(sample_ids)} observations, "
            f"{len(trip_rows)} rows, "
            f"coverage {summary[-1]['coverage_percent']:.2f}%",
            flush=True
        )

    except Exception as exc:
        print(f"ERROR in trip {trip_id}: {exc}", flush=True)

df = pd.DataFrame(all_rows)
df.to_csv(BASE / "all_training_candidates.csv", index=False)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(BASE / "candidate_coverage_summary.csv", index=False)

print("\nMULTI-TRIP DATASET COMPLETE")
print("Trips processed:", len(summary_df))
print("Candidate rows:", len(df))
print("Positive labels:", int(df["is_correct_arc"].sum()))
print(
    "Mean trip coverage:",
    round(summary_df["coverage_percent"].mean(), 2),
    "%"
)
print("Elapsed seconds:", round(time.time() - start_time, 1))
print("\nSaved all_training_candidates.csv")
print("Saved candidate_coverage_summary.csv")
