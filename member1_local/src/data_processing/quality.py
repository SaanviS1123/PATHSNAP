
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data_processing.loaders import load_trip

ROOT = Path("Student folder")
OUTPUT = Path("results/metrics")
OUTPUT.mkdir(parents=True, exist_ok=True)


def analyze_trip(folder, partition):
    trip = load_trip(folder)

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    timestamps = track[:, 2]
    gaps = np.diff(timestamps)

    # Treat the network as undirected for weak connectivity.
    adjacency = coo_matrix(
        (
            np.ones(len(arcs) * 2, dtype=np.int8),
            (
                np.concatenate([arcs[:, 0], arcs[:, 1]]),
                np.concatenate([arcs[:, 1], arcs[:, 0]])
            )
        ),
        shape=(len(nodes), len(nodes))
    ).tocsr()

    components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True
    )

    degrees = np.bincount(
        np.concatenate([arcs[:, 0], arcs[:, 1]]),
        minlength=len(nodes)
    )

    used_nodes = np.count_nonzero(degrees)

    file_sizes = {
        extension: (
            folder / f"{folder.name}.{extension}"
        ).stat().st_size
        for extension in ("track", "nodes", "arcs", "route")
    }

    inventory = {
        "partition": partition,
        "trip_id": folder.name,
        "gps_points": len(track),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "route_arcs": len(route),
        "track_bytes": file_sizes["track"],
        "nodes_bytes": file_sizes["nodes"],
        "arcs_bytes": file_sizes["arcs"],
        "route_bytes": file_sizes["route"]
    }

    quality = {
        "partition": partition,
        "trip_id": folder.name,
        "track_dtype": str(track.dtype),
        "nodes_dtype": str(nodes.dtype),
        "arcs_dtype": str(arcs.dtype),
        "missing_or_nonfinite": int(
            np.size(track) - np.isfinite(track).sum()
            + np.size(nodes) - np.isfinite(nodes).sum()
        ),
        "invalid_gps_coordinates": int(
            np.sum(
                (track[:, 0] < -180)
                | (track[:, 0] > 180)
                | (track[:, 1] < -90)
                | (track[:, 1] > 90)
            )
        ),
        "non_increasing_timestamps": int(
            np.sum(gaps <= 0)
        ),
        "gaps_over_120s": int(
            np.sum(gaps > 120)
        ),
        "max_gap_seconds": float(
            np.max(gaps) if len(gaps) else 0
        ),
        "min_longitude": float(track[:, 0].min()),
        "max_longitude": float(track[:, 0].max()),
        "min_latitude": float(track[:, 1].min()),
        "max_latitude": float(track[:, 1].max()),
        "weak_components": int(components),
        "largest_component_nodes": int(
            np.bincount(labels).max()
        ),
        "isolated_nodes": int(
            len(nodes) - used_nodes
        ),
        "directed_network": True,
        "reference_links_disconnected": int(
            np.sum(
                arcs[route[:-1], 1]
                != arcs[route[1:], 0]
            )
        )
    }

    return inventory, quality


def main():
    inventory_rows = []
    quality_rows = []

    for partition in ("train", "student_test"):
        folders = sorted(
            folder
            for folder in (ROOT / partition).iterdir()
            if folder.is_dir()
        )

        print(
            f"\nProcessing {partition}: "
            f"{len(folders)} trips"
        )

        for index, folder in enumerate(folders, 1):
            inventory, quality = analyze_trip(
                folder, partition
            )

            inventory_rows.append(inventory)
            quality_rows.append(quality)

            if index % 10 == 0 or index == len(folders):
                print(
                    f"  Completed {index}/{len(folders)}"
                )

    inventory_df = pd.DataFrame(inventory_rows)
    quality_df = pd.DataFrame(quality_rows)

    inventory_df.to_csv(
        OUTPUT / "inventory.csv",
        index=False
    )

    quality_df.to_csv(
        OUTPUT / "quality_report.csv",
        index=False
    )

    print("\nDATASET SUMMARY")
    print(
        inventory_df.groupby("partition")[
            ["trip_id", "gps_points", "nodes", "arcs"]
        ].agg({
            "trip_id": "count",
            "gps_points": "sum",
            "nodes": "sum",
            "arcs": "sum"
        })
    )

    print(
        "\nTotal isolated nodes:",
        quality_df["isolated_nodes"].sum()
    )

    print(
        "Trips with timestamp problems:",
        int(
            (quality_df[
                "non_increasing_timestamps"
            ] > 0).sum()
        )
    )

    print(
        "Disconnected reference links:",
        quality_df[
            "reference_links_disconnected"
        ].sum()
    )

    print("\nSaved:")
    print(OUTPUT / "inventory.csv")
    print(OUTPUT / "quality_report.csv")


if __name__ == "__main__":
    main()
