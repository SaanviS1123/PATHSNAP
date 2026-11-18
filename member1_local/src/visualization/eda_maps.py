
from pathlib import Path

import folium
import numpy as np

from src.data_processing.loaders import load_trip

ROOT = Path("Student folder") / "train"
OUTPUT = Path("maps")
TRIP_IDS = ["00000001", "00000033", "00000054"]


def make_map(trip_id):
    trip = load_trip(ROOT / trip_id)

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    center = [
        float(np.mean(track[:, 1])),
        float(np.mean(track[:, 0]))
    ]

    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # GPS trajectory
    gps_points = [
        [float(lat), float(lon)]
        for lon, lat in track[:, :2]
    ]

    folium.PolyLine(
        gps_points,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Observed GPS trajectory"
    ).add_to(m)

    # Reference route arcs
    route_layer = folium.FeatureGroup(
        name="Reference route",
        show=True
    )

    for arc_id in route:
        start_node, end_node = arcs[int(arc_id)]

        start_lon, start_lat = nodes[int(start_node)]
        end_lon, end_lat = nodes[int(end_node)]

        folium.PolyLine(
            [
                [float(start_lat), float(start_lon)],
                [float(end_lat), float(end_lon)]
            ],
            color="red",
            weight=3,
            opacity=0.65
        ).add_to(route_layer)

    route_layer.add_to(m)

    # Starting and ending points
    folium.CircleMarker(
        gps_points[0],
        radius=6,
        color="green",
        fill=True,
        tooltip="GPS start"
    ).add_to(m)

    folium.CircleMarker(
        gps_points[-1],
        radius=6,
        color="black",
        fill=True,
        tooltip="GPS end"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.fit_bounds(gps_points)

    output_path = OUTPUT / f"eda_trip_{trip_id}.html"
    m.save(str(output_path))

    return output_path, len(track), len(route)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("A8 INTERACTIVE MAP GENERATION")

    for trip_id in TRIP_IDS:
        path, gps_count, route_count = make_map(trip_id)

        print(
            f"Trip {trip_id}: "
            f"{gps_count} GPS points, "
            f"{route_count} reference arcs"
        )
        print("Saved:", path)

    print("\nA8 MAPS COMPLETE")


if __name__ == "__main__":
    main()
