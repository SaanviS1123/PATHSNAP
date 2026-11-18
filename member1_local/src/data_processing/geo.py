
import numpy as np


class LocalMetricProjection:
    """
    Local equirectangular projection.

    Input: longitude, latitude in degrees.
    Output: approximate east/north coordinates in metres.

    Intended for local trip-scale geometry.
    """

    def __init__(self, coordinates):
        coordinates = np.asarray(
            coordinates, dtype=float
        )

        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError(
                "Coordinates must have longitude and latitude columns"
            )

        self.lon0 = float(np.mean(coordinates[:, 0]))
        self.lat0 = float(np.mean(coordinates[:, 1]))

        self.radius = 6371008.8
        self.cos_lat0 = np.cos(
            np.radians(self.lat0)
        )

    def transform(self, coordinates):
        coordinates = np.asarray(
            coordinates, dtype=float
        )

        lon = coordinates[..., 0]
        lat = coordinates[..., 1]

        x = (
            np.radians(lon - self.lon0)
            * self.radius
            * self.cos_lat0
        )

        y = (
            np.radians(lat - self.lat0)
            * self.radius
        )

        return np.stack([x, y], axis=-1)

    def inverse(self, xy):
        xy = np.asarray(xy, dtype=float)

        x = xy[..., 0]
        y = xy[..., 1]

        lon = (
            self.lon0
            + np.degrees(
                x / (
                    self.radius * self.cos_lat0
                )
            )
        )

        lat = (
            self.lat0
            + np.degrees(
                y / self.radius
            )
        )

        return np.stack([lon, lat], axis=-1)


if __name__ == "__main__":
    from pathlib import Path
    from src.data_processing.loaders import load_trip

    train_root = Path("Student folder") / "train"

    trip_folders = sorted(
        folder
        for folder in train_root.iterdir()
        if folder.is_dir()
    )

    # Choose two trips with the greatest difference
    # in mean latitude, if available.
    trip_locations = []

    for folder in trip_folders:
        trip = load_trip(folder)

        trip_locations.append((
            float(np.mean(trip["track"][:, 1])),
            folder,
            trip
        ))

    trip_locations.sort(key=lambda item: item[0])

    selected = [
        trip_locations[0],
        trip_locations[-1]
    ]

    print("A3 LOCAL PROJECTION TEST")

    for _, folder, trip in selected:
        gps = trip["track"][:, :2]

        projection = LocalMetricProjection(
            trip["nodes"]
        )

        xy = projection.transform(gps)
        restored = projection.inverse(xy)

        error = np.max(
            np.abs(restored - gps)
        )

        print("\nTrip:", folder.name)
        print(
            "Mean latitude:",
            round(projection.lat0, 5)
        )
        print(
            "First GPS point (metres):",
            np.round(xy[0], 2)
        )
        print(
            "Maximum inverse error (degrees):",
            error
        )

        assert error < 1e-8

    print("\nA3 PROJECTION TEST PASSED")
