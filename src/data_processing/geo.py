"""Per-trip WGS84 -> local azimuthal equidistant projection, always lon/lat."""
from dataclasses import dataclass
import numpy as np
from pyproj import CRS, Transformer
from src.data_processing.loaders import Trip


@dataclass
class MetricTrip:
    trip: Trip
    xy: np.ndarray
    node_xy: np.ndarray
    start: np.ndarray
    vector: np.ndarray
    length: np.ndarray
    crs: str


def project_trip(trip: Trip) -> MetricTrip:
    # Circular mean avoids a wrong centre for trajectories crossing the dateline.
    radians = np.deg2rad(trip.track[:, 0])
    lon = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    lat = trip.track[:, 1].mean()
    crs = f'+proj=aeqd +lat_0={lat:.10f} +lon_0={lon:.10f} +datum=WGS84 +units=m'
    transform = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(crs), always_xy=True)
    xy = np.column_stack(transform.transform(*trip.track[:, :2].T))
    node_xy = np.column_stack(transform.transform(*trip.nodes.T))
    start = node_xy[trip.arcs[:, 0]]
    vector = node_xy[trip.arcs[:, 1]] - start
    return MetricTrip(trip, xy, node_xy, start, vector, np.linalg.norm(vector, axis=1), crs)


def motion(metric: MetricTrip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = np.zeros_like(metric.xy)
    delta[1:] = np.diff(metric.xy, axis=0)
    dt = np.zeros(len(delta))
    dt[1:] = np.diff(metric.trip.track[:, 2])
    distance = np.linalg.norm(delta, axis=1)
    valid = (dt > 0) & (distance > 1)
    unit = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
    speed = np.divide(distance, dt, out=np.zeros_like(dt), where=dt > 0)
    return unit, dt, speed, valid


def segment_projection(points: np.ndarray, start: np.ndarray, vector: np.ndarray):
    denominator = np.sum(vector * vector, axis=-1)
    fraction = np.divide(np.sum((points - start) * vector, axis=-1), denominator,
                         out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    fraction = np.clip(fraction, 0, 1)
    distance = np.linalg.norm(points - start - fraction[..., None] * vector, axis=-1)
    return distance, fraction
