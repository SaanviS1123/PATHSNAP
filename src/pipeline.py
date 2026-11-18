"""Live inference adapter. No reference files or saved predictions are read here."""
import re
import numpy as np
from api.config import STUDENT_TEST_DIR
from src.hmm.predict import predict_trip as infer_trip
from api.models import registry


def predict_trip(trip_id: str, model_name: str = "best", *, scorer=None, params=None):
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise ValueError("Trip ID must contain exactly eight digits.")
    if model_name not in registry.specs:
        raise ValueError("Unknown model.")
    folder = STUDENT_TEST_DIR / trip_id
    if not (folder / f"{trip_id}.track").is_file():
        raise FileNotFoundError("Trip not found in student_test.")
    return infer_trip(folder, scorer=scorer or registry.get(model_name), params=params or registry.params)


def response_geometry(result):
    """Return only observed GPS and predicted/network geometries in lat/lon."""
    trip = result.network.metric.trip

    def geometry(ids):
        ids = np.asarray(list(ids), dtype=int)
        ids = np.unique(ids[ids >= 0])
        return trip.nodes[trip.arcs[ids]][:, :, ::-1].tolist()

    metric = result.network.metric
    lower, upper = metric.xy.min(axis=0) - 200, metric.xy.max(axis=0) + 200
    endpoints = metric.node_xy[trip.arcs]
    near = np.flatnonzero((endpoints.max(axis=1) >= lower).all(axis=1) &
                          (endpoints.min(axis=1) <= upper).all(axis=1))[:10000]
    return dict(gps_geometry=trip.track[:, :2][:, ::-1].tolist(),
                timestamps=trip.track[:, 2].tolist(), road_geometry=geometry(near),
                classifier_arcs=result.classifier.arc_id.astype(int).tolist(),
                classifier_geometry=geometry(result.classifier.arc_id),
                predicted_arcs=result.decoded.arc_id.astype(int).tolist(),
                predicted_geometry=geometry(arc for part in result.segments for arc in part))
