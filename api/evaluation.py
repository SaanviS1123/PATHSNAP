"""Server-only reference boundary. Call only after inference has completed."""
from dataclasses import replace
from src.data_processing.loaders import load_trip
from src.features.labels import align_labels
from src.evaluation.route_metrics import pointwise_accuracy, route_continuity, sequence_metrics
from src.hmm.viterbi import route_segments
from api.schemas import MethodMetrics, PredictionMetrics


def evaluate_prediction(folder, result, params):
    reference_trip = load_trip(folder, include_route=True)
    labels = align_labels(replace(result.network.metric, trip=reference_trip))

    def measure(predictions, segments):
        frame = predictions.assign(trip_id=reference_trip.trip_id)
        point = pointwise_accuracy(frame, labels)
        seq = sequence_metrics(segments, reference_trip.route)
        return MethodMetrics(**{key: seq[key] for key in ("sequence_precision", "sequence_recall", "sequence_f1")},
                             route_continuity=route_continuity(frame, network=result.network),
                             pointwise_accuracy=point["pointwise_accuracy"])

    return PredictionMetrics(
        classifier=measure(result.classifier, route_segments(result.classifier, result.network, params, bridge=False)),
        viterbi=measure(result.decoded, result.segments),
        n_known_obs=int((labels.ref_arc >= 0).sum()), n_unknown_obs=int((labels.ref_arc < 0).sum()))
