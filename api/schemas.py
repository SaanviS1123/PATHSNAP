"""Public contract: reference IDs, labels and geometry are absent."""
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(min_length=1, max_length=32)
    model_name: str = "best"


class MethodMetrics(BaseModel):
    sequence_precision: float
    sequence_recall: float
    sequence_f1: float
    route_continuity: float | None
    pointwise_accuracy: float | None


class PredictionMetrics(BaseModel):
    classifier: MethodMetrics
    viterbi: MethodMetrics
    n_known_obs: int
    n_unknown_obs: int


class PredictResponse(BaseModel):
    trip_id: str
    model: str
    status: str = "success"
    message: str
    n_observations: int
    n_road_segments: int
    gps_geometry: list[list[float]]
    timestamps: list[float]
    road_geometry: list[list[list[float]]]
    classifier_arcs: list[int]
    classifier_geometry: list[list[list[float]]]
    predicted_arcs: list[int]
    predicted_geometry: list[list[list[float]]]
    metrics: PredictionMetrics | None = None
    timing_ms: float
    stages_ms: dict[str, float]
    diagnostics: dict
