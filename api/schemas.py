
"""Request and response schemas for the PATHSNAP API."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str


class TripListResponse(BaseModel):
    trips: list[str]
    count: int


class PredictRequest(BaseModel):
    trip_id: str = Field(
        ...,
        min_length=1,
        description="ID of a permitted student-testing trip",
    )


class MethodMetrics(BaseModel):
    sequence_precision: float
    sequence_recall: float
    sequence_f1: float
    route_continuity: float


class PredictionMetrics(BaseModel):
    classifier: MethodMetrics
    viterbi: MethodMetrics


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    message: str
    n_observations: int | None = None
    n_road_segments: int | None = None

    gps_geometry: list[Any] = Field(default_factory=list)
    road_geometry: list[Any] = Field(default_factory=list)

    classifier_arcs: list[int] = Field(default_factory=list)
    classifier_geometry: list[Any] = Field(default_factory=list)

    predicted_arcs: list[int] = Field(default_factory=list)
    predicted_geometry: list[Any] = Field(default_factory=list)

    metrics: PredictionMetrics | None = None
    timing_ms: float | None = None
