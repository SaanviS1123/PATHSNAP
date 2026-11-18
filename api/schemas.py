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


class PredictionMetrics(BaseModel):
    top1_accuracy: float | None = None
    route_continuity: float | None = None


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    message: str
    n_observations: int | None = None
    predicted_arcs: list[Any] = Field(default_factory=list)
    predicted_geometry: list[Any] = Field(default_factory=list)
    metrics: PredictionMetrics | None = None
    timing_ms: float | None = None
