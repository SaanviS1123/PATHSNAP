
"""Main FastAPI application for PATHSNAP."""

import logging
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_HOST, API_PORT, LOG_LEVEL, STUDENT_TEST_DIR
from api.schemas import (
    HealthResponse,
    MethodMetrics,
    PredictRequest,
    PredictResponse,
    PredictionMetrics,
    TripListResponse,
)
from src.pipeline import predict_trip as run_prediction


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("PATHSNAP")
TRIP_ID_PATTERN = re.compile(r"^[0-9]{8}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PATHSNAP API starting...")
    yield
    logger.info("PATHSNAP API shutting down...")


app = FastAPI(
    title="PATHSNAP API",
    description="Distributed Map-Matching and Route Reconstruction",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def collapse_consecutive(sequence):
    """Remove repeated adjacent arc IDs from a route sequence."""
    collapsed = []

    for value in sequence:
        value = int(value)
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)

    return collapsed


def lcs_length(first, second):
    """Return longest common subsequence length using linear memory."""
    if len(second) > len(first):
        first, second = second, first

    previous = [0] * (len(second) + 1)

    for value in first:
        current = [0] * (len(second) + 1)

        for index, other in enumerate(second, start=1):
            if value == other:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(
                    previous[index],
                    current[index - 1],
                )

        previous = current

    return previous[-1]


def calculate_method_metrics(predicted, reference, arcs):
    """Calculate sequence precision, recall, F1 and route continuity."""
    predicted = collapse_consecutive(predicted)

    matched = lcs_length(predicted, reference)

    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(reference) if reference else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    if len(predicted) > 1:
        connected = sum(
            arcs[first, 1] == arcs[second, 0]
            for first, second in zip(predicted[:-1], predicted[1:])
        )
        continuity = connected / (len(predicted) - 1)
    else:
        continuity = 0.0

    return MethodMetrics(
        sequence_precision=float(precision),
        sequence_recall=float(recall),
        sequence_f1=float(f1),
        route_continuity=float(continuity),
    )


def evaluate_prediction(trip_id, result):
    """Evaluate predictions server-side; never return reference arc IDs."""
    trip_dir = STUDENT_TEST_DIR / trip_id

    route_file = trip_dir / f"{trip_id}.route"
    arcs_file = trip_dir / f"{trip_id}.arcs"

    if not route_file.is_file() or not arcs_file.is_file():
        raise FileNotFoundError(
            "Reference route or road network is unavailable for evaluation."
        )

    reference = np.loadtxt(
        route_file,
        dtype=int,
        ndmin=1,
    ).ravel().tolist()

    arcs = np.loadtxt(
        arcs_file,
        dtype=int,
        ndmin=2,
    )

    if not reference:
        raise ValueError("Reference route is empty.")

    if min(reference) < 0 or max(reference) >= len(arcs):
        raise ValueError("Reference route contains invalid arc IDs.")

    return PredictionMetrics(
        classifier=calculate_method_metrics(
            result["classifier_arcs"],
            reference,
            arcs,
        ),
        viterbi=calculate_method_metrics(
            result["predicted_arcs"],
            reference,
            arcs,
        ),
    )


@app.get("/", tags=["General"])
def root():
    return {
        "project": "PATHSNAP",
        "message": "PATHSNAP API is running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy",
        project="PATHSNAP",
        version="1.0.0",
    )


@app.get("/trips", response_model=TripListResponse, tags=["Trips"])
def list_trips():
    if not STUDENT_TEST_DIR.exists():
        logger.warning(
            "Student-testing directory does not exist: %s",
            STUDENT_TEST_DIR,
        )
        return TripListResponse(trips=[], count=0)

    trip_ids = sorted(
        path.stem
        for path in STUDENT_TEST_DIR.glob("*/*.track")
        if path.is_file()
    )

    return TripListResponse(
        trips=trip_ids,
        count=len(trip_ids),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
)
def predict_trip_endpoint(request: PredictRequest):
    trip_id = request.trip_id.strip()

    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid trip ID format.",
        )

    track_file = STUDENT_TEST_DIR / trip_id / f"{trip_id}.track"

    if not track_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Trip not found in the student-testing partition.",
        )

    logger.info("Prediction requested for trip %s", trip_id)

    try:
        result = run_prediction(trip_id, model_name="mlp")

        logger.info(
            "Prediction completed for trip %s in %.2f ms",
            trip_id,
            result["timing_ms"],
        )

        metrics = None
        message = result["message"]

        try:
            metrics = evaluate_prediction(trip_id, result)
            logger.info(
                "Server-side evaluation completed for trip %s",
                trip_id,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning(
                "Evaluation unavailable for trip %s: %s",
                trip_id,
                exc,
            )
            message += " Reference evaluation is unavailable."

        return PredictResponse(
            trip_id=result["trip_id"],
            status=result["status"],
            message=message,
            n_observations=result["n_observations"],
            n_road_segments=result["n_road_segments"],
            gps_geometry=result["gps_geometry"],
            road_geometry=result["road_geometry"],
            classifier_arcs=result["classifier_arcs"],
            classifier_geometry=result["classifier_geometry"],
            predicted_arcs=result["predicted_arcs"],
            predicted_geometry=result["predicted_geometry"],
            metrics=metrics,
            timing_ms=result["timing_ms"],
        )

    except FileNotFoundError as exc:
        logger.error("Prediction input not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except (ValueError, RuntimeError, KeyError) as exc:
        logger.exception("Prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
