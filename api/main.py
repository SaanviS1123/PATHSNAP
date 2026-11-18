"""Main FastAPI application for PATHSNAP."""

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import (
    API_HOST,
    API_PORT,
    LOG_LEVEL,
    STUDENT_TEST_DIR,
)
from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TripListResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("PATHSNAP")

TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("PATHSNAP API starting...")
    yield
    logger.info("PATHSNAP API shutting down...")


app = FastAPI(
    title="PATHSNAP API",
    description="Distributed Map-Matching and Route Reconstruction",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend to communicate with the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
    """Return API health status."""
    return HealthResponse(
        status="healthy",
        project="PATHSNAP",
        version="1.0.0",
    )


@app.get("/trips", response_model=TripListResponse, tags=["Trips"])
def list_trips():
    """List permitted student-testing trip IDs."""
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
def predict_trip(request: PredictRequest):
    """Predict a route for an allowed student-testing trip.

    Ground-truth .route files are never returned by this endpoint.
    The real pipeline will be connected during team integration.
    """
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


    # Real prediction pipeline will be integrated here.
    raise HTTPException(
        status_code=503,
        detail=(
            "Prediction pipeline is not connected yet. "
            "No prediction has been generated."
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
