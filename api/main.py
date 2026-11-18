"""Host-local API with bounded serial model jobs and server-only evaluation."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import re
from threading import Lock
import time
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from api.config import STUDENT_TEST_DIR, LOG_LEVEL
from api.models import registry
from api.schemas import PredictRequest, PredictResponse
from api.evaluation import evaluate_prediction
from src.pipeline import predict_trip, response_geometry

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PATHSNAP")


def run_prediction(request):
    started = time.perf_counter()
    result = predict_trip(request.trip_id, request.model_name)
    geometry = response_geometry(result)
    metrics = None
    message = "Live classifier and HMM prediction completed."
    try:
        metrics = evaluate_prediction(STUDENT_TEST_DIR / request.trip_id, result, registry.params)
    except (FileNotFoundError, ValueError, OSError):
        logger.warning("Reference evaluation unavailable for %s", request.trip_id, exc_info=True)
        message += " Reference evaluation unavailable."
    elapsed = (time.perf_counter() - started) * 1000
    payload = PredictResponse(trip_id=request.trip_id, model=request.model_name, message=message,
                              n_observations=len(result.decoded), n_road_segments=len(geometry["road_geometry"]),
                              **geometry, metrics=metrics, timing_ms=elapsed, stages_ms=result.timing_ms,
                              diagnostics=result.decoded.attrs.get("diagnostics", {}))
    logger.info("trip=%s model=%s points=%d latency_ms=%.2f classifier_accuracy=%s hmm_accuracy=%s",
                request.trip_id, request.model_name, len(result.decoded), elapsed,
                metrics.classifier.pointwise_accuracy if metrics else "unavailable",
                metrics.viterbi.pointwise_accuracy if metrics else "unavailable")
    return payload


def execute_job(request):
    try:
        return run_prediction(request)
    except Exception:
        logger.exception("trip=%s model=%s prediction failed", request.trip_id, request.model_name)
        raise


class JobStore:
    """One model worker, at most four outstanding requests, twelve retained results."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prediction")
        self.lock = Lock()
        self.jobs = {}

    def submit(self, request):
        with self.lock:
            if sum(not item["future"].done() for item in self.jobs.values()) >= 4:
                raise HTTPException(429, "Demo queue is full. Try again after the current jobs finish.")
            completed = [key for key, item in self.jobs.items() if item["future"].done()]
            while len(self.jobs) >= 12 and completed:
                del self.jobs[completed.pop(0)]
            job_id = uuid4().hex
            self.jobs[job_id] = dict(request=request, future=self.executor.submit(execute_job, request))
            return job_id

    def get(self, job_id):
        with self.lock:
            item = self.jobs.get(job_id)
            if item is None:
                raise HTTPException(404, "Job expired or not found. Run the prediction again.")
            return item


@asynccontextmanager
async def lifespan(app):
    registry.startup()
    app.state.jobs = JobStore()
    logger.info("Ready: best model + frozen HMM loaded; other scorers load once on demand")
    yield
    app.state.jobs.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="PATHSNAP API", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return dict(status="healthy", project="PATHSNAP", version="2.0.0", ready=registry.params is not None)


@app.get("/trips")
def trips():
    ids = sorted(path.name for path in STUDENT_TEST_DIR.glob("*")
                 if re.fullmatch(r"[0-9]{8}", path.name) and (path / f"{path.name}.track").is_file())
    return dict(trips=ids, count=len(ids))


@app.get("/models")
def models():
    return dict(models=registry.describe(), default="best",
                selection="Highest mean Top-1 across the five frozen training-trip folds; not selected on student test.",
                decoder="Frozen HMM tuned for the selected model. Other model + HMM runs are demonstrations, not separately tuned benchmarks.")


def validate(request):
    if not re.fullmatch(r"[0-9]{8}", request.trip_id):
        raise HTTPException(400, "Invalid trip ID format.")
    if request.model_name not in registry.specs:
        raise HTTPException(400, "Unknown model.")
    if not (STUDENT_TEST_DIR / request.trip_id / f"{request.trip_id}.track").is_file():
        raise HTTPException(404, "Trip not found in the student-testing partition.")


def job_result(item):
    try:
        return item["future"].result()
    except Exception as exc:
        logger.exception("Prediction job failed")
        raise HTTPException(500, "Prediction failed. Check the host API log for details.") from exc


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    validate(request)
    job_id = app.state.jobs.submit(request)
    return job_result(app.state.jobs.get(job_id))


@app.post("/jobs", status_code=202)
def create_job(request: PredictRequest):
    validate(request)
    return dict(job_id=app.state.jobs.submit(request))


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    item = app.state.jobs.get(job_id)
    future = item["future"]
    status = "failed" if future.done() and future.exception() else "complete" if future.done() else "running" if future.running() else "queued"
    return dict(job_id=job_id, status=status, trip_id=item["request"].trip_id, model=item["request"].model_name)


@app.get("/jobs/{job_id}/result", response_model=PredictResponse)
def completed_job(job_id: str):
    item = app.state.jobs.get(job_id)
    if not item["future"].done():
        raise HTTPException(409, "Prediction is still running.")
    return job_result(item)
