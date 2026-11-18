"""Exercise real HTTP inference; persist measurements without reference data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def wait_prediction(base, trip, model, timeout=900):
    response = requests.post(base + "/jobs", json=dict(trip_id=trip, model_name=model), timeout=20)
    response.raise_for_status()
    job = response.json()["job_id"]
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        state = requests.get(base + f"/jobs/{job}", timeout=20)
        state.raise_for_status()
        if state.json()["status"] in ("complete", "failed"):
            response = requests.get(base + f"/jobs/{job}/result", timeout=60)
            response.raise_for_status()
            return response.json()
        time.sleep(.25)
    raise TimeoutError(f"Prediction {trip}/{model} still running after {timeout}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--all-models", action="store_true", help="all six configurations on every trip, rather than baselines on the first trip")
    args = parser.parse_args()
    base = args.api_url.rstrip("/")
    def get(path):
        response = requests.get(base + path, timeout=20)
        response.raise_for_status()
        return response.json()
    assert get("/health")["ready"]
    trips = get("/trips")["trips"]
    models = get("/models")["models"]
    assert len(trips) == 10, f"Expected ten student trips, found {len(trips)}"
    requests_to_run = [(trip, "best") for trip in trips]
    requests_to_run += [(trip, model["id"]) for trip in (trips if args.all_models else trips[:1]) for model in models if model["id"] != "best"]
    rows = []
    for trip, model in requests_to_run:
        started = time.perf_counter()
        result = wait_prediction(base, trip, model)
        wall_ms = (time.perf_counter()-started)*1000
        assert result["trip_id"] == trip and result["model"] == model
        assert len(result["classifier_arcs"]) == len(result["predicted_arcs"]) == result["n_observations"]
        assert all(arc >= 0 for arc in result["predicted_arcs"])
        assert not {"reference", "reference_route", "reference_geometry", "ref_arc", "route_position", "is_ref"}.intersection(result)
        metrics = result["metrics"]
        assert metrics is not None, "Demo reference evaluation must be available"
        rows.append(dict(trip_id=trip, model=model, n_observations=result["n_observations"],
                         server_ms=result["timing_ms"], http_wall_ms=wall_ms,
                         classifier_accuracy=metrics["classifier"]["pointwise_accuracy"],
                         hmm_accuracy=metrics["viterbi"]["pointwise_accuracy"],
                         hmm_sequence_f1=metrics["viterbi"]["sequence_f1"],
                         hmm_continuity=metrics["viterbi"]["route_continuity"],
                         response_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()))
        print(f"PASS trip={trip} model={model} points={result['n_observations']} server_ms={result['timing_ms']:.2f} classifier={rows[-1]['classifier_accuracy']:.4f} hmm={rows[-1]['hmm_accuracy']:.4f}", flush=True)
    target = ROOT / "results/metrics/integration_run.csv"
    pd.DataFrame(rows).to_csv(target,index=False)
    print(f"Saved {len(rows)} real HTTP results to {target}")


if __name__ == "__main__":
    main()
