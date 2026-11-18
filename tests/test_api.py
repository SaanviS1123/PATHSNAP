
"""Automated tests for the PATHSNAP API."""

from fastapi.testclient import TestClient

from api.main import app
import api.main as api_main


def create_test_trip(root, trip_id):
    """Create a nested student-testing trip fixture."""
    trip_dir = root / trip_id
    trip_dir.mkdir(parents=True)

    track_file = trip_dir / f"{trip_id}.track"
    track_file.write_text(
        "54.963519 56.251559 1.000000\n"
        "54.963600 56.251600 2.000000\n"
    )

    # A reference file may exist, but the prediction endpoint
    # must not return its contents.
    route_file = trip_dir / f"{trip_id}.route"
    route_file.write_text("100 200 300\n")


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_trips(tmp_path, monkeypatch):
    create_test_trip(tmp_path, "00000000")
    create_test_trip(tmp_path, "00000007")

    monkeypatch.setattr(
        api_main, "STUDENT_TEST_DIR", tmp_path
    )

    client = TestClient(app)
    response = client.get("/trips")

    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert response.json()["trips"] == [
        "00000000",
        "00000007",
    ]


def test_valid_trip_returns_pipeline_not_connected(
    tmp_path, monkeypatch
):
    create_test_trip(tmp_path, "00000000")

    monkeypatch.setattr(
        api_main, "STUDENT_TEST_DIR", tmp_path
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"trip_id": "00000000"},
    )

    assert response.status_code == 503
    assert "pipeline is not connected" in response.json()["detail"]


def test_invalid_trip_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main, "STUDENT_TEST_DIR", tmp_path
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"trip_id": "../bad"},
    )

    assert response.status_code == 400


def test_nonexistent_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        api_main, "STUDENT_TEST_DIR", tmp_path
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"trip_id": "99999999"},
    )

    assert response.status_code == 404

