"""Boundary tests for real inference, async jobs and reference isolation."""
import time
from concurrent.futures import Future
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
import api.main as api_main
import src.pipeline as pipeline
from api.main import app, JobStore
from api.schemas import PredictResponse, PredictRequest
from src.data_processing.synthetic import write_fixture


@pytest.fixture
def client(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(api_main, "STUDENT_TEST_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "STUDENT_TEST_DIR", tmp_path)
    with TestClient(app) as client:
        yield client


def test_catalog_and_validation(client):
    assert client.get('/health').json()['ready']
    assert client.get('/trips').json()['count'] == 5
    catalog = client.get('/models').json()
    assert len(catalog['models']) == 6
    assert sum(item['selected'] for item in catalog['models']) == 1
    for payload, status in [({'trip_id':'../bad'},400),({'trip_id':'99999999'},404),
                            ({'trip_id':'00000001','model_name':'../../model'},400),
                            ({'trip_id':'00000001','ref_arc':[0]},422)]:
        assert client.post('/predict',json=payload).status_code == status


def test_real_sync_prediction_and_response_boundary(client):
    response = client.post('/predict',json={'trip_id':'00000001'})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['model'] == 'best' and len(data['predicted_arcs']) == data['n_observations'] == 12
    assert data['metrics']['n_known_obs'] > 0
    schema = str(PredictResponse.model_json_schema()).lower()
    for forbidden in ('ref_arc', 'reference_geometry', 'reference_route', 'route_position', 'is_ref'):
        assert forbidden not in schema and forbidden not in data
    assert np.isfinite(data['timing_ms']) and data['timing_ms'] > 0


def test_inference_never_reads_reference_or_processed_labels(tmp_path,monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(pipeline,'STUDENT_TEST_DIR',tmp_path)
    original=np.loadtxt
    def guarded(path,*args,**kwargs):
        assert not str(path).endswith('.route')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'loadtxt',guarded)
    import pandas as pd
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k: pytest.fail('Private table read during inference'))
    result=pipeline.predict_trip('00000001')
    assert result.network.metric.trip.route is None
    assert len(result.decoded)==12


def test_missing_reference_still_returns_prediction(client,tmp_path):
    (tmp_path/'00000001/00000001.route').unlink()
    response=client.post('/predict',json={'trip_id':'00000001'})
    assert response.status_code == 200
    assert response.json()['metrics'] is None
    assert response.json()['predicted_geometry']


def test_async_job_lifecycle_and_errors(client,monkeypatch):
    created=client.post('/jobs',json={'trip_id':'00000001','model_name':'mlp'})
    assert created.status_code == 202
    job=created.json()['job_id']
    for _ in range(100):
        status=client.get(f'/jobs/{job}').json()['status']
        if status in ('complete','failed'): break
        time.sleep(.02)
    assert status == 'complete'
    assert client.get(f'/jobs/{job}/result').json()['model'] == 'mlp'
    assert client.get('/jobs/not-a-job').status_code == 404
    def fail(_): raise ValueError('private/path should not reach client')
    monkeypatch.setattr(api_main,'run_prediction',fail)
    response=client.post('/predict',json={'trip_id':'00000001'})
    assert response.status_code == 500 and 'private/path' not in response.text


def test_queue_is_bounded_and_incomplete_result_is_not_read(client):
    store=client.app.state.jobs
    pending=[Future() for _ in range(4)]
    with store.lock:
        store.jobs={str(i):dict(future=f,request=PredictRequest(trip_id='00000001')) for i,f in enumerate(pending)}
    try:
        assert client.post('/jobs',json={'trip_id':'00000001'}).status_code == 429
        assert client.get('/jobs/0/result').status_code == 409
    finally:
        for future in pending: future.cancel()
