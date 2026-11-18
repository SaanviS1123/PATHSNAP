"""Acceptance evidence must remain consistent with delivered inputs and PDF."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def test_live_api_results_match_frozen_decoder_results():
    integration=pd.read_csv(ROOT/'results/metrics/integration_run.csv',dtype={'trip_id':str})
    best=integration[integration.model=='best']
    assert len(best)==10 and best.trip_id.nunique()==10
    assert set(integration.model)=={'best','knn','mlp','logistic_regression','random_forest','hist_gradient_boosting'}
    expected=pd.read_csv(ROOT/'results/metrics/hmm_comparison_by_trip.csv',dtype={'trip_id':str})
    expected=expected[(expected.partition=='student_test') & (expected.variant=='hmm')].set_index('trip_id')
    for row in best.itertuples():
        assert abs(row.hmm_accuracy-expected.loc[row.trip_id,'pointwise_accuracy'])<1e-12
        assert row.server_ms>0 and row.http_wall_ms>=row.server_ms
        assert len(row.response_sha256)==64


def test_browser_and_supervisor_acceptance_evidence():
    browser=json.loads((ROOT/'results/metrics/ui_integration.json').read_text())
    assert len(browser['checks'])==16 and not browser['page_errors']
    assert not browser['physical_second_laptop']
    assert browser['hmm_toggle_verified'] and browser['batch_demo_verified']
    assert browser['model_switches_verified']==6
    deployment=json.loads((ROOT/'results/metrics/deployment_checks.json').read_text())
    assert deployment['18000']==deployment['18501']==200
    assert deployment['shutdown_children_stopped']
    assert not deployment['public_tunnel_verified']
    assert not deployment['physical_second_laptop_verified']


def test_consolidated_pdf_and_manifest():
    manifest=json.loads((ROOT/'results/metrics/report_manifest.json').read_text())
    pdf=ROOT/manifest['report']
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==manifest['report_sha256']
    for name,expected in manifest['inputs'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    reader=PdfReader(pdf)
    text='\n'.join(page.extract_text() for page in reader.pages)
    for heading in ['Objective and delivered system','Dataset understanding','Bayesian hyperparameter optimisation','HMM/Viterbi','API, dashboard and deployment','References and evidence index']:
        assert heading in text
    assert 'Browser acceptance evidence is not yet available' not in text
    assert 'Public tunnel verification remains blocked' in text
    assert len(reader.pages)>20 and '\ufffd' not in text
