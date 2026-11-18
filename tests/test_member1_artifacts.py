"""Portable real fitted artifacts; skip until model runs have finished."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('name', ['knn', 'mlp'])
def test_model_roundtrip_in_fresh_process(name, tmp_path):
    folder = Path('models/member1') / name
    if not (folder / 'model_card.json').exists():
        pytest.skip('Run the full model experiment first')
    output = tmp_path / 'predictions.npy'
    # A new interpreter verifies the standard artifacts and scorer adapter.
    script = '''import numpy as np,pandas as pd,sys
from src.models.member1_harness import load_scorer
scorer=load_scorer(sys.argv[1])
table=pd.read_parquet('tests/fixtures/member1/candidates.parquet')
probability=scorer.predict_proba(table)
assert probability.shape==(len(table),)
assert np.isfinite(probability).all()
assert ((probability>=0)&(probability<=1)).all()
np.save(sys.argv[2],probability)
'''
    subprocess.run([sys.executable, '-c', script, str(folder), str(output)], check=True)
    assert np.load(output).size > 0
    card = json.loads((folder / 'model_card.json').read_text())
    assert card['fit']['n_fit_trips'] == 80
    assert card['max_train_rows'] >= 0
    experiments = pd.read_csv(f'results/metrics/{name}_experiments.csv')
    assert experiments[experiments.partition == 'cv'].fold.nunique() == 5
    if Path('data/processed/obs_labels.parquet').exists():
        labels = pd.read_parquet('data/processed/obs_labels.parquet')
        student_labels = labels[labels.partition == 'student_test']
        assert experiments[experiments.partition == 'student_test'].n_known_obs.iloc[0] == int((student_labels.ref_arc >= 0).sum())
        assert card['student_test']['n_unknown_obs'] == int((student_labels.ref_arc < 0).sum())
    for path in folder.glob('*.joblib'):
        assert path.stat().st_size < 100_000_000, f'Artifact exceeds GitHub file limit: {path}'
