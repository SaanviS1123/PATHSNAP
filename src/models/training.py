"""Schema-checked loading, reproducible signatures and the public scorer export."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.harness import _positive_probability, fit_estimator, validate_table

ROOT=Path(__file__).resolve().parents[2]
TRAIN_PATH=ROOT/'data/processed/train.parquet'
SCHEMA_PATH=ROOT/'models/feature_schema.json'
MODEL_DIR=ROOT/'models'


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),allow_nan=False)+'\n')


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def input_signature(features,estimator=None,extra=None):
    import sklearn
    paths=['data/processed/train.parquet','data/processed/student_test.parquet','data/processed/obs_labels.parquet','configs/folds.json','models/feature_schema.json','results/metrics/data_manifest.csv','src/evaluation/harness.py','src/evaluation/continuity.py','src/models/training.py','src/models/specs.py']
    inputs={p:file_hash(ROOT/p) for p in paths if (ROOT/p).exists()}
    payload=dict(inputs=inputs,features=features,sklearn_version=sklearn.__version__,config=estimator.get_params() if estimator is not None else None,extra=extra)
    key=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
    return key,payload


def load_training_data(path=TRAIN_PATH):
    frame=pd.read_parquet(path)
    schema=json.loads(SCHEMA_PATH.read_text())
    features=[f['name'] for f in schema['features']]
    validate_table(frame,features)
    return frame,features


class CandidateScorer:
    """Public contract: raw DataFrame -> positive probabilities, shape (n,)."""
    def __init__(self,estimator,preprocessor,feature_columns,calibrator=None):
        self.estimator=estimator
        self.preprocessor=preprocessor
        self.feature_columns=list(feature_columns)
        self.calibrator=calibrator

    def predict_proba(self,X):
        if not set(self.feature_columns).issubset(X.columns):
            raise ValueError('Missing schema feature columns')
        values=X[self.feature_columns]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Nonfinite features')
        if len(values)==0:
            return np.empty(0)
        if not isinstance(self.preprocessor,str):
            values=self.preprocessor.transform(values)
        scores=_positive_probability(self.estimator,values)
        if self.calibrator is not None:
            scores=self.calibrator.predict(scores)
        # Finite log emissions; no reference files are read at inference.
        return np.clip(scores,np.finfo(float).eps,1-np.finfo(float).eps)


def scorer_from_fitted(estimator,features,calibrator=None):
    if isinstance(estimator,Pipeline):
        return CandidateScorer(estimator.steps[-1][1],estimator[:-1],features,calibrator)
    return CandidateScorer(estimator,'passthrough',features,calibrator)


def export_scorer(scorer,name,features,output_dir,card):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    schema=json.loads(SCHEMA_PATH.read_text())
    with tempfile.TemporaryDirectory(prefix='export-',suffix='.tmp',dir=out) as temporary:
        stage=Path(temporary)
        joblib.dump(scorer,stage/'model.joblib',compress=3)
        joblib.dump(scorer.preprocessor,stage/'preprocessor.joblib',compress=3)
        for path in stage.glob('*.joblib'):
            if path.stat().st_size>=100_000_000:
                raise ValueError(f'Artifact exceeds 100 MB: {path.name}')
        write_json(stage/'feature_schema.json',schema)
        card=dict(card,model_name=name,feature_schema_version=schema['version'],feature_columns=features,
                  probability_output='model.joblib accepts raw schema features and includes preprocessing/calibration; do not preprocess twice',
                  artifacts_sha256={p.name:file_hash(p) for p in stage.iterdir()})
        write_json(stage/'model_card.json',card)
        for filename in ['model.joblib','preprocessor.joblib','feature_schema.json','model_card.json']:
            (stage/filename).replace(out/filename)
    return card


def load_scorer(model_dir=MODEL_DIR):
    folder=Path(model_dir)
    card=json.loads((folder/'model_card.json').read_text())
    schema=json.loads((folder/'feature_schema.json').read_text())
    if card['feature_schema_version'] != schema['version']:
        raise ValueError('Model/schema version mismatch')
    for name,expected in card['artifacts_sha256'].items():
        if file_hash(folder/name)!=expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return joblib.load(folder/'model.joblib')


def fit_and_export(estimator,name,frame,features,*,scale=False,random_seed=42,metrics=None,output_dir=MODEL_DIR):
    if scale:
        estimator=make_pipeline(StandardScaler(),estimator)
    fitted,fit=fit_estimator(estimator,frame,features)
    scorer=scorer_from_fitted(fitted,features)
    signature,inputs=input_signature(features,estimator)
    card=export_scorer(scorer,name,features,output_dir,dict(status='provisional',seed=random_seed,fit=fit,metrics=metrics or {},signature=signature,inputs=inputs))
    return scorer,card
