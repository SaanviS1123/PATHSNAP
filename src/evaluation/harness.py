"""Shared, trip-grouped candidate-classification evaluation (B1)."""
from __future__ import annotations
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
FOLDS_PATH = ROOT / 'configs/folds.json'
REQUIRED_COLUMNS = {'trip_id', 'obs_idx', 'arc_id', 'is_ref'}


def make_group_splits(groups, n_splits=5, folds_path=None, seed=42):
    groups = np.asarray(groups).astype(str)
    path = Path(folds_path) if folds_path is not None else FOLDS_PATH
    if path.exists():
        raw = json.loads(path.read_text())
        mapping = raw.get('trip_to_fold', raw)
        if not isinstance(mapping, dict) or not all(type(v) is int for v in mapping.values()):
            raise ValueError('Fold assignments must map trip IDs to integer fold IDs')
        if set(groups)-set(mapping):
            raise ValueError('Frozen folds do not assign every trip')
        assignments = np.array([mapping[g] for g in groups])
        folds = sorted(set(assignments))
        if len(folds) < 2:
            raise ValueError('Need at least two represented validation folds')
        for fold in folds:
            yield np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
    else:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(np.zeros(len(groups)), groups=groups)


def _positive_probability(estimator, X):
    if len(X) == 0:
        return np.empty(0)
    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.ndim == 2:
        classes = list(estimator.classes_)
        if 1 not in classes:
            raise ValueError('Estimator has no positive class 1')
        probabilities = probabilities[:, classes.index(1)]
    if probabilities.shape != (len(X),) or not np.isfinite(probabilities).all():
        raise ValueError('Probabilities must be a finite shape-(n,) vector')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must lie in [0,1]')
    return probabilities.astype(float)


def winners(frame, scores):
    result = frame[['trip_id', 'obs_idx', 'arc_id']].copy().reset_index(drop=True)
    result['score'] = scores
    return result.sort_values(['trip_id', 'obs_idx', 'score', 'arc_id'], ascending=[True, True, False, True]).drop_duplicates(['trip_id', 'obs_idx'])


def calibration_bins(y, scores, n_bins=10):
    rows = []
    bins = np.minimum((np.asarray(scores)*n_bins).astype(int), n_bins-1)
    for i in range(n_bins):
        selected = bins == i
        if selected.any():
            rows.append(dict(bin=i, count=int(selected.sum()), mean_probability=float(np.mean(scores[selected])), positive_fraction=float(np.mean(y[selected]))))
    return rows


def candidate_miss_rate(candidates, obs_labels):
    labels = obs_labels[obs_labels.ref_arc >= 0][['trip_id', 'obs_idx', 'ref_arc']]
    if labels.empty:
        return None
    present = candidates[['trip_id', 'obs_idx', 'arc_id']].drop_duplicates()
    matched = labels.merge(present, left_on=['trip_id','obs_idx','ref_arc'], right_on=['trip_id','obs_idx','arc_id'], how='left', indicator=True, validate='one_to_one')
    return float((matched['_merge'] == 'left_only').mean())


def score_predictions(frame, scores, obs_labels, continuity_fn=None):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all() or ((scores < 0)|(scores > 1)).any():
        raise ValueError('Invalid probability vector')
    known = frame.is_ref.to_numpy() >= 0
    labels = obs_labels[obs_labels.ref_arc >= 0]
    if not known.any() or labels.empty:
        raise ValueError('No known labels to evaluate')
    if labels.duplicated(['trip_id','obs_idx']).any():
        raise ValueError('Duplicate observation labels')
    y, p = frame.is_ref.to_numpy()[known], scores[known]
    predicted = p >= .5
    best = winners(frame, scores)
    joined = labels[['trip_id','obs_idx','ref_arc']].merge(best, on=['trip_id','obs_idx'], how='left', validate='one_to_one')
    bins = calibration_bins(y, p)
    return dict(candidate_accuracy=float(accuracy_score(y,predicted)),
                precision=float(precision_score(y,predicted,zero_division=0)),
                recall=float(recall_score(y,predicted,zero_division=0)),
                f1=float(f1_score(y,predicted,zero_division=0)),
                macro_f1=float(f1_score(y,predicted,labels=[0,1],average='macro',zero_division=0)),
                log_loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
                brier_score=float(brier_score_loss(y,p)),
                ece_10=float(sum(r['count']/len(y)*abs(r['mean_probability']-r['positive_fraction']) for r in bins)),
                calibration_bins=bins,
                top1_accuracy=float((joined.arc_id == joined.ref_arc).mean()),
                trip_mean_top1=float(joined.assign(correct=joined.arc_id == joined.ref_arc).groupby('trip_id').correct.mean().mean()),
                candidate_miss_rate=candidate_miss_rate(frame,obs_labels),
                route_continuity=continuity_fn(best) if continuity_fn else None,
                n_known_obs=len(labels), n_unknown_obs=int((obs_labels.ref_arc < 0).sum()),
                n_candidate_rows=len(frame), n_known_candidate_rows=int(known.sum()),
                confusion_matrix=confusion_matrix(y,predicted,labels=[0,1]).tolist())


def validate_table(frame, features):
    if not REQUIRED_COLUMNS.issubset(frame):
        raise ValueError('Missing candidate identifiers or labels')
    if not set(frame.is_ref.unique()).issubset({-1,0,1}):
        raise ValueError('is_ref must be -1, 0 or 1')
    if not features or set(features) & REQUIRED_COLUMNS:
        raise ValueError('Features must exclude identifiers and labels')
    if not np.isfinite(frame[features].to_numpy()).all():
        raise ValueError('Nonfinite features')
    if frame.duplicated(['trip_id','obs_idx','arc_id']).any():
        raise ValueError('Duplicate candidate keys')


def fit_estimator(estimator, frame, features):
    known = frame.is_ref >= 0
    if frame.loc[known,'is_ref'].nunique() != 2:
        raise ValueError('Fitting requires both known classes')
    model = clone(estimator)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        model.fit(frame.loc[known,features],frame.loc[known,'is_ref'].astype(int))
    fit = dict(train_time_sec=time.perf_counter()-started, n_fit_rows=int(known.sum()), n_fit_trips=int(frame.loc[known,'trip_id'].nunique()), warnings=[str(w.message) for w in caught])
    return model, fit


def evaluate_model(estimator, model_name, data, feature_columns=None, config=None,
                   random_seed=42, obs_labels=None, n_splits=5, folds_path=None,
                   continuity_fn=None, output_path=None, log=True, oof_path=None):
    features = feature_columns or [c for c in data.columns if c not in REQUIRED_COLUMNS]
    validate_table(data,features)
    if obs_labels is None:
        raise ValueError('Independent observation labels are required for miss-aware Top-1')
    oof = np.full(len(data),np.nan)
    folds, histories = [], []
    for fold,(train_idx,val_idx) in enumerate(make_group_splits(data.trip_id,n_splits,folds_path,random_seed)):
        print(f'{model_name}: CV fold {fold}',flush=True)
        training, validation = data.iloc[train_idx],data.iloc[val_idx]
        model, fit = fit_estimator(estimator,training,features)
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            scores = _positive_probability(model,validation[features])
        inference = time.perf_counter()-started
        oof[val_idx] = scores
        labels = obs_labels[obs_labels.trip_id.isin(validation.trip_id.unique())]
        result = score_predictions(validation,scores,labels,continuity_fn)
        folds.append(dict(fold=fold,**result,**fit,inference_time_sec=inference))
        terminal = model.steps[-1][1] if hasattr(model,'steps') else model
        if hasattr(terminal,'loss_curve_'):
            histories.append(dict(fold=fold,kind='training_loss',values=list(map(float,terminal.loss_curve_))))
        if hasattr(terminal,'staged_predict_proba'):
            # Real per-stage empirical training loss on a fixed <=10k fit-row
            # diagnostic. No invented history or row-level early-stop validation.
            diagnostic = training[training.is_ref >= 0].sample(n=min(10000,int((training.is_ref >= 0).sum())),random_state=random_seed)
            with threadpool_limits(limits=2):
                losses = [float(log_loss(diagnostic.is_ref,prob,labels=[0,1])) for prob in terminal.staged_predict_proba(diagnostic[features])]
            histories.append(dict(fold=fold,kind='fit_sample_log_loss',values=losses))
        if output_path:
            Path(output_path).parent.mkdir(parents=True,exist_ok=True)
            Path(output_path).write_text(json.dumps(dict(model_name=model_name,folds=folds,histories=histories),indent=2)+'\n')
    if not np.isfinite(oof).all():
        raise ValueError('OOF coverage incomplete')
    if oof_path:
        Path(oof_path).parent.mkdir(parents=True,exist_ok=True)
        np.save(oof_path,oof.astype('float64'))
    keys = ['candidate_accuracy','precision','recall','f1','macro_f1','log_loss','brier_score','ece_10','top1_accuracy','trip_mean_top1','candidate_miss_rate','route_continuity','train_time_sec','inference_time_sec']
    summary = {key:float(np.mean([r[key] for r in folds])) if all(r[key] is not None for r in folds) else None for key in keys}
    summary.update(model_name=model_name,random_seed=random_seed,n_folds=len(folds),folds=folds,histories=histories,
                   macro_f1_std=float(np.std([r['macro_f1'] for r in folds],ddof=1)),
                   config=config or estimator.get_params(),fold_metrics_json=json.dumps(folds),training_history_json=json.dumps(histories))
    return summary


def evaluate_holdout(estimator,train_data,test_data,obs_labels,feature_columns,*,model_name='model',continuity_fn=None,return_estimator=False):
    if set(train_data.trip_id) & set(test_data.trip_id):
        raise ValueError('Training and holdout trips overlap')
    validate_table(train_data,feature_columns)
    validate_table(test_data,feature_columns)
    model,fit = fit_estimator(estimator,train_data,feature_columns)
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        scores=_positive_probability(model,test_data[feature_columns])
    inference=time.perf_counter()-started
    labels=obs_labels[obs_labels.trip_id.isin(test_data.trip_id.unique())]
    result=dict(model_name=model_name,**score_predictions(test_data,scores,labels,continuity_fn),**fit,inference_time_sec=inference)
    return (result,model,scores) if return_estimator else result
