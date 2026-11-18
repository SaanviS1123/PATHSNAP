"""Training-trip-only audit and OOF sigmoid calibration of a selected model."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from src.evaluation.harness import fit_estimator,_positive_probability,score_predictions
from src.models.training import ROOT,write_json


class SigmoidCalibrator:
    def __init__(self):
        self.model=LogisticRegression(C=1e6,max_iter=200,random_state=42)

    @staticmethod
    def logits(scores):
        p=np.clip(np.asarray(scores,dtype=float),1e-12,1-1e-12)
        return (np.log(p)-np.log1p(-p)).reshape(-1,1)

    def fit(self,scores,y):
        with threadpool_limits(limits=2):
            self.model.fit(self.logits(scores),y)
        return self

    def predict(self,scores):
        return self.model.predict_proba(self.logits(scores))[:,1]

    @property
    def increasing(self):
        return bool(self.model.coef_[0,0]>0)


def audit_calibration(estimator,frame,features,labels):
    fold_map=json.loads((ROOT/'configs/folds.json').read_text())['trip_to_fold']
    assignment=frame.trip_id.map(fold_map)
    groups=sorted(set(fold_map.values()))
    calibration_fold,audit_fold=groups[-1],groups[-2]
    fitting=frame[~assignment.isin([calibration_fold,audit_fold])]
    calibration=frame[assignment==calibration_fold]
    audit=frame[assignment==audit_fold]
    model,fit=fit_estimator(estimator,fitting,features)
    with threadpool_limits(limits=2):
        calibration_scores=_positive_probability(model,calibration[features])
        audit_scores=_positive_probability(model,audit[features])
    known=calibration.is_ref>=0
    calibrator=SigmoidCalibrator().fit(calibration_scores[known],calibration.loc[known,'is_ref'])
    audit_labels=labels[labels.trip_id.isin(audit.trip_id.unique())]
    raw=score_predictions(audit,audit_scores,audit_labels)
    calibrated=score_predictions(audit,calibrator.predict(audit_scores),audit_labels)
    accepted=calibrator.increasing and calibrated['log_loss']<raw['log_loss'] and calibrated['brier_score']<=raw['brier_score']
    result=dict(method='sigmoid on clipped log odds',accepted=bool(accepted),
                rule='Strictly increasing slope, lower audit log loss and no higher audit Brier score',
                base_fit_trip_ids=sorted(fitting.trip_id.unique()),calibration_trip_ids=sorted(calibration.trip_id.unique()),
                audit_trip_ids=sorted(audit.trip_id.unique()),raw=raw,calibrated=calibrated,fit=fit,
                note='Training-only calibration audit after family/parameter selection; not an unbiased estimate of the full selected pipeline. Student test is not consulted.')
    write_json(ROOT/'results/metrics/calibration_audit.json',result)
    return bool(accepted),result


def fit_oof_calibrator(frame,oof):
    known=frame.is_ref.to_numpy()>=0
    calibrator=SigmoidCalibrator().fit(np.asarray(oof)[known],frame.is_ref.to_numpy()[known])
    if not calibrator.increasing:
        raise ValueError('Final calibration must preserve candidate ranking')
    return calibrator
