"""Persistent, seeded TPE optimisation of the observed baseline CV winner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import joblib
import pandas as pd
from src.evaluation.harness import evaluate_model
from src.evaluation.continuity import route_continuity
from src.models.specs import make_estimator,suggest_params,decode_trial_params
from src.models.training import ROOT,load_training_data,input_signature,file_hash,write_json


def tune(data_path=None,trials=10,seed=42,output_path=None,model_name=None,frame=None,features=None,obs_labels=None):
    import optuna
    if trials<4:
        raise ValueError('At least four trials are required: three startup plus a TPE proposal')
    if frame is None:
        frame,features=load_training_data(data_path) if data_path else load_training_data()
    if obs_labels is None:
        obs_labels=pd.read_parquet(ROOT/'data/processed/obs_labels.parquet')
        obs_labels=obs_labels[obs_labels.partition=='train']
    if model_name is None:
        selection=json.loads((ROOT/'results/metrics/model_selection.json').read_text())
        model_name=selection['baseline_winner']
    signature,inputs=input_signature(features,extra=dict(family=model_name,seed=seed,search_source=file_hash(__file__)))
    cache=ROOT/'data/processed/member2'
    cache.mkdir(parents=True,exist_ok=True)
    records=ROOT/'results/metrics/member2'
    records.mkdir(parents=True,exist_ok=True)
    study_name=f'{model_name}-{signature[:16]}'
    sampler_path=cache/f'{study_name}_sampler.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=seed,n_startup_trials=3)
    study=optuna.create_study(direction='maximize',sampler=sampler,study_name=study_name,
                             storage=f'sqlite:///{cache / (study_name+".sqlite3")}',load_if_exists=True)
    started=time.perf_counter()

    def objective(trial):
        params=suggest_params(trial,model_name)
        estimator=make_estimator(model_name,params,seed)
        trial_started=time.perf_counter()
        result=evaluate_model(estimator,f'{model_name} TPE trial {trial.number}',frame,features,
                              obs_labels=obs_labels,continuity_fn=route_continuity,
                              output_path=records/f'trial_{trial.number}_progress.json',
                              oof_path=cache/f'{study_name}_trial_{trial.number}_oof.npy')
        result.update(signature=signature,trial=trial.number,params=params,elapsed_s=time.perf_counter()-trial_started)
        write_json(records/f'trial_{trial.number}.json',result)
        trial.set_user_attr('elapsed_s',result['elapsed_s'])
        trial.set_user_attr('macro_f1',result['macro_f1'])
        trial.set_user_attr('log_loss',result['log_loss'])
        return result['top1_accuracy']

    def persist(study,trial):
        joblib.dump(study.sampler,sampler_path)
        rows=[]
        for item in study.trials:
            rows.append(dict(trial=item.number,state=item.state.name,objective_top1_accuracy=item.value,
                             proposal='startup' if item.number<3 else 'TPE',params_json=json.dumps(item.params),
                             elapsed_s=item.user_attrs.get('elapsed_s'),macro_f1=item.user_attrs.get('macro_f1'),
                             log_loss=item.user_attrs.get('log_loss')))
        pd.DataFrame(rows).to_csv(ROOT/'results/metrics/tuning_trials.csv',index=False)

    study.optimize(objective,n_trials=max(0,trials-len(study.trials)),callbacks=[persist])
    persist(study,None)
    best=study.best_trial
    result=json.loads((records/f'trial_{best.number}.json').read_text())
    summary=dict(model_name=model_name,objective='mean observation Top-1 across all five frozen trip folds',
                 n_trials=len(study.trials),n_startup_trials=3,n_tpe_trials=sum(t.number>=3 for t in study.trials),
                 best_trial=best.number,best_top1_accuracy=best.value,
                 best_params=decode_trial_params(model_name,best.params),sampler='Optuna TPESampler',seed=seed,
                 total_compute_seconds=sum(t.user_attrs.get('elapsed_s',0) for t in study.trials),
                 current_invocation_seconds=time.perf_counter()-started,signature=signature,inputs=inputs,
                 search_space={k:str(v) for k,v in best.distributions.items()},
                 oof_path=str((cache/f'{study_name}_trial_{best.number}_oof.npy').relative_to(ROOT)))
    write_json(ROOT/'results/metrics/tuning_best.json',summary)
    return summary,result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--model',dest='model_name')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(tune(trials=args.trials,seed=args.seed,model_name=args.model)[0])
