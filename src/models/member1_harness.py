"""Member 1's reusable runner while B1 is absent; keeps B's folders untouched.

Consumes the shared frozen schema/folds. B can reuse these results or call
run_model from its eventual common harness. Route continuity belongs to C1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.features.build import FEATURES, SCHEMA_VERSION

SEED = 42


class CandidateScorer:
    """Contract adapter: return P(is_ref=1), shape (n,), with schema ordering."""
    def __init__(self, model, preprocessor, features=FEATURES):
        self.model, self.preprocessor, self.features = model, preprocessor, list(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not set(self.features).issubset(X.columns):
            raise ValueError('Missing required feature columns')
        values = X[self.features]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Features must be finite')
        transformed = self.preprocessor.transform(values)
        positive = int(np.flatnonzero(self.model.classes_ == 1)[0])
        # Bounded prediction batches keep KNN query memory predictable.
        return np.concatenate([self.model.predict_proba(transformed[i:i+10000])[:, positive]
                               for i in range(0, len(transformed), 10000)]) if len(values) else np.empty(0)


def evaluate(table, labels, probabilities):
    """Known-label denominators include candidate misses; unknown labels excluded."""
    known_labels = labels[labels.ref_arc >= 0]
    known = table.is_ref.to_numpy() >= 0
    y, p = table.is_ref.to_numpy()[known], np.asarray(probabilities)[known]
    predicted = (p >= .5).astype(int)
    scored = table[['trip_id', 'obs_idx', 'arc_id']].copy()
    scored['probability'] = probabilities
    # Stable arc-ID tie breaking, independent of DataFrame input order.
    best = scored.sort_values(['trip_id', 'obs_idx', 'probability', 'arc_id'],
                              ascending=[True, True, False, True]).drop_duplicates(['trip_id', 'obs_idx'])
    joined = known_labels[['trip_id', 'obs_idx', 'ref_arc']].merge(best, on=['trip_id', 'obs_idx'], how='left', validate='one_to_one')
    hits = table.loc[table.is_ref == 1, ['trip_id', 'obs_idx']].drop_duplicates()
    per_trip = joined.assign(correct=joined.arc_id == joined.ref_arc).groupby('trip_id').correct.mean()
    return dict(candidate_accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                macro_f1=float(f1_score(y, predicted, labels=[0, 1], average='macro', zero_division=0)),
                log_loss=float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
                top1_acc=float((joined.arc_id == joined.ref_arc).mean()),
                trip_mean_top1=float(per_trip.mean()),
                candidate_miss_rate=float(1 - len(hits) / len(known_labels)),
                n_known_obs=len(known_labels), n_unknown_obs=int((labels.ref_arc < 0).sum()),
                n_candidate_rows=len(table), n_known_candidate_rows=int(known.sum()),
                confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist())


def training_sample(table, max_rows):
    known = table[table.is_ref >= 0]
    if max_rows and len(known) > max_rows:
        # Proportional sampling within every trip/class prevents long trips from
        # entirely crowding out shorter trips and retains observed class balance.
        fraction = max_rows / len(known)
        parts = [group.sample(n=max(1, int(round(len(group)*fraction))), random_state=SEED)
                 for _, group in known.groupby(['trip_id', 'is_ref'], sort=True)]
        return pd.concat(parts, ignore_index=True)
    return known


def estimator(name):
    if name == 'knn':
        return KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='kd_tree', leaf_size=40, n_jobs=2)
    if name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                             alpha=.001, batch_size=512, learning_rate_init=.001,
                             max_iter=40, early_stopping=False, n_iter_no_change=8, random_state=SEED)
    raise ValueError(name)


def fit_model(name, table, max_rows):
    sample = training_sample(table, max_rows)
    scaler = StandardScaler().fit(sample[FEATURES])
    model = estimator(name)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(scaler.transform(sample[FEATURES]), sample.is_ref)
    return CandidateScorer(model, scaler), dict(train_time_s=time.perf_counter()-start,
               n_fit_rows=len(sample), n_fit_trips=sample.trip_id.nunique(),
               fit_warnings=[str(x.message) for x in captured])


def run_model(name, processed_dir=Path('data/processed'), max_train_rows=120000):
    processed_dir = Path(processed_dir)
    train = pd.read_parquet(processed_dir / 'train.parquet')
    student = pd.read_parquet(processed_dir / 'student_test.parquet')
    labels = pd.read_parquet(processed_dir / 'obs_labels.parquet')
    folds = json.loads(Path('configs/folds.json').read_text())['trip_to_fold']
    if set(train.trip_id) != set(folds):
        raise ValueError('Fold assignments do not match training trips')
    if set(train.trip_id) & set(student.trip_id):
        raise ValueError('Training and student-test trips overlap')
    if json.loads(Path('models/feature_schema.json').read_text())['version'] != SCHEMA_VERSION:
        raise ValueError('Schema version mismatch')
    out = Path('results/metrics')
    rows, histories = [], []
    for fold in sorted(set(folds.values())):
        print(f'{name}: fold {fold}', flush=True)
        validation_ids = {trip_id for trip_id, value in folds.items() if value == fold}
        training = train[~train.trip_id.isin(validation_ids)]
        validation = train[train.trip_id.isin(validation_ids)]
        scorer, fit = fit_model(name, training, max_train_rows)
        start = time.perf_counter()
        with threadpool_limits(limits=2):
            probabilities = scorer.predict_proba(validation)
        inference = time.perf_counter()-start
        result = evaluate(validation, labels[(labels.partition == 'train') & labels.trip_id.isin(validation_ids)], probabilities)
        row = dict(model=name, partition='cv', fold=fold, **result, **fit, inference_time_s=inference)
        rows.append(row)
        if hasattr(scorer.model, 'loss_curve_'):
            histories.extend(dict(model=name, run=f'fold_{fold}', epoch=i+1, train_loss=float(loss))
                             for i, loss in enumerate(scorer.model.loss_curve_))
        Path(out / f'{name}_progress.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'{name}: refit and student-test evaluation', flush=True)
    scorer, fit = fit_model(name, train, max_train_rows)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        probabilities = scorer.predict_proba(student)
    inference = time.perf_counter()-start
    result = evaluate(student, labels[labels.partition == 'student_test'], probabilities)
    rows.append(dict(model=name, partition='student_test', fold=-1, **result, **fit, inference_time_s=inference))
    # A bounded fit-sample diagnostic, explicitly not full training performance.
    diagnostic = training_sample(train, max_train_rows).sample(n=min(10000, fit['n_fit_rows']), random_state=SEED)
    diagnostic_p = scorer.predict_proba(diagnostic)
    diagnostic_f1 = float(f1_score(diagnostic.is_ref, diagnostic_p >= .5, average='macro', labels=[0, 1], zero_division=0))
    model_dir = Path('models/member1') / name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer.model, model_dir / 'model.joblib', compress=3)
    joblib.dump(scorer.preprocessor, model_dir / 'preprocessor.joblib', compress=3)
    joblib.dump(scorer, model_dir / 'scorer.joblib', compress=3)
    if hasattr(scorer.model, 'loss_curve_'):
        histories.extend(dict(model=name, run='refit', epoch=i+1, train_loss=float(loss))
                         for i, loss in enumerate(scorer.model.loss_curve_))
        pd.DataFrame(histories).to_csv(out / f'{name}_loss_history.csv', index=False)
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in {'confusion_matrix', 'fit_warnings'}} for row in rows])
    frame.to_csv(out / f'{name}_experiments.csv', index=False)
    Path(out / f'{name}_experiments.json').write_text(json.dumps(rows, indent=2) + '\n')
    card = dict(name=name, schema_version=SCHEMA_VERSION, seed=SEED,
                configuration=scorer.model.get_params(), max_train_rows=max_train_rows,
                sampling='Proportional within training trip and class; no validation/student rows used in fitting',
                training_diagnostic_macro_f1=diagnostic_f1, training_diagnostic_rows=len(diagnostic),
                training_diagnostic_scope='10k or fewer rows from the fitted sample; KNN includes self neighbours',
                cv_macro_f1_mean=float(frame[frame.partition == 'cv'].macro_f1.mean()),
                cv_macro_f1_std=float(frame[frame.partition == 'cv'].macro_f1.std()),
                student_test=result, fit=fit,
                calibration='Uncalibrated binary probabilities; C/B must assess HMM suitability',
                route_continuity='Pending C1 metric; not fabricated here',
                input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [processed_dir / 'train.parquet', processed_dir / 'student_test.parquet',
                                           processed_dir / 'obs_labels.parquet', Path('configs/folds.json'),
                                           Path('models/feature_schema.json'), Path(__file__)]})
    Path(model_dir / 'model_card.json').write_text(json.dumps(card, indent=2) + '\n')
    from src.visualization.member1_models import plot_model
    plot_model(name, rows, histories)
    print(f'{name}: saved; student Top-1={result["top1_acc"]:.4f}', flush=True)
    return card


def main(default=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['knn', 'mlp', 'all'], default=default or 'all')
    parser.add_argument('--processed-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--max-train-rows', type=int, default=120000,
                        help='Proportional training sample cap; 0 fits all known rows')
    args = parser.parse_args()
    if args.max_train_rows < 0:
        parser.error('--max-train-rows must be nonnegative')
    for name in ['knn', 'mlp'] if args.model == 'all' else [args.model]:
        run_model(name, args.processed_dir, args.max_train_rows)


if __name__ == '__main__':
    main()
