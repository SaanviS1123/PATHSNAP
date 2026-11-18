#!/usr/bin/env bash
# Train, tune and evaluate models. Every target needs scripts/prep.sh to have run.
#
# Usage: scripts/train.sh <target> [options passed to the Python module]
#   member1                KNN then MLP on all known-label rows: 5 trip folds + student test.
#                          Slow, KNN scoring alone took ~25 min in the recorded run.
#                          Option: --max-train-rows N (recorded proportional sample)
#   baselines              Logistic Regression, Random Forest, HistGradientBoosting on the
#                          frozen folds, reuse the saved KNN/MLP, write the five-model
#                          comparison and the CV winner. Needs models/member1 (run `member1` first).
#   tune                   Optuna TPE on one model. Options: --model NAME, --trials N (default 10), --seed N
#   full                   baselines + tuning + calibration audit + final export to models/.
#                          Options: --trials N, --seed N, --output-dir DIR, --student-test PATH
#   one <name>             fit one model on all training rows and export it.
#                          name: logreg | rf | boosting | knn | mlp
#   report                 regenerate report/member1/eda_and_models.md from saved metrics
#
# Output is tee'd into logs/ (git-ignored). Refuses to start while another run is active.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

target="${1:-}"; shift || true
case "$target" in
  ""|-h|--help) usage "$0"; exit 0 ;;
esac
need_venv

case "$target" in
  member1)
    refuse_if_training; need_prepped
    run_logged train-knn "$PY" -m src.models.train_knn "$@"
    run_logged train-mlp "$PY" -m src.models.train_mlp "$@"
    say "Done. Next: scripts/train.sh report, then scripts/train.sh baselines"
    ;;
  baselines)
    refuse_if_training; need_prepped
    need_file models/member1/knn/model_card.json "run scripts/train.sh member1 first"
    need_file models/member1/mlp/model_card.json "run scripts/train.sh member1 first"
    run_logged train-baselines "$PY" -m src.models.run_all --baseline-only "$@"
    ;;
  tune)
    refuse_if_training; need_prepped
    run_logged train-tune "$PY" -m src.models.tuning.bayes_opt "$@"
    ;;
  full)
    refuse_if_training; need_prepped
    need_file models/member1/knn/model_card.json "run scripts/train.sh member1 first"
    # run_all imports this at the very end; without the check a run would crash after
    # the tuning and calibration steps have already spent their compute.
    need_file src/models/member2_report.py \
      "run_all imports it after tuning, so a full run would crash at the end. Use 'baselines' until it exists"
    run_logged train-full "$PY" -m src.models.run_all "$@"
    ;;
  one)
    name="${1:?usage: scripts/train.sh one <logreg|rf|boosting|knn|mlp>}"; shift
    case "$name" in
      logreg|rf|boosting|knn|mlp) module="src.models.train_$name" ;;
      *) die "unknown model '$name' (logreg | rf | boosting | knn | mlp)" ;;
    esac
    refuse_if_training; need_prepped
    run_logged "train-$name" "$PY" -m "$module" "$@"
    ;;
  report)
    need_prepped
    run_logged report "$PY" -m src.data_processing.member1_report "$@"
    ;;
  *) die "unknown target '$target' (see scripts/train.sh --help)" ;;
esac
