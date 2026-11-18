#!/usr/bin/env bash
# Show live API result metrics in the host terminal (no giant geometry dump).
# Usage: scripts/predict.sh 00000000 [best|knn|mlp|logistic_regression|random_forest|hist_gradient_boosting]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
exec "$PY" "$ROOT/scripts/predict_demo.py" "$@"
