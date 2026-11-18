#!/usr/bin/env bash
# Offline demo of what exists today: the model comparison, then one trip scored by the
# exported classifier and drawn on a Folium map. Classifier-only: no HMM/Viterbi and no
# API/UI yet. For the two-laptop demo use scripts/serve.sh once workstream D lands.
#
# Usage: scripts/demo.sh [--trip ID] [--model-dir DIR] [--with-reference] [--no-open]
#   --trip ID           8-digit trip id (default: the smallest student-test trip)
#   --model-dir DIR     scorer folder (default: models/ if exported, else models/member1/mlp)
#   --with-reference    also draw the reference .route (offline/report use only, never inference)
#   --no-open           write the map but do not open a browser
# Output: maps/demo_<trip>.html
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage "$0"; exit 0; }
need_venv; need_prepped
exec "$PY" scripts/demo_trip.py "$@"
