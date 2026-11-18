#!/usr/bin/env bash
# Optional isolated browser test, with real predictions and dashboard screenshots.
# Usage: scripts/ui-smoke.sh [--url http://127.0.0.1:8501]
# First: uv pip install --python .venv/bin/python -r requirements-member3-render.txt
#        .venv/bin/python -m playwright install chromium
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
exec "$PY" "$ROOT/scripts/ui_smoke.py" "$@"
