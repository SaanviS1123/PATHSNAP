#!/usr/bin/env bash
# Check ten real API trips and all model configurations. API must already run.
# Usage: scripts/integration.sh [--all-models] [--api-url http://127.0.0.1:8000]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
run_logged integration "$PY" "$ROOT/scripts/integration.py" "$@"
