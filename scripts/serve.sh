#!/usr/bin/env bash
# Run one service on the model-host laptop; use separate terminals.
# Usage: scripts/serve.sh api | ui
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-2}"
case "${1:-}" in
  api|ui) exec "$PY" "$ROOT/scripts/serve.py" "$1" ;;
  *) usage "$0"; exit 1 ;;
esac
