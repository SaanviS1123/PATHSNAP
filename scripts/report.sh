#!/usr/bin/env bash
# Assemble the complete measured report; run integration/browser checks first.
# Usage: scripts/report.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
exec "$PY" "$ROOT/scripts/build_report.py"
