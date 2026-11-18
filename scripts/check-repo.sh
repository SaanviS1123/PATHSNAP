#!/usr/bin/env bash
# Check tracked raw data, environment files, common secret patterns and file sizes.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
exec "$PY" "$ROOT/scripts/check_repo.py"
