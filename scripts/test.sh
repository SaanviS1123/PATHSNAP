#!/usr/bin/env bash
# Run the pytest suite.
#
# Usage: scripts/test.sh [pytest args]
#   scripts/test.sh                       everything (Member 1 took ~19 s in the recorded run)
#   scripts/test.sh -k member2            only Member 2's tests
#   scripts/test.sh tests/test_member1.py one file
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage "$0"; exit 0; }
need_venv
exec "$PY" -m pytest -q "$@"
