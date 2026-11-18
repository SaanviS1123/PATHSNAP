#!/usr/bin/env bash
# Start the two-laptop system. NOT BUILT YET: api/ and ui/ are empty scaffolds (workstream D).
# The preflight checks below tell you what is missing, and this script works once D lands.
#
# Usage: scripts/serve.sh api     # Laptop 1: FastAPI on 0.0.0.0:$PORT, request logs on screen
#        scripts/serve.sh ui      # Laptop 2: UI pointed at $API_URL
#
# Environment (defaults follow .github/HACKATHON_PLAN.md; edit when D chooses otherwise):
#   API_APP         api.main:app          uvicorn app path
#   PORT            8000
#   EVAL_DATA_DIR   data/raw/student_test directory the API reads trips from
#   UI_ENTRY        ui/app.py             launched with `streamlit run`
#   API_URL         http://127.0.0.1:8000 set to http://<laptop-1-ip>:8000 on Laptop 2
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-}" in
  api)
    need_venv
    API_APP="${API_APP:-api.main:app}"; PORT="${PORT:-8000}"
    module_file="${API_APP%%:*}"; module_file="${module_file//./\/}.py"
    need_file "$module_file" "workstream D's FastAPI app has not landed; api/ only has .gitkeep"
    "$PY" -c 'import fastapi, uvicorn' 2>/dev/null ||
      die "fastapi/uvicorn not installed: add requirements-api.txt and rerun scripts/setup.sh"
    need_file models/model_card.json "export a scorer first (scripts/train.sh full or one)"
    export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ROOT/data/raw/student_test}"
    say "Laptop 1 API: http://$(lan_ip || echo '<this-laptop-ip>'):$PORT (docs at /docs), data: $EVAL_DATA_DIR"
    exec "$PY" -m uvicorn "$API_APP" --host 0.0.0.0 --port "$PORT" --log-level info
    ;;
  ui)
    need_venv
    UI_ENTRY="${UI_ENTRY:-ui/app.py}"; export API_URL="${API_URL:-http://127.0.0.1:8000}"
    need_file "$UI_ENTRY" "workstream D's UI has not landed; ui/ only has .gitkeep"
    "$PY" -c 'import streamlit' 2>/dev/null ||
      die "streamlit not installed: add requirements-ui.txt and rerun scripts/setup.sh (or set UI_ENTRY for another framework)"
    say "Laptop 2 UI, API at $API_URL"
    curl -fsS --max-time 3 "$API_URL/health" >/dev/null ||
      warn "API not reachable at $API_URL/health: wrong IP, firewall, or different network?"
    exec "$PY" -m streamlit run "$UI_ENTRY"
    ;;
  *) usage "$0"; exit 1 ;;
esac
