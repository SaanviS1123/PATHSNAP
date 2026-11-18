#!/usr/bin/env bash
# Run a service on the model-host laptop; use separate terminals.
# Usage: scripts/serve.sh api | ui
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
need_venv
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-2}"
case "${1:-}" in
  api)
    need_file models/model_card.json "run scripts/train.sh full first"
    exec "$PY" -m uvicorn api.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}" --log-level info ;;
  ui)
    export API_URL="${API_URL:-http://127.0.0.1:${API_PORT:-8000}}"
    exec "$PY" -m streamlit run "$ROOT/ui/app.py" --server.address "${UI_HOST:-127.0.0.1}" --server.port "${UI_PORT:-8501}" --server.headless true --server.enableWebsocketCompression false --browser.gatherUsageStats false ;;
  *) usage "$0"; exit 1 ;;
esac
