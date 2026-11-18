#!/usr/bin/env bash
# Run the complete host stack. Ctrl-C stops the tunnel, UI and API started here.
# Usage: scripts/host.sh [--local]    # omit --local to print a public dashboard URL
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$ROOT/scripts/_demo-env.sh"
[[ "${1:-}" == "" || "${1:-}" == "--local" ]] || die "usage: scripts/host.sh [--local]"
need_venv
demo_logs="${DEMO_LOG_DIR:-$ROOT/logs}"
mkdir -p "$demo_logs"
pids=()
cleanup() { trap - EXIT INT TERM; for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
wait_ready() {
  local url="$1" pid="$2"
  for ((i=0; i<90; i++)); do
    kill -0 "$pid" 2>/dev/null || die "service stopped during startup; inspect logs/"
    if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then return; fi
    sleep 1
  done
  die "startup timed out: $url"
}
# Fail if another stack occupies either port, rather than tunnel the wrong app.
"$PY" - <<'PY'
import os,socket
for name,default in [('API_PORT',8000),('UI_PORT',8501)]:
    with socket.socket() as sock:
        try: sock.bind(('127.0.0.1',int(os.getenv(name,default))))
        except OSError: raise SystemExit(f'{name} is occupied; stop the previous host stack first')
PY
scripts/serve.sh api > >(tee "$demo_logs/api.log") 2>&1 & pids+=("$!")
wait_ready "http://127.0.0.1:${API_PORT:-8000}/health" "${pids[0]}"
scripts/serve.sh ui > >(tee "$demo_logs/dashboard.log") 2>&1 & pids+=("$!")
wait_ready "http://127.0.0.1:${UI_PORT:-8501}/_stcore/health" "${pids[1]}"
say "Host ready: http://127.0.0.1:${UI_PORT:-8501} · API docs: http://127.0.0.1:${API_PORT:-8000}/docs"
if [[ "${1:-}" != "--local" ]]; then
  scripts/tunnel.sh > >(tee "$demo_logs/tunnel.log") 2>&1 & pids+=("$!")
fi
while true; do
  for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null || die "a service stopped; shutting down this stack"; done
  sleep 2
done
