#!/usr/bin/env bash
# Shared helpers, sourced by the other scripts. Not meant to be run directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="${PATHSNAP_PYTHON:-$ROOT/.venv/bin/python}"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Print the leading comment block of a script (the usage text).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$1"; }

need_venv()  { [[ -x "$PY" ]] || die "no Python at $PY: run scripts/setup.sh first"; }
need_file()  { [[ -e "$1" ]] || die "missing $1${2:+ ($2)}"; }
need_prepped() {
  local f
  for f in train student_test obs_labels; do
    need_file "data/processed/$f.parquet" "run scripts/prep.sh first"
  done
}

# Training writes progress files, caches and model artifacts that other runs read,
# so two runs at once corrupt each other. Override only if you know they are disjoint.
refuse_if_training() {
  if [[ -z "${PATHSNAP_ALLOW_PARALLEL:-}" ]] &&
     pgrep -f 'src\.(models\.(run_all|member1_harness|train_|tuning)|data_processing\.run_member1)' >/dev/null; then
    die "a training/prep run is already active (see scripts/status.sh). Wait for it, or set PATHSNAP_ALLOW_PARALLEL=1."
  fi
}

# run_logged <name> <command...>: echo the command, tee output into logs/ (git-ignored).
run_logged() {
  local name="$1"; shift
  mkdir -p "$ROOT/logs"
  local log stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  log="logs/${name}-${stamp}.log"
  say "$* (log: $log)"
  "$@" 2>&1 | tee "$ROOT/$log"
}

open_file() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1"
  else say "open this file in a browser: $1"; fi
}

lan_ip() {
  ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true
}
