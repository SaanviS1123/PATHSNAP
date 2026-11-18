#!/usr/bin/env bash
# Create .venv (Python 3.13) and install the pinned requirements.
#
# Usage: scripts/setup.sh
#
# Installs requirements-member2.txt (which includes requirements-member1.txt), plus
# requirements-api.txt / requirements-ui.txt if workstream D adds them.
# Uses uv when available, otherwise python3.13 -m venv + pip.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage "$0"; exit 0; }

if [[ ! -x "$PY" ]]; then
  say "Creating .venv (Python 3.13)"
  if command -v uv >/dev/null 2>&1; then uv venv --python 3.13 .venv
  elif command -v python3.13 >/dev/null 2>&1; then python3.13 -m venv .venv
  else die "need uv or python3.13 on PATH"; fi
fi
"$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))' ||
  die ".venv is not Python 3.13: delete it and rerun"

install() {
  say "Installing $1"
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" -r "$1"
  else "$PY" -m pip install -r "$1"; fi
}
install requirements-member2.txt
for extra in requirements-api.txt requirements-ui.txt; do
  if [[ -f "$extra" ]]; then install "$extra"; fi
done

say "Installed versions"
"$PY" - <<'EOF'
import importlib
for name in ["numpy", "pandas", "scipy", "sklearn", "shapely", "pyproj",
             "pyarrow", "folium", "optuna", "joblib", "pytest"]:
    print(f"  {name:<10} {importlib.import_module(name).__version__}")
EOF
say "Done. Next: scripts/prep.sh"
