#!/usr/bin/env bash
# Read-only snapshot: what is installed, prepared, trained, exported, built, and running.
#
# Usage: scripts/status.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage "$0"; exit 0; }

row() { printf '  %-12s %-8s %s\n' "$1" "$2" "$3"; }
count_trips() { find "data/raw/$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '; }

say "PATHSNAP status"
if [[ -x "$PY" ]]; then row setup ok "$("$PY" --version 2>&1)"; else row setup MISSING "scripts/setup.sh"; fi

n_train="$(count_trips train)"; n_test="$(count_trips student_test)"
if (( n_train == 80 && n_test == 10 )); then row raw-data ok "train $n_train, student_test $n_test"
else row raw-data MISSING "train $n_train, student_test $n_test (expected 80 / 10): scripts/prep.sh --source DIR"; fi

have=""; for f in train student_test obs_labels; do [[ -f "data/processed/$f.parquet" ]] && have+="$f "; done
if [[ "$have" == "train student_test obs_labels " ]]; then row processed ok "$have"
else row processed MISSING "have: ${have:-none}. scripts/prep.sh"; fi

m1=""; for m in knn mlp; do [[ -f "models/member1/$m/model_card.json" ]] && m1+="$m "; done
row member1 "$([[ "$m1" == "knn mlp " ]] && echo ok || echo PARTIAL)" "models: ${m1:-none}"

m2=""; for d in models/member2/*/; do [[ -f "${d}model_card.json" ]] && m2+="$(basename "$d") "; done
row member2 "$([[ -n "$m2" ]] && echo ok || echo none)" "exported: ${m2:-none}"

if [[ -f models/model_card.json && -x "$PY" ]]; then
  row final ok "$("$PY" -c "import json; c=json.load(open('models/model_card.json')); print('models/model.joblib =', c.get('model_name'), '(status:', str(c.get('status'))+')')")"
else row final MISSING "no models/model_card.json"; fi

if [[ -f src/models/member2_report.py ]]; then row full-run ok "src/models/member2_report.py present"
else row full-run BLOCKED "src/models/member2_report.py missing: train.sh full would crash at the end"; fi
if [[ -f api/main.py ]]; then row api ok "api/main.py"; else row api MISSING "api/main.py (workstream D)"; fi
if [[ -f ui/app.py ]]; then row ui ok "ui/app.py"; else row ui MISSING "ui/app.py (workstream D)"; fi
if [[ -n "$(find src/hmm -name '*.py' 2>/dev/null)" ]]; then row hmm ok "src/hmm"; else row hmm MISSING "src/hmm (workstream C)"; fi

procs="$(pgrep -fl 'src\.(models|data_processing\.run_member1)' || true)"
if [[ -n "$procs" ]]; then row running yes "$(echo "$procs" | sed 's/.*-m //' | tr '\n' ';')"; else row running no "nothing active"; fi
