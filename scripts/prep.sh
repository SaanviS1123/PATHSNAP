#!/usr/bin/env bash
# Validate the raw data layout and build every processed table (Member 1 pipeline).
#
# Usage: scripts/prep.sh [--source DIR]
#   --source DIR   organiser folder containing train/ and student_test/; copied into data/raw/
#
# Reads data/raw/{train,student_test}/<8-digit trip>/<trip>.{track,nodes,arcs,route}.
# Writes data/processed/*.parquet, configs/{candidates,folds}.json, models/feature_schema.json,
# results/metrics/*, results/plots/*, maps/eda_*.html and tests/fixtures/development/.
# Re-running rewrites the tables Member 2's cached runs are fingerprinted against.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SRC=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SRC="${2:?--source needs a directory}"; shift 2 ;;
    -h|--help) usage "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
need_venv

if [[ -n "$SRC" ]]; then
  for part in train student_test; do
    [[ -d "$SRC/$part" ]] || die "$SRC/$part not found"
    mkdir -p "data/raw/$part"
    cp -R "$SRC/$part/." "data/raw/$part/"
  done
fi

check_partition() {
  local part="$1" n=0 bad=0 d id ext
  for d in "data/raw/$part"/*/; do
    [[ -d "$d" ]] || continue
    id="$(basename "$d")"; n=$((n + 1))
    for ext in track nodes arcs route; do
      [[ -f "$d$id.$ext" ]] || { warn "missing $d$id.$ext"; bad=$((bad + 1)); }
    done
  done
  (( n > 0 )) || die "data/raw/$part has no trip folders (pass --source DIR)"
  (( bad == 0 )) || die "$bad files missing under data/raw/$part"
  say "$part: $n trips"
  COUNT="$n"
}
check_partition train; n_train="$COUNT"
check_partition student_test; n_test="$COUNT"
(( n_train == 80 && n_test == 10 )) || warn "expected 80 train / 10 student_test trips, found $n_train / $n_test (Member 2's run_all asserts these)"

refuse_if_training
run_logged prep "$PY" -m src.data_processing.run_member1
say "Processed tables"
ls -lh data/processed/*.parquet
say "Done. Next: scripts/train.sh member1   (or scripts/train.sh baselines if models/member1 already exists)"
