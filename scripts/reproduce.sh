#!/usr/bin/env bash
# Rebuild everything from raw data, in order: setup, prep, models, tests.
#
# Usage: scripts/reproduce.sh [--source DIR] [--full] [--skip-setup]
#   --source DIR    organiser folder with train/ and student_test/ (see prep.sh)
#   --full          also tune, audit calibration and export the final model (needs
#                   src/models/member2_report.py; otherwise stops at the baselines)
#   --skip-setup    reuse the existing .venv
#
# Runs scripts/setup.sh, prep.sh, train.sh member1, train.sh report, train.sh baselines
# (or full), then test.sh. Long: the KNN scoring alone took ~25 min in the recorded run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

source_args=(); full=0; setup=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) source_args=(--source "${2:?--source needs a directory}"); shift 2 ;;
    --full) full=1; shift ;;
    --skip-setup) setup=0; shift ;;
    -h|--help) usage "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

here="$ROOT/scripts"
(( setup )) && "$here/setup.sh"
"$here/prep.sh" ${source_args[@]+"${source_args[@]}"}
"$here/train.sh" member1
"$here/train.sh" report
if (( full )); then "$here/train.sh" full; else "$here/train.sh" baselines; fi
"$here/test.sh"
say "Reproduction finished. Try scripts/demo.sh"
