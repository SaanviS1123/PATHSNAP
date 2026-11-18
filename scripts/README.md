# scripts/

Every command to set up, prepare, train, test and demo PathSNAP. Run them from anywhere; each script
changes into the repo root itself. Run output is tee'd into `logs/` (git-ignored).

| Script | Does | Recorded time |
|---|---|---|
| `setup.sh` | Create `.venv` (Python 3.13), install `requirements-member2.txt` (includes member1) | n/a |
| `prep.sh [--source DIR]` | Check `data/raw` layout, build all processed tables, configs, plots, EDA map | 59 s |
| `train.sh member1` | KNN and MLP, 5 trip folds plus student test | KNN scoring ~25 min, MLP fit ~6 min |
| `train.sh baselines` | Logistic Regression, Random Forest, HistGradientBoosting; reuse KNN/MLP; CV winner | not yet recorded |
| `train.sh tune [--model M] [--trials N]` | Optuna TPE on one model | not yet recorded |
| `train.sh full [--trials N]` | Baselines, tuning, calibration audit, final export to `models/` | not yet recorded |
| `train.sh one logreg\|rf\|boosting\|knn\|mlp` | Fit and export one model | varies |
| `train.sh report` | Regenerate `report/member1/eda_and_models.md` from saved metrics | n/a |
| `test.sh [pytest args]` | pytest | about 20 s |
| `demo.sh [--trip ID] [--with-reference]` | Print the comparison table, score one trip, write `maps/demo_<trip>.html` | a few seconds |
| `serve.sh api` / `serve.sh ui` | Two-laptop system (Laptop 1 API, Laptop 2 UI) | not built yet |
| `status.sh` | Read-only: what is installed, prepared, trained, exported, built, running | instant |
| `reproduce.sh [--source DIR] [--full]` | Setup, prep, member1, report, baselines (or full), tests | hours |

## Fresh machine

```bash
scripts/setup.sh
scripts/prep.sh --source /path/to/organiser/folder   # folder holding train/ and student_test/
scripts/train.sh member1
scripts/train.sh baselines
scripts/test.sh
scripts/demo.sh
```

Or `scripts/reproduce.sh --source /path/to/organiser/folder`. `scripts/status.sh` shows where you are.

## Things to know

- **Demo scope.** `demo.sh` is classifier-only: each GPS point takes its highest-scoring candidate arc. There is no HMM
  or Viterbi decoding yet, so the continuity number it prints is the interim classifier-adjacency metric.
  Its "agreement" figures use the aligned reference labels, which are approximate supervision, not point-level truth.
  `--with-reference` draws the `.route` for offline report maps only; keep it off the live demo.
- **Not built yet.** `api/` and `ui/` hold only `.gitkeep`, and `src/hmm/` is empty. `serve.sh` stops with a clear message
  until `api/main.py` and `ui/app.py` exist. It assumes `api.main:app`, `streamlit run ui/app.py` and the plan's
  `EVAL_DATA_DIR` / `API_URL` variables; override them (see the header of `serve.sh`) if workstream D chooses otherwise.
- **One run at a time.** `prep.sh` and every `train.sh` target refuse to start while another prep or training run is active,
  because they write the same caches and artifacts. Override with `PATHSNAP_ALLOW_PARALLEL=1` only if you know the runs are disjoint.
- **`train.sh full` guard.** `run_all` imports `src/models/member2_report.py` only after tuning and calibration finish,
  so `full` refuses to start if that file is missing rather than crash at the end.
- **Re-running `prep.sh`** rewrites the tables that Member 2's cached results are fingerprinted against;
  a run only resumes if the fingerprints still match.
- **Never commit** `data/raw`, `logs/`, or anything from the evaluator partition. `.gitignore` covers the first two.
