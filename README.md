# PATHSNAP

Distributed map matching and route reconstruction with five classifier families and a directed HMM/Viterbi decoder.

The model laptop hosts the saved models, FastAPI and the Streamlit dashboard. A Cloudflare tunnel exposes the dashboard; the second laptop opens its HTTPS link in a browser. Predictions are computed live. Reference routes stay on the host and are read only by the separate evaluation step after inference.

```bash
scripts/setup.sh
scripts/host.sh
```

Open the printed `trycloudflare.com` URL on the second laptop. Use **Demo every model** to run KNN, MLP, Logistic Regression, Random Forest, baseline Histogram Boosting and the selected tuned/calibrated Histogram Boosting. Select a completed model to inspect its map and classifier/HMM metrics. Keep the host terminal open to show request logs; `Ctrl-C` stops the stack.

Requires Python 3.13, the saved `models/` and `configs/`, local `data/raw/student_test/<8-digit-id>/<id>.track,.nodes,.arcs` files, and `cloudflared` ([installation](https://developers.cloudflare.com/tunnel/downloads/)). `.route` files enable server-side evaluation but are not inference inputs. The receiving laptop needs no project installation. For fresh data preparation use `scripts/prep.sh --source /path/to/organiser`; train/student_test are the only permitted development partitions. Do not prepare or inspect evaluator-test data.

See **[all run commands and two-laptop fallback](scripts/README.md)** for separate-terminal startup, environment settings, LAN/hotspot access, validation and training. The report and real integration evidence are in **[Member 4 delivery notes](report/member4/README.md)**. Cloudflare Quick Tunnels are temporary test links; their URL changes on restart. A named tunnel can be configured separately if a persistent hostname is needed.

The selected model won on mean observation Top-1 across the five frozen training-trip folds. The recorded student-test result improves approximate known-point accuracy from **80.36% classifier-only to 88.46% with HMM** and direct continuity from **90.38% to 97.92%**. One trip worsens with HMM. Alternative classifiers use the same frozen decoder in the demo; those are not separately tuned HMM benchmarks. See `results/metrics/hmm_comparison.csv` and the reports for definitions, label-alignment limitations and the failure case.

| Component | Source / evidence |
| --- | --- |
| Data, candidates, features, KNN/MLP | [Member 1](report/member1/README.md) |
| Model comparison, Bayesian tuning, calibration | [Member 2](report/member2/README.md) |
| HMM, route metrics, prediction maps | [Member 3](report/member3/README.md) |
| API, dashboard, deployment, report | [Member 4](report/member4/README.md) |
| Concise implementation decisions | [DECISIONS.md](DECISIONS.md) |

```bash
scripts/test.sh
scripts/integration.sh       # API must be running
scripts/report.sh
```

The API offers `/health`, `/models`, `/trips`, synchronous `/predict` and asynchronous `/jobs` plus status/result endpoints. Default binds are loopback ports 8000/8501. The UI polls bounded jobs so slow KNN calls do not hold a tunnel request open. No browser-to-API CORS exception is needed. Models are never trained by the API, saved prediction files never substitute for live inference, and API responses contain no reference-route geometry or labels.

Code, analysis and report drafting used Codex AI assistance. Team members remain responsible for understanding, verifying and explaining the delivered work. No identities, member names, physical-laptop evidence or deployment success have been fabricated.
