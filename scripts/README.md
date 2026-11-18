# Run commands

Run these from the repository. Scripts find the repository root automatically. Python 3.13 is required on the model host. Laptop 2 needs only a browser and Internet access.

## Demo on two laptops

**Host laptop, one terminal:**

```bash
scripts/setup.sh                 # once; installs pinned model/API/UI/report dependencies
scripts/status.sh               # check that local student data and six model exports exist
scripts/host.sh                  # API + dashboard + Cloudflare Quick Tunnel
```

Keep the terminal open. Open the printed `https://….trycloudflare.com` dashboard URL on Laptop 2. No API address needs to be changed there. `Ctrl-C` stops the services started by `host.sh`. The model host must stay awake and connected.

On the dashboard: select a trip, choose a model, click **Run selected model**, or click **Demo every model**. The latter runs KNN, MLP, Logistic Regression, Random Forest, baseline Histogram Boosting, then the selected tuned/calibrated model. Results accumulate in a per-trip table. Select any completed model to view its map; toggle HMM or individual map layers. The winner is chosen by recorded training-fold CV, not by one demo trip.

Install `cloudflared` separately: `brew install cloudflared` on macOS, or follow [Cloudflare installation](https://developers.cloudflare.com/tunnel/downloads/) for Windows/Linux. [Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) print a different temporary URL each time, have no uptime guarantee, and are intended for testing. Polling jobs keep individual requests short. A Quick Tunnel is a public demo link; stop it when finished.

**Separate terminals, if you want clearer API logs:**

```bash
scripts/serve.sh api             # terminal 1: host-local port 8000, request metrics logged
scripts/serve.sh ui              # terminal 2: host-local port 8501
scripts/tunnel.sh                # terminal 3: dashboard tunnel only
```

The API stays on loopback. The Streamlit server calls it on the same laptop; Laptop 2 talks only to the dashboard. `.env.example` documents configurable paths/ports; copy it to `.env` if needed. Shell exports override `.env`. Use a single API worker so models load once and the in-memory job queue stays coherent.

**Laptop 2 connectivity check (optional terminal):**

```bash
curl --fail --max-time 30 https://YOUR-LINK.trycloudflare.com/_stcore/health
# If this repository is present on Laptop 2:
bash scripts/check-dashboard.sh https://YOUR-LINK.trycloudflare.com
```

Then run a prediction in the browser. Capture that browser and the host's trip/model/points/latency/accuracy log for the physical demo evidence. Laptop 2 does not need the raw files or Python environment.

## Network fallback

If Quick Tunnel creation fails or a link stops working, restart `scripts/tunnel.sh` and use the new URL. If outbound QUIC is blocked, use `TUNNEL_PROTOCOL=http2 scripts/tunnel.sh`. A failure to create a tunnel at `api.trycloudflare.com` happens before protocol selection; try a different network/hotspot.

For a same-Wi-Fi/hotspot demo without Cloudflare, stop the previous stack and run:

```bash
UI_HOST=0.0.0.0 scripts/host.sh --local
# Laptop 2: open http://HOST_LAN_IP:8501 (allow inbound 8501 on host firewall).
# Host locally: http://127.0.0.1:8501
```

The API still stays local. If both Internet and tile access fail, predictions/metrics still run, and map line overlays remain available on an empty basemap. Keep saved offline report screenshots available for context. Do not present a static screenshot as live inference.

## Validation and report

With the API/dashboard running:

```bash
scripts/integration.sh           # 10 trips with best + the 5 baseline configurations on first trip
scripts/integration.sh --all-models  # optional: all 6 configurations on all 10 trips; KNN is slow
scripts/test.sh
scripts/check-repo.sh          # tracked data, credential patterns, file sizes
scripts/predict.sh 00000000 best  # compact real response in the host terminal
scripts/report.sh               # assemble report/PATHSNAP_report.pdf from measured sources
# Optional real browser tests + screenshots:
uv pip install --python .venv/bin/python -r requirements-member3-render.txt
.venv/bin/python -m playwright install chromium
scripts/ui-smoke.sh
```

`integration.sh` produces `results/metrics/integration_run.csv` and logs under `logs/`. The browser check exercises all ten trips, the batch demo, model switching and HMM toggle; it writes `ui_integration.json` and real screenshots under `report/figures/`. These checks do not establish that a second physical laptop or public tunnel worked.

## Prepare or reproduce training (not required before a demo with saved exports)

| Command | Purpose |
| --- | --- |
| `scripts/prep.sh --source /path/to/organiser` | Copy only train/student_test, validate, build features/EDA |
| `scripts/train.sh member1` | Train KNN and MLP with frozen trip folds |
| `scripts/train.sh baselines` | LR, RF and boosting comparison |
| `scripts/train.sh full --trials 10` | Baselines, Bayesian tuning, calibration and final export |
| `scripts/train.sh one logreg\|rf\|boosting\|knn\|mlp` | Fit a single model |
| `.venv/bin/python -m src.hmm.run_member3` | Reproduce decoder tuning, evaluation and offline maps |
| `scripts/demo.sh --trip 00000000` | Classifier-only standalone Folium preview; use the dashboard for HMM |
| `scripts/reproduce.sh --source /path/to/organiser --full` | Full data/model workflow; may take hours |

Raw and processed data and live logs stay ignored. Never copy evaluator-test data into development. The model artifacts were exported with the exact scikit-learn version in `requirements.txt`; use that version when loading them.
