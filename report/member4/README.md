# Member 4 delivery

The host runs `api.main:app` and `ui/app.py`. `scripts/host.sh` supervises both and starts a temporary dashboard tunnel. The second laptop uses a browser only. Read [run commands](../../scripts/README.md) before the demo.

- Real HTTP evidence: `results/metrics/integration_run.csv` (ten student trips with the selected model, five other configurations on the first trip).
- Browser evidence: `results/metrics/ui_integration.json`, with screenshots in `report/figures/` after `scripts/ui-smoke.sh`.
- Consolidated report: `report/PATHSNAP_report.pdf`, rebuilt by `scripts/report.sh`; source sections retain the original measured experiment reports.
- Physical second-laptop check and public tunnel: still require the demo network. The development environment reset connections to `api.trycloudflare.com` during tunnel creation. No working public link or second-laptop screenshot is claimed.

API logs show trip ID, model, point count, measured server latency and both point accuracies. Inference reads only track/nodes/arcs; `api/evaluation.py` opens reference routes afterwards and uses Member 3's metrics. Approximate alignment labels are regenerated server-side; the client receives only aggregate metrics.

The selected best model/HMM load at startup. Other scorers load once on demand. One worker serializes model work; at most four jobs are outstanding and twelve completed/current jobs are retained. A missing reference returns predictions with metrics unavailable. Missing/corrupt model artifacts prevent startup or fail the affected model job with a readable host log. Job IDs expire on eviction or server restart; repeat a prediction if necessary. Use one API process.

Acceptance checks, including measured latencies and any remaining external limitations, are recorded in section 7 of the consolidated report. Live logs are local and ignored; committed metrics and screenshots contain no reference geometry from the live API.
