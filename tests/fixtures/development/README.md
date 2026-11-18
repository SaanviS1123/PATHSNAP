# G1 development feature fixture

Real candidate features and offline labels for the five smallest **training** trips.
No raw GPS coordinates, raw networks, or student-test data are included.
This small table is the committed G1 sample requested by the team plan; full
regenerable feature tables remain gitignored under `data/processed/`.
Use `models/feature_schema.json` for order/dtypes and `manifest.json` for IDs.
Reference labels are offline-only; `-1` means unknown. For invented raw
networks and geometry edge cases, use the separate `../member1/` fixture.
Regenerate with `python -m src.data_processing.fixtures` after preparation.
