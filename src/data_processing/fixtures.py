"""Small real training-table handoff plus synthetic geometry regression fixtures."""
import json
from pathlib import Path
import pandas as pd
from src.data_processing.synthetic import write_fixture
from src.features.build import SCHEMA_VERSION


def write_fixtures(processed_dir=Path('data/processed')):
    processed_dir = Path(processed_dir)
    write_fixture()
    quality = pd.read_csv('results/metrics/quality_report.csv', dtype={'trip_id': str})
    selected = quality[quality.partition == 'train'].sort_values(['n_obs', 'trip_id']).head(5).trip_id.tolist()
    if len(selected) != 5:
        raise ValueError('The G1 handoff requires five training trips')
    root = Path('tests/fixtures/development')
    root.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(processed_dir / 'train.parquet', filters=[('trip_id', 'in', selected)])
    labels = pd.read_parquet(processed_dir / 'obs_labels.parquet', filters=[('trip_id', 'in', selected), ('partition', '==', 'train')])
    table.to_parquet(root / 'candidates.parquet', index=False)
    labels.to_parquet(root / 'obs_labels.parquet', index=False)
    manifest = dict(partition='train', schema_version=SCHEMA_VERSION, trip_ids=selected,
                    selection='Five smallest training trips by observation count; trip ID breaks ties',
                    n_candidate_rows=len(table), n_observations=len(labels),
                    no_raw_coordinates=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (root / 'README.md').write_text(
        '# G1 development feature fixture\n\n'
        'Real candidate features and offline labels for the five smallest **training** trips.\n'
        'No raw GPS coordinates, raw networks, or student-test data are included.\n'
        'This small table is the committed G1 sample requested by the team plan; full\n'
        'regenerable feature tables remain gitignored under `data/processed/`.\n'
        'Use `models/feature_schema.json` for order/dtypes and `manifest.json` for IDs.\n'
        'Reference labels are offline-only; `-1` means unknown. For invented raw\n'
        'networks and geometry edge cases, use the separate `../member1/` fixture.\n'
        'Regenerate with `python -m src.data_processing.fixtures` after preparation.\n')


if __name__ == '__main__':
    write_fixtures()
