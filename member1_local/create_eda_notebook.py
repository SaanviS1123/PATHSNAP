
from pathlib import Path

import nbformat as nbf

ROOT = Path("notebooks")
ROOT.mkdir(parents=True, exist_ok=True)

notebook = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# PathSnap — Exploratory Data Analysis\n\n"
    "Member 1: Data, Candidates, Features and EDA\n\n"
    "This notebook summarizes the dataset inventory, "
    "GPS quality, candidate generation, and "
    "provisional-label coverage."
))

cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "ROOT = Path('..') if Path('../results').exists() else Path('.')\n"
    "METRICS = ROOT / 'results' / 'metrics'\n"
    "PLOTS = ROOT / 'results' / 'plots'\n"
    "print('Project root:', ROOT.resolve())"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 1. Dataset inventory"
))

cells.append(nbf.v4.new_code_cell(
    "inventory = pd.read_csv(METRICS / 'inventory.csv')\n"
    "display(inventory.head())\n"
    "print('Inventory rows:', len(inventory))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 2. GPS and road-network quality"
))

cells.append(nbf.v4.new_code_cell(
    "quality = pd.read_csv(METRICS / 'quality_report.csv')\n"
    "display(quality.head())\n"
    "print('Quality-report rows:', len(quality))\n"
    "display(quality.describe(include='all').T)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 3. Candidate feature generation"
))

cells.append(nbf.v4.new_code_cell(
    "report = pd.read_csv(\n"
    "    METRICS / 'feature_build_report.csv'\n"
    ")\n"
    "display(report.groupby('split').agg(\n"
    "    trips=('trip_id', 'count'),\n"
    "    gps_observations=('gps_observations', 'sum'),\n"
    "    candidate_rows=('candidate_rows', 'sum')\n"
    "))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 4. Provisional-label coverage\n\n"
    "Labels are produced by heuristic alignment to "
    "reference-route arcs. They are not independently "
    "verified ground truth."
))

cells.append(nbf.v4.new_code_cell(
    "train = report[report['split'] == 'train'].copy()\n"
    "train['trip_id'] = train['trip_id'].astype(str).str.zfill(8)\n"
    "overall = (\n"
    "    train['positive_labels'].sum()\n"
    "    / train['gps_observations'].sum()\n"
    ")\n"
    "print(f'Observation-weighted coverage: {overall:.2%}')\n"
    "print('Lowest-coverage trips:')\n"
    "display(train.nsmallest(10, 'candidate_coverage')"
    "[['trip_id', 'gps_observations', 'candidate_coverage']])"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 5. Candidate configuration sweep"
))

cells.append(nbf.v4.new_code_cell(
    "sweep = pd.read_csv(METRICS / 'candidate_sweep.csv')\n"
    "display(sweep.sort_values('miss_rate').head(10))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 6. EDA plots"
))

cells.append(nbf.v4.new_code_cell(
    "from IPython.display import Image, display\n"
    "\n"
    "for filename in [\n"
    "    'eda_trip_sizes.png',\n"
    "    'eda_candidate_coverage.png',\n"
    "    'eda_candidate_distances.png'\n"
    "]:\n"
    "    print(filename)\n"
    "    display(Image(filename=str(PLOTS / filename)))"
))

cells.append(nbf.v4.new_markdown_cell(
    "## 7. Interactive route maps\n\n"
    "Open the generated HTML maps in a browser:\n\n"
    "- `maps/eda_trip_00000001.html`\n"
    "- `maps/eda_trip_00000033.html`\n"
    "- `maps/eda_trip_00000054.html`\n\n"
    "Blue shows the GPS trajectory; red shows the "
    "reference road arcs."
))

cells.append(nbf.v4.new_markdown_cell(
    "## 8. Findings and limitations\n\n"
    "- The dataset contains 80 training trips and "
    "10 student-test trips.\n"
    "- Candidate features use a per-trip local "
    "metric projection.\n"
    "- The baseline uses up to 30 candidates "
    "within 100 metres per GPS observation.\n"
    "- Training labels are provisional route-aligned "
    "arc assignments.\n"
    "- Trips 33 and 54 have particularly low "
    "candidate-label coverage.\n"
    "- Increasing candidate count and radius "
    "improves recall on sampled observations, "
    "but increases computation.\n"
    "- Student-test reference routes are not "
    "used to construct model input features."
))

notebook.cells = cells
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    }
}

output = ROOT / "01_eda.ipynb"

nbf.write(notebook, output)

print("A8 NOTEBOOK CREATED")
print("Saved:", output)
print("Notebook cells:", len(cells))

# Validate notebook structure.
nbf.validate(nbf.read(output, as_version=4))
print("Notebook validation passed")
