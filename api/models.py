"""Explicit artifact registry; models load once, never retrained by the API."""
import json
import pandas as pd
from api.config import MODEL_DIR, CONFIG_DIR, RESULTS_DIR
from src.hmm.predict import load_params
from src.models.training import load_scorer
from src.models.member1_harness import load_scorer as load_member1


class ModelRegistry:
    def __init__(self):
        self.specs = {
            "best": ("Selected best · tuned + calibrated Histogram Boosting", MODEL_DIR, "hist_gradient_boosting_tuned"),
            "knn": ("K-Nearest Neighbours", MODEL_DIR / "member1/knn", "knn"),
            "mlp": ("Multi-Layer Perceptron", MODEL_DIR / "member1/mlp", "mlp"),
            "logistic_regression": ("Logistic Regression", MODEL_DIR / "member2/logistic_regression", "logistic_regression"),
            "random_forest": ("Random Forest", MODEL_DIR / "member2/random_forest", "random_forest"),
            "hist_gradient_boosting": ("Histogram Boosting · baseline", MODEL_DIR / "member2/hist_gradient_boosting", "hist_gradient_boosting"),
        }
        self.loaded = {}
        self.params = None

    def get(self, name):
        if name not in self.loaded:
            _, folder, _ = self.specs[name]
            schema = json.loads((MODEL_DIR / "feature_schema.json").read_text())
            card = json.loads((folder / "model_card.json").read_text())
            version = card.get("feature_schema_version", card.get("schema_version"))
            if version != schema["version"]:
                raise ValueError(f"Feature schema mismatch for {name}")
            self.loaded[name] = (load_member1 if name in ("knn", "mlp") else load_scorer)(folder)
        return self.loaded[name]

    def startup(self):
        self.params = load_params(CONFIG_DIR / "hmm.json")
        self.get("best")

    def describe(self):
        table = pd.read_csv(RESULTS_DIR / "metrics/model_comparison.csv").set_index("model_name")
        result = []
        for name, (label, folder, metric_name) in self.specs.items():
            row = table.loc[metric_name]
            result.append(dict(id=name, label=label, selected=name == "best",
                               available=all((folder / f).is_file() for f in ("model.joblib", "preprocessor.joblib", "model_card.json")),
                               cv_top1=float(row.cv_top1_accuracy), student_top1=float(row.student_top1_accuracy),
                               slow=name == "knn"))
        return result


registry = ModelRegistry()
