"""Train and export the Logistic Regression candidate scorer."""

import argparse

from sklearn.linear_model import LogisticRegression

from src.models.training import fit_and_export, load_training_data


def train(data_path=None, random_seed: int = 42, output_dir=None):
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    model = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                               random_state=random_seed, solver="lbfgs")
    return fit_and_export(model, "logistic_regression", frame, features,
                          scale=True, random_seed=random_seed, output_dir=output_dir or "models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    train(args.data_path, args.seed, args.output_dir)
