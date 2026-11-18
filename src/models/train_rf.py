"""Train and export the Random Forest candidate scorer."""

from sklearn.ensemble import RandomForestClassifier

from src.models.training import fit_and_export, load_training_data


def train(data_path=None, random_seed: int = 42, output_dir="models/member2/random_forest"):
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    estimator = RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=5,
        max_samples=0.1, class_weight="balanced_subsample", n_jobs=2, random_state=random_seed)
    return fit_and_export(estimator, "random_forest", frame, features,
                          random_seed=random_seed, output_dir=output_dir)


if __name__ == "__main__":
    train()
