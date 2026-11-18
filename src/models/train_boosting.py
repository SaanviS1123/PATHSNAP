"""Train and export a histogram gradient boosting candidate scorer."""

from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.training import fit_and_export, load_training_data


def train(data_path=None, random_seed: int = 42, output_dir="models"):
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=random_seed)
    return fit_and_export(estimator, "hist_gradient_boosting", frame, features,
                          random_seed=random_seed, output_dir=output_dir)


if __name__ == "__main__":
    train()
