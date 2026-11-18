import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    log_loss, 
    accuracy_score
)

# Define paths based on your project structure
RESULTS_DIR = "../../results/metrics"
EXPERIMENTS_CSV = os.path.join(RESULTS_DIR, "experiments.csv")

def calculate_custom_metrics(y_true, y_pred, y_prob, groups):
    """
    Placeholder for observation-level Top-1, candidate miss rate, 
    and route continuity metrics. 
    (You will refine these once Member 3's HMM data structures are ready).
    """
    # Top-1 Accuracy: This assumes y_pred is already the Top-1 candidate chosen by the classifier.
    top1_accuracy = accuracy_score(y_true, y_pred)
    
    # Candidate Miss Rate & Route Continuity need specific grouping logic based on raw data shapes.
    # Returning 0.0 placeholders so the harness runs immediately for Member 1.
    candidate_miss_rate = 0.0 
    route_continuity = 0.0    
    
    return top1_accuracy, candidate_miss_rate, route_continuity

def evaluate_model(estimator, model_name, X, y, groups, config, random_seed=42):
    """
    Evaluates a scikit-learn estimator using GroupKFold cross-validation.
    
    Args:
        estimator: The scikit-learn model to evaluate.
        model_name (str): Name of the model (e.g., "LogisticRegression").
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Ground truth labels.
        groups (pd.Series or np.ndarray): Trip IDs to group by.
        config (dict): Hyperparameter configuration to log.
        random_seed (int): Seed used for reproducibility.
    """
    print(f"Starting evaluation for {model_name}...")
    
    gkf = GroupKFold(n_splits=5)
    
    # Storage for fold metrics
    fold_metrics = {
        "precision": [], "recall": [], "f1_macro": [], "log_loss": [],
        "top1_acc": [], "miss_rate": [], "continuity": [],
        "train_time": [], "inference_time": []
    }

    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups):
        # 1. Split data (ensuring no trip data overlaps)
        X_train, y_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], \
                           y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        X_val, y_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx], \
                       y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
        
        # 2. Train and track time
        start_train = time.time()
        estimator.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # 3. Predict and track time
        start_infer = time.time()
        y_pred = estimator.predict(X_val)
        # Handle models that don't output probabilities (like some baseline SVMs)
        try:
            y_prob = estimator.predict_proba(X_val)
            fold_log_loss = log_loss(y_val, y_prob)
        except (AttributeError, ValueError):
            y_prob = None
            fold_log_loss = np.nan
        inference_time = time.time() - start_infer
        
        # 4. Calculate standard metrics
        fold_metrics["precision"].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        fold_metrics["f1_macro"].append(f1_score(y_val, y_pred, average='macro', zero_division=0))
        fold_metrics["log_loss"].append(fold_log_loss)
        
        # 5. Calculate custom hackathon metrics
        top1, miss_rate, continuity = calculate_custom_metrics(y_val, y_pred, y_prob, groups.iloc[val_idx] if isinstance(groups, pd.Series) else groups[val_idx])
        fold_metrics["top1_acc"].append(top1)
        fold_metrics["miss_rate"].append(miss_rate)
        fold_metrics["continuity"].append(continuity)
        
        fold_metrics["train_time"].append(train_time)
        fold_metrics["inference_time"].append(inference_time)
        
        print(f"Fold {fold} complete | Macro F1: {fold_metrics['f1_macro'][-1]:.4f}")
        fold += 1

    # 6. Aggregate results
    results = {
        "model_name": model_name,
        "config": json.dumps(config),
        "random_seed": random_seed,
        "cv_precision": np.mean(fold_metrics["precision"]),
        "cv_recall": np.mean(fold_metrics["recall"]),
        "cv_macro_f1": np.mean(fold_metrics["f1_macro"]),
        "cv_log_loss": np.nanmean(fold_metrics["log_loss"]),
        "cv_top1_acc": np.mean(fold_metrics["top1_acc"]),
        "cv_miss_rate": np.mean(fold_metrics["miss_rate"]),
        "cv_continuity": np.mean(fold_metrics["continuity"]),
        "avg_train_time_sec": np.mean(fold_metrics["train_time"]),
        "avg_inference_time_sec": np.mean(fold_metrics["inference_time"])
    }
    
    # 7. Log to CSV
    log_results(results)
    print(f"Evaluation complete. Macro F1 Average: {results['cv_macro_f1']:.4f}")
    return results

def log_results(results_dict):
    """Appends the evaluation results to the experiments.csv file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df = pd.DataFrame([results_dict])
    
    if not os.path.isfile(EXPERIMENTS_CSV):
        df.to_csv(EXPERIMENTS_CSV, index=False)
    else:
        df.to_csv(EXPERIMENTS_CSV, mode='a', header=False, index=False)
    
    print(f"Results logged to {EXPERIMENTS_CSV}")