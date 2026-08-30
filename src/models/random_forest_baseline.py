"""Random Forest baseline for ASVspoof 2019 LA bonafide/spoof classification.

Reuses the same cached 26-dim MFCC-statistics feature matrices built by the
SVM baseline (src.models.svm_baseline) so MFCCs are extracted once and shared
across models, and reports the same evaluation metrics for a direct SVM vs.
Random Forest comparison.
"""

import sys
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_dev_protocol, load_train_protocol
from src.models.svm_baseline import (
    build_or_load_feature_matrix,
    class_distribution,
    evaluate_predictions,
)

RANDOM_STATE = 42
N_ESTIMATORS = 200


def build_random_forest() -> RandomForestClassifier:
    """Build a class-balanced Random Forest classifier (baseline config, no tuning)."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def run_full_experiment() -> None:
    """Train and evaluate the Random Forest baseline on the full train/dev protocols.

    Reuses the cached MFCC-statistics feature matrices from the SVM baseline
    instead of re-extracting MFCCs.
    """
    train_df = load_train_protocol()
    dev_df = load_dev_protocol()

    print(f"train protocol entries: {len(train_df)}")
    print(f"dev protocol entries:   {len(dev_df)}")

    print("\nloading train feature matrix...")
    X_train, y_train, _ = build_or_load_feature_matrix(train_df, "train_mfcc_stats")
    print("\nloading dev feature matrix...")
    X_dev, y_dev, _ = build_or_load_feature_matrix(dev_df, "dev_mfcc_stats")

    model = build_random_forest()
    print("\ntraining Random Forest on full train set...")
    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    y_pred = model.predict(X_dev)
    metrics = evaluate_predictions(y_dev, y_pred)

    print(f"\ntrain samples:      {len(y_train)}")
    print(f"dev samples:        {len(y_dev)}")
    print(f"feature dimension:  {X_train.shape[1]}")
    print(f"train class dist:   {class_distribution(y_train)}")
    print(f"dev class dist:     {class_distribution(y_dev)}")
    print(f"positive class:     spoof (label=1); bonafide=0 is negative")
    print(f"training time:      {training_time:.2f}s")
    print(f"accuracy:           {metrics['accuracy']:.4f}")
    print(f"precision:          {metrics['precision']:.4f}")
    print(f"recall:             {metrics['recall']:.4f}")
    print(f"f1 score:           {metrics['f1']:.4f}")
    print("confusion matrix (rows=true, cols=pred, order=[bonafide, spoof]):")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    run_full_experiment()
