"""SVM baseline for ASVspoof 2019 LA bonafide/spoof classification.

Builds fixed-size MFCC-statistics features (via extract_mfcc_statistics) from
selected protocol rows, then trains/evaluates a StandardScaler + SVC pipeline.
Kept modular so the same feature-building and evaluation functions can later
run on the full train/dev splits and be reused for a Random Forest baseline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_dev_protocol, load_train_protocol
from src.features.feature_prep import extract_mfcc_statistics

RANDOM_STATE = 42


def sample_subset(
    df: pd.DataFrame, n_samples: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Randomly sample n_samples protocol rows, guaranteeing both classes appear."""
    if n_samples >= len(df):
        return df.reset_index(drop=True)

    sampled = df.sample(n=n_samples, random_state=random_state)
    while sampled["label"].nunique() < 2:
        random_state += 1
        sampled = df.sample(n=n_samples, random_state=random_state)
    return sampled.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert protocol rows into (X, y): 26-dim MFCC-statistics features + binary labels."""
    X = np.stack([extract_mfcc_statistics(path) for path in df["path"]])
    y = df["label"].to_numpy()
    return X, y


def build_svm_pipeline() -> Pipeline:
    """Build a feature-scaling + class-balanced SVM pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute accuracy/precision/recall/F1/confusion-matrix for spoof (label=1) as positive class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def class_distribution(y: np.ndarray) -> dict:
    return {"bonafide": int((y == 0).sum()), "spoof": int((y == 1).sum())}


def run_sanity_check(n_train: int = 500, n_dev: int = 200) -> None:
    """Train and evaluate the SVM baseline on a small train/dev subset."""
    train_df = sample_subset(load_train_protocol(), n_train)
    dev_df = sample_subset(load_dev_protocol(), n_dev)

    X_train, y_train = build_feature_matrix(train_df)
    X_dev, y_dev = build_feature_matrix(dev_df)

    pipeline = build_svm_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_dev)

    metrics = evaluate_predictions(y_dev, y_pred)

    print(f"train samples:      {len(y_train)}")
    print(f"dev samples:        {len(y_dev)}")
    print(f"feature dimension:  {X_train.shape[1]}")
    print(f"train class dist:   {class_distribution(y_train)}")
    print(f"dev class dist:     {class_distribution(y_dev)}")
    print(f"accuracy:           {metrics['accuracy']:.4f}")
    print(f"precision:          {metrics['precision']:.4f}")
    print(f"recall:             {metrics['recall']:.4f}")
    print(f"f1 score:           {metrics['f1']:.4f}")
    print(f"confusion matrix:\n{metrics['confusion_matrix']}")


if __name__ == "__main__":
    run_sanity_check()
