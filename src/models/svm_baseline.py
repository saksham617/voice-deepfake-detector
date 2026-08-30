"""SVM baseline for ASVspoof 2019 LA bonafide/spoof classification.

Builds fixed-size MFCC-statistics features (via extract_mfcc_statistics) from
selected protocol rows, then trains/evaluates a StandardScaler + SVC pipeline.
Kept modular so the same feature-building and evaluation functions can later
run on the full train/dev splits and be reused for a Random Forest baseline.
"""

import sys
import time
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

# Cached feature matrices are dataset-derived artifacts, not source, and are
# gitignored (data/processed/*) so the full-dataset run doesn't need to
# re-extract MFCCs on every invocation.
FEATURE_CACHE_DIR = _PROJECT_ROOT / "data" / "processed" / "features"


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


def build_feature_matrix(
    df: pd.DataFrame, progress_every: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert protocol rows into (X, y): 26-dim MFCC-statistics features + binary labels.

    Processes one audio file at a time (via extract_mfcc_statistics) rather than
    loading the whole dataset into memory at once. If progress_every > 0, prints
    a progress line every that many files.
    """
    total = len(df)
    features = []
    for i, path in enumerate(df["path"], start=1):
        features.append(extract_mfcc_statistics(path))
        if progress_every and (i % progress_every == 0 or i == total):
            print(f"  extracted {i}/{total} files...")
    X = np.stack(features)
    y = df["label"].to_numpy()
    return X, y


def build_or_load_feature_matrix(
    df: pd.DataFrame, cache_name: str, progress_every: int = 2000
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a cached (X, y) feature matrix if present, else build and cache it.

    Returns (X, y, extraction_seconds). extraction_seconds is 0.0 on a cache hit.
    """
    cache_path = FEATURE_CACHE_DIR / f"{cache_name}.npz"
    if cache_path.exists():
        print(f"loading cached features from {cache_path}")
        cached = np.load(cache_path)
        return cached["X"], cached["y"], 0.0

    print(f"extracting features for {len(df)} files (no cache found)...")
    start = time.perf_counter()
    X, y = build_feature_matrix(df, progress_every=progress_every)
    extraction_seconds = time.perf_counter() - start

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    print(f"saved features to {cache_path}")
    return X, y, extraction_seconds


def build_svm_pipeline(probability: bool = False) -> Pipeline:
    """Build a feature-scaling + class-balanced SVM pipeline.

    probability=False (default, unchanged behavior) keeps training fast via
    plain hinge-loss SVC. Pass probability=True to additionally fit Platt
    scaling so predict_proba() is available (needed for probability-based
    ensembling); this costs extra training time and is opt-in.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    class_weight="balanced",
                    probability=probability,
                    random_state=RANDOM_STATE,
                ),
            ),
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


def run_full_experiment() -> None:
    """Train and evaluate the SVM baseline on the full ASVspoof 2019 LA train/dev protocols."""
    train_df = load_train_protocol()
    dev_df = load_dev_protocol()

    print(f"train protocol entries: {len(train_df)}")
    print(f"dev protocol entries:   {len(dev_df)}")

    print("\nbuilding train feature matrix...")
    X_train, y_train, train_extract_time = build_or_load_feature_matrix(
        train_df, "train_mfcc_stats"
    )
    print("\nbuilding dev feature matrix...")
    X_dev, y_dev, dev_extract_time = build_or_load_feature_matrix(dev_df, "dev_mfcc_stats")

    pipeline = build_svm_pipeline()
    print("\ntraining SVM on full train set...")
    train_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    y_pred = pipeline.predict(X_dev)
    metrics = evaluate_predictions(y_dev, y_pred)

    print(f"\ntrain samples:          {len(y_train)}")
    print(f"dev samples:            {len(y_dev)}")
    print(f"feature dimension:      {X_train.shape[1]}")
    print(f"train class dist:       {class_distribution(y_train)}")
    print(f"dev class dist:         {class_distribution(y_dev)}")
    print(f"positive class:         spoof (label=1); bonafide=0 is negative")
    print(
        f"feature extraction time: train={train_extract_time:.1f}s "
        f"(0.0s = loaded from cache), dev={dev_extract_time:.1f}s"
    )
    print(f"training time:          {training_time:.1f}s")
    print(f"accuracy:               {metrics['accuracy']:.4f}")
    print(f"precision:              {metrics['precision']:.4f}")
    print(f"recall:                 {metrics['recall']:.4f}")
    print(f"f1 score:               {metrics['f1']:.4f}")
    print(f"confusion matrix (rows=true, cols=pred, order=[bonafide, spoof]):")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    if "--full" in sys.argv:
        run_full_experiment()
    else:
        run_sanity_check()
