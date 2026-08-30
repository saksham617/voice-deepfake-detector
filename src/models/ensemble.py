"""Probability-averaging ensemble of SVM + Random Forest + CNN.

Combines each model's spoof-class probability estimate to test whether
ensembling improves generalization to the ASVspoof 2019 LA eval protocol's
unseen attacks (A07-A19), especially the historically hard A17-A19.

Reuses existing code rather than re-implementing it:
- src.models.svm_baseline.build_or_load_feature_matrix / build_svm_pipeline
  for the cached MFCC-statistics features and SVM (SVM/RF have no saved
  model checkpoint, so they are refit here on the cached train features --
  this is fast, ~seconds, not a full retraining experiment).
- src.models.random_forest_baseline.build_random_forest for the RF model.
- src.models.inference.load_model for the already-trained CNN checkpoint
  (the CNN itself is NOT retrained) and src.models.cnn_baseline.predict_proba_all
  for batched CNN inference.
- src.evaluation.cnn_evaluation.attack_wise_results for the attack-wise
  breakdown, and src.models.svm_baseline.evaluate_predictions for metrics,
  so all four models are scored identically.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_eval_protocol, load_train_protocol
from src.evaluation.cnn_evaluation import attack_wise_results
from src.models.cnn_baseline import get_device, make_dataloader, predict_proba_all
from src.models.inference import CHECKPOINT_PATH, load_model
from src.models.random_forest_baseline import build_random_forest
from src.models.svm_baseline import (
    build_or_load_feature_matrix,
    build_svm_pipeline,
    class_distribution,
    evaluate_predictions,
    sample_subset,
)

# Configurable ensemble weights, initially equal (1/3 each). Renormalized in
# combine_probabilities() so they don't need to sum to exactly 1.
DEFAULT_WEIGHTS = {"svm": 1 / 3, "random_forest": 1 / 3, "cnn": 1 / 3}

MODEL_NAMES = ["svm", "random_forest", "cnn", "ensemble"]


def train_svm_rf(train_df: pd.DataFrame):
    """Fit SVM (probability=True) and Random Forest on the cached train MFCC-stat features."""
    X_train, y_train, _ = build_or_load_feature_matrix(train_df, "train_mfcc_stats")

    svm_pipeline = build_svm_pipeline(probability=True)
    svm_pipeline.fit(X_train, y_train)

    rf_model = build_random_forest()
    rf_model.fit(X_train, y_train)

    return svm_pipeline, rf_model


def get_svm_rf_probabilities(
    svm_pipeline, rf_model, eval_df: pd.DataFrame, cache_name: str = "eval_mfcc_stats"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build/load MFCC-stat features for eval_df and return (y_true, svm_spoof_proba, rf_spoof_proba).

    cache_name must be unique per distinct eval_df (e.g. a smoke-test subset
    must use a different cache_name than the full eval set), since the cache
    is keyed only by name, not by which rows were used to build it.
    """
    X_eval, y_eval, _ = build_or_load_feature_matrix(eval_df, cache_name)
    svm_proba = svm_pipeline.predict_proba(X_eval)[:, 1]
    rf_proba = rf_model.predict_proba(X_eval)[:, 1]
    return y_eval, svm_proba, rf_proba


def get_cnn_probabilities(
    eval_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    progress_every: int = 0,
) -> np.ndarray:
    """Run the already-trained CNN checkpoint over eval_df; return spoof-class probabilities."""
    model, device = load_model(CHECKPOINT_PATH)
    loader = make_dataloader(eval_df, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probabilities = predict_proba_all(model, loader, device, progress_every=progress_every)
    return probabilities[:, 1]


def combine_probabilities(
    svm_proba: np.ndarray,
    rf_proba: np.ndarray,
    cnn_proba: np.ndarray,
    weights: dict = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Weighted average of per-model spoof probabilities (weights renormalized to sum to 1)."""
    total_weight = weights["svm"] + weights["random_forest"] + weights["cnn"]
    w_svm = weights["svm"] / total_weight
    w_rf = weights["random_forest"] / total_weight
    w_cnn = weights["cnn"] / total_weight
    return w_svm * svm_proba + w_rf * rf_proba + w_cnn * cnn_proba


def labels_from_proba(spoof_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (spoof_proba >= threshold).astype(int)


def evaluate_all(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    weights: dict = DEFAULT_WEIGHTS,
    cnn_batch_size: int = 32,
    cnn_num_workers: int = 4,
    cnn_progress_every: int = 0,
    eval_cache_name: str = "eval_mfcc_stats",
) -> dict:
    """Train SVM/RF, run CNN inference, build the ensemble, and score all four.

    Returns {"y_true": ..., "svm": {...}, "random_forest": {...}, "cnn": {...},
    "ensemble": {...}}, each of the four with "proba", "y_pred", "metrics".

    eval_cache_name must be distinct for differently-sized/sampled eval_df
    inputs (e.g. a smoke-test subset vs. the full eval set) -- see
    get_svm_rf_probabilities.
    """
    svm_pipeline, rf_model = train_svm_rf(train_df)
    y_true, svm_proba, rf_proba = get_svm_rf_probabilities(
        svm_pipeline, rf_model, eval_df, cache_name=eval_cache_name
    )
    cnn_proba = get_cnn_probabilities(
        eval_df,
        batch_size=cnn_batch_size,
        num_workers=cnn_num_workers,
        progress_every=cnn_progress_every,
    )
    ensemble_proba = combine_probabilities(svm_proba, rf_proba, cnn_proba, weights)

    results = {"y_true": y_true}
    for name, proba in [
        ("svm", svm_proba),
        ("random_forest", rf_proba),
        ("cnn", cnn_proba),
        ("ensemble", ensemble_proba),
    ]:
        y_pred = labels_from_proba(proba)
        results[name] = {
            "proba": proba,
            "y_pred": y_pred,
            "metrics": evaluate_predictions(y_true, y_pred),
        }
    return results


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n[{name}]")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1 score:  {metrics['f1']:.4f}")
    print("  confusion matrix (rows=true, cols=pred, order=[bonafide, spoof]):")
    print(f"  {metrics['confusion_matrix']}")


def print_comparison(results: dict) -> None:
    print("\n" + "=" * 70)
    print("comparison summary (accuracy / precision / recall / f1)")
    print("=" * 70)
    for name in MODEL_NAMES:
        m = results[name]["metrics"]
        print(
            f"{name:15s} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}"
        )


def print_attack_comparison(eval_df: pd.DataFrame, results: dict) -> None:
    attack_accuracies = {
        display_name: attack_wise_results(eval_df, results[name]["y_pred"])["accuracy"]
        for name, display_name in zip(MODEL_NAMES, ["SVM", "Random Forest", "CNN", "Ensemble"])
    }
    comparison = pd.DataFrame(attack_accuracies)

    print("\nattack-wise accuracy per model (A07-A19 + bonafide '-'):")
    print(comparison.to_string())

    hard_attacks = [a for a in ["A17", "A18", "A19"] if a in comparison.index]
    if hard_attacks:
        print("\nfocus: hardest unseen attacks")
        print(comparison.loc[hard_attacks].to_string())


def run_smoke_test(n_eval_samples: int = 40) -> None:
    """Small-subset check: SVM/RF fit, CNN inference, ensembling, and metrics all wire up."""
    train_df = load_train_protocol()
    eval_df = sample_subset(load_eval_protocol(), n_eval_samples)

    print(f"smoke-test eval subset size: {len(eval_df)}")
    print(f"smoke-test eval class dist:  {class_distribution(eval_df['label'].to_numpy())}")

    results = evaluate_all(
        train_df,
        eval_df,
        cnn_batch_size=8,
        cnn_num_workers=2,
        eval_cache_name="smoke_test_eval_mfcc_stats",
    )

    for name in MODEL_NAMES:
        print_metrics(name, results[name]["metrics"])

    print(
        "\nsmoke test passed: SVM/RF training, CNN inference, probability "
        "ensembling, and metrics all succeeded."
    )


def run_full_evaluation(weights: dict = DEFAULT_WEIGHTS) -> None:
    """Train SVM/RF and run the CNN + ensemble over the full ASVspoof 2019 LA eval protocol."""
    train_df = load_train_protocol()
    eval_df = load_eval_protocol()

    print(f"train samples:    {len(train_df)}")
    print(f"eval samples:     {len(eval_df)}")
    print(f"eval class dist:  {class_distribution(eval_df['label'].to_numpy())}")
    print("positive class:   spoof (label=1); bonafide=0 is negative")
    print(f"ensemble weights: {weights}")

    device = get_device()
    print(f"device:           {device}")

    print("\ntraining SVM (probability=True) + Random Forest, running CNN inference...")
    start = time.perf_counter()
    results = evaluate_all(train_df, eval_df, cnn_progress_every=200)
    total_time = time.perf_counter() - start
    print(f"\ntotal runtime: {total_time:.1f}s")

    for name in MODEL_NAMES:
        print_metrics(name, results[name]["metrics"])

    print_comparison(results)
    print_attack_comparison(eval_df, results)


if __name__ == "__main__":
    print("running smoke test...\n")
    run_smoke_test()

    if "--full" in sys.argv:
        print("\n" + "=" * 60)
        print("running full evaluation...\n")
        run_full_evaluation()
