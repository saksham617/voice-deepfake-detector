"""Tune SVM+RF+CNN ensemble weights on the dev set, then evaluate on eval.

The equal-weight ensemble (src.models.ensemble, commit 614b7f1) underperforms
the best individual model on the hardest unseen attacks (A17-A19): it blindly
averages in the CNN's confidently-wrong spoof probability on those attacks,
dragging the combined score toward bonafide. This module searches ensemble
weights on the DEV set only (never the eval set) to avoid overfitting the
reported generalization numbers, then applies the selected weights to the
held-out eval set (unseen attacks) for the final comparison.

Reuses everything from src.models.ensemble: train_svm_rf, probability
extraction (get_svm_rf_probabilities/get_cnn_probabilities), combine_probabilities,
labels_from_proba, print_metrics -- this module only adds the weight grid
search and the dev-then-eval evaluation flow.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_dev_protocol, load_eval_protocol, load_train_protocol
from src.evaluation.cnn_evaluation import attack_wise_results
from src.models.ensemble import (
    DEFAULT_WEIGHTS,
    combine_probabilities,
    get_cnn_probabilities,
    get_svm_rf_probabilities,
    labels_from_proba,
    print_metrics,
    train_svm_rf,
)
from src.models.svm_baseline import class_distribution, evaluate_predictions, sample_subset

WEIGHT_STEP = 0.1
TUNING_METRIC = "f1"


def generate_weight_grid(step: float = WEIGHT_STEP) -> list[dict]:
    """All (svm, random_forest, cnn) weight combinations on a step-sized simplex grid summing to 1."""
    n_steps = int(round(1 / step))
    grid = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            grid.append(
                {
                    "svm": round(i * step, 2),
                    "random_forest": round(j * step, 2),
                    "cnn": round(k * step, 2),
                }
            )
    return grid


def tune_weights(
    y_true: np.ndarray,
    svm_proba: np.ndarray,
    rf_proba: np.ndarray,
    cnn_proba: np.ndarray,
    metric: str = TUNING_METRIC,
    step: float = WEIGHT_STEP,
) -> dict:
    """Grid-search ensemble weights maximizing `metric` on the given probabilities.

    Intended to be called with DEV-set probabilities so weight selection
    never sees the eval set (which stays a clean generalization test).
    """
    best = {"weights": None, "score": -1.0, "metrics": None}
    for weights in generate_weight_grid(step):
        ensemble_proba = combine_probabilities(svm_proba, rf_proba, cnn_proba, weights)
        y_pred = labels_from_proba(ensemble_proba)
        metrics = evaluate_predictions(y_true, y_pred)
        score = metrics[metric]
        if score > best["score"]:
            best = {"weights": weights, "score": score, "metrics": metrics}
    return best


def run_smoke_test(n_samples: int = 40) -> None:
    """Small-subset check: weight grid search wiring, dev probabilities, metric computation."""
    train_df = load_train_protocol()
    dev_subset = sample_subset(load_dev_protocol(), n_samples)

    print(f"smoke-test dev subset size: {len(dev_subset)}")
    print(f"smoke-test dev class dist:  {class_distribution(dev_subset['label'].to_numpy())}")

    svm_pipeline, rf_model = train_svm_rf(train_df)
    y_dev, svm_proba, rf_proba = get_svm_rf_probabilities(
        svm_pipeline, rf_model, dev_subset, cache_name="smoke_test_dev_mfcc_stats"
    )
    cnn_proba = get_cnn_probabilities(dev_subset, batch_size=8, num_workers=2)

    grid = generate_weight_grid(step=0.25)
    print(f"smoke-test weight grid size (step=0.25): {len(grid)}")

    best = tune_weights(y_dev, svm_proba, rf_proba, cnn_proba, step=0.25)
    print(f"smoke-test best weights: {best['weights']}")
    print(f"smoke-test best f1:      {best['score']:.4f}")

    print("\nsmoke test passed: weight grid search and metric computation succeeded.")


def run_full_tuning_and_evaluation(metric: str = TUNING_METRIC, step: float = WEIGHT_STEP) -> None:
    """Tune ensemble weights on the full dev set, then evaluate on the full eval set."""
    train_df = load_train_protocol()
    dev_df = load_dev_protocol()
    eval_df = load_eval_protocol()

    print(f"train samples: {len(train_df)}")
    print(f"dev samples:   {len(dev_df)}")
    print(f"eval samples:  {len(eval_df)}")

    print("\ntraining SVM (probability=True) + Random Forest on cached train features...")
    svm_pipeline, rf_model = train_svm_rf(train_df)

    print("building dev-set probabilities (weight tuning happens here, not on eval)...")
    y_dev, svm_dev_proba, rf_dev_proba = get_svm_rf_probabilities(
        svm_pipeline, rf_model, dev_df, cache_name="dev_mfcc_stats"
    )
    cnn_dev_proba = get_cnn_probabilities(dev_df, progress_every=200)

    print(f"\nsearching weight grid (step={step}, metric={metric})...")
    grid = generate_weight_grid(step)
    print(f"grid size: {len(grid)} combinations")
    best = tune_weights(y_dev, svm_dev_proba, rf_dev_proba, cnn_dev_proba, metric=metric, step=step)

    print(f"\nbest weights (selected on dev): {best['weights']}")
    print(f"best dev {metric}: {best['score']:.4f}")
    print_metrics("tuned ensemble (dev)", best["metrics"])

    print("\n" + "=" * 70)
    print("applying tuned weights to the held-out EVAL set (unseen attacks)")
    print("=" * 70)

    y_eval, svm_eval_proba, rf_eval_proba = get_svm_rf_probabilities(
        svm_pipeline, rf_model, eval_df, cache_name="eval_mfcc_stats"
    )
    cnn_eval_proba = get_cnn_probabilities(eval_df, progress_every=200)

    equal_proba = combine_probabilities(svm_eval_proba, rf_eval_proba, cnn_eval_proba, DEFAULT_WEIGHTS)
    tuned_proba = combine_probabilities(svm_eval_proba, rf_eval_proba, cnn_eval_proba, best["weights"])

    eval_results = {}
    for name, proba in [
        ("svm", svm_eval_proba),
        ("random_forest", rf_eval_proba),
        ("cnn", cnn_eval_proba),
        ("ensemble_equal", equal_proba),
        ("ensemble_tuned", tuned_proba),
    ]:
        y_pred = labels_from_proba(proba)
        eval_results[name] = {"y_pred": y_pred, "metrics": evaluate_predictions(y_eval, y_pred)}

    model_order = ["svm", "random_forest", "cnn", "ensemble_equal", "ensemble_tuned"]
    for name in model_order:
        print_metrics(name, eval_results[name]["metrics"])

    print("\n" + "=" * 70)
    print("final comparison on EVAL set (accuracy / precision / recall / f1)")
    print("=" * 70)
    for name in model_order:
        m = eval_results[name]["metrics"]
        print(
            f"{name:16s} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}"
        )

    print("\nattack-wise accuracy per model (A07-A19 + bonafide '-'):")
    display_names = ["SVM", "Random Forest", "CNN", "Ensemble (equal)", "Ensemble (tuned)"]
    attack_accuracies = {
        display: attack_wise_results(eval_df, eval_results[name]["y_pred"])["accuracy"]
        for name, display in zip(model_order, display_names)
    }
    comparison = pd.DataFrame(attack_accuracies)
    print(comparison.to_string())

    hard_attacks = [a for a in ["A17", "A18", "A19"] if a in comparison.index]
    if hard_attacks:
        print("\nfocus: hardest unseen attacks")
        print(comparison.loc[hard_attacks].to_string())


if __name__ == "__main__":
    print("running smoke test...\n")
    run_smoke_test()

    if "--full" in sys.argv:
        print("\n" + "=" * 60)
        print("running full tuning + evaluation...\n")
        run_full_tuning_and_evaluation()
