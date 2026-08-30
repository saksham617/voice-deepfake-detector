"""CNN generalization evaluation on the ASVspoof 2019 LA eval CM protocol.

The eval protocol uses 13 spoofing systems (A07-A19) never seen during
training or dev (train/dev only contain A01-A06), so this measures how well
the already-trained CNN generalizes to unseen attacks -- it does not retrain
the model or modify the raw dataset.

Reuses the exact training/inference preprocessing and model-loading code:
- src.data.asvspoof_cm_loader.load_eval_protocol for the protocol/paths
- src.models.cnn_baseline.SpectrogramDataset/make_dataloader/predict_all for
  on-the-fly spectrogram extraction and batched inference
- src.models.inference.load_model for loading the trained checkpoint
- src.models.svm_baseline.evaluate_predictions/class_distribution for metrics,
  so results are directly comparable to the SVM/RF/CNN dev-set numbers
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_eval_protocol
from src.features.spectrogram import extract_log_mel_spectrogram
from src.models.cnn_baseline import get_device, make_dataloader, predict_all
from src.models.inference import CHECKPOINT_PATH, load_model
from src.models.svm_baseline import class_distribution, evaluate_predictions, sample_subset

BATCH_SIZE = 32
NUM_WORKERS = 4


def run_evaluation(
    eval_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    progress_every: int = 0,
) -> dict:
    """Run the trained CNN checkpoint over `eval_df` and return predictions + metrics."""
    model, device = load_model(CHECKPOINT_PATH)
    loader = make_dataloader(eval_df, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    start = time.perf_counter()
    y_pred = predict_all(model, loader, device, progress_every=progress_every)
    runtime_sec = time.perf_counter() - start

    y_true = eval_df["label"].to_numpy()
    metrics = evaluate_predictions(y_true, y_pred)

    return {"y_true": y_true, "y_pred": y_pred, "metrics": metrics, "runtime_sec": runtime_sec}


def attack_wise_results(eval_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-system_id accuracy, one row per spoofing attack plus one for bonafide.

    Bonafide rows carry system_id='-' in the ASVspoof protocol, so grouping by
    system_id naturally separates bonafide accuracy from each of the 13
    eval-only spoofing systems (A07-A19).
    """
    breakdown = eval_df[["system_id", "label"]].copy()
    breakdown["correct"] = (y_pred == breakdown["label"].to_numpy()).astype(int)

    grouped = breakdown.groupby("system_id").agg(
        n_samples=("label", "size"),
        true_label=("label", "first"),
        accuracy=("correct", "mean"),
    )
    grouped["true_label"] = grouped["true_label"].map({0: "bonafide", 1: "spoof"})
    return grouped.sort_index()


def print_metrics(metrics: dict) -> None:
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1 score:  {metrics['f1']:.4f}")
    print("confusion matrix (rows=true, cols=pred, order=[bonafide, spoof]):")
    print(metrics["confusion_matrix"])


def run_smoke_test(n_samples: int = 40) -> None:
    """Small-subset check: protocol parsing, checkpoint loading, spectrogram shape, predictions, metrics."""
    eval_df = load_eval_protocol()
    print(f"eval protocol entries: {len(eval_df)}")
    print(f"eval class dist:       {class_distribution(eval_df['label'].to_numpy())}")

    subset_df = sample_subset(eval_df, n_samples)
    print(f"\nsmoke-test subset size: {len(subset_df)}")
    print(f"smoke-test class dist:  {class_distribution(subset_df['label'].to_numpy())}")

    sample_spectrogram = extract_log_mel_spectrogram(subset_df.iloc[0]["path"])
    print(f"sample spectrogram shape: {sample_spectrogram.shape}")

    result = run_evaluation(subset_df, batch_size=8, num_workers=2)
    print(f"\nsmoke-test runtime: {result['runtime_sec']:.2f}s")
    print_metrics(result["metrics"])

    print("\nsmoke test passed: protocol parsing, checkpoint loading, spectrogram shape, "
          "predictions, and metrics all succeeded.")


def run_full_evaluation() -> None:
    """Run the trained CNN over the full ASVspoof 2019 LA eval protocol."""
    eval_df = load_eval_protocol()

    print(f"total eval samples: {len(eval_df)}")
    print(f"class distribution: {class_distribution(eval_df['label'].to_numpy())}")
    print("positive class:     spoof (label=1); bonafide=0 is negative")

    device = get_device()
    print(f"device:             {device}")
    print(f"checkpoint:         {CHECKPOINT_PATH}")

    print("\nrunning inference over eval set...")
    result = run_evaluation(eval_df, progress_every=200)

    print(f"\nruntime:            {result['runtime_sec']:.1f}s")
    print_metrics(result["metrics"])

    print("\nattack-wise results (accuracy per system_id; '-' = bonafide):")
    attack_df = attack_wise_results(eval_df, result["y_pred"])
    print(attack_df.to_string())


if __name__ == "__main__":
    print("running smoke test...\n")
    run_smoke_test()

    if "--full" in sys.argv:
        print("\n" + "=" * 60)
        print("running full evaluation...\n")
        run_full_evaluation()
