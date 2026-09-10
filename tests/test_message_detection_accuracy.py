"""Empirical accuracy check for src/models/message_detection.py.

Uses the public SMS Spam Collection Dataset (Almeida & Hidalgo, 2011, UCI ML
Repository) -- 5,169 messages after de-duplication, downloaded from
https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip into
data/raw/sms_spam_collection/SMSSpamCollection (gitignored like the rest of
data/raw/, so this test is skipped on a fresh clone rather than failing).

Retrains a pipeline on the same 80/20 stratified split train_classifier()
uses (fixed random_state, so this is deterministic and reproducible) and
checks two things against the held-out 20%: (1) the model clears sensible
accuracy/precision/recall bars -- catching a broken pipeline, not demanding
perfection -- and (2) SUSPICIOUS_THRESHOLD in message_detection.py is close
to the F1-maximizing threshold found by sweeping every distinct predicted
probability, so the constant doesn't quietly drift out of sync with the
data it's supposed to reflect.

Run standalone with `-s` to see the full metrics report.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import message_detection as md

pytestmark = pytest.mark.skipif(
    not md.DATASET_PATH.exists(),
    reason=(
        f"SMS Spam Collection Dataset not present at {md.DATASET_PATH} -- "
        f"download from {md.DATASET_URL} to run this accuracy check."
    ),
)


@pytest.fixture(scope="module")
def test_split_scores() -> dict:
    """Reproduce train_classifier()'s exact split, fit a pipeline on train,
    and return predicted probabilities + labels for the held-out test set."""
    df = md._load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=md.TEST_SIZE,
        random_state=md.RANDOM_STATE,
        stratify=df["label"],
    )
    pipeline = md._build_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {"proba": proba, "y_test": y_test.to_numpy(), "n_train": len(X_train)}


def _best_f1_threshold(proba: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Sweep every distinct predicted probability as a candidate threshold
    and return the one that maximizes F1, plus that best F1."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.unique(proba):
        predicted = (proba >= threshold).astype(int)
        f1 = f1_score(y_test, predicted)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold, best_f1


def test_classifier_clears_accuracy_bars_on_held_out_split(test_split_scores):
    proba = test_split_scores["proba"]
    y_test = test_split_scores["y_test"]
    predicted = (proba >= md.SUSPICIOUS_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)

    print("\n=== Message detection accuracy report (SMS Spam Collection, held-out 20%) ===")
    print(f"train={test_split_scores['n_train']}  test={len(y_test)}  spam_in_test={int(y_test.sum())}")
    print(f"SUSPICIOUS_THRESHOLD={md.SUSPICIOUS_THRESHOLD}")
    print(f"accuracy={accuracy:.4f}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")

    # Sanity bars: catches a broken pipeline (e.g. vectorizer/label mismatch)
    # or a genuinely unusable model, without demanding unrealistic perfection.
    assert accuracy > 0.95, f"accuracy {accuracy:.4f} is suspiciously low for this dataset"
    assert precision > 0.85, f"precision {precision:.4f} too low -- too many false 'suspicious' flags"
    assert recall > 0.75, f"recall {recall:.4f} too low -- missing too much real spam/phishing"


def test_suspicious_threshold_is_close_to_f1_maximizing_threshold(test_split_scores):
    proba = test_split_scores["proba"]
    y_test = test_split_scores["y_test"]

    best_threshold, best_f1 = _best_f1_threshold(proba, y_test)
    default_predicted = (proba >= md.SUSPICIOUS_THRESHOLD).astype(int)
    default_f1 = f1_score(y_test, default_predicted)

    print(f"\nF1-maximizing threshold on this split: {best_threshold:.4f} (f1={best_f1:.4f})")
    print(f"SUSPICIOUS_THRESHOLD={md.SUSPICIOUS_THRESHOLD} achieves f1={default_f1:.4f}")

    assert default_f1 >= best_f1 - 0.01, (
        f"SUSPICIOUS_THRESHOLD={md.SUSPICIOUS_THRESHOLD} (f1={default_f1:.4f}) has drifted "
        f"away from the F1-maximizing threshold {best_threshold:.4f} (f1={best_f1:.4f})"
    )


def test_classify_message_flags_phishing_style_text():
    result = md.classify_message(
        "URGENT! Your account will be suspended. Click here to verify your "
        "details now: http://bit.ly/verify-now or you will lose access."
    )
    assert result["verdict"] == "suspicious"
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_message_passes_normal_text():
    result = md.classify_message("Hey, are we still on for dinner tonight at 7?")
    assert result["verdict"] == "safe"
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_message_raises_on_empty_text():
    with pytest.raises(ValueError):
        md.classify_message("   ")
