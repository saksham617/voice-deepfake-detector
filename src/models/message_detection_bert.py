"""BERT-backed message phishing classifier — inference side.

Loads the DistilBERT model fine-tuned by ``scripts/train_message_bert.py`` and exposes the
SAME interface as the TF-IDF ``message_detection.py`` (``classify_message`` returning
verdict/confidence/spam_probability), so it is a drop-in backend swap behind the
``/check_message`` API.

Unlike the TF-IDF module this one imports torch/transformers, so it is only loaded lazily
(when the BERT backend is actually selected — see ``message_detection.classify_message``),
keeping the default text path free of the voice stack's heavy dependencies.

The fine-tuned model (~268 MB) is NOT committed; run the training script to produce it. If the
model directory is absent, ``get_model()`` raises ``BertModelNotFoundError`` so the caller can
fall back to the TF-IDF classifier instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "data" / "processed" / "models" / "message_bert"

# Fallback if metrics.json (written by training) is missing; training normally records the
# F1-optimal cutoff, mirroring message_detection.SUSPICIOUS_THRESHOLD.
DEFAULT_THRESHOLD = 0.5

_cached = None  # (tokenizer, model, device, threshold)


class BertModelNotFoundError(Exception):
    """Raised when the fine-tuned BERT model directory is absent (not trained yet)."""


def _load_threshold() -> float:
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            return float(json.loads(metrics_path.read_text(encoding="utf-8"))["best_threshold"])
        except (ValueError, KeyError):
            pass
    return DEFAULT_THRESHOLD


def get_model():
    """Lazy singleton: (tokenizer, model, device, threshold). Loads on first use."""
    global _cached
    if _cached is not None:
        return _cached

    if not (MODEL_DIR / "config.json").exists():
        raise BertModelNotFoundError(
            f"No fine-tuned BERT model at {MODEL_DIR}. Train it first: "
            f"python scripts/train_message_bert.py"
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    _cached = (tokenizer, model, device, _load_threshold())
    return _cached


def ensure_model_loaded() -> None:
    get_model()


def classify_message(text: str) -> dict:
    """Classify one message with the fine-tuned BERT model.

    Returns the same schema as message_detection.classify_message:
    {verdict: "safe"|"suspicious", confidence: float, spam_probability: float}.
    Raises ValueError on empty text; BertModelNotFoundError if the model isn't trained.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    import torch

    tokenizer, model, device, threshold = get_model()
    enc = tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        spam_probability = float(torch.softmax(logits, dim=1)[0, 1].item())

    is_suspicious = spam_probability >= threshold
    return {
        "verdict": "suspicious" if is_suspicious else "safe",
        "confidence": spam_probability if is_suspicious else 1.0 - spam_probability,
        "spam_probability": spam_probability,
    }
