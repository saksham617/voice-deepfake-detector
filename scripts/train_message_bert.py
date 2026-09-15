"""Fine-tune a BERT (DistilBERT by default) message-phishing classifier.

A drop-in upgrade path for the shipped TF-IDF + LogisticRegression model in
``src/models/message_detection.py``. It reuses that module's exact dataset loading and 80/20
stratified split (random_state=42), so the metrics printed here are directly comparable to the
baseline documented there (97.9% acc, F1 0.915 on the held-out 1,034 messages).

Trains with a small manual PyTorch loop (no Trainer/accelerate needed), on GPU if available.
Saves the fine-tuned model + tokenizer to ``data/processed/models/message_bert/`` and writes a
``metrics.json`` (incl. the F1-optimal spam-probability threshold, mirroring the baseline).

Usage:
    python scripts/download_datasets.py   # (or the curl in message_detection.py's docstring)
    python scripts/train_message_bert.py --epochs 3 --batch 16

The saved model is ~268 MB (DistilBERT) — it is NOT committed to git (see .gitignore). Produce
it by running this script, the same way the voice checkpoints are produced by training/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.message_detection import RANDOM_STATE, TEST_SIZE, _load_dataset  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "models" / "message_bert"


def _best_f1_threshold(y_true: np.ndarray, spam_prob: np.ndarray) -> tuple[float, float]:
    """Sweep every distinct predicted probability for the F1-maximising cutoff (as the
    TF-IDF baseline does), so the two models are compared on the same footing."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(spam_prob):
        f1 = f1_score(y_true, (spam_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    df = _load_dataset()  # raises DatasetNotFoundError with fetch instructions if missing
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )
    print(f"Dataset: {len(df)} messages -> {len(X_train)} train / {len(X_test)} test "
          f"({int(sum(y_test))} spam in test)")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tok(texts, truncation=True, padding="max_length",
                   max_length=args.max_len, return_tensors="pt")

    enc_tr, enc_te = encode(X_train), encode(X_test)
    ds_tr = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"],
                          torch.tensor(y_train, dtype=torch.long))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for input_ids, attn, labels in dl_tr:
            opt.zero_grad()
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                        labels=labels.to(device))
            out.loss.backward()
            opt.step()
            running += out.loss.item()
        print(f"  epoch {epoch + 1}/{args.epochs}  loss={running / len(dl_tr):.4f}")
    print(f"Trained in {time.time() - t0:.0f}s")

    # -- evaluate on the held-out split -------------------------------------
    model.eval()
    spam_prob = np.empty(len(X_test), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X_test), args.batch):
            ids = enc_te["input_ids"][i : i + args.batch].to(device)
            am = enc_te["attention_mask"][i : i + args.batch].to(device)
            probs = torch.softmax(model(input_ids=ids, attention_mask=am).logits, dim=1)
            spam_prob[i : i + args.batch] = probs[:, 1].cpu().numpy()

    y_test_arr = np.array(y_test)
    y_pred_05 = (spam_prob >= 0.5).astype(int)
    best_t, best_f1 = _best_f1_threshold(y_test_arr, spam_prob)
    metrics = {
        "model": args.model,
        "accuracy": float(accuracy_score(y_test_arr, y_pred_05)),
        "precision": float(precision_score(y_test_arr, y_pred_05, zero_division=0)),
        "recall": float(recall_score(y_test_arr, y_pred_05, zero_division=0)),
        "f1": float(f1_score(y_test_arr, y_pred_05, zero_division=0)),
        "best_threshold": best_t,
        "f1_at_best_threshold": best_f1,
        "n_train": len(X_train), "n_test": len(X_test), "n_spam_test": int(y_test_arr.sum()),
        "epochs": args.epochs, "device": device,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== BERT message classifier — held-out metrics ===")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:<10} {metrics[k]:.4f}")
    print(f"  F1-optimal threshold: {best_t:.3f} (F1={best_f1:.4f})")
    print(f"\nBaseline (TF-IDF+LogReg):  acc 0.979  F1 0.915  (see message_detection.py)")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
