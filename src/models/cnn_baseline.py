"""CNN baseline for spectrogram-based ASVspoof 2019 LA spoof detection.

Reuses the existing fixed-size log-Mel spectrogram utility (src.features.
spectrogram) as the CNN's input, rather than re-implementing spectrogram
extraction. Spectrograms are generated on the fly per audio file (via a
PyTorch Dataset), not pre-rendered/saved in bulk.

Architecture is intentionally simple (3 conv blocks + one hidden FC layer)
so every layer can be explained: each block is Conv2d -> ReLU -> MaxPool2d,
which halves the spatial size while learning local time/frequency patterns;
Dropout after each block and before the final layer reduces overfitting;
the classifier head flattens to a fixed-size vector and predicts one of two
classes (bonafide=0, spoof=1).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.spectrogram import FIXED_FRAMES, N_MELS, extract_log_mel_spectrogram

# NOTE: load_train_protocol/load_dev_protocol (src.data.asvspoof_cm_loader) and
# sample_subset/class_distribution/evaluate_predictions (src.models.svm_baseline)
# are intentionally imported locally inside run_smoke_test()/run_full_experiment()
# below, not at module level. svm_baseline pulls in scikit-learn and the full
# MFCC/feature-prep training stack; keeping that out of this module's top-level
# imports lets a deployment that only needs SpoofCNN/get_device/CHECKPOINT_DIR
# (src.models.inference) skip installing scikit-learn entirely.
NUM_CLASSES = 2  # 0 = bonafide, 1 = spoof
RANDOM_STATE = 42

# Checkpoints are dataset-derived artifacts, not source, and are gitignored
# (data/processed/*) rather than committed.
CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "processed" / "models"


class SpectrogramDataset(Dataset):
    """Maps protocol DataFrame rows to (spectrogram, label) tensors.

    Spectrograms are computed on demand from audio files, one at a time, so
    the dataset never holds more than a batch's worth of spectrograms in
    memory at once.
    """

    def __init__(self, protocol_df: pd.DataFrame):
        self.paths = protocol_df["path"].tolist()
        self.labels = protocol_df["label"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = extract_log_mel_spectrogram(self.paths[index])  # (n_mels, n_frames)
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return spectrogram_tensor, label_tensor


class SpoofCNN(nn.Module):
    """A small 3-block CNN classifying (1, n_mels, n_frames) spectrograms as bonafide/spoof."""

    def __init__(self, n_mels: int = N_MELS, n_frames: int = FIXED_FRAMES):
        super().__init__()

        # Log-Mel values are raw dB magnitudes (roughly -80 to 0), which is too
        # large/unscaled to feed a randomly-initialized CNN directly -- it makes
        # activations and gradients blow up. BatchNorm2d on the single input
        # channel standardizes each batch to zero mean / unit variance first.
        self.input_norm = nn.BatchNorm2d(num_features=1)

        # Each block: Conv2d (learn local patterns) -> ReLU (non-linearity)
        # -> MaxPool2d (halve height/width, keep strongest activations)
        # -> Dropout (randomly zero activations during training to reduce overfitting).
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        # Three 2x2 max-pools halve each spatial dimension three times.
        pooled_mels = n_mels // (2 ** 3)
        pooled_frames = n_frames // (2 ** 3)
        flattened_size = 64 * pooled_mels * pooled_frames

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)  # raw logits, shape (batch, NUM_CLASSES)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_smoke_test(n_samples: int = 8, batch_size: int = 4) -> None:
    """Build a tiny DataLoader batch and run one forward pass to verify shapes."""
    from src.data.asvspoof_cm_loader import load_train_protocol
    from src.models.svm_baseline import sample_subset

    subset_df = sample_subset(load_train_protocol(), n_samples)
    print(f"smoke-test subset size: {len(subset_df)}")
    print(f"smoke-test label counts: {subset_df['label'].value_counts().to_dict()}")

    dataset = SpectrogramDataset(subset_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    spectrograms, labels = next(iter(loader))
    print(f"batch spectrogram shape: {tuple(spectrograms.shape)}")
    print(f"batch label shape:       {tuple(labels.shape)}")

    device = get_device()
    print(f"device:                  {device}")

    model = SpoofCNN().to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(spectrograms.to(device))

    print(f"model output shape:      {tuple(outputs.shape)}")
    print(f"model output (logits):\n{outputs}")

    assert outputs.shape == (spectrograms.shape[0], NUM_CLASSES), (
        "expected model output shape (batch_size, NUM_CLASSES)"
    )
    print("\nsmoke test passed: forward pass succeeded with expected shapes.")


def compute_class_weights(labels: pd.Series) -> torch.Tensor:
    """Inverse-frequency class weights, matching sklearn's class_weight='balanced'.

    weight_c = n_samples / (n_classes * count_c), so the ~9:1 spoof:bonafide
    imbalance is compensated for in the loss rather than in the data itself.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_dataloader(
    protocol_df: pd.DataFrame, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    dataset = SpectrogramDataset(protocol_df)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average per-batch loss."""
    model.train()
    total_loss = 0.0
    for spectrograms, labels in loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_proba_all(
    model: nn.Module, loader: DataLoader, device: torch.device, progress_every: int = 0
) -> np.ndarray:
    """Run the model over a full DataLoader and return softmax class probabilities.

    Returns an array of shape (n_samples, NUM_CLASSES) (column 0 = bonafide,
    column 1 = spoof). If progress_every > 0, prints a progress line every
    that many batches (useful for large evaluation sets); silent by default.
    """
    model.eval()
    all_probabilities = []
    total_batches = len(loader)
    for i, (spectrograms, _) in enumerate(loader, start=1):
        outputs = model(spectrograms.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        all_probabilities.append(probabilities.cpu().numpy())
        if progress_every and (i % progress_every == 0 or i == total_batches):
            print(f"  evaluated batch {i}/{total_batches}")
    return np.concatenate(all_probabilities, axis=0)


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device, progress_every: int = 0
) -> np.ndarray:
    """Run the model over a full DataLoader and return hard predicted labels.

    Equivalent to argmax(predict_proba_all(...), axis=1); kept as a separate
    convenience function since most callers only need the hard label.
    """
    probabilities = predict_proba_all(model, loader, device, progress_every=progress_every)
    return probabilities.argmax(axis=1)


def run_full_experiment(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_workers: int | None = None,
) -> None:
    """Train and evaluate the CNN baseline on the full ASVspoof 2019 LA train/dev protocols.

    Spectrograms are extracted on the fly (not cached to disk) because the
    available disk space is too limited to persist the full spectrogram set;
    a multi-worker DataLoader parallelizes extraction across CPU cores instead.
    """
    from src.data.asvspoof_cm_loader import load_dev_protocol, load_train_protocol
    from src.models.svm_baseline import class_distribution, evaluate_predictions

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    torch.manual_seed(RANDOM_STATE)

    train_df = load_train_protocol()
    dev_df = load_dev_protocol()

    print(f"train protocol entries: {len(train_df)}")
    print(f"dev protocol entries:   {len(dev_df)}")
    print(f"train class dist:       {class_distribution(train_df['label'].to_numpy())}")
    print(f"dev class dist:         {class_distribution(dev_df['label'].to_numpy())}")
    print("positive class:         spoof (label=1); bonafide=0 is negative")

    device = get_device()
    print(f"device:                 {device}")
    print(f"num_workers:            {num_workers}")

    train_loader = make_dataloader(train_df, batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = make_dataloader(dev_df, batch_size, shuffle=False, num_workers=num_workers)

    model = SpoofCNN().to(device)
    class_weights = compute_class_weights(train_df["label"].to_numpy()).to(device)
    print(f"class weights (balanced): {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\ntraining for {epochs} epochs...")
    train_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_losses.append(avg_loss)
        print(
            f"  epoch {epoch}/{epochs}: loss={avg_loss:.4f} "
            f"({time.perf_counter() - epoch_start:.1f}s)"
        )
    training_time = time.perf_counter() - train_start

    print("\nevaluating on dev set...")
    y_pred = predict_all(model, dev_loader, device)
    y_dev = dev_df["label"].to_numpy()
    metrics = evaluate_predictions(y_dev, y_pred)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "cnn_baseline.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nsaved model checkpoint to: {checkpoint_path}")

    print(f"\nepochs:             {epochs}")
    print(f"final train loss:   {epoch_losses[-1]:.4f}")
    print(f"loss per epoch:     {[round(l, 4) for l in epoch_losses]}")
    print(f"training time:      {training_time:.1f}s")
    print(f"train samples:      {len(train_df)}")
    print(f"dev samples:        {len(dev_df)}")
    print(f"accuracy:           {metrics['accuracy']:.4f}")
    print(f"precision:          {metrics['precision']:.4f}")
    print(f"recall:             {metrics['recall']:.4f}")
    print(f"f1 score:           {metrics['f1']:.4f}")
    print("confusion matrix (rows=true, cols=pred, order=[bonafide, spoof]):")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    if "--full" in sys.argv:
        run_full_experiment()
    else:
        run_smoke_test()
