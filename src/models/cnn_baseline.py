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

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.spectrogram import FIXED_FRAMES, N_MELS, extract_log_mel_spectrogram
from src.models.svm_baseline import sample_subset

NUM_CLASSES = 2  # 0 = bonafide, 1 = spoof


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


if __name__ == "__main__":
    run_smoke_test()
