"""Inference for the trained CNN spoof-detection model.

Reuses the exact preprocessing pipeline used during training
(src.features.spectrogram.extract_log_mel_spectrogram: audio -> 16 kHz mono
-> 64x400 log-Mel spectrogram) and the exact model architecture and device
selection used during training (src.models.cnn_baseline), so inference sees
the model in the same conditions it was trained under. Does not retrain or
re-define the architecture, and does not cache spectrograms to disk.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.spectrogram import extract_log_mel_spectrogram
from src.models.cnn_baseline import CHECKPOINT_DIR, SpoofCNN, get_device

CHECKPOINT_PATH = CHECKPOINT_DIR / "cnn_baseline.pt"
LABEL_NAMES = {0: "bonafide", 1: "spoof"}

# Module-level cache so repeated predict_audio() calls (e.g. from a backend
# request handler) don't reload the checkpoint from disk every time.
_cached_model = None
_cached_device = None


def load_model(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[SpoofCNN, torch.device]:
    """Load the trained SpoofCNN checkpoint onto the best available device.

    Device preference: MPS, then CUDA, then CPU (src.models.cnn_baseline.get_device).
    """
    device = get_device()
    model = SpoofCNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def _get_model() -> tuple[SpoofCNN, torch.device]:
    global _cached_model, _cached_device
    if _cached_model is None:
        _cached_model, _cached_device = load_model()
    return _cached_model, _cached_device


def ensure_model_loaded() -> torch.device:
    """Eagerly load (and cache) the model -- e.g. at application startup.

    Raises if the checkpoint is missing/corrupt, so a caller (like a FastAPI
    startup hook) can treat that as a readiness failure. Subsequent
    predict_audio() calls reuse this same cached model.
    """
    _, device = _get_model()
    return device


def preprocess(path: str) -> torch.Tensor:
    """Convert one audio file into the (1, 1, n_mels, n_frames) tensor the CNN expects."""
    spectrogram = extract_log_mel_spectrogram(path)  # (n_mels, n_frames)
    return torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict_audio(path: str, verbose: bool = False) -> dict:
    """Run one audio file through the full inference pipeline.

    Returns a dict with predicted_label ("bonafide"/"spoof"), predicted_class
    (0/1), bonafide_probability, and spoof_probability.
    """
    model, device = _get_model()

    input_tensor = preprocess(path).to(device)
    if verbose:
        print(f"input tensor shape: {tuple(input_tensor.shape)}")

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    predicted_class = int(torch.argmax(probabilities).item())

    return {
        "predicted_label": LABEL_NAMES[predicted_class],
        "predicted_class": predicted_class,
        "bonafide_probability": probabilities[0].item(),
        "spoof_probability": probabilities[1].item(),
    }


if __name__ == "__main__":
    from src.data.asvspoof_cm_loader import load_train_protocol

    model, device = _get_model()
    print(f"checkpoint loaded from: {CHECKPOINT_PATH}")
    print(f"device: {device}")

    train_df = load_train_protocol()
    bonafide_path = train_df.loc[train_df["label"] == 0, "path"].iloc[0]
    spoof_path = train_df.loc[train_df["label"] == 1, "path"].iloc[0]

    for name, path in [("known bonafide", bonafide_path), ("known spoof", spoof_path)]:
        print(f"\n--- {name} file ---")
        print(f"path: {path}")
        result = predict_audio(path, verbose=True)
        print(f"predicted_label:       {result['predicted_label']}")
        print(f"predicted_class:       {result['predicted_class']}")
        print(f"bonafide_probability:  {result['bonafide_probability']:.4f}")
        print(f"spoof_probability:     {result['spoof_probability']:.4f}")
