"""Minimal audio loading/inspection utility for ASVspoof 2019 LA .flac files.

Loads a single audio file at a time and reports basic properties
(sample rate, sample count, duration, channel count). Does not batch-load
the dataset or extract features.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_train_protocol

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load a single audio file and return (waveform, sample_rate)."""
    waveform, sample_rate = sf.read(path, dtype="float32")
    return waveform, sample_rate


def inspect_audio(path: str) -> dict:
    """Load a single audio file and report its basic properties."""
    waveform, sample_rate = load_audio(path)

    num_samples = waveform.shape[0]
    num_channels = 1 if waveform.ndim == 1 else waveform.shape[1]
    duration_sec = num_samples / sample_rate

    info = {
        "path": path,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration_sec": duration_sec,
        "num_channels": num_channels,
    }

    print(f"path:          {info['path']}")
    print(f"sample_rate:   {info['sample_rate']} Hz")
    print(f"num_samples:   {info['num_samples']}")
    print(f"duration_sec:  {info['duration_sec']:.4f} s")
    print(f"num_channels:  {info['num_channels']}")

    return info


def verify_asvspoof_format(info: dict) -> None:
    """Assert that an inspected audio file matches ASVspoof LA's expected format."""
    assert info["sample_rate"] == EXPECTED_SAMPLE_RATE, (
        f"expected {EXPECTED_SAMPLE_RATE} Hz, got {info['sample_rate']} Hz"
    )
    assert info["num_channels"] == EXPECTED_CHANNELS, (
        f"expected {EXPECTED_CHANNELS} channel (mono), got {info['num_channels']}"
    )
    print(f"\nFormat check OK: {EXPECTED_SAMPLE_RATE} Hz, mono")


def _get_sample_train_path() -> str:
    """Return the path of the first file listed in the train CM protocol."""
    train_df = load_train_protocol()
    return train_df.iloc[0]["path"]


if __name__ == "__main__":
    sample_path = _get_sample_train_path()
    audio_info = inspect_audio(sample_path)
    verify_asvspoof_format(audio_info)
