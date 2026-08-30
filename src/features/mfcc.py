"""MFCC feature extraction utility for ASVspoof 2019 LA audio.

Extracts MFCCs from a single audio file at a time using librosa. Reuses
the existing audio-loading utility rather than reading files directly.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.audio_io import load_audio

# Standard short-time analysis config for 16 kHz speech: 25 ms window, 10 ms hop.
N_MFCC = 13
N_FFT = 400  # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz


def extract_mfcc(
    path: str,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Load one audio file and extract its MFCC matrix.

    Returns an array of shape (n_mfcc, n_frames).
    """
    waveform, sample_rate = load_audio(path)
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc


def inspect_mfcc(path: str) -> np.ndarray:
    """Extract MFCCs for one file and print duration/shape info."""
    waveform, sample_rate = load_audio(path)
    duration_sec = waveform.shape[0] / sample_rate

    mfcc = extract_mfcc(path)
    n_coeffs, n_frames = mfcc.shape

    print(f"input duration:   {duration_sec:.4f} s")
    print(f"sample_rate:      {sample_rate} Hz")
    print(f"mfcc shape:       {mfcc.shape}")
    print(f"n_mfcc_coeffs:    {n_coeffs}")
    print(f"n_time_frames:    {n_frames}")

    return mfcc


if __name__ == "__main__":
    from src.data.asvspoof_cm_loader import load_train_protocol

    sample_path = load_train_protocol().iloc[0]["path"]
    inspect_mfcc(sample_path)
