"""Minimal audio loading/inspection utility for ASVspoof 2019 LA .flac files.

Loads a single audio file at a time and reports basic properties
(sample rate, sample count, duration, channel count). Does not batch-load
the dataset or extract features.
"""

import sys
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.asvspoof_cm_loader import load_train_protocol

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1

# Every ASVspoof 2019 LA file is peak-normalized to ~1.0 (confirmed across a
# 50-file random sample of the train protocol: mean/median/min peak all
# 1.0000), so matching that convention -- not RMS-matching, which varies
# file to file even within the dataset -- is what makes an external
# recording's absolute loudness comparable to what the model was trained on.
# Target is just under full scale to leave headroom against any tiny
# overshoot from resampling.
TARGET_PEAK_AMPLITUDE = 0.99

# Silence below this level (relative to the clip's own peak, in dB) is
# trimmed from the start/end before normalization. Deliberately librosa's
# own default -- a lower threshold (30) was tried and rejected: on a real
# ASVspoof bonafide file it trimmed 28% of the clip (0.8s of a 3.46s clip)
# as "leading silence" when that was actually quiet speech content (soft
# onset/consonants), flipping a correct bonafide prediction to spoof at
# 100% confidence. top_db=60 only removes near-total silence: 34ms on that
# same bonafide file, 192ms of genuine lead-in gap on a real external
# recording.
SILENCE_TRIM_TOP_DB = 60


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load a single audio file and return (waveform, sample_rate)."""
    waveform, sample_rate = sf.read(path, dtype="float32")
    return waveform, sample_rate


def load_audio_resampled(
    path: str, target_sr: int = EXPECTED_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resampling to target_sr and mixing down to mono.

    Inference-time safety net for arbitrary user uploads, which -- unlike
    the ASVspoof dataset load_audio() assumes -- aren't guaranteed to
    already be 16 kHz mono. librosa.load(sr=..., mono=True) is a no-op
    when the source already matches (no resample call, no channel mixing),
    so this is safe to point at already-conforming files too.

    Also trims leading/trailing silence and peak-normalizes to
    TARGET_PEAK_AMPLITUDE, matching the loudness/silence characteristics
    every ASVspoof training file already has -- an external recording (quiet
    mic gain, room-noise lead-in) otherwise produces a log-mel spectrogram
    with a substantially different value distribution than anything the
    model trained on, even after resampling/mono-mixing alone.
    """
    waveform, sample_rate = librosa.load(path, sr=target_sr, mono=True)

    trimmed, _ = librosa.effects.trim(waveform, top_db=SILENCE_TRIM_TOP_DB)
    # A pathological all-silence/all-noise clip can trim to (near-)empty;
    # fall back to the untrimmed waveform rather than normalize by ~0.
    if trimmed.size > 0:
        waveform = trimmed

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform * (TARGET_PEAK_AMPLITUDE / peak)

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
