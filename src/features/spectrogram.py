"""Log-Mel spectrogram feature extraction for CNN input on ASVspoof 2019 LA audio.

Extracts a fixed-size 2D log-Mel spectrogram from a single audio file at a
time using librosa, reusing the existing audio-loading utility rather than
reading files directly. The time axis is padded/truncated to a fixed number
of frames so that clips of different durations all map to the same CNN input
shape. Does not process the dataset in bulk or persist spectrogram images.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.audio_io import load_audio

# Config tuned for 16 kHz speech, consistent with the MFCC pipeline's hop size.
SAMPLE_RATE = 16000
N_FFT = 512  # ~32 ms window at 16 kHz
HOP_LENGTH = 160  # 10 ms hop at 16 kHz, matches mfcc.py
N_MELS = 64  # mel filterbank resolution: enough spectral detail, compact CNN input

# Fixed clip length so every spectrogram has the same time dimension.
FIXED_DURATION_SEC = 4.0
FIXED_FRAMES = int(FIXED_DURATION_SEC * SAMPLE_RATE / HOP_LENGTH)  # 400 frames


def _pad_or_truncate(spec: np.ndarray, fixed_frames: int = FIXED_FRAMES) -> np.ndarray:
    """Pad (with the clip's own silence floor) or truncate a spectrogram's time axis."""
    n_frames = spec.shape[1]
    if n_frames == fixed_frames:
        return spec
    if n_frames > fixed_frames:
        return spec[:, :fixed_frames]

    pad_width = fixed_frames - n_frames
    return np.pad(
        spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=spec.min()
    )


def extract_log_mel_spectrogram(
    path: str,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fixed_frames: int = FIXED_FRAMES,
    loader=load_audio,
) -> np.ndarray:
    """Load one audio file and extract a fixed-size log-Mel spectrogram.

    `loader` defaults to load_audio (assumes the file is already 16 kHz
    mono, true for every ASVspoof dataset file -- used unchanged by
    training). Inference on arbitrary uploads passes
    audio_io.load_audio_resampled instead, so non-conforming files get
    resampled/mixed to mono before feature extraction.

    Returns an array of shape (n_mels, fixed_frames) in dB scale.
    """
    waveform, sample_rate = loader(path)
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    return _pad_or_truncate(mel_db, fixed_frames)


def inspect_spectrogram(path: str) -> np.ndarray:
    """Extract a log-Mel spectrogram for one file and print shape/duration info."""
    waveform, sample_rate = load_audio(path)
    duration_sec = waveform.shape[0] / sample_rate

    spec = extract_log_mel_spectrogram(path)
    n_mel_bins, n_time_frames = spec.shape

    print(f"sample_rate:        {sample_rate} Hz")
    print(f"input duration:     {duration_sec:.4f} s")
    print(f"spectrogram shape:  {spec.shape}")
    print(f"n_mel_bins:         {n_mel_bins}")
    print(f"n_time_frames:      {n_time_frames}")

    return spec


def save_spectrogram_preview(spec: np.ndarray, out_path: str) -> None:
    """Save a single spectrogram as an image for visual verification only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        spec,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Log-Mel spectrogram (sample)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    from src.data.asvspoof_cm_loader import load_train_protocol

    sample_path = load_train_protocol().iloc[0]["path"]
    spectrogram = inspect_spectrogram(sample_path)

    preview_path = _PROJECT_ROOT / "data" / "processed" / "spectrogram_sample.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    save_spectrogram_preview(spectrogram, str(preview_path))
    print(f"saved sample spectrogram preview to: {preview_path}")
