"""Generate the VAD test fixtures under tests/fixtures/.

    python scripts/make_vad_fixtures.py

Writes:
  * background_noise.wav — synthetic room-tone (broadband noise + faint 50/60 Hz mains hum),
    no speech at all. Reproduces the bug this feature fixes: fed straight to AASIST/XLS-R it
    drifts into a sustained HIGH-risk alert (see VOICEGUARD_HANDOFF.md).
  * real_speech.wav — a short real utterance, NOT synthetic like tests/fixtures/sample_*.wav
    (see scripts/make_sample_audio.py's own docstring on why those don't work for VAD tests:
    they're additive-harmonics placeholders, not real speech, and Silero VAD correctly scores
    them as mostly non-speech). Trimmed from LibriSpeech dev-clean
    (1272-128104-0000.flac, CC BY 4.0, https://www.openslr.org/12) so it's small enough to
    commit and free of any AASIST-training-set overlap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
LIBRISPEECH_SRC = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "librispeech_dev_clean_subset" / "1272-128104-0000.flac"
)


def _write(name: str, sig: np.ndarray, sr: int = SR) -> None:
    path = OUT / name
    sf.write(path, sig.astype(np.float32), sr)
    print(f"wrote {path}  ({sig.size / sr:.1f}s)")


def make_background_noise(duration_s: float = 10.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * duration_s)
    t = np.arange(n) / SR
    white = rng.normal(0, 1, n).astype(np.float32)
    hum = 0.15 * np.sin(2 * np.pi * 60 * t) + 0.05 * np.sin(2 * np.pi * 120 * t)
    return (0.02 * white + 0.03 * hum.astype(np.float32)).astype(np.float32)


def make_real_speech() -> np.ndarray:
    if not LIBRISPEECH_SRC.exists():
        raise FileNotFoundError(
            f"{LIBRISPEECH_SRC} not found -- fetch the LibriSpeech dev-clean subset first "
            "(see CLAUDE.md section 5)."
        )
    wav, sr = sf.read(LIBRISPEECH_SRC, dtype="float32")
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    return wav


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write("background_noise.wav", make_background_noise())
    if LIBRISPEECH_SRC.exists():
        _write("real_speech.wav", make_real_speech())
    else:
        print(f"skipping real_speech.wav: {LIBRISPEECH_SRC} not found")


if __name__ == "__main__":
    main()
