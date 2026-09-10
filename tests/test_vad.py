"""SileroVAD unit tests. Uses the real Silero model -- it ships its own ~2 MB JIT weights
inside the package (no network access, no HF Hub), so this stays fast and offline like the
rest of the suite."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from backend.audio import SileroVAD

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> torch.Tensor:
    wav, sr = sf.read(FIXTURES / name, dtype="float32")
    assert sr == 16000
    return torch.from_numpy(wav)


def test_background_noise_is_not_speech():
    vad = SileroVAD()
    noise = _load("background_noise.wav")
    assert vad.speech_ratio(noise) < 0.05
    assert not vad.is_speech(noise)


def test_silence_is_not_speech():
    vad = SileroVAD()
    silence = torch.zeros(4 * 16000)
    assert vad.speech_ratio(silence) == 0.0
    assert not vad.is_speech(silence)


def test_pure_tone_is_not_speech():
    """A steady sine tone (used elsewhere in the suite as synthetic PCM) is not speech
    either -- it isn't a stand-in for real voice, just a convenient deterministic signal."""
    vad = SileroVAD()
    t = np.arange(4 * 16000) / 16000
    tone = torch.from_numpy((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    assert not vad.is_speech(tone)


def test_real_speech_is_detected():
    vad = SileroVAD()
    speech = _load("real_speech.wav")
    assert vad.speech_ratio(speech) > 0.5
    assert vad.is_speech(speech)


def test_min_speech_ratio_threshold_is_respected():
    vad = SileroVAD(min_speech_ratio=0.99)
    speech = _load("real_speech.wav")
    # a near-impossible bar: even real speech has some non-speech frames (pauses, breaths)
    assert not vad.is_speech(speech)


def test_short_window_below_one_frame_is_not_speech():
    vad = SileroVAD()
    assert vad.speech_ratio(torch.zeros(100)) == 0.0
