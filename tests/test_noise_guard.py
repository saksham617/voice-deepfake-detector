"""Synthetic, data-free unit tests for the background-noise warning guard.

These validate the *mechanics* of ``NoiseGuard`` with numpy-generated signals so they run
anywhere, with no datasets: the blind SNR estimate must fall monotonically as we mix in more
noise, clean audio must stay SILENT, and only genuinely severe noise must trigger a warning.

The exact dB threshold that separates "warn" from "don't" is NOT asserted here — that is what
``scripts/calibrate_noise_warning.py`` (real MUSAN + speech) and the Part-2 real-recording
validation are for. These tests only pin the behaviour the calibration must not break.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.audio.noise_guard import SEVERITY_ORDER, NoiseAssessment, NoiseGuard

SR = 16000
RNG = np.random.default_rng(0)


def _speech_like(seconds: float = 1.6, sr: int = SR) -> np.ndarray:
    """A crude voiced signal: tone bursts (with envelopes) separated by silent gaps.

    The gaps matter — the guard reads its noise floor from the quietest frames, so a signal
    with word-like pauses is what makes a blind SNR estimate meaningful (a gapless tone would
    have no floor to measure)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    burst = 0.2  # seconds of voice, then 0.2 s silence
    step = int(2 * burst * sr)
    blen = int(burst * sr)
    for start in range(0, n - blen, step):
        seg = np.arange(blen) / sr
        env = np.hanning(blen)
        # two formant-ish tones so it isn't a pure sinusoid
        voice = (np.sin(2 * np.pi * 180 * seg) + 0.5 * np.sin(2 * np.pi * 320 * seg)) * env
        sig[start : start + blen] = voice
    return sig / (np.max(np.abs(sig)) + 1e-9)


def _mix_at_snr(speech: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at a target SNR — same power-ratio maths as EnvironmentalAugment."""
    noise = rng.standard_normal(speech.shape[0])
    ps = float(np.mean(speech**2)) + 1e-12
    pn = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return (speech + noise * scale).astype(np.float64)


def test_clean_audio_is_silent():
    guard = NoiseGuard()
    clean = _mix_at_snr(_speech_like(), snr_db=40.0, rng=RNG)  # ~pristine
    a = guard.assess(clean)
    assert a.severity in ("clean", "mild")
    assert a.warn is False


def test_severe_noise_triggers_warning():
    guard = NoiseGuard()
    noisy = _mix_at_snr(_speech_like(), snr_db=-6.0, rng=RNG)  # speech drowning in noise
    a = guard.assess(noisy)
    assert a.severity == "severe"
    assert a.warn is True


def test_estimated_snr_is_monotonic_in_true_snr():
    """More added noise (lower true SNR) must not read as *cleaner*."""
    guard = NoiseGuard()
    speech = _speech_like()
    true_snrs = [40.0, 25.0, 15.0, 6.0, -6.0]
    est = [guard.estimate_snr_db(_mix_at_snr(speech, s, RNG)) for s in true_snrs]
    for earlier, later in zip(est, est[1:]):
        assert earlier >= later - 1.0  # allow tiny non-monotonic jitter from randomness


def test_severity_grades_span_the_range():
    """Sweeping noise should produce clean at the top and severe at the bottom."""
    guard = NoiseGuard()
    speech = _speech_like()
    grades = {guard.assess(_mix_at_snr(speech, s, RNG)).severity for s in (40, 25, 12, 3, -8)}
    assert "clean" in grades
    assert "severe" in grades


def test_warn_only_fires_at_or_below_configured_severity():
    lenient = NoiseGuard(warn_min_severity="moderate")
    speech = _speech_like()
    moderate_ish = _mix_at_snr(speech, 8.0, RNG)
    # A window graded 'moderate' should warn under the lenient policy but not the default one.
    a_default = NoiseGuard().assess(moderate_ish)
    a_lenient = lenient.assess(moderate_ish)
    if a_default.severity == "moderate":
        assert a_default.warn is False
        assert a_lenient.warn is True


def test_from_config_duck_types():
    class Cfg:
        clean_snr_db, mild_snr_db, moderate_snr_db = 20.0, 12.0, 6.0
        warn_min_severity = "severe"
        frame_ms, hop_ms = 25.0, 10.0
        noise_percentile, speech_percentile = 10.0, 95.0

    guard = NoiseGuard.from_config(Cfg(), sample_rate=SR)
    assert isinstance(guard.assess(_speech_like()), NoiseAssessment)


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        NoiseGuard(clean_snr_db=10.0, mild_snr_db=12.0)  # clean must exceed mild
    with pytest.raises(ValueError):
        NoiseGuard(warn_min_severity="catastrophic")


def test_severity_order_constant():
    assert SEVERITY_ORDER == ("clean", "mild", "moderate", "severe")
