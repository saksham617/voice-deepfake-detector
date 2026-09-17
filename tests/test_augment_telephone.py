import numpy as np
import pytest

from training.augment_telephone import TelephoneChannelAugment, alaw_roundtrip


def _tone(n=16000, freq=220, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(n)).astype("float32")


def test_preserves_shape_dtype_finiteness():
    wav = _tone()
    aug = TelephoneChannelAugment(p=1.0, seed=1)
    out = aug(wav)
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-4


def test_probability_gate_zero_is_noop():
    wav = np.ones(4000, dtype="float32") * 0.2
    aug = TelephoneChannelAugment(p=0.0, seed=2)
    for _ in range(10):
        assert np.array_equal(aug(wav), wav)


def test_actually_changes_signal():
    wav = _tone(seed=3)
    aug = TelephoneChannelAugment(p=1.0, seed=4)
    out = aug(wav)
    assert not np.allclose(out, wav)


def test_attenuates_energy_outside_telephone_band():
    n = 16000
    t = np.arange(n) / 16000
    low_tone = (0.3 * np.sin(2 * np.pi * 100 * t)).astype("float32")   # below 300Hz cutoff
    mid_tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype("float32")  # inside passband
    aug = TelephoneChannelAugment(p=1.0, snr_min_db=60, snr_max_db=60, seed=5)
    low_out = aug(low_tone)
    mid_out = aug(mid_tone)
    # bandlimiting should suppress the out-of-band tone much more than the in-band one
    assert np.sqrt(np.mean(low_out**2)) < 0.3 * np.sqrt(np.mean(mid_out**2))


def test_alaw_roundtrip_preserves_shape_and_is_bounded():
    rng = np.random.default_rng(6)
    x = (rng.standard_normal(8000) * 0.5).astype("float32")
    out = alaw_roundtrip(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= np.max(np.abs(x)) + 1e-3


def test_alaw_roundtrip_introduces_quantization_error():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal(8000) * 0.5).astype("float32")
    out = alaw_roundtrip(x, bits=8)
    assert not np.allclose(out, x)
    # but should stay close-ish (codec compression, not a totally different signal)
    assert np.sqrt(np.mean((out - x) ** 2)) < 0.2


def test_alaw_roundtrip_handles_silence():
    x = np.zeros(4000, dtype="float32")
    out = alaw_roundtrip(x)
    assert np.allclose(out, 0.0, atol=1e-6)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_reproducible_with_seed(seed):
    wav = _tone(seed=seed)
    out_a = TelephoneChannelAugment(p=1.0, seed=42)(wav.copy())
    out_b = TelephoneChannelAugment(p=1.0, seed=42)(wav.copy())
    assert np.array_equal(out_a, out_b)
