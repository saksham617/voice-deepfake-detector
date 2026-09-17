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


# ---- regression: found via a real-training silent-failure investigation (dev EER came back
# `inf`, no batch loss ever printed). A large scan of real ASVspoof clips at the actual e2e
# 4s crop size never produced NaN/Inf, including deliberately-quiet clips and all-zero/tiny
# synthetic input — the existing epsilon guards in alaw_roundtrip/_add_line_noise already
# cover near-silent input correctly. Two other edge cases did break, fixed below.

def test_empty_input_does_not_crash():
    # dataset.py's tile-pad crop can hand an augmenter a zero-length array if a manifest row
    # ever resolves to a corrupt/zero-duration clip; sosfilt previously raised ValueError
    # ("cannot reshape array of size 0 into shape (0)") on this.
    wav = np.zeros(0, dtype=np.float32)
    aug = TelephoneChannelAugment(p=1.0, seed=20)
    out = aug(wav)
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_extreme_amplitude_does_not_overflow_to_nan():
    # mean(x**2) in _add_line_noise previously overflowed float32 (max ~3.4e38) for
    # amplitudes beyond ~1e19, producing inf -> the final peak-normalize's inf/inf then
    # turned the whole clip to NaN. Not reachable from real [-1,1] decoded audio, but a real
    # numerical-safety gap in the SNR-scaling math.
    rng = np.random.default_rng(21)
    for scale in (1e10, 1e20, 1e30):
        wav = (rng.standard_normal(16000).astype(np.float32) * np.float32(scale))
        out = TelephoneChannelAugment(p=1.0, seed=22)(wav)
        assert np.isfinite(out).all(), f"NaN/Inf at amplitude scale {scale:g}"


def test_non_finite_input_passed_through_not_amplified():
    # if a non-finite value ever reaches here from an upstream decode error, don't let the
    # codec/filter math turn it into a full-clip NaN — pass it through unchanged (an upstream
    # data problem, not this augmenter's to fix).
    wav = np.full(4000, np.inf, dtype=np.float32)
    out = TelephoneChannelAugment(p=1.0, seed=23)(wav)
    assert np.array_equal(out, wav)


def test_bandpass_filter_is_stable():
    # explicit stability check for the Butterworth bandpass sos, since filter instability on
    # near-zero input is a known related failure mode in IIR filters generally (ruled out
    # here — the filter's stability doesn't depend on input amplitude, only its own poles).
    from scipy.signal import sos2tf, tf2zpk

    aug = TelephoneChannelAugment(seed=24)
    b, a = sos2tf(aug._sos)
    _, poles, _ = tf2zpk(b, a)
    assert np.all(np.abs(poles) < 1.0), f"unstable pole(s): {poles[np.abs(poles) >= 1.0]}"
