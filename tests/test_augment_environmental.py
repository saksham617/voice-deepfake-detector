import numpy as np
import pytest
import soundfile as sf

from training.augment_environmental import EnvironmentalAugment


def _tone(n=16000, freq=220, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(n)).astype("float32")


def _write_noise_dir(tmp_path, n_files=3, length=8000, seed=1):
    d = tmp_path / "musan_noise"
    d.mkdir()
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        sf.write(d / f"noise{i}.wav", (0.2 * rng.standard_normal(length)).astype("float32"), 16000)
    return d


def _write_rir_dir(tmp_path, lengths, seed=2):
    d = tmp_path / "rirs"
    d.mkdir()
    rng = np.random.default_rng(seed)
    for i, length in enumerate(lengths):
        rir = rng.standard_normal(length).astype("float32") * np.exp(-np.arange(length) / (length / 4 + 1))
        sf.write(d / f"rir{i}.wav", rir, 16000)
    return d


@pytest.mark.parametrize("mode", ["noise", "reverb", "both"])
def test_preserves_shape_dtype_finiteness(tmp_path, mode):
    wav = _tone()
    musan = _write_noise_dir(tmp_path)
    rir = _write_rir_dir(tmp_path, [4000, 12000])
    aug = EnvironmentalAugment(musan_dir=musan, rir_dir=rir, mode=mode, p=1.0, seed=3)
    out = aug(wav)
    assert out.shape == wav.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-4


def test_probability_gate_zero_is_noop(tmp_path):
    wav = _tone()
    musan = _write_noise_dir(tmp_path)
    rir = _write_rir_dir(tmp_path, [4000])
    aug = EnvironmentalAugment(musan_dir=musan, rir_dir=rir, mode="both", p=0.0, seed=4)
    for _ in range(10):
        assert np.array_equal(aug(wav), wav)


def test_actually_changes_signal(tmp_path):
    wav = _tone(seed=5)
    musan = _write_noise_dir(tmp_path)
    rir = _write_rir_dir(tmp_path, [4000])
    aug = EnvironmentalAugment(musan_dir=musan, rir_dir=rir, mode="both", p=1.0, seed=6)
    out = aug(wav)
    assert not np.allclose(out, wav)


def test_rir_longer_than_clip_does_not_crash(tmp_path):
    wav = _tone(n=4000, seed=7)  # 0.25s clip
    rir = _write_rir_dir(tmp_path, [32000])  # 2s RIR, 8x longer than the clip
    aug = EnvironmentalAugment(musan_dir=tmp_path / "missing_musan", rir_dir=rir,
                               mode="reverb", p=1.0, seed=8)
    out = aug(wav)
    assert out.shape == wav.shape
    assert np.isfinite(out).all()


def test_rir_shorter_than_clip_does_not_crash(tmp_path):
    wav = _tone(n=32000, seed=9)
    rir = _write_rir_dir(tmp_path, [800])  # short RIR, well under the clip length
    aug = EnvironmentalAugment(musan_dir=tmp_path / "missing_musan", rir_dir=rir,
                               mode="reverb", p=1.0, seed=10)
    out = aug(wav)
    assert out.shape == wav.shape
    assert np.isfinite(out).all()


def test_missing_corpora_is_noop_not_crash(tmp_path):
    wav = _tone(seed=11)
    aug = EnvironmentalAugment(musan_dir=tmp_path / "no_musan", rir_dir=tmp_path / "no_rir",
                               mode="both", p=1.0, seed=12)
    assert np.array_equal(aug(wav), wav)


def test_degrades_to_noise_only_when_rir_missing(tmp_path):
    wav = _tone(seed=13)
    musan = _write_noise_dir(tmp_path)
    aug = EnvironmentalAugment(musan_dir=musan, rir_dir=tmp_path / "no_rir",
                               mode="both", p=1.0, seed=14)
    out = aug(wav)
    assert not np.allclose(out, wav)


def test_degrades_to_reverb_only_when_musan_missing(tmp_path):
    wav = _tone(seed=15)
    rir = _write_rir_dir(tmp_path, [4000])
    aug = EnvironmentalAugment(musan_dir=tmp_path / "no_musan", rir_dir=rir,
                               mode="both", p=1.0, seed=16)
    out = aug(wav)
    assert not np.allclose(out, wav)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        EnvironmentalAugment(mode="bogus")
