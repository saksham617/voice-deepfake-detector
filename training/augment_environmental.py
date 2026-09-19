"""MUSAN-style background noise + RIR-style room reverb augmentation.

Complements ``training/augment.py``'s RawBoost, which simulates channel/codec artefacts but
has no model of real ambient noise or room acoustics. This module adds the two pieces
RawBoost doesn't cover:

  1. additive noise mixed in from real recorded clips (MUSAN "noise" subset) at a randomised
     SNR — cafe/street/HVAC/crowd-style ambient sound, not synthetic Gaussian noise.
  2. convolution with a real/simulated room impulse response (RIR corpus) — actual
     reverberation, not RawBoost's notch-filter channel colouration.

Meant to run alongside RawBoost (see ``training/train.py::_run_e2e``), not replace it.

Sources (fetch first with ``python scripts/download_datasets.py --only musan,rir_noises``):
  MUSAN noise subset : data/raw/musan/musan/noise/**/*.wav        (OpenSLR 17)
  RIR corpus         : data/raw/rir_noises/RIRS_NOISES/simulated_rirs/**/*.wav (OpenSLR 28)

If neither corpus has been downloaded yet, ``EnvironmentalAugment`` degrades to a no-op
(with a one-time warning) instead of failing training — the multi-GB downloads are opt-in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
MUSAN_NOISE_DIR = ROOT / "data" / "raw" / "musan" / "musan" / "noise"
RIR_DIR = ROOT / "data" / "raw" / "rir_noises" / "RIRS_NOISES" / "simulated_rirs"


def _list_wavs(d: Path) -> list[Path]:
    return sorted(d.rglob("*.wav")) if d.exists() else []


def _norm_wav(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) + 1e-9
    return x / peak if peak > 1.0 else x


def _fit_length(a: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random-crop or tile-pad ``a`` to exactly ``n`` samples (mirrors dataset.py's clip crop)."""
    m = a.shape[0]
    if m == n:
        return a
    if m > n:
        start = int(rng.integers(0, m - n + 1))
        return a[start : start + n]
    reps = int(np.ceil(n / max(m, 1)))
    return np.tile(a, reps)[:n]


def _load_mono_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from backend.audio import resample_to_16k

        wav = resample_to_16k(wav, sr)
    return wav.astype(np.float32)


class EnvironmentalAugment:
    """mode: 'noise' | 'reverb' | 'both' (default). Single probability ``p`` gates the whole
    call, mirroring ``RawBoost``'s shape so a sample can independently draw RawBoost and/or
    this augmentation — see ``training/train.py::_run_e2e``.

    Resilient to missing corpora: if a requested effect's directory has no wav files (not
    downloaded yet), that effect is silently skipped rather than raising; if nothing is
    available at all, the call is a no-op.
    """

    def __init__(
        self,
        musan_dir: str | Path | None = None,
        rir_dir: str | Path | None = None,
        mode: str = "both",
        p: float = 0.5,
        p_spoof: float | None = None,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        seed: int | None = None,
    ) -> None:
        if mode not in ("noise", "reverb", "both"):
            raise ValueError(f"mode must be noise|reverb|both, got {mode!r}")
        self.mode = mode
        self.p = p
        self.p_spoof = p_spoof
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.rng = np.random.default_rng(seed)
        self._noise_files = _list_wavs(Path(musan_dir) if musan_dir else MUSAN_NOISE_DIR)
        self._rir_files = _list_wavs(Path(rir_dir) if rir_dir else RIR_DIR)
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            print(
                "[EnvironmentalAugment] no MUSAN noise / RIR wav files found "
                f"(mode={self.mode}) — run "
                "`python scripts/download_datasets.py --only musan,rir_noises` first. "
                "Skipping environmental augmentation until then."
            )
            self._warned = True

    def _mix_noise(self, x: np.ndarray) -> np.ndarray:
        path = self._noise_files[int(self.rng.integers(len(self._noise_files)))]
        noise = _fit_length(_load_mono_16k(path), x.shape[0], self.rng)
        snr_db = float(self.rng.uniform(self.snr_min_db, self.snr_max_db))
        ps = float(np.mean(x**2)) + 1e-9
        pn = float(np.mean(noise**2)) + 1e-9
        scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
        return x + noise * scale

    def _apply_reverb(self, x: np.ndarray) -> np.ndarray:
        path = self._rir_files[int(self.rng.integers(len(self._rir_files)))]
        rir = _load_mono_16k(path)
        if rir.size == 0 or not np.any(rir):
            return x
        rir = rir / (np.max(np.abs(rir)) + 1e-9)
        n = x.shape[0]
        wet = fftconvolve(x, rir, mode="full").astype(np.float32)
        # align the RIR's direct-path peak to t=0 so reverb doesn't just look like a delay;
        # works regardless of whether the RIR is longer or shorter than the clip, since the
        # 'full' convolution is always >= n + direct samples long.
        direct = int(np.argmax(np.abs(rir)))
        wet = wet[direct : direct + n]
        if wet.shape[0] < n:  # defensive: degenerate/near-empty RIR
            wet = np.pad(wet, (0, n - wet.shape[0]))
        # convolution changes overall energy (reverb sums reflections) — rescale to the dry
        # clip's RMS so loudness stays comparable across augmented/unaugmented samples.
        rms_dry = np.sqrt(np.mean(x**2)) + 1e-9
        rms_wet = np.sqrt(np.mean(wet**2)) + 1e-9
        return wet * (rms_dry / rms_wet)

    def __call__(self, wav: np.ndarray, label: int | None = None) -> np.ndarray:
        p = self.p_spoof if (self.p_spoof is not None and label == 1) else self.p
        if self.rng.random() > p:
            return wav.astype(np.float32)
        x = wav.astype(np.float32)
        want_noise = self.mode in ("noise", "both") and bool(self._noise_files)
        want_reverb = self.mode in ("reverb", "both") and bool(self._rir_files)
        if not want_noise and not want_reverb:
            self._warn_once()
            return x
        # reverb first (models the room the speech was uttered in), then mix in ambient
        # noise (picked up independently by the mic) — order is a modelling choice, not a
        # hard physical requirement, but keeps the two effects from compounding oddly.
        if want_reverb:
            x = self._apply_reverb(x)
        if want_noise:
            x = self._mix_noise(x)
        return _norm_wav(x).astype(np.float32)
