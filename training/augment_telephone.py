"""Telephone-channel simulation: bandlimit + G.711 A-law codec round-trip + line noise.

Targets a specific real-world gap: eval shows 73% accuracy on clean AI-cloned voice vs. 6.5%
once that same voice has been played back and re-recorded over an actual phone call. The
working hypothesis is that a telephone channel (narrow ~300-3400Hz bandwidth, lossy companded
codec, line noise) washes out the vocoder artefacts the model currently keys on, and training
never shows it that condition.

Complements ``training/augment.py``'s RawBoost (synthetic channel/codec-style distortion via
notch-filter convolutive noise) and ``training/augment_environmental.py``'s MUSAN/RIR (real
ambient noise + room reverb) — this module is the third, narrowly-targeted piece: an actual
PSTN/VoIP-style channel, not a general distortion or room model.

Three effects, applied in signal-chain order (mic -> bandlimit -> codec -> line noise):
  1. Causal Butterworth bandpass to the telephone voice band (~300-3400 Hz) — causal (a real
     line's filtering is causal too), not zero-phase.
  2. G.711 A-law compand -> quantize -> expand round-trip. This approximates the codec's
     characteristic non-uniform quantization noise (fine steps near zero, coarse at high
     amplitude); it is not a bit-exact ITU-T G.711 implementation (no bitstream/framing).
  3. Light additive white line noise at a randomised SNR — lighter than
     ``EnvironmentalAugment``'s ambient-noise mixing, modelling residual line hiss rather than
     a background environment.

Applied to BOTH bonafide and spoof clips, unconditionally — ``ManifestDataset`` applies the
composed augment chain regardless of label (see ``training/dataset.py``). This is required, not
incidental: if only spoof clips were degraded, the model could learn "degraded audio = fake" as
a shortcut instead of learning to detect vocoder artefacts under any channel condition. This
project already made that mistake once; the fix must not repeat it.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

A_LAW = 87.6  # ITU-T G.711 A-law compression parameter


def _bandpass_sos(low_hz: float, high_hz: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    return butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")


def _alaw_compress(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    thresh = 1.0 / A_LAW
    small = ax < thresh
    y = np.empty_like(x)
    y[small] = A_LAW * ax[small] / (1.0 + np.log(A_LAW))
    y[~small] = (1.0 + np.log(A_LAW * ax[~small])) / (1.0 + np.log(A_LAW))
    return np.sign(x) * y


def _alaw_expand(y: np.ndarray) -> np.ndarray:
    ay = np.abs(y)
    thresh = 1.0 / (1.0 + np.log(A_LAW))
    small = ay < thresh
    x = np.empty_like(y)
    x[small] = ay[small] * (1.0 + np.log(A_LAW)) / A_LAW
    x[~small] = np.exp(ay[~small] * (1.0 + np.log(A_LAW)) - 1.0) / A_LAW
    return np.sign(y) * x


def alaw_roundtrip(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """Compand -> uniformly quantize the compressed value to ``bits`` -> expand.

    Simulates G.711 A-law's quantization noise floor without implementing the ITU-T
    bitstream/framing itself.
    """
    peak = float(np.max(np.abs(x))) + 1e-9
    xn = x / peak if peak > 1.0 else x
    compressed = _alaw_compress(xn)
    half_levels = 2 ** bits / 2 - 1
    quantized = np.round(compressed * half_levels) / half_levels
    expanded = _alaw_expand(quantized)
    return (expanded * peak).astype(np.float32)


class TelephoneChannelAugment:
    """Bandlimit + A-law codec round-trip + line noise, applied together (a real phone call
    doesn't let you pick just one of these). Single probability ``p`` gates the whole call,
    mirroring ``RawBoost``/``EnvironmentalAugment`` so a sample independently draws this
    alongside (or instead of) them — see ``training/train.py::_run_e2e``.
    """

    def __init__(
        self,
        low_hz: float = 300.0,
        high_hz: float = 3400.0,
        fs: int = 16000,
        codec_bits: int = 8,
        p: float = 0.45,
        snr_min_db: float = 15.0,
        snr_max_db: float = 35.0,
        seed: int | None = None,
    ) -> None:
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.codec_bits = codec_bits
        self.p = p
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.rng = np.random.default_rng(seed)
        self._sos = _bandpass_sos(low_hz, high_hz, fs)

    def _add_line_noise(self, x: np.ndarray) -> np.ndarray:
        noise = self.rng.standard_normal(x.shape[0]).astype(np.float32)
        snr_db = float(self.rng.uniform(self.snr_min_db, self.snr_max_db))
        # power in float64: x**2 can overflow float32 (max ~3.4e38) well before it overflows
        # float64, which then propagates to inf/nan through the scale factor below.
        ps = float(np.mean(x.astype(np.float64) ** 2)) + 1e-9
        pn = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-9
        scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
        return x + noise * np.float32(scale)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        # degenerate input (empty clip, or non-finite from an upstream decode error) — skip
        # rather than let sosfilt/the codec math choke on it.
        if wav.size == 0 or not np.all(np.isfinite(wav)):
            return wav.astype(np.float32)
        if self.rng.random() > self.p:
            return wav.astype(np.float32)
        x = wav.astype(np.float32)
        x = sosfilt(self._sos, x).astype(np.float32)
        x = alaw_roundtrip(x, bits=self.codec_bits)
        x = self._add_line_noise(x)
        peak = float(np.max(np.abs(x))) + 1e-9
        return (x / peak if peak > 1.0 else x).astype(np.float32)
