"""Background-noise quality guard for the live-call pipeline.  [noise warning]

``SileroVAD`` (see ``vad.py``) answers *is there speech in this window?* and skips windows
that have none. This guard answers a different question about the windows that DO contain
speech: *is that speech buried in so much background noise that a verdict can't be trusted?*
When it is, the caller can surface a user-facing warning ("high background noise — move
somewhere quieter") instead of, or alongside, a low-confidence spoof/bonafide result.

Why a blind (no-reference) estimate
------------------------------------
On a live call there is no clean copy of the speech to compare against, so true SNR is
unknowable. We estimate it from the window's own frame-energy distribution: chop the window
into short frames, and read the noise floor off the quietest frames (the gaps between words,
which in clean audio are near-silent and in noisy audio sit at the ambient level) and the
speech level off the loudest frames. Their ratio is a robust proxy for SNR that needs no
reference and no model — just numpy — so it is cheap enough to run on every window and has
no heavy dependency (unlike the VAD, this module is import-light and unit-testable offline).

Thresholds are PROVISIONAL
--------------------------
The dB cut-points below (``clean``/``mild``/``moderate``/``severe``) are sensible defaults,
not calibrated truth. They are meant to be tuned by the synthetic-mixing sweep in
``scripts/calibrate_noise_warning.py`` (Part 1) and then validated on real recordings
(Part 2) — after which the winning values go into ``config/config.yaml`` under ``noise_guard``.
By design the guard stays SILENT on clean/mild audio and only warns on genuinely severe noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Severity ordered from best to worst; index doubles as an ordinal for comparisons.
SEVERITY_ORDER: tuple[str, ...] = ("clean", "mild", "moderate", "severe")


@dataclass(frozen=True)
class NoiseAssessment:
    """Result of assessing one analysis window for background-noise severity."""

    snr_db: float           # estimated (blind) speech-to-noise ratio, in dB
    severity: str           # one of SEVERITY_ORDER
    warn: bool              # True => surface a "too noisy" warning to the user
    reason: str             # short human-readable explanation

    def __str__(self) -> str:
        flag = "WARN" if self.warn else "ok"
        return f"[{flag}] {self.severity:<8} snr≈{self.snr_db:5.1f} dB — {self.reason}"


class NoiseGuard:
    """Estimate background-noise severity of a 16 kHz mono window and decide whether to warn.

    Parameters mirror the ``config/config.yaml::noise_guard`` section so a ``NoiseGuardConfig``
    can be splatted straight in. All dB thresholds are upper-exclusive band edges:
      snr >= ``clean_snr_db``            -> clean
      ``mild_snr_db``  <= snr < clean    -> mild
      ``moderate_snr_db`` <= snr < mild  -> moderate
      snr < ``moderate_snr_db``          -> severe
    A window warns when its severity is at or worse than ``warn_min_severity``.
    """

    def __init__(
        self,
        clean_snr_db: float = 20.0,
        mild_snr_db: float = 12.0,
        moderate_snr_db: float = 6.0,
        warn_min_severity: str = "severe",
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        noise_percentile: float = 10.0,
        speech_percentile: float = 95.0,
        sample_rate: int = 16000,
    ) -> None:
        if not (clean_snr_db > mild_snr_db > moderate_snr_db):
            raise ValueError("thresholds must satisfy clean > mild > moderate (dB)")
        if warn_min_severity not in SEVERITY_ORDER:
            raise ValueError(f"warn_min_severity must be one of {SEVERITY_ORDER}")
        if not (0.0 <= noise_percentile < speech_percentile <= 100.0):
            raise ValueError("require 0 <= noise_percentile < speech_percentile <= 100")
        self.clean_snr_db = clean_snr_db
        self.mild_snr_db = mild_snr_db
        self.moderate_snr_db = moderate_snr_db
        self.warn_min_severity = warn_min_severity
        self.sample_rate = sample_rate
        self.frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
        self.hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
        self.noise_percentile = noise_percentile
        self.speech_percentile = speech_percentile

    @classmethod
    def from_config(cls, cfg: Any, sample_rate: int = 16000) -> "NoiseGuard":
        """Build from a ``NoiseGuardConfig`` (duck-typed, so no import cycle with core.config)."""
        return cls(
            clean_snr_db=cfg.clean_snr_db,
            mild_snr_db=cfg.mild_snr_db,
            moderate_snr_db=cfg.moderate_snr_db,
            warn_min_severity=cfg.warn_min_severity,
            frame_ms=cfg.frame_ms,
            hop_ms=cfg.hop_ms,
            noise_percentile=cfg.noise_percentile,
            speech_percentile=cfg.speech_percentile,
            sample_rate=sample_rate,
        )

    # -- core estimate -------------------------------------------------------

    def _frame_powers(self, wav: np.ndarray) -> np.ndarray:
        """Per-frame mean power (energy) for the window."""
        x = np.asarray(wav, dtype=np.float64).reshape(-1)
        if x.shape[0] < self.frame_len:
            return np.array([float(np.mean(x**2))]) if x.size else np.array([0.0])
        n_frames = 1 + (x.shape[0] - self.frame_len) // self.hop_len
        powers = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * self.hop_len
            frame = x[s : s + self.frame_len]
            powers[i] = float(np.mean(frame**2))
        return powers

    def estimate_snr_db(self, wav: np.ndarray) -> float:
        """Blind SNR estimate in dB from the frame-energy distribution.

        noise floor = low percentile of frame powers (word gaps / ambient),
        speech level = high percentile (voiced peaks, = speech + noise). We subtract the
        floor from the peak so the numerator approximates speech-only power, matching the
        additive-noise model used by ``training/augment_environmental.py``'s SNR mixing.
        """
        powers = self._frame_powers(wav)
        eps = 1e-12
        noise = float(np.percentile(powers, self.noise_percentile))
        peak = float(np.percentile(powers, self.speech_percentile))
        speech = max(peak - noise, 0.0)
        if speech <= eps:
            # No energy above the floor: effectively no speech / flat signal. Report a very
            # low SNR so a flat noisy hiss is treated as worst-case; callers that care about
            # "is there speech at all" should consult the VAD, not this number.
            return -30.0
        snr = 10.0 * math.log10((speech + eps) / (noise + eps))
        # Clamp to a sane display range; clean digital silence would otherwise read +inf.
        return float(max(-30.0, min(60.0, snr)))

    # -- decision ------------------------------------------------------------

    def severity(self, snr_db: float) -> str:
        if snr_db >= self.clean_snr_db:
            return "clean"
        if snr_db >= self.mild_snr_db:
            return "mild"
        if snr_db >= self.moderate_snr_db:
            return "moderate"
        return "severe"

    def _warns(self, severity: str) -> bool:
        return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(self.warn_min_severity)

    def assess(self, wav: np.ndarray) -> NoiseAssessment:
        """Full assessment for one window: estimate SNR, grade it, decide whether to warn."""
        snr_db = self.estimate_snr_db(wav)
        sev = self.severity(snr_db)
        warn = self._warns(sev)
        if warn:
            reason = "background noise is high enough to make a verdict unreliable"
        elif sev == "moderate":
            reason = "some background noise, still within tolerance"
        else:
            reason = "background noise is low"
        return NoiseAssessment(snr_db=round(snr_db, 2), severity=sev, warn=warn, reason=reason)
