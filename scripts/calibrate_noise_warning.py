"""Part 1 — calibrate the background-noise warning threshold by synthetic mixing.

Mixes a handful of clean speech clips with real MUSAN noise at a sweep of SNRs (clean / mild /
moderate / severe), runs ``NoiseGuard`` over each mix, and prints where its blind SNR estimate
and severity grades actually land. Use the table to pick the ``noise_guard`` dB thresholds in
config/config.yaml so the guard stays SILENT on clean/mild and only warns on severe noise.

Usage:
    python scripts/calibrate_noise_warning.py                 # uses repo default paths
    python scripts/calibrate_noise_warning.py --n-speech 8 --n-noise 6
    python scripts/calibrate_noise_warning.py --speech-dir data/raw/indictts \\
        --noise-dir data/raw/musan/musan/noise --snr 30 18 9 0

Data (fetch first if missing):
    python scripts/download_datasets.py --only indictts,musan

The SNR mixing here mirrors training/augment_environmental.py::_mix_noise so the calibration
matches how noisy audio is actually produced during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.audio.chunker import resample_to_16k  # noqa: E402
from backend.audio.noise_guard import SEVERITY_ORDER, NoiseGuard  # noqa: E402
from backend.core.config import get_config  # noqa: E402

DEFAULT_SPEECH_DIR = ROOT / "data" / "raw" / "indictts"
DEFAULT_NOISE_DIR = ROOT / "data" / "raw" / "musan" / "musan" / "noise"
# Nominal SNRs (dB) that stand in for the four severity buckets the plan asks for.
# "clean" is a dry copy (no noise added); the rest add MUSAN noise at these ratios.
DEFAULT_SNR_LEVELS = [20.0, 10.0, 3.0]  # mild, moderate, severe


def _load_mono_16k(path: Path) -> np.ndarray | None:
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed — `pip install soundfile` (or use the backend env).")
        raise
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:  # unreadable/short file — skip it, don't abort the sweep
        print(f"  ! skip {path.name}: {e}")
        return None
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_to_16k(wav, sr)
    return wav.astype(np.float32)


def _fit_length(a: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = a.shape[0]
    if m == n:
        return a
    if m > n:
        start = int(rng.integers(0, m - n + 1))
        return a[start : start + n]
    return np.tile(a, int(np.ceil(n / max(m, 1))))[:n]


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """speech + scaled noise at a target SNR — identical maths to EnvironmentalAugment."""
    ps = float(np.mean(speech**2)) + 1e-9
    pn = float(np.mean(noise**2)) + 1e-9
    scale = np.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return speech + noise * scale


def _wavs(d: Path, limit: int, rng: np.random.Generator) -> list[Path]:
    files = sorted(d.rglob("*.wav")) if d.exists() else []
    if len(files) > limit:
        idx = rng.choice(len(files), size=limit, replace=False)
        files = [files[i] for i in sorted(idx)]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speech-dir", type=Path, default=DEFAULT_SPEECH_DIR)
    ap.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    ap.add_argument("--n-speech", type=int, default=6)
    ap.add_argument("--n-noise", type=int, default=5)
    ap.add_argument("--snr", type=float, nargs="+", default=DEFAULT_SNR_LEVELS,
                    help="noisy SNR levels in dB (a dry 'clean' level is always included)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    speech_files = _wavs(args.speech_dir, args.n_speech, rng)
    noise_files = _wavs(args.noise_dir, args.n_noise, rng)

    if not speech_files or not noise_files:
        print("No audio found for calibration:")
        print(f"  speech: {len(speech_files)} wavs under {args.speech_dir}")
        print(f"  noise : {len(noise_files)} wavs under {args.noise_dir}")
        print("Fetch first:  python scripts/download_datasets.py --only indictts,musan")
        return 0  # not an error — data is an opt-in download

    speech = [w for w in (_load_mono_16k(p) for p in speech_files) if w is not None]
    noises = [w for w in (_load_mono_16k(p) for p in noise_files) if w is not None]
    if not speech or not noises:
        print("Could not load any audio (all files unreadable).")
        return 0

    guard = NoiseGuard.from_config(get_config().noise_guard)
    # nominal label -> list of estimated SNRs and warn flags across all speech×noise pairs
    levels: list[tuple[str, float | None]] = [("clean", None)] + [
        (f"snr{int(s)}", s) for s in args.snr
    ]

    print(f"\nCalibration: {len(speech)} speech × {len(noises)} noise clips per level")
    print(f"Guard thresholds (dB): clean>={guard.clean_snr_db} mild>={guard.mild_snr_db} "
          f"moderate>={guard.moderate_snr_db}  warn@{guard.warn_min_severity}\n")
    header = f"{'level':<8}{'est_SNR mean':>13}{'  [min..max]':>16}  {'severity spread':<28}{'warn%':>6}"
    print(header)
    print("-" * len(header))

    per_level_est: dict[str, list[float]] = {}
    for label, snr in levels:
        ests: list[float] = []
        warns = 0
        sev_counts = {s: 0 for s in SEVERITY_ORDER}
        for sp in speech:
            for nz in noises if snr is not None else [None]:
                clip = sp if snr is None else _mix_at_snr(sp, _fit_length(nz, sp.shape[0], rng), snr)
                a = guard.assess(clip)
                ests.append(a.snr_db)
                sev_counts[a.severity] += 1
                warns += int(a.warn)
        per_level_est[label] = ests
        n = len(ests)
        spread = " ".join(f"{s[:4]}:{sev_counts[s]}" for s in SEVERITY_ORDER if sev_counts[s])
        print(f"{label:<8}{np.mean(ests):>13.1f}{f'[{min(ests):.0f}..{max(ests):.0f}]':>16}  "
              f"{spread:<28}{100*warns/n:>5.0f}%")

    # Suggest a warn boundary: between the cleanest 'severe' target and the noisiest 'clean/mild'.
    noisy = per_level_est.get(f"snr{int(args.snr[-1])}")
    reference = per_level_est["clean"]
    if noisy and reference:
        print(f"\nSuggestion: the severe level (snr{int(args.snr[-1])}) estimates around "
              f"{np.median(noisy):.0f} dB; clean estimates around {np.median(reference):.0f} dB. "
              f"Set noise_guard.moderate_snr_db (the warn boundary) between them so severe warns "
              f"and clean/mild stay silent, then re-run to confirm the warn% column.")
    print("\nNext: validate on real recordings — scripts/validate_noise_warning_realworld.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
