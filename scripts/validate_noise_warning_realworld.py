"""Part 2 — validate the noise warning on real recordings.

Runs ``NoiseGuard`` (with the thresholds calibrated in Part 1, read from config) over a folder
of genuine recordings — e.g. Arnav's real-world test set ``1_real_voice/`` — and reports each
file's estimated SNR, severity and warn flag, sorted noisiest-first. There are no ground-truth
labels here, so this is an eyeball check: confirm clean recordings stay SILENT and audibly
noisy ones correctly WARN. If they don't, re-run Part 1 and adjust the thresholds.

Usage:
    python scripts/validate_noise_warning_realworld.py --dir path/to/1_real_voice
    python scripts/validate_noise_warning_realworld.py --dir data/real_world/1_real_voice

Note: the real-world set is still being actively collected, so treat any run as a snapshot —
re-run once more files land. This script cannot fetch the folder for you; get it from whoever
manages the real-world dataset first.
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

AUDIO_EXTS = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")


def _load_mono_16k(path: Path) -> np.ndarray | None:
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed — `pip install soundfile` (or use the backend env).")
        raise
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        print(f"  ! skip {path.name}: {e}")
        return None
    wav = np.asarray(wav)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_to_16k(wav, sr)
    return wav.astype(np.float32)


def _find_audio(d: Path) -> list[Path]:
    files: list[Path] = []
    for pat in AUDIO_EXTS:
        files.extend(d.rglob(pat))
    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ROOT / "data" / "real_world" / "1_real_voice",
                    help="folder of real recordings to assess (searched recursively)")
    args = ap.parse_args()

    files = _find_audio(args.dir)
    if not files:
        print(f"No audio files found under {args.dir}")
        print("Get the real-world test folder (e.g. 1_real_voice/) from whoever manages it, "
              "then point --dir at it.")
        return 0

    guard = NoiseGuard.from_config(get_config().noise_guard)
    print(f"\nAssessing {len(files)} file(s) in {args.dir}")
    print(f"Guard: clean>={guard.clean_snr_db} mild>={guard.mild_snr_db} "
          f"moderate>={guard.moderate_snr_db} dB  warn@{guard.warn_min_severity}\n")

    rows: list[tuple[float, str, bool, str]] = []
    for p in files:
        wav = _load_mono_16k(p)
        if wav is None or wav.size == 0:
            continue
        a = guard.assess(wav)
        rows.append((a.snr_db, a.severity, a.warn, p.name))

    rows.sort(key=lambda r: r[0])  # noisiest (lowest SNR) first
    print(f"{'est_SNR':>8}  {'severity':<9}{'warn':<6}file")
    print("-" * 60)
    for snr_db, sev, warn, name in rows:
        print(f"{snr_db:>8.1f}  {sev:<9}{'WARN' if warn else '':<6}{name}")

    n = len(rows)
    warned = sum(1 for r in rows if r[2])
    counts = {s: sum(1 for r in rows if r[1] == s) for s in SEVERITY_ORDER}
    print("\nSummary:")
    print("  by severity: " + "  ".join(f"{s}={counts[s]}" for s in SEVERITY_ORDER))
    print(f"  warned: {warned}/{n} ({100*warned/max(n,1):.0f}%)")
    print("\nEyeball check: do the WARN rows match the ones that actually sound noisy? "
          "If a clean file warns or a noisy one doesn't, adjust noise_guard thresholds (Part 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
