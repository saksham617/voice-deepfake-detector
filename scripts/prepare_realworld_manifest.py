"""Turn raw real-world call/phone recordings into training manifest rows.

Context: the core AASIST+wav2vec2-XLS-R checkpoint was trained on ASVspoof2019 LA +
IndicTTS bonafide audio only (studio-clean). It scores 0% accuracy on real phone/laptop
recordings (misclassified as "fake") because it never learned what real telephone-quality
acoustic signatures (compression, ~8kHz effective bandwidth, real room noise/reverb) look
like as genuine speech -- despite 100% accuracy on AI-voice-replayed fakes. Pooled
real-world EER was 36%. A first fine-tune attempt with only 47 real-world files made this
WORSE (39% EER), i.e. too little signal, risk of overfitting. The fix is a proper amount of
real-world "bonafide" audio mixed into the main training manifest -- this script builds it.

What it does, per source folder you point it at:
  1. finds audio files recursively (wav/flac/mp3/m4a/ogg/aac)
  2. splits stereo/dual-channel files into separate per-channel mono tracks (call-center
     recordings are often one speaker per channel -- mixing them down would blend the two)
  3. resamples every track to 16 kHz
  4. slices each track into --window-seconds windows (default 4.0s, matching
     training/config_train.yaml's crop_seconds) with --hop-seconds stride
  5. drops near-silent / no-speech windows (Silero VAD if available -- it's already a repo
     dependency and runs fully offline once its weights are cached -- else an RMS-energy gate)
  6. writes the surviving windows as 16 kHz PCM wav files
  7. emits a bonafide manifest (utt_id, path, label, language, source, dataset -- same schema
     scripts/prepare_manifests.py uses) with each surviving row repeated --oversample times,
     because training.train's WeightedRandomSampler balances bonafide vs spoof but NOT within
     the bonafide class -- without oversampling, a few hundred real-world rows would be
     drowned out by tens of thousands of ASVspoof/IndicTTS bonafide rows.

Run once per source folder (each likely needs its own --dataset-name / --language), e.g. the
Google Drive `new_realworld_data` folders mentioned in project notes:

    python scripts/prepare_realworld_manifest.py \
        --input data/raw/realworld/axonlabs_callcenter \
        --dataset-name realworld_axonlabs --language hi \
        --merge-into data/manifests/train.tsv

    python scripts/prepare_realworld_manifest.py \
        --input data/raw/realworld/hindi_dualchannel_callcenter \
        --dataset-name realworld_hindi_callcenter --language hi \
        --merge-into data/manifests/train.tsv

    python scripts/prepare_realworld_manifest.py \
        --input data/raw/realworld/indian_languages_youtube \
        --dataset-name realworld_indic_youtube --language und --oversample 5 \
        --merge-into data/manifests/train.tsv   # weaker licensing -- lower weight, don't feature in the pitch

Requires ffmpeg on PATH for mp3/m4a decoding (librosa fallback): apt-get install -y ffmpeg

Do a --dry-run first to see file/window counts before anything is written.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
COLUMNS = ["utt_id", "path", "label", "language", "source", "dataset"]


@dataclass
class Row:
    utt_id: str
    path: str
    label: str
    language: str
    source: str
    dataset: str


def find_audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Returns (channels, sr): channels has shape (n_channels, n_samples), float32."""
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n_samples, n_ch)
        return wav.T, sr
    except Exception:
        import librosa

        wav, sr = librosa.load(str(path), sr=None, mono=False)
        wav = np.asarray(wav, dtype="float32")
        if wav.ndim == 1:
            wav = wav[None, :]
        return wav, sr


def resample(track: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return track
    import librosa

    return librosa.resample(track, orig_sr=sr, target_sr=target_sr)


_SILERO = {"model": None, "get_ts": None, "failed": False}


def _silero():
    if _SILERO["failed"] or _SILERO["model"] is not None:
        return _SILERO["model"], _SILERO["get_ts"]
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        _SILERO["model"] = load_silero_vad()
        _SILERO["get_ts"] = get_speech_timestamps
        print("  [vad] using Silero VAD")
    except Exception as e:  # pragma: no cover - depends on optional weights/network
        print(f"  [vad] Silero VAD unavailable ({e}); falling back to RMS-energy gate")
        _SILERO["failed"] = True
    return _SILERO["model"], _SILERO["get_ts"]


def speech_ratio_silero(window: np.ndarray, sr: int) -> float | None:
    model, get_ts = _silero()
    if model is None:
        return None
    import torch

    ts = get_ts(torch.from_numpy(window), model, sampling_rate=sr, return_seconds=True)
    speech = sum(seg["end"] - seg["start"] for seg in ts)
    return speech / (len(window) / sr)


def speech_ratio_energy(window: np.ndarray, min_rms_db: float) -> float:
    rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2) + 1e-12))
    db = 20 * np.log10(rms + 1e-12)
    return 1.0 if db >= min_rms_db else 0.0


def process_file(path: Path, out_dir: Path, sample_rate: int, window_s: float, hop_s: float,
                  min_speech_ratio: float, min_rms_db: float, vad_mode: str,
                  stats: dict) -> list[tuple[str, str]]:
    try:
        chans, sr = load_audio(path)
    except Exception as e:
        print(f"  [skip] {path.name}: failed to load ({e})")
        stats["load_failed"] += 1
        return []

    n_ch = chans.shape[0]
    if n_ch > 2:  # unexpected layout (not the mono/stereo call-recording case) -- downmix
        chans = chans.mean(axis=0, keepdims=True)
        n_ch = 1

    win_len = int(round(window_s * sample_rate))
    hop_len = int(round(hop_s * sample_rate))
    stem = path.stem.replace(" ", "_")
    kept: list[tuple[str, str]] = []

    for ch_idx in range(n_ch):
        track = resample(chans[ch_idx], sr, sample_rate)
        n = track.shape[0]
        pos, seg_idx = 0, 0
        while pos + win_len <= n:
            window = track[pos : pos + win_len]
            pos += hop_len

            ratio = None
            if vad_mode in ("auto", "silero"):
                ratio = speech_ratio_silero(window, sample_rate)
            if ratio is None:
                if vad_mode == "silero":
                    seg_idx += 1
                    continue  # forced silero but it's unavailable -- can't score this window
                ratio = speech_ratio_energy(window, min_rms_db)

            stats["windows_total"] += 1
            if ratio < min_speech_ratio:
                stats["windows_silent"] += 1
                seg_idx += 1
                continue

            tag = f"{stem}_ch{ch_idx}_seg{seg_idx:04d}"
            out_path = out_dir / f"{tag}.wav"
            sf.write(str(out_path), window, sample_rate, subtype="PCM_16")
            kept.append((tag, str(out_path.resolve())))
            stats["windows_kept"] += 1
            seg_idx += 1

    return kept


def build_rows(kept: list[tuple[str, str]], dataset_name: str, language: str,
               oversample: int) -> list[Row]:
    rows = []
    for tag, out_path in kept:
        for k in range(oversample):
            suffix = "" if oversample == 1 else f"_dup{k}"
            rows.append(Row(f"rw_{dataset_name}_{tag}{suffix}", out_path, "bonafide",
                             language, "human", dataset_name))
    return rows


def write_manifest(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def append_to_manifest(rows: list[Row], target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    before = 0
    if target.exists() and target.stat().st_size > 0:
        with target.open(encoding="utf-8") as fh:
            before = max(sum(1 for _ in fh) - 1, 0)
    is_new = before == 0
    with target.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    return before


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, type=Path,
                     help="one or more folders of raw audio for this dataset/source")
    ap.add_argument("--dataset-name", required=True, help="tag for the manifest 'dataset' column")
    ap.add_argument("--language", default="und", help="tag for the manifest 'language' column")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="default: data/raw/realworld/<dataset-name>")
    ap.add_argument("--manifest-out", type=Path, default=None,
                     help="default: data/manifests/realworld_<dataset-name>.tsv")
    ap.add_argument("--merge-into", type=Path, default=None,
                     help="also append the oversampled rows onto this manifest, e.g. "
                          "data/manifests/train.tsv")
    ap.add_argument("--window-seconds", type=float, default=4.0)
    ap.add_argument("--hop-seconds", type=float, default=None, help="default: = window-seconds (no overlap)")
    ap.add_argument("--min-speech-ratio", type=float, default=0.3,
                     help="minimum fraction of a window that must be speech to keep it")
    ap.add_argument("--min-rms-db", type=float, default=-40.0, help="RMS-energy fallback threshold")
    ap.add_argument("--vad", choices=["auto", "silero", "energy"], default="auto")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--oversample", type=int, default=15,
                     help="repeat each surviving window this many times in the manifest")
    ap.add_argument("--limit", type=int, default=None, help="cap source files processed (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    out_dir = args.out_dir or (ROOT / "data" / "raw" / "realworld" / args.dataset_name)
    manifest_out = args.manifest_out or (ROOT / "data" / "manifests" / f"realworld_{args.dataset_name}.tsv")
    hop_s = args.hop_seconds if args.hop_seconds is not None else args.window_seconds
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for inp in args.input:
        if not inp.exists():
            print(f"warning: input path does not exist: {inp}")
            continue
        files += find_audio_files(inp)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("no audio files found under the given --input paths")
        return

    print(f"found {len(files)} source file(s) under {[str(i) for i in args.input]}")

    stats = {"load_failed": 0, "windows_total": 0, "windows_silent": 0, "windows_kept": 0}
    all_kept: list[tuple[str, str]] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        if args.dry_run:
            continue
        all_kept += process_file(path, out_dir, args.sample_rate, args.window_seconds, hop_s,
                                  args.min_speech_ratio, args.min_rms_db, args.vad, stats)

    print()
    if args.dry_run:
        print("(dry run -- no audio or manifest written)")
        return

    print(f"windows: {stats['windows_total']} scored, {stats['windows_silent']} dropped (silent), "
          f"{stats['windows_kept']} kept  [{stats['load_failed']} file(s) failed to load]")

    rows = build_rows(all_kept, args.dataset_name, args.language, args.oversample)
    write_manifest(rows, manifest_out)
    try:
        rel = manifest_out.relative_to(ROOT)
    except ValueError:
        rel = manifest_out
    print(f"wrote {len(rows)} row(s) ({len(all_kept)} unique window(s) x {args.oversample}x "
          f"oversample) -> {rel}")

    if args.merge_into:
        before = append_to_manifest(rows, args.merge_into)
        try:
            rel2 = args.merge_into.relative_to(ROOT)
        except ValueError:
            rel2 = args.merge_into
        print(f"merged into {rel2}: {before} -> {before + len(rows)} row(s)")


if __name__ == "__main__":
    main()
