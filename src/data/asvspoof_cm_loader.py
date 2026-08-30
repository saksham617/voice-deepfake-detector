"""Metadata loader for the ASVspoof 2019 LA countermeasure (CM) protocols.

Builds pandas DataFrames of (speaker_id, filename, system_id, label, path)
for the train/dev splits. Does not read or load any audio.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LA_ROOT = PROJECT_ROOT / "data" / "raw" / "asvspoof2019" / "LA"
CM_PROTOCOL_DIR = LA_ROOT / "ASVspoof2019_LA_cm_protocols"

TRAIN_PROTOCOL_PATH = CM_PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL_PATH = CM_PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"

TRAIN_AUDIO_DIR = LA_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = LA_ROOT / "ASVspoof2019_LA_dev" / "flac"

PROTOCOL_COLUMNS = ["speaker_id", "filename", "unused", "system_id", "key"]
LABEL_MAP = {"bonafide": 0, "spoof": 1}


def _load_cm_protocol(protocol_path: Path, audio_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(protocol_path, sep=r"\s+", names=PROTOCOL_COLUMNS, header=None)

    invalid_keys = set(df["key"].unique()) - set(LABEL_MAP)
    if invalid_keys:
        raise ValueError(
            f"Unexpected label value(s) in {protocol_path.name}: {sorted(invalid_keys)}"
        )

    df["label"] = df["key"].map(LABEL_MAP)
    df["path"] = df["filename"].apply(lambda name: str(audio_dir / f"{name}.flac"))

    exists = df["path"].apply(lambda p: Path(p).exists())
    missing = df.loc[~exists]
    if not missing.empty:
        raise FileNotFoundError(
            f"{len(missing)} audio file(s) referenced in {protocol_path.name} are "
            f"missing, e.g. {missing.iloc[0]['path']}"
        )

    return df[["speaker_id", "filename", "system_id", "label", "path"]].reset_index(
        drop=True
    )


def load_train_protocol() -> pd.DataFrame:
    """Load the ASVspoof 2019 LA train CM protocol as a metadata DataFrame."""
    return _load_cm_protocol(TRAIN_PROTOCOL_PATH, TRAIN_AUDIO_DIR)


def load_dev_protocol() -> pd.DataFrame:
    """Load the ASVspoof 2019 LA dev CM protocol as a metadata DataFrame."""
    return _load_cm_protocol(DEV_PROTOCOL_PATH, DEV_AUDIO_DIR)


def validate_split(
    df: pd.DataFrame,
    expected_total: int,
    expected_bonafide: int,
    expected_spoof: int,
    split_name: str,
) -> None:
    """Assert label validity, path existence, and expected counts for a split."""
    assert set(df["label"].unique()) <= {0, 1}, f"{split_name}: labels must be 0 or 1 only"
    assert df["path"].apply(lambda p: Path(p).exists()).all(), (
        f"{split_name}: some referenced audio paths do not exist"
    )
    assert len(df) == expected_total, (
        f"{split_name}: expected {expected_total} entries, got {len(df)}"
    )

    bonafide_count = int((df["label"] == 0).sum())
    spoof_count = int((df["label"] == 1).sum())
    assert bonafide_count == expected_bonafide, (
        f"{split_name}: expected {expected_bonafide} bonafide, got {bonafide_count}"
    )
    assert spoof_count == expected_spoof, (
        f"{split_name}: expected {expected_spoof} spoof, got {spoof_count}"
    )

    print(f"[{split_name}] total={len(df)} bonafide={bonafide_count} spoof={spoof_count}  OK")


def run_validation() -> None:
    """Load train/dev protocols and validate them against known ASVspoof 2019 LA counts."""
    train_df = load_train_protocol()
    dev_df = load_dev_protocol()

    validate_split(train_df, 25380, 2580, 22800, "train")
    validate_split(dev_df, 24844, 2548, 22296, "dev")

    print("\nSample rows (train):")
    print(train_df.head(3).to_string(index=False))
    print("\nSample rows (dev):")
    print(dev_df.head(3).to_string(index=False))


if __name__ == "__main__":
    run_validation()
