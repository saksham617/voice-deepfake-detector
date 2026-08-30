"""Fixed-size feature preparation for traditional ML models (SVM, Random Forest).

Traditional ML classifiers expect one fixed-length feature vector per example,
but extract_mfcc() returns a variable-size (n_mfcc, n_time_frames) matrix since
audio clips have different durations. This module reduces that matrix to a
fixed-size vector by summarizing each MFCC coefficient's distribution over
time, so it can be reused unchanged for any clip length across train/dev.
"""

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.mfcc import extract_mfcc


def extract_mfcc_statistics(path: str) -> np.ndarray:
    """Convert one audio file's MFCC matrix into a fixed-size feature vector.

    Computes the mean and standard deviation of each MFCC coefficient across
    time frames, then concatenates them: [mean_0..mean_n, std_0..std_n].
    For n_mfcc=13, this yields a 26-length vector regardless of clip duration.
    """
    mfcc = extract_mfcc(path)
    mean_per_coeff = np.mean(mfcc, axis=1)
    std_per_coeff = np.std(mfcc, axis=1)
    return np.concatenate([mean_per_coeff, std_per_coeff])


if __name__ == "__main__":
    from src.data.asvspoof_cm_loader import load_train_protocol
    from src.features.mfcc import extract_mfcc

    sample_path = load_train_protocol().iloc[0]["path"]

    mfcc = extract_mfcc(sample_path)
    feature_vector = extract_mfcc_statistics(sample_path)

    print(f"original mfcc shape:   {mfcc.shape}")
    print(f"feature vector shape:  {feature_vector.shape}")
    print(f"feature vector values:\n{feature_vector}")
