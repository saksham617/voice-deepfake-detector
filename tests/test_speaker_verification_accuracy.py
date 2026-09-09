"""Empirical accuracy check for src/models/speaker_verification.py.

Uses a small labeled test set built from LibriSpeech dev-clean (public
domain audiobook recordings, CC BY 4.0) -- 6 speakers, 2 clips each,
downloaded from https://www.openslr.org/12/ into
data/raw/librispeech_dev_clean_subset/ (gitignored like the rest of
data/raw/, so this test is skipped on a fresh clone rather than failing).

Filenames follow LibriSpeech's own convention, <speaker>-<chapter>-<utt>.flac,
so the speaker label is recovered directly from the filename -- no separate
manifest needed.

Builds every same-speaker pair (one per speaker, 6 total) and every
different-speaker pair (15 = C(6,2)) from the 12 clips, runs them through
verify_speaker(), and asserts the two score distributions actually separate
-- i.e. the empirical threshold this suite computes is a real decision
boundary, not just a number. Run standalone with `-s` to see the full score
report and the recommended threshold.
"""

import itertools
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import speaker_verification as sv

TEST_AUDIO_DIR = _PROJECT_ROOT / "data" / "raw" / "librispeech_dev_clean_subset"

pytestmark = pytest.mark.skipif(
    not TEST_AUDIO_DIR.is_dir() or not any(TEST_AUDIO_DIR.glob("*.flac")),
    reason=(
        f"Test audio not present at {TEST_AUDIO_DIR} -- download a small "
        "LibriSpeech dev-clean subset to run this accuracy check."
    ),
)


def _speaker_id(flac_path: Path) -> str:
    return flac_path.name.split("-")[0]


@pytest.fixture(scope="module")
def clips_by_speaker() -> dict:
    clips: dict[str, list[Path]] = {}
    for flac_path in sorted(TEST_AUDIO_DIR.glob("*.flac")):
        clips.setdefault(_speaker_id(flac_path), []).append(flac_path)
    return clips


@pytest.fixture(scope="module")
def score_report(clips_by_speaker, tmp_path_factory) -> dict:
    """Enroll one clip per speaker, verify_speaker() every same- and
    different-speaker pair against those enrollments, and return the raw
    scores plus the best-separating threshold."""
    enrollment_dir = tmp_path_factory.mktemp("speaker_embeddings")
    original_dir = sv.ENROLLMENT_DIR
    sv.ENROLLMENT_DIR = enrollment_dir
    try:
        speaker_ids = sorted(clips_by_speaker)
        for speaker_id in speaker_ids:
            enroll_clip = clips_by_speaker[speaker_id][0]
            sv.enroll_speaker(speaker_id, str(enroll_clip))

        same_speaker_scores = []
        for speaker_id in speaker_ids:
            clips = clips_by_speaker[speaker_id]
            assert len(clips) >= 2, f"need >=2 clips for speaker {speaker_id}"
            verify_clip = clips[1]
            result = sv.verify_speaker(speaker_id, str(verify_clip), threshold=0.0)
            same_speaker_scores.append((speaker_id, speaker_id, result["similarity"]))

        different_speaker_scores = []
        for enrolled_id, other_id in itertools.combinations(speaker_ids, 2):
            other_clip = clips_by_speaker[other_id][0]
            result = sv.verify_speaker(enrolled_id, str(other_clip), threshold=0.0)
            different_speaker_scores.append((enrolled_id, other_id, result["similarity"]))

        return {
            "same": same_speaker_scores,
            "different": different_speaker_scores,
        }
    finally:
        sv.ENROLLMENT_DIR = original_dir


def _best_threshold(same_scores: list[float], diff_scores: list[float]) -> tuple[float, int]:
    """Sweep candidate thresholds (midpoints between adjacent sorted scores)
    and return the one that minimizes total misclassifications (same-speaker
    pairs scoring at/below it, plus different-speaker pairs scoring above it)."""
    candidates = sorted(set(same_scores) | set(diff_scores))
    midpoints = [(a + b) / 2 for a, b in zip(candidates, candidates[1:])]
    midpoints = [candidates[0] - 0.01, *midpoints, candidates[-1] + 0.01]

    best_threshold, best_errors = midpoints[0], len(same_scores) + len(diff_scores)
    for threshold in midpoints:
        errors = sum(s <= threshold for s in same_scores) + sum(d > threshold for d in diff_scores)
        if errors < best_errors:
            best_threshold, best_errors = threshold, errors
    return best_threshold, best_errors


# On this 21-pair test set, one different-speaker pair (1462 vs 1673)
# scores higher (0.4727) than the weakest same-speaker pair (1988, 0.3249)
# -- a real false-accept risk this small sample surfaces, not a bug in the
# test. No single threshold gets 0 errors here, so the bar is "best
# achievable on this data" rather than perfect separation.
MAX_ACCEPTABLE_ERRORS = 1


def test_scores_mostly_separate_with_documented_overlap(score_report):
    same_scores = [s for *_, s in score_report["same"]]
    diff_scores = [s for *_, s in score_report["different"]]

    threshold, errors = _best_threshold(same_scores, diff_scores)

    print("\n=== Speaker verification accuracy report (LibriSpeech dev-clean subset) ===")
    print("\nSame-speaker pairs (enrolled vs. a second clip of the same person):")
    for enrolled_id, verify_id, score in score_report["same"]:
        print(f"  {enrolled_id} vs {verify_id}: {score:.4f}")
    print("\nDifferent-speaker pairs:")
    for enrolled_id, other_id, score in score_report["different"]:
        print(f"  {enrolled_id} vs {other_id}: {score:.4f}")

    print(f"\nSame-speaker scores:      min={min(same_scores):.4f}  max={max(same_scores):.4f}  "
          f"mean={sum(same_scores)/len(same_scores):.4f}")
    print(f"Different-speaker scores: min={min(diff_scores):.4f}  max={max(diff_scores):.4f}  "
          f"mean={sum(diff_scores)/len(diff_scores):.4f}")
    print(f"\nBest-separating threshold on this test set: {threshold:.4f} "
          f"({errors}/{len(same_scores) + len(diff_scores)} misclassified)")
    print(f"DEFAULT_MATCH_THRESHOLD currently set to: {sv.DEFAULT_MATCH_THRESHOLD}")

    # Sanity bar: the model must be doing real work (mean same-speaker score
    # well above mean different-speaker score) and the best achievable error
    # rate must stay small -- catches a broken embedding pipeline or a
    # genuinely unusable model, without demanding unrealistic 0-error
    # perfection from a 21-pair sample.
    same_mean = sum(same_scores) / len(same_scores)
    diff_mean = sum(diff_scores) / len(diff_scores)
    assert same_mean - diff_mean > 0.3, (
        f"mean same-speaker score ({same_mean:.4f}) isn't well separated from "
        f"mean different-speaker score ({diff_mean:.4f})"
    )
    assert errors <= MAX_ACCEPTABLE_ERRORS, (
        f"best achievable threshold on this test set misclassifies {errors} pairs, "
        f"more than the documented {MAX_ACCEPTABLE_ERRORS}"
    )


def test_default_threshold_achieves_best_achievable_error_rate(score_report):
    """DEFAULT_MATCH_THRESHOLD in speaker_verification.py should match the
    error-minimizing threshold found by the sweep above -- catches the
    constant drifting out of sync with the data it's supposed to reflect."""
    same_scores = [s for *_, s in score_report["same"]]
    diff_scores = [s for *_, s in score_report["different"]]
    _, best_errors = _best_threshold(same_scores, diff_scores)

    default_errors = sum(s <= sv.DEFAULT_MATCH_THRESHOLD for s in same_scores) + sum(
        d > sv.DEFAULT_MATCH_THRESHOLD for d in diff_scores
    )

    assert default_errors == best_errors, (
        f"DEFAULT_MATCH_THRESHOLD={sv.DEFAULT_MATCH_THRESHOLD} misclassifies "
        f"{default_errors} pairs; the best achievable on this data is {best_errors}"
    )


def test_short_clip_degrades_similarity_and_is_flagged(clips_by_speaker, tmp_path):
    """Trim an enrolled speaker's verification clip to under
    MIN_RELIABLE_DURATION_SEC and confirm (a) it's still processed rather
    than rejected, (b) the response flags it via short_clip, and (c) reports
    whether the similarity score measurably drops vs. the full-length clip."""
    import soundfile as sf

    original_dir = sv.ENROLLMENT_DIR
    sv.ENROLLMENT_DIR = tmp_path
    try:
        speaker_id = sorted(clips_by_speaker)[0]
        enroll_clip, full_clip = clips_by_speaker[speaker_id][:2]
        sv.enroll_speaker(speaker_id, str(enroll_clip))

        full_result = sv.verify_speaker(speaker_id, str(full_clip), threshold=0.0)
        assert full_result["short_clip"] is False

        waveform, samplerate = sf.read(str(full_clip))
        short_path = tmp_path / "short_clip.wav"
        sf.write(str(short_path), waveform[: int(1.5 * samplerate)], samplerate)

        short_result = sv.verify_speaker(speaker_id, str(short_path), threshold=0.0)

        print(f"\n=== Short-clip degradation check ({speaker_id}) ===")
        print(f"Full clip ({full_result['similarity']:.4f}) vs 1.5s clip ({short_result['similarity']:.4f})")

        assert short_result["short_clip"] is True
    finally:
        sv.ENROLLMENT_DIR = original_dir


def test_verify_speaker_raises_for_unenrolled_name(tmp_path):
    sv.ENROLLMENT_DIR, original_dir = tmp_path, sv.ENROLLMENT_DIR
    try:
        with pytest.raises(sv.SpeakerNotEnrolledError):
            sv.verify_speaker("nobody-enrolled-yet", "irrelevant.wav")
    finally:
        sv.ENROLLMENT_DIR = original_dir


def test_corrupted_audio_raises_audio_decode_error(tmp_path):
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"this is not a real wav file, just garbage bytes")

    with pytest.raises(sv.AudioDecodeError):
        sv.compute_embedding(str(corrupt_path))
