"""Speaker verification using SpeechBrain's pretrained ECAPA-TDNN model
(speechbrain/spkrec-ecapa-voxceleb, trained on VoxCeleb1+2).

Two operations, both keyed by a contact name:
    enroll_speaker(name, audio_path) -- compute an embedding and store it
    verify_speaker(name, audio_path) -- compare a new clip's embedding
        against the named contact's stored embedding via cosine similarity

Does not train or fine-tune anything -- this wraps the pretrained model's
embedding extraction plus a small on-disk enrollment store. See
tests/test_speaker_verification_accuracy.py for the empirical basis of
DEFAULT_MATCH_THRESHOLD.
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from speechbrain.inference.speaker import SpeakerRecognition

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVEDIR = _PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"

# Where enrolled speaker embeddings are persisted, one .pt file per contact.
# *.pt is already gitignored project-wide (see .gitignore), so enrollments
# -- runtime user data, not project assets -- are never committed.
ENROLLMENT_DIR = _PROJECT_ROOT / "data" / "processed" / "speaker_embeddings"

# ECAPA-TDNN's expected input rate; speechbrain's own load_audio()
# resamples/downmixes to this automatically (see _load_waveform), matching
# how src/data/audio_io.py's load_audio_resampled handles the CNN pipeline
# elsewhere in this project.
SAMPLE_RATE = 16000

# Below this, embeddings are measurably less reliable -- see the
# accuracy-test report for the empirical basis. Not a hard cutoff: short
# clips are still processed, just flagged via the `short_clip` field so
# callers (e.g. the API layer) can surface a caveat instead of silently
# trusting a score computed from too little audio.
MIN_RELIABLE_DURATION_SEC = 2.0

# Cosine-similarity cutoff above which two embeddings are considered the
# same speaker. This is the error-minimizing threshold on a labeled
# LibriSpeech dev-clean test set (6 speakers, 6 same-speaker pairs, 15
# different-speaker pairs): same-speaker scores ranged 0.325-0.830, and
# different-speaker scores ranged -0.065-0.473 -- the two distributions
# mostly separate but do overlap by one pair (one different-speaker pair
# scored higher than the weakest same-speaker pair), so no threshold gets
# 0 errors on this data; 0.30 is the value that minimizes total
# misclassifications (1/21). See
# tests/test_speaker_verification_accuracy.py, which recomputes this sweep
# and asserts the constant still matches it, rather than trusting a stale
# number. Not speechbrain's own default (0.25 in
# SpeakerRecognition.verify_batch), which is tuned on a much larger dataset
# at a different operating point.
DEFAULT_MATCH_THRESHOLD = 0.30

_cached_model = None


class AudioDecodeError(Exception):
    """Raised when an audio file can't be read/decoded into a waveform."""


class SpeakerNotEnrolledError(Exception):
    """Raised by verify_speaker() when no enrollment exists for the given name."""


def _get_model() -> SpeakerRecognition:
    global _cached_model
    if _cached_model is None:
        _cached_model = SpeakerRecognition.from_hparams(
            source=MODEL_SOURCE, savedir=str(MODEL_SAVEDIR)
        )
    return _cached_model


def ensure_model_loaded() -> None:
    """Eagerly load (and cache) the model -- e.g. at application startup."""
    _get_model()


def _sanitize_name(name: str) -> str:
    """Turn a contact name into a safe filename component.

    The name comes from API callers, so it's untrusted: without this, a name
    like "../../etc/passwd" would let enroll_speaker/verify_speaker read or
    write outside ENROLLMENT_DIR. Keeps only alphanumerics/dash/underscore,
    collapses everything else (spaces, punctuation) into single underscores.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not slug:
        raise ValueError(f"'{name}' does not contain a usable name (only letters/digits/-/_).")
    return slug.lower()


def _embedding_path(name: str) -> Path:
    return ENROLLMENT_DIR / f"{_sanitize_name(name)}.pt"


def contact_id_for_name(name: str) -> str:
    """The stable, URL/form-safe id a contact is addressed by everywhere
    outside this module (API responses, verify_speaker lookups). Derived
    from the name rather than a separately-generated id/counter, since
    enrollment is already one file per sanitized name -- see
    _embedding_path. Re-sanitizing an id this function already produced is
    a no-op, so callers may pass either a raw name or a prior contact_id.
    """
    return _sanitize_name(name)


def _load_waveform(audio_path: str) -> torch.Tensor:
    model = _get_model()
    try:
        return model.load_audio(str(audio_path))
    except Exception as exc:
        raise AudioDecodeError(f"Could not read/decode audio file: {audio_path}") from exc


def compute_embedding(audio_path: str) -> tuple[torch.Tensor, float]:
    """Compute a speaker embedding for one audio file.

    Returns (embedding, duration_sec). embedding is a 1-D tensor (192-dim
    for this model). Raises AudioDecodeError if the file can't be read.
    """
    model = _get_model()
    waveform = _load_waveform(audio_path)
    duration_sec = waveform.shape[-1] / SAMPLE_RATE

    with torch.no_grad():
        embedding = model.encode_batch(waveform.unsqueeze(0), normalize=False)
    return embedding.reshape(-1), duration_sec


def enroll_speaker(name: str, audio_path: str) -> dict:
    """Compute a speaker embedding from audio_path and store it under `name`,
    overwriting any prior enrollment for that name.

    Returns a dict describing what was stored: name, duration_sec, short_clip
    (True if under MIN_RELIABLE_DURATION_SEC -- enrollment still succeeds,
    but a caller may want to ask for a longer sample).
    """
    embedding, duration_sec = compute_embedding(audio_path)
    enrolled_at = datetime.now(timezone.utc).isoformat()

    ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": name,
            "embedding": embedding,
            "duration_sec": duration_sec,
            "enrolled_at": enrolled_at,
        },
        _embedding_path(name),
    )

    return {
        "name": name,
        "contact_id": contact_id_for_name(name),
        "duration_sec": duration_sec,
        "short_clip": duration_sec < MIN_RELIABLE_DURATION_SEC,
        "enrolled_at": enrolled_at,
    }


def verify_speaker(
    name: str, audio_path: str, threshold: float = DEFAULT_MATCH_THRESHOLD
) -> dict:
    """Compare a new clip against the named contact's stored embedding.

    Raises SpeakerNotEnrolledError if `name` has no enrollment on file.
    Returns a dict: name, similarity (cosine similarity in [-1, 1]),
    is_match (similarity > threshold), threshold, short_clip (True if
    either the enrolled or the new clip is under MIN_RELIABLE_DURATION_SEC).
    """
    enrollment_path = _embedding_path(name)
    if not enrollment_path.exists():
        raise SpeakerNotEnrolledError(f"No enrolled speaker found for name '{name}'.")

    stored = torch.load(enrollment_path)
    new_embedding, new_duration_sec = compute_embedding(audio_path)

    similarity = torch.nn.functional.cosine_similarity(
        stored["embedding"], new_embedding, dim=0
    ).item()

    short_clip = (
        new_duration_sec < MIN_RELIABLE_DURATION_SEC
        or stored["duration_sec"] < MIN_RELIABLE_DURATION_SEC
    )

    is_match = similarity > threshold
    return {
        "name": stored["name"],
        "contact_id": contact_id_for_name(name),
        "similarity": similarity,
        "is_match": is_match,
        "match": is_match,
        "threshold": threshold,
        "short_clip": short_clip,
    }


def list_enrolled_speakers() -> list[dict]:
    """List every enrolled contact: contact_id, name, enrolled_at.

    Reads back the per-contact .pt files written by enroll_speaker rather
    than keeping a separate index, so there's exactly one source of truth
    for "who's enrolled" -- can't drift out of sync with the embeddings
    themselves. enrolled_at falls back to the file's mtime for enrollments
    written before that field existed.
    """
    if not ENROLLMENT_DIR.exists():
        return []

    contacts = []
    for path in sorted(ENROLLMENT_DIR.glob("*.pt")):
        try:
            stored = torch.load(path, map_location="cpu")
        except Exception:
            continue
        enrolled_at = stored.get("enrolled_at")
        if not enrolled_at:
            enrolled_at = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        contacts.append(
            {"contact_id": path.stem, "name": stored.get("name", path.stem), "enrolled_at": enrolled_at}
        )
    return contacts
