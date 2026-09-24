"""End-to-end tests for the speaker enrollment/verification endpoints
(/enroll_speaker, /verify_speaker, /contacts) in backend/main.py, driven
through the real FastAPI app via TestClient -- same style as
tests/test_backend.py and tests/test_backend_reports.py.

This exists to lock in the frontend<->backend contract described in
frontend/src/types/speaker.ts: enroll_speaker must return a contact_id,
verify_speaker must accept one (not a name) and return a `match` field.
Before this fix, the frontend and backend disagreed on both, so every real
(non-mocked) call to either endpoint would fail -- see backend/api/legacy.py.

Uses real LibriSpeech clips (same fixture set as
test_speaker_verification_accuracy.py) so the model actually runs; skipped
if that data isn't present, matching that test's own skip condition.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.main import app
from src.models import speaker_verification as sv

TEST_AUDIO_DIR = _PROJECT_ROOT / "data" / "raw" / "librispeech_dev_clean_subset"

pytestmark = pytest.mark.skipif(
    not TEST_AUDIO_DIR.is_dir() or not any(TEST_AUDIO_DIR.glob("*.flac")),
    reason=f"Test audio not present at {TEST_AUDIO_DIR}.",
)


@pytest.fixture(scope="module")
def clips() -> dict[str, list[Path]]:
    by_speaker: dict[str, list[Path]] = {}
    for flac_path in sorted(TEST_AUDIO_DIR.glob("*.flac")):
        by_speaker.setdefault(flac_path.name.split("-")[0], []).append(flac_path)
    # Two speakers, two clips each, is all these tests need.
    speakers = list(by_speaker)[:2]
    return {sid: by_speaker[sid][:2] for sid in speakers}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    original_dir = sv.ENROLLMENT_DIR
    sv.ENROLLMENT_DIR = tmp_path_factory.mktemp("speaker_embeddings")
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        sv.ENROLLMENT_DIR = original_dir


def test_enroll_returns_contact_id(client, clips):
    (speaker_id, speaker_clips), = list(clips.items())[:1]
    with open(speaker_clips[0], "rb") as f:
        response = client.post(
            "/enroll_speaker",
            data={"name": "Aditi Rao"},
            files={"file": ("clip.flac", f, "audio/flac")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Aditi Rao"
    assert body["contact_id"] == "aditi_rao"
    assert isinstance(body["enrolled_at"], str) and body["enrolled_at"]


def test_contacts_lists_enrolled_speaker(client):
    response = client.get("/contacts")
    assert response.status_code == 200
    body = response.json()
    assert any(c["contact_id"] == "aditi_rao" and c["name"] == "Aditi Rao" for c in body)


def test_verify_speaker_accepts_contact_id_and_matches_self(client, clips):
    (speaker_id, speaker_clips), = list(clips.items())[:1]
    with open(speaker_clips[1], "rb") as f:
        response = client.post(
            "/verify_speaker",
            data={"contact_id": "aditi_rao"},
            files={"file": ("clip2.flac", f, "audio/flac")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["match"] == body["is_match"]
    assert body["contact_id"] == "aditi_rao"
    assert -1.0 <= body["similarity"] <= 1.0


def test_verify_speaker_unknown_contact_id_returns_404(client, clips):
    (speaker_id, speaker_clips), = list(clips.items())[:1]
    with open(speaker_clips[0], "rb") as f:
        response = client.post(
            "/verify_speaker",
            data={"contact_id": "nobody-enrolled"},
            files={"file": ("clip.flac", f, "audio/flac")},
        )

    assert response.status_code == 404
