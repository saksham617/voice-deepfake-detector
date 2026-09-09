"""Tests for the FastAPI inference backend (backend/main.py).

Uses the real trained CNN checkpoint and real dataset audio files -- no
mocking of the model or preprocessing -- consistent with how every other
smoke test in this project verifies against the actual inference pipeline
rather than a stub.
"""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api.legacy import GENERIC_PREDICTION_ERROR_DETAIL, MAX_UPLOAD_SIZE_BYTES
from backend.main import app
from src.data.asvspoof_cm_loader import load_train_protocol


@pytest.fixture(scope="module")
def client():
    # `with` triggers the app's lifespan startup, which loads the real model.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def bonafide_path() -> str:
    train_df = load_train_protocol()
    return train_df.loc[train_df["label"] == 0, "path"].iloc[0]


@pytest.fixture(scope="module")
def spoof_path() -> str:
    train_df = load_train_protocol()
    return train_df.loc[train_df["label"] == 1, "path"].iloc[0]


def test_health_reports_model_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_predict_known_bonafide_file(client, bonafide_path):
    with open(bonafide_path, "rb") as f:
        response = client.post("/predict", files={"file": ("bonafide.flac", f, "audio/flac")})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "bonafide"
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["bonafide_probability"] <= 1.0
    assert 0.0 <= body["spoof_probability"] <= 1.0


def test_predict_known_spoof_file(client, spoof_path):
    with open(spoof_path, "rb") as f:
        response = client.post("/predict", files={"file": ("spoof.flac", f, "audio/flac")})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "spoof"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_unsupported_extension(client):
    fake_file = io.BytesIO(b"not real audio")
    response = client.post("/predict", files={"file": ("test.mp3", fake_file, "audio/mpeg")})

    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client):
    oversized = io.BytesIO(b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024))
    response = client.post("/predict", files={"file": ("big.wav", oversized, "audio/wav")})

    assert response.status_code == 413


def test_predict_returns_generic_error_without_leaking_exception_details(client):
    """A valid extension but unparseable content should fail inside
    predict_audio(), and the client must see only the generic message --
    not the underlying library's exception text (e.g. file paths, the
    exception class name, or soundfile's own error string)."""
    corrupt_file = io.BytesIO(b"this is not a real wav file, just garbage bytes")
    response = client.post("/predict", files={"file": ("corrupt.wav", corrupt_file, "audio/wav")})

    assert response.status_code == 500
    # Exact equality: proves both that the generic message is present AND
    # that nothing else (no extra leaked text/fields) is in the body.
    assert response.json() == {"detail": GENERIC_PREDICTION_ERROR_DETAIL}
