"""End-to-end tests for the /check_message endpoint in backend/main.py,
driven through the real FastAPI app via TestClient -- same style as
tests/test_backend.py and tests/test_backend_reports.py.

Uses the real trained TF-IDF + Logistic Regression classifier (the
committed data/processed/models/message_classifier.joblib), not a mock --
consistent with how every other smoke test in this project verifies
against the actual inference pipeline. See
tests/test_message_detection_accuracy.py for the held-out accuracy/
precision/recall numbers behind this model.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.api.legacy import GENERIC_MESSAGE_ERROR_DETAIL
from backend.main import app

PHISHING_STYLE_MESSAGES = [
    "Your account will be suspended. Click here to verify your details now: "
    "http://bit.ly/verify-now or you will lose access.",
    "WINNER! You have been selected to receive a 1000 dollar reward! Call now to claim your prize.",
    "URGENT: Your Netflix payment failed. Update your billing info now to avoid "
    "service interruption: netflix-verify.com",
]

NORMAL_MESSAGES = [
    "Hey, are we still on for dinner tonight at 7?",
    "Can you pick up milk on your way home?",
    "Meeting moved to 3pm tomorrow, see you then.",
]


@pytest.fixture(scope="module")
def client():
    # `with` triggers the app's lifespan startup, which loads the real model.
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_message_model_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["message_model_ready"] is True


@pytest.mark.parametrize("text", PHISHING_STYLE_MESSAGES)
def test_check_message_flags_phishing_style_text_as_suspicious(client, text):
    response = client.post("/check_message", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "suspicious"
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["spam_probability"] <= 1.0


@pytest.mark.parametrize("text", NORMAL_MESSAGES)
def test_check_message_passes_normal_text_as_safe(client, text):
    response = client.post("/check_message", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "safe"
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["spam_probability"] <= 1.0


def test_check_message_rejects_empty_text_with_400_not_500(client):
    response = client.post("/check_message", json={"text": "   "})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_check_message_failure_returns_generic_500_detail(client):
    """Break the cached classifier (simulating a corrupt/incompatible model
    object) and confirm the client sees only the generic message -- not the
    underlying exception text -- same contract as /predict's equivalent
    test in test_backend.py."""
    from src.models import message_detection as md

    class _BrokenClassifier:
        def predict_proba(self, texts):
            raise RuntimeError("simulated classifier failure with internal details")

    original_classifier = md._cached_classifier
    md._cached_classifier = _BrokenClassifier()
    try:
        response = client.post("/check_message", json={"text": "irrelevant"})
    finally:
        md._cached_classifier = original_classifier

    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_MESSAGE_ERROR_DETAIL}
