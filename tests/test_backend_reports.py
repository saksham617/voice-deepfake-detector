"""End-to-end tests for the reporting endpoints (/report, /reports,
/reports/{id}) in backend/main.py, driven through the real FastAPI app via
TestClient -- same style as tests/test_backend.py.

Uses mock report data only (no real CNN/speaker-verification results
needed, since those models aren't the thing under test here) to exercise
the full create -> list -> get -> delete lifecycle end-to-end, plus input
validation. Runs against an isolated temp SQLite file, never the real
data/processed/reports.db.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.main import app
from src.models import reporting

MOCK_REPORTS = [
    {
        "type": "voice",
        "verdict": "spoof",
        "confidence_score": 0.92,
        "claimed_identity": "Boss",
        "user_notes": "Sounded robotic",
    },
    {
        "type": "message",
        "verdict": "suspicious",
        "confidence_score": 0.61,
        "user_notes": "Urgent wire transfer request",
    },
    {
        "type": "speaker_verification",
        "verdict": "safe",
        "confidence_score": 0.88,
        "claimed_identity": "Mom",
    },
    {
        "type": "voice",
        "verdict": "bonafide",
        "confidence_score": 0.77,
        "claimed_identity": "Bank official",
        "user_notes": "Verified via callback",
    },
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # `with` triggers the app's lifespan startup (loads the CNN and speaker
    # models too, same as test_backend.py) -- but every test in this file
    # only exercises the reports feature, which doesn't depend on either.
    original_db_path = reporting.DB_PATH
    original_db_ready = reporting._db_ready
    reporting.DB_PATH = tmp_path_factory.mktemp("reports_db") / "test_reports.db"
    reporting._db_ready = False
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        reporting.DB_PATH = original_db_path
        reporting._db_ready = original_db_ready


def test_health_reports_reports_db_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["reports_db_ready"] is True


# The tests below intentionally run as one sequential narrative against the
# shared module-scoped client/DB (create all mock reports, then list, then
# fetch one, then delete one) -- matching the create -> list -> get ->
# delete lifecycle this feature needs to support end-to-end.


def test_create_report_endpoint_accepts_each_mock_report(client):
    created_ids = []
    for payload in MOCK_REPORTS:
        response = client.post("/report", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["type"] == payload["type"]
        assert body["verdict"] == payload["verdict"]
        assert body["confidence_score"] == payload["confidence_score"]
        assert body["claimed_identity"] == payload.get("claimed_identity")
        created_ids.append(body["id"])

    assert len(set(created_ids)) == len(MOCK_REPORTS)


def test_list_reports_returns_all_sorted_newest_first(client):
    response = client.get("/reports")
    assert response.status_code == 200
    body = response.json()

    assert len(body) == len(MOCK_REPORTS)
    timestamps = [r["timestamp"] for r in body]
    assert timestamps == sorted(timestamps, reverse=True)
    # Newest-first means the last mock report posted comes back first.
    assert body[0]["verdict"] == MOCK_REPORTS[-1]["verdict"]
    assert body[-1]["verdict"] == MOCK_REPORTS[0]["verdict"]


def test_get_single_report_returns_correct_details(client):
    listed = client.get("/reports").json()
    target = listed[-1]  # oldest = the first mock report created

    response = client.get(f"/reports/{target['id']}")
    assert response.status_code == 200
    assert response.json() == target


def test_get_nonexistent_report_returns_404_not_crash(client):
    response = client.get("/reports/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "No report found with id 999999."}


def test_delete_report_removes_it_and_subsequent_get_is_404(client):
    listed = client.get("/reports").json()
    target_id = listed[0]["id"]

    delete_response = client.delete(f"/reports/{target_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True, "id": target_id}

    get_response = client.get(f"/reports/{target_id}")
    assert get_response.status_code == 404
    assert get_response.json() == {"detail": f"No report found with id {target_id}."}

    second_delete_response = client.delete(f"/reports/{target_id}")
    assert second_delete_response.status_code == 404

    remaining = client.get("/reports").json()
    assert len(remaining) == len(MOCK_REPORTS) - 1
    assert target_id not in [r["id"] for r in remaining]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "banana", "verdict": "spoof", "confidence_score": 0.5},
        {"type": "voice", "verdict": "banana", "confidence_score": 0.5},
        {"type": "voice", "verdict": "spoof", "confidence_score": 1.5},
        {"type": "voice", "verdict": "spoof", "confidence_score": -0.1},
    ],
)
def test_create_report_rejects_invalid_values_with_400_not_500(client, payload):
    response = client.post("/report", json=payload)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_list_reports_for_number_returns_only_matching_reports(client):
    number = "+1 202-555-0199"
    other_number = "+1 202-555-0100"

    matching_ids = []
    for verdict in ("spoof", "bonafide"):
        response = client.post(
            "/report",
            json={"type": "voice", "verdict": verdict, "confidence_score": 0.5, "phone_number": number},
        )
        assert response.status_code == 200
        matching_ids.append(response.json()["id"])

    client.post(
        "/report",
        json={"type": "voice", "verdict": "spoof", "confidence_score": 0.5, "phone_number": other_number},
    )
    client.post("/report", json={"type": "voice", "verdict": "spoof", "confidence_score": 0.5})

    response = client.get(f"/reports/number/{number}")
    assert response.status_code == 200
    body = response.json()

    assert {r["id"] for r in body} == set(matching_ids)
    assert all(r["phone_number"] == number for r in body)
    timestamps = [r["timestamp"] for r in body]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_reports_for_number_returns_empty_for_unknown_number(client):
    response = client.get("/reports/number/+1 000-000-0000")
    assert response.status_code == 200
    assert response.json() == []
