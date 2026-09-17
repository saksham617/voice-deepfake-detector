"""Twilio-facing HTTP routes, mounted at the app root under /twilio.

Chunk 1 (this file, phone verification + profile/history):
    POST /twilio/verify/start   {phone}         -> send OTP (real Twilio Verify or dev-mock)
    POST /twilio/verify/check   {phone, code}   -> verify OTP, mark profile phone_verified
    GET  /twilio/profile                        -> current verified-phone profile
    GET  /twilio/calls                          -> call history (most recent first)

Chunk 2 adds the auto-answer voice webhook (POST /twilio/voice) and the Media Stream
WebSocket. All Twilio REST calls happen server-side here; credentials never reach the browser.
"""

from __future__ import annotations

import logging
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.telephony import (
    VerificationError,
    check_verification,
    get_telephony_config,
    start_verification,
)
from backend.telephony import store

logger = logging.getLogger("backend.telephony")

router = APIRouter(prefix="/twilio", tags=["telephony"])


class VerifyStartRequest(BaseModel):
    phone: str = Field(..., description="Phone number in E.164 format, e.g. +919812345678")


class VerifyCheckRequest(BaseModel):
    phone: str = Field(..., description="Phone number in E.164 format")
    code: str = Field(..., min_length=3, max_length=10, description="OTP the user received")


@router.post("/verify/start")
def verify_start(payload: VerifyStartRequest) -> dict:
    try:
        result = start_verification(payload.phone)
    except VerificationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, **result}


@router.post("/verify/check")
def verify_check(payload: VerifyCheckRequest) -> dict:
    try:
        approved = check_verification(payload.phone, payload.code)
    except VerificationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not approved:
        raise HTTPException(status_code=400, detail="Incorrect or expired code.")
    profile = store.set_phone_verified(payload.phone)
    return {"ok": True, "profile": profile}


@router.get("/profile")
def get_profile() -> dict:
    cfg = get_telephony_config()
    profile = store.get_profile()
    # Surface the number callers should dial for the demo, and whether OTP is dev-mocked.
    return {
        "profile": profile,
        "protected_number": cfg.phone_number or None,
        "dev_mode": cfg.dev_mode or not cfg.real_verify_ready,
    }


@router.get("/calls")
def list_calls(limit: int = 100) -> dict:
    return {"calls": store.list_calls(limit=limit)}


@router.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Twilio 'A call comes in' webhook. Returns TwiML that AUTO-ANSWERS the call (no ring,
    no human pickup) and forks a live Media Stream of the audio to our WebSocket for analysis.

    <Start><Stream> is non-blocking (it copies the audio out-of-band while the call proceeds),
    so the trailing <Pause> keeps the caller connected — and the audio flowing — while the
    dashboard analyzes it in real time. Point the number's webhook at <PUBLIC_BASE_URL>/twilio/voice.
    """
    form = await request.form()
    from_number = str(form.get("From", "unknown"))
    to_number = str(form.get("To", get_telephony_config().phone_number or ""))
    stream_url = get_telephony_config().stream_wss_url()

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Say>Connected to Voice Guard. This call is being analyzed for A I generated voice.</Say>"
        "<Start>"
        f"<Stream url={quoteattr(stream_url)}>"
        f"<Parameter name=\"from\" value={quoteattr(from_number)}/>"
        f"<Parameter name=\"to\" value={quoteattr(to_number)}/>"
        "</Stream>"
        "</Start>"
        "<Pause length=\"3600\"/>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")
