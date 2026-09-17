"""Phone-number OTP verification via Twilio Verify, with a dev-mock fallback.

Two modes, chosen by TelephonyConfig.dev_mode (see config.py):

  * dev-mock (default): no network, no SMS. ``start_verification`` is a no-op that "sends"
    the fixed dev code; ``check_verification`` accepts exactly that code. This keeps the
    demo working regardless of India's DLT / sender-ID SMS restrictions, which can block or
    delay real OTP delivery to +91 numbers on trial/unregistered accounts.

  * real: calls Twilio Verify's REST API server-side (credentials never reach the browser).
    The ``twilio`` package is imported lazily so the app runs without it in dev-mock mode.

Both modes raise VerificationError on failure so the API layer can return a clean 4xx.
"""

from __future__ import annotations

import logging

from backend.telephony.config import TelephonyConfig, get_telephony_config

logger = logging.getLogger("backend.telephony")


class VerificationError(Exception):
    """Raised when starting or checking a verification fails."""


def start_verification(phone: str, cfg: TelephonyConfig | None = None) -> dict:
    """Send an OTP to ``phone`` (E.164, e.g. +9198XXXXXXXX). Returns a small status dict."""
    cfg = cfg or get_telephony_config()
    phone = phone.strip()
    if not phone.startswith("+") or len(phone) < 8:
        raise VerificationError("Phone number must be in E.164 format, e.g. +919812345678.")

    if cfg.dev_mode or not cfg.real_verify_ready:
        if not cfg.dev_mode:
            logger.warning("Twilio Verify not configured; falling back to dev-mock OTP.")
        logger.info("[dev-mock] OTP for %s is %s", phone, cfg.dev_code)
        return {"status": "pending", "channel": "dev-mock", "dev_mode": True}

    try:
        from twilio.rest import Client

        client = Client(cfg.account_sid, cfg.auth_token)
        v = client.verify.v2.services(cfg.verify_service_sid).verifications.create(
            to=phone, channel="sms"
        )
        return {"status": v.status, "channel": "sms", "dev_mode": False}
    except Exception as e:  # noqa: BLE001 — surface any Twilio/SDK error as a clean 4xx
        logger.exception("Twilio Verify start failed for %s", phone)
        raise VerificationError(f"Could not send verification code: {e}") from e


def check_verification(phone: str, code: str, cfg: TelephonyConfig | None = None) -> bool:
    """Return True iff ``code`` is the valid OTP for ``phone``."""
    cfg = cfg or get_telephony_config()
    phone, code = phone.strip(), code.strip()

    if cfg.dev_mode or not cfg.real_verify_ready:
        return code == cfg.dev_code

    try:
        from twilio.rest import Client

        client = Client(cfg.account_sid, cfg.auth_token)
        check = client.verify.v2.services(cfg.verify_service_sid).verification_checks.create(
            to=phone, code=code
        )
        return check.status == "approved"
    except Exception as e:  # noqa: BLE001
        logger.exception("Twilio Verify check failed for %s", phone)
        raise VerificationError(f"Could not verify code: {e}") from e
