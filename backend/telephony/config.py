"""Twilio / telephony configuration, read from environment (never hardcoded).

Loads a repo-root ``.env`` (gitignored) if present so local dev doesn't need the vars exported
manually — no python-dotenv dependency, just a tiny parser. Real Twilio calls only happen when
``dev_mode`` is False AND the credentials are present; otherwise the OTP flow runs in a
dev-mock mode (a fixed code, no real SMS) so the demo never depends on flaky +91 SMS delivery.

Required for the REAL flow (put these in .env — see .env.example):
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SERVICE_SID
    TWILIO_PHONE_NUMBER      the provisioned VoiceGuard number callers dial
    PUBLIC_BASE_URL          public https URL of this backend (e.g. an ngrok URL), used to
                             build the wss:// Media Stream URL in the voice webhook's TwiML
Toggle:
    TELEPHONY_DEV_MODE=true|false   (default true — dev-mock OTP, safe for demos)
    TELEPHONY_DEV_CODE=123456       (the code accepted in dev-mock mode)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO_ROOT / ".env"


def _load_dotenv(path: Path = _ENV_PATH) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file (does not overwrite vars already
    set in the real environment). Silently does nothing if the file is absent."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TelephonyConfig:
    account_sid: str = ""
    auth_token: str = ""
    verify_service_sid: str = ""
    phone_number: str = ""
    public_base_url: str = ""
    dev_mode: bool = True
    dev_code: str = "123456"

    @classmethod
    def from_env(cls) -> "TelephonyConfig":
        _load_dotenv()
        return cls(
            account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            verify_service_sid=os.environ.get("TWILIO_VERIFY_SERVICE_SID", ""),
            phone_number=os.environ.get("TWILIO_PHONE_NUMBER", ""),
            public_base_url=os.environ.get("PUBLIC_BASE_URL", "").rstrip("/"),
            dev_mode=_as_bool(os.environ.get("TELEPHONY_DEV_MODE"), default=True),
            dev_code=os.environ.get("TELEPHONY_DEV_CODE", "123456"),
        )

    @property
    def real_verify_ready(self) -> bool:
        """True only if we have everything needed to call Twilio Verify for real."""
        return bool(self.account_sid and self.auth_token and self.verify_service_sid)

    def stream_wss_url(self) -> str:
        """wss:// URL Twilio Media Streams should connect to (derived from PUBLIC_BASE_URL)."""
        base = self.public_base_url or "http://localhost:8000"
        wss = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{wss}/twilio/stream"


_cached: TelephonyConfig | None = None


def get_telephony_config(reload: bool = False) -> TelephonyConfig:
    global _cached
    if _cached is None or reload:
        _cached = TelephonyConfig.from_env()
    return _cached
