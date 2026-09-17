"""Telephony subsystem: phone verification, Twilio voice/media-stream handling, and the
live-call profile + history store. Kept separate from the audio/ML packages so the real-call
plumbing can evolve independently of the model."""

from backend.telephony.config import TelephonyConfig, get_telephony_config
from backend.telephony.verify import VerificationError, check_verification, start_verification

__all__ = [
    "TelephonyConfig",
    "get_telephony_config",
    "start_verification",
    "check_verification",
    "VerificationError",
]
