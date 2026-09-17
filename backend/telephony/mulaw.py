"""G.711 µ-law decode for Twilio Media Streams.

Twilio sends call audio as 8 kHz mono µ-law, base64-encoded in each ``media`` event. Python
3.13 removed the stdlib ``audioop`` module (the usual decoder), so we expand µ-law to linear
PCM with a precomputed 256-entry lookup table in numpy — fast and dependency-free.

Output is float32 in [-1, 1] at 8 kHz; resampling to the model's 16 kHz is handled downstream
by ``backend.audio.resample_to_16k``.
"""

from __future__ import annotations

import base64

import numpy as np

_BIAS = 0x84  # 132
_CLIP = 32635


def _build_table() -> np.ndarray:
    table = np.empty(256, dtype=np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + _BIAS) << exponent
        sample -= _BIAS
        table[i] = -sample if sign else sample
    return table


_ULAW_TABLE = _build_table()


def ulaw_bytes_to_float32(ulaw: bytes) -> np.ndarray:
    """Decode raw µ-law bytes to float32 PCM in [-1, 1]."""
    if not ulaw:
        return np.zeros(0, dtype=np.float32)
    idx = np.frombuffer(ulaw, dtype=np.uint8)
    pcm16 = _ULAW_TABLE[idx].astype(np.float32)
    return pcm16 / 32768.0


def decode_media_payload(payload_b64: str) -> np.ndarray:
    """Decode a Twilio ``media.payload`` (base64 µ-law) to float32 PCM in [-1, 1] @ 8 kHz."""
    return ulaw_bytes_to_float32(base64.b64decode(payload_b64))
