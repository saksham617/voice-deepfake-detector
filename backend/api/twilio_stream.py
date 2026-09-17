"""Twilio Media Streams ingestion + the dashboard fan-out WebSocket.

  WS /twilio/stream     <- Twilio connects here (from the voice webhook's <Stream>) and sends
                           JSON events: connected / start / media (base64 µ-law 8 kHz) / stop.
                           We decode, buffer into ~4 s windows, run the (stubbed) inference,
                           smooth with the rolling RiskEngine, and publish verdicts to the hub.
  WS /twilio/dashboard  <- the browser dashboard connects here and receives call_started /
                           score / call_ended events for every call, live. No accept button.

The inference call is the swappable stub (backend/inference/live_infer.infer); this file never
needs to change when the real model is wired in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.audio import AudioChunker
from backend.core import get_config
from backend.inference.live_infer import fake_probability, infer
from backend.telephony import store
from backend.telephony.call_hub import hub
from backend.telephony.mulaw import decode_media_payload

logger = logging.getLogger("backend.telephony")
router = APIRouter()

_DISCONNECT_ERRORS = (WebSocketDisconnect, OSError, RuntimeError)
_FLAG_THRESHOLD = 0.5  # final verdict = "fake" if smoothed risk >= this (tunable)


@router.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket) -> None:
    await ws.accept()
    cfg = get_config()
    from backend.scoring import RiskEngine

    risk = RiskEngine.from_config(cfg.risk)
    # Twilio audio is 8 kHz µ-law -> decoded to float32; chunker resamples to the model's 16 kHz.
    chunker = AudioChunker(
        sample_rate=cfg.audio.sample_rate,
        chunk_seconds=cfg.audio.chunk_seconds,
        hop_seconds=cfg.audio.hop_seconds,
        input_sample_rate=8000,
        pcm_format="float32",
    )

    call_id: str | None = None
    from_number = "unknown"
    to_number = ""
    index = 0
    last_state = None
    last_verdict = None

    try:
        while True:
            raw = await ws.receive_text()  # Twilio Media Streams frames are text JSON
            data = json.loads(raw)
            event = data.get("event")

            if event == "start":
                start = data.get("start", {})
                call_id = start.get("callSid") or start.get("streamSid") or f"call_{int(time.time())}"
                params = start.get("customParameters") or {}
                from_number = params.get("from", "unknown")
                to_number = params.get("to", "")
                store.record_call_start(call_id, from_number, to_number)
                await hub.call_started(call_id, from_number, to_number, time.time())
                logger.info("twilio stream started call=%s from=%s", call_id, from_number)

            elif event == "media":
                payload = data.get("media", {}).get("payload")
                if not payload:
                    continue
                pcm = decode_media_payload(payload)  # float32 @ 8 kHz
                for window in chunker.push(pcm):
                    index += 1
                    verdict = await asyncio.to_thread(infer, window)  # {is_fake, confidence}
                    p_fake = fake_probability(verdict)
                    state = risk.update(p_fake)
                    last_state, last_verdict = state, verdict
                    await hub.score(
                        call_id or "unknown",
                        {
                            "index": index,
                            "is_fake": verdict["is_fake"],
                            "confidence": verdict["confidence"],
                            "fake_prob": round(p_fake, 4),
                            "risk": {
                                "score": state.score,
                                "level": state.level.value,
                                "consecutive_high": state.consecutive_high,
                            },
                            "alert": state.alert,
                        },
                    )

            elif event == "stop":
                break
    except _DISCONNECT_ERRORS:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("twilio stream error (call=%s)", call_id)
    finally:
        if call_id:
            if last_state is not None:
                final_fake = last_state.score >= _FLAG_THRESHOLD
                verdict = "fake" if final_fake else "real"
                store.record_call_end(call_id, verdict, last_state.score)
                await hub.call_ended(call_id, verdict, round(last_state.score, 4))
            else:
                store.record_call_end(call_id, "unknown", 0.0)
                await hub.call_ended(call_id, "unknown", 0.0)
        with contextlib.suppress(*_DISCONNECT_ERRORS):
            await ws.close()


@router.websocket("/twilio/dashboard")
async def twilio_dashboard(ws: WebSocket) -> None:
    """Browser dashboard subscribes here; receives every call's live events. Read-only."""
    await ws.accept()
    await hub.subscribe(ws)
    try:
        while True:
            await ws.receive_text()  # ignore anything the client sends; this is a push channel
    except _DISCONNECT_ERRORS:
        pass
    finally:
        hub.unsubscribe(ws)
