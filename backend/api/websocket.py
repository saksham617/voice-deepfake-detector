"""WebSocket streaming endpoint.  [Day 2 -> Day 5 full loop, Day 3 real-time hardening]

  WS /ws/stream

Client (receiver browser tab) sends:
  * binary frames  -> raw PCM (int16 or float32 LE, mono, sample_rate from /config)
  * text  {"type":"config", "input_sample_rate": 48000}      (optional, first message)
  * text  {"type":"end"}                                     (graceful close)

Server sends, per analysis window it actually scores:
  {"type":"score", "index":N, "fake_prob":0.xx, "risk":{...}, "latency_ms":..,
   "dropped":K, "alert":false}
  ``dropped`` = analysis windows skipped since the last score because inference could not
  keep up with the incoming audio (see below).

Real-time strategy
------------------
SSL inference on CPU (~0.4-1.0 s per 1 s chunk) can be slower than audio arrives. To keep
latency bounded instead of growing without limit:
  * a receiver task drains the socket continuously and feeds the chunker;
  * only the *most recent* pending analysis window is kept — older ones are dropped and
    counted;
  * inference runs in a worker thread (``asyncio.to_thread``) so receiving never blocks.
Under load the risk score simply updates less often; it never falls behind real time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.audio import AudioChunker
from backend.core import get_config

router = APIRouter()
logger = logging.getLogger("backend")

# Full-call transcripts, kept in memory keyed by session_id so they're available at session end
# (e.g. to attach to a report later — NOT wired into reporting here, that's a separate track).
SESSION_TRANSCRIPTS: dict[str, str] = {}
_MAX_TRANSCRIPTS = 50

# A client that vanishes mid-send (tab closed, network drop) can surface as
# WebSocketDisconnect on receive, but as OSError/RuntimeError from the ASGI server on send
# (e.g. uvicorn's ClientDisconnected, a subclass of OSError). Treat both as a plain disconnect.
_DISCONNECT_ERRORS = (WebSocketDisconnect, OSError, RuntimeError)


def _score_window(pipeline, vad, window):
    """VAD-gate then (maybe) run the classifier, off the event loop.

    Returns ``None`` for a window VAD classifies as non-speech -- it is never handed to
    the fake-detection model, which was only ever trained on real/fake *speech* and has no
    reliable behaviour on pure background noise.
    """
    if vad is not None and not vad.is_speech(window):
        return None
    return pipeline.infer_chunk(window)


# --------------------------------------------------------------------------- signaling relay
# Minimal room-based WebRTC signaling so caller/receiver auto-connect by a shared code
# instead of copy-pasting SDP. Not for production (no auth, in-memory, single worker).
_rooms: dict[str, list[WebSocket]] = {}


@router.websocket("/ws/signal/{room}")
async def signal(ws: WebSocket, room: str) -> None:
    await ws.accept()
    peers = _rooms.setdefault(room, [])
    if len(peers) >= 2:
        await ws.send_json({"type": "full"})
        await ws.close()
        return
    peers.append(ws)
    await ws.send_json({"type": "joined", "role": "caller" if len(peers) == 1 else "receiver"})
    if len(peers) == 2:
        for p in peers:
            await p.send_json({"type": "ready"})
    try:
        while True:
            msg = await ws.receive_text()
            for p in list(peers):
                if p is not ws:
                    with contextlib.suppress(Exception):
                        await p.send_text(msg)
    except _DISCONNECT_ERRORS:
        pass
    finally:
        if ws in peers:
            peers.remove(ws)
        for p in list(peers):
            with contextlib.suppress(Exception):
                await p.send_json({"type": "peer-left"})
        if not peers:
            _rooms.pop(room, None)


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    cfg = get_config()
    session_id = uuid.uuid4().hex[:12]

    # lazy import so Day 1/2 work before the pipeline is ready
    from backend.main import get_pipeline, get_vad, get_webhook
    from backend.scoring import RiskEngine
    from backend.scoring.session_report import SessionRiskAccumulator

    pipeline = get_pipeline()
    vad = get_vad()  # None if vad.enabled=false in config
    risk = RiskEngine.from_config(cfg.risk)
    webhook = get_webhook()
    # Accumulates this session's per-window risk so we can auto-log one report when it ends.
    acc = SessionRiskAccumulator()
    session_start = time.monotonic()

    chunker = AudioChunker(
        sample_rate=cfg.audio.sample_rate,
        chunk_seconds=cfg.audio.chunk_seconds,
        hop_seconds=cfg.audio.hop_seconds,
        input_sample_rate=cfg.audio.sample_rate,
        pcm_format=cfg.audio.pcm_format,
    )

    # Transcription: an INDEPENDENT, never-drop path. It taps the raw resampled stream (every
    # sample once, via chunker.add_tap) into its own buffer, and a separate task transcribes
    # non-overlapping segments off the event loop — so it can lag the risk score but never
    # drops audio and never blocks the receive loop. See backend/audio/transcriber.py.
    from backend.audio import StreamingTranscriber
    from backend.main import get_transcription_engine

    transcriber = None
    transcript_ready = asyncio.Event()
    _engine = get_transcription_engine()
    if _engine is not None and cfg.transcription.enabled:
        transcriber = StreamingTranscriber(
            _engine,
            segment_seconds=cfg.transcription.segment_seconds,
            language=cfg.transcription.language,
            sample_rate=cfg.audio.sample_rate,
        )

        def _feed_transcription(wav) -> None:
            transcriber.feed(wav)
            transcript_ready.set()

        chunker.add_tap(_feed_transcription)

    # shared state between the receiver task and the inference loop
    pending: dict = {"window": None, "dropped": 0}
    got_window = asyncio.Event()
    stop = asyncio.Event()

    async def receiver() -> None:
        try:
            while not stop.is_set():
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if (text := msg.get("text")) is not None:
                    data = json.loads(text)
                    if data.get("type") == "config" and data.get("input_sample_rate"):
                        chunker.input_sample_rate = int(data["input_sample_rate"])
                    elif data.get("type") == "end":
                        break
                    continue

                pcm = msg.get("bytes")
                if not pcm:
                    continue

                for window in chunker.push(pcm):
                    if pending["window"] is not None:
                        pending["dropped"] += 1  # inference is behind; skip the older window
                    pending["window"] = window
                    got_window.set()
        finally:
            stop.set()
            got_window.set()  # wake the inference loop so it can exit
            transcript_ready.set()  # wake the transcription loop so it can drain + exit

    async def transcription_loop() -> None:
        """Consume non-overlapping segments in order and emit transcript messages. Never drops
        audio; runs ASR off the loop via to_thread so it can't block risk scoring."""
        if transcriber is None:
            return
        try:
            while True:
                if not transcriber.has_segment():
                    if stop.is_set():
                        break
                    await transcript_ready.wait()
                    transcript_ready.clear()
                    continue
                seg = transcriber.take_segment()
                try:
                    text = await asyncio.to_thread(transcriber.transcribe_segment, seg)
                except Exception:  # noqa: BLE001 — one bad segment must not kill the transcript
                    logger.exception("transcription failed for a segment (session %s)", session_id)
                    continue
                await _emit_transcript(ws, transcriber, transcriber.record(text))
        finally:
            # Flush the end-of-call remainder so the last words aren't lost.
            tail = transcriber.take_tail()
            if tail is not None:
                with contextlib.suppress(Exception):
                    text = await asyncio.to_thread(transcriber.transcribe_segment, tail)
                    await _emit_transcript(ws, transcriber, transcriber.record(text))

    await ws.send_json({"type": "ready", "session_id": session_id})
    recv_task = asyncio.create_task(receiver())
    trans_task = asyncio.create_task(transcription_loop())
    window_index = 0  # every window seen, scored or VAD-skipped

    try:
        while True:
            await got_window.wait()
            got_window.clear()

            window = pending["window"]
            if window is None:
                if stop.is_set():
                    break
                continue
            pending["window"] = None
            dropped = pending["dropped"]
            pending["dropped"] = 0
            window_index += 1

            result = await asyncio.to_thread(_score_window, pipeline, vad, window)
            if result is None:
                # VAD says non-speech: skip scoring entirely rather than force a fake_prob
                # out of a model that was never trained on non-speech audio.
                await ws.send_json({"type": "vad_skip", "index": window_index, "dropped": dropped})
                continue
            state = risk.update(result.fake_prob)
            acc.add(state)

            await ws.send_json(
                {
                    "type": "score",
                    "index": result.index,
                    "fake_prob": round(result.fake_prob, 4),
                    "latency_ms": round(result.latency_ms, 1),
                    "dropped": dropped,
                    "risk": {
                        "score": state.score,
                        "level": state.level.value,
                        "consecutive_high": state.consecutive_high,
                    },
                    "alert": state.alert,
                }
            )
            if state.alert:
                import time as _time

                from backend.alerts import AlertPayload

                await webhook.fire(
                    AlertPayload(
                        session_id=session_id,
                        score=state.score,
                        level=state.level.value,
                        raw_prob=state.raw_prob,
                        consecutive_high=state.consecutive_high,
                        n_chunks=state.n_chunks,
                        ts=_time.time(),
                    )
                )
    except _DISCONNECT_ERRORS:
        pass
    finally:
        stop.set()
        transcript_ready.set()
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await recv_task
        # Let transcription drain its remaining segments + end-of-call tail before we close,
        # but bound it so a slow/hung ASR call can't hang the teardown.
        try:
            await asyncio.wait_for(trans_task, timeout=20.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            trans_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await trans_task
        if transcriber is not None and transcriber.full_text:
            _store_transcript(session_id, transcriber.full_text)
        with contextlib.suppress(*_DISCONNECT_ERRORS):
            await ws.close()
        # Auto-log one report for the finished session (skipped if nothing was ever scored).
        # Runs regardless of how the session ended (client 'end', disconnect, or error), and a
        # DB failure here must never surface as a WS error — the socket is already closed.
        await _log_session_report(acc, time.monotonic() - session_start, session_id)


async def _emit_transcript(ws: WebSocket, transcriber, text: str) -> None:
    """Send a transcript update (distinct message type from score/vad_skip/ready). No-op for
    empty text (e.g. a silent segment ASR returned nothing for)."""
    if not text:
        return
    with contextlib.suppress(*_DISCONNECT_ERRORS):
        await ws.send_json(
            {
                "type": "transcript",
                "text": text,  # the latest segment
                "session_running_text": transcriber.full_text,  # everything so far
            }
        )


def _store_transcript(session_id: str, full_text: str) -> None:
    """Keep the full-call transcript in memory (capped) for retrieval at/after session end."""
    SESSION_TRANSCRIPTS[session_id] = full_text
    if len(SESSION_TRANSCRIPTS) > _MAX_TRANSCRIPTS:
        oldest = next(iter(SESSION_TRANSCRIPTS))
        SESSION_TRANSCRIPTS.pop(oldest, None)
    logger.info("session %s transcript (%d chars): %s",
                session_id, len(full_text), full_text[:200])


async def _log_session_report(
    acc, duration_seconds: float, session_id: str, phone_number: str | None = None
) -> None:
    """Persist a report for a completed live-call session. phone_number is None for now —
    /ws/stream carries no caller ID (see reporting.create_report's docstring)."""
    kwargs = acc.to_report_kwargs(duration_seconds, phone_number=phone_number)
    if kwargs is None:
        return  # no scored windows -> no meaningful report
    try:
        from src.models import reporting

        report = await asyncio.to_thread(reporting.create_report, **kwargs)
        logger.info(
            "live-call session %s logged as report %s (%s, conf=%.2f)",
            session_id, report["id"], report["verdict"], report["confidence_score"],
        )
    except Exception:
        logger.exception("failed to auto-log report for session %s", session_id)
