"""In-memory pub/sub between inbound Twilio calls and the live dashboard(s).

A Twilio Media Stream (one per call) publishes call lifecycle + per-window verdict events
here; every connected dashboard WebSocket is subscribed and receives them, so a card appears
and its confidence meter updates automatically — no one clicks "accept". Also keeps a snapshot
of currently-active calls so a dashboard that connects mid-call immediately shows it.

Single-process, in-memory (like the existing WebRTC signaling relay) — fine for the hackathon
demo; a multi-worker deployment would swap this for Redis pub/sub behind the same interface.

Event shapes broadcast to dashboards:
  {"type":"call_started", "call_id","from","to","started_at"}
  {"type":"score",        "call_id","index","is_fake","confidence","fake_prob","risk":{...}}
  {"type":"call_ended",   "call_id","final_verdict","final_confidence"}
"""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import WebSocket


class CallHub:
    def __init__(self) -> None:
        self._dashboards: set[WebSocket] = set()
        self._active: dict[str, dict[str, Any]] = {}

    # -- dashboard subscribers ----------------------------------------------
    async def subscribe(self, ws: WebSocket) -> None:
        self._dashboards.add(ws)
        # Replay currently-active calls so a late-joining dashboard isn't blank.
        for state in list(self._active.values()):
            with contextlib.suppress(Exception):
                await ws.send_json(state["started_event"])
            if state.get("last_score"):
                with contextlib.suppress(Exception):
                    await ws.send_json(state["last_score"])

    def unsubscribe(self, ws: WebSocket) -> None:
        self._dashboards.discard(ws)

    async def _broadcast(self, msg: dict) -> None:
        for ws in list(self._dashboards):
            try:
                await ws.send_json(msg)
            except Exception:  # noqa: BLE001 — drop dead dashboard sockets
                self._dashboards.discard(ws)

    # -- publishers (called by the media-stream handler) --------------------
    async def call_started(self, call_id: str, from_number: str, to_number: str,
                           started_at: float) -> None:
        event = {
            "type": "call_started", "call_id": call_id,
            "from": from_number, "to": to_number, "started_at": started_at,
        }
        self._active[call_id] = {"started_event": event, "last_score": None}
        await self._broadcast(event)

    async def score(self, call_id: str, payload: dict) -> None:
        event = {"type": "score", "call_id": call_id, **payload}
        if call_id in self._active:
            self._active[call_id]["last_score"] = event
        await self._broadcast(event)

    async def call_ended(self, call_id: str, final_verdict: str, final_confidence: float) -> None:
        self._active.pop(call_id, None)
        await self._broadcast({
            "type": "call_ended", "call_id": call_id,
            "final_verdict": final_verdict, "final_confidence": final_confidence,
        })


hub = CallHub()
