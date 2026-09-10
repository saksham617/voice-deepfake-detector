"""VAD gate on the live-call WS stream: non-speech windows are skipped before they ever
reach the fake-detection model; real speech is scored normally.

The rest of the suite runs with VG_VAD__ENABLED=false (see conftest.py) so the sine-wave
PCM used to test chunking/pacing plumbing elsewhere isn't itself misread as a VAD test.
Here VAD is turned on explicitly with the real (fast, offline) Silero model; the classifier
is stubbed so these tests don't need the ~1.2 GB AASIST/XLS-R checkpoint.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

FIXTURES = Path(__file__).parent / "fixtures"
FRAME_S = 0.1  # matches test_websocket.py's real-time pacing so windows arrive one at a
                # time instead of bursting and collapsing into a single "most recent" window


class _AlwaysHighPipeline:
    """Would drive the risk engine straight to HIGH if VAD didn't filter it out first --
    proof that a skipped window truly never reaches the classifier."""

    def __init__(self):
        self._i = 0

    def infer_chunk(self, waveform):
        from backend.inference.pipeline import ChunkResult

        self._i += 1
        return ChunkResult(fake_prob=0.99, n_samples=int(waveform.numel()), n_frames=50,
                           latency_ms=1.0, index=self._i)


class _RecordingWebhook:
    def __init__(self):
        self.calls = []

    async def fire(self, payload):
        self.calls.append(payload)
        return True


@pytest.fixture
def app_with_vad(monkeypatch):
    import backend.main as m
    from backend.audio import SileroVAD

    pipe, hook = _AlwaysHighPipeline(), _RecordingWebhook()
    monkeypatch.setattr(m, "get_pipeline", lambda: pipe)
    monkeypatch.setattr(m, "get_vad", lambda: SileroVAD())
    monkeypatch.setattr(m, "get_webhook", lambda: hook)
    return m.app, pipe, hook


def _wav_pcm16(name: str) -> bytes:
    wav, sr = sf.read(FIXTURES / name, dtype="float32")
    assert sr == 16000
    return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()


def _send_paced(ws, pcm: bytes, sr: int = 16000):
    """Real-time-paced send so each analysis window is processed as it completes instead
    of bursting and collapsing into a single 'most recent window' (see websocket.py's
    real-time strategy docstring)."""
    frame_bytes = int(FRAME_S * sr) * 2  # int16 mono
    for i in range(0, len(pcm), frame_bytes):
        ws.send_bytes(pcm[i : i + frame_bytes])
        time.sleep(FRAME_S)


def _drain(ws, n_events: int, timeout: float = 10.0) -> list[dict]:
    out: list[dict] = []
    deadline = time.time() + timeout
    while len(out) < n_events and time.time() < deadline:
        out.append(ws.receive_json())
    return out


def test_background_noise_is_skipped_not_scored(app_with_vad):
    """The bug this feature fixes: background noise (no speech) must never reach the
    classifier, so it can never trip a HIGH alert."""
    app, pipe, hook = app_with_vad
    pcm = _wav_pcm16("background_noise.wav")  # 10 s, no speech

    with TestClient(app) as client:
        with client.websocket_connect("/live-call/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 16000}))
            _send_paced(ws, pcm)

            events = _drain(ws, 7)  # 10 s of audio -> windows at 4..10 s (7 windows)
            ws.send_text(json.dumps({"type": "end"}))

    assert len(events) == 7, events
    assert all(e["type"] == "vad_skip" for e in events), events
    assert pipe._i == 0, "classifier must never be called on non-speech audio"
    assert hook.calls == [], "no alert should fire for audio with no speech in it"


def test_real_speech_is_scored_normally(app_with_vad):
    """VAD must not filter out genuine speech -- the classifier still sees it and the
    existing score/alert path is unaffected."""
    app, pipe, hook = app_with_vad
    pcm = _wav_pcm16("real_speech.wav")  # ~5.9 s of real speech

    with TestClient(app) as client:
        with client.websocket_connect("/live-call/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_text(json.dumps({"type": "config", "input_sample_rate": 16000}))
            _send_paced(ws, pcm)

            events = _drain(ws, 2)  # ~5.9 s -> windows at 4 s and 5 s (2 windows)
            ws.send_text(json.dumps({"type": "end"}))

    assert len(events) == 2, events
    assert all(e["type"] == "score" for e in events), events
    assert all(0.0 <= e["fake_prob"] <= 1.0 for e in events)
    assert pipe._i == len(events)
    # the stub always returns fake_prob=0.99 -- confirms the existing alert path still
    # works once VAD lets a window through (3 consecutive-high needed; here just 2 windows
    # arrive so no alert yet, but nothing was skipped that should have scored)
    assert hook.calls == []
