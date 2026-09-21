"""Live-call transcription — a SEPARATE, never-drop path from the risk-scoring loop.

The risk loop in backend/api/websocket.py feeds AudioChunker overlapping 4 s windows and
drops the oldest when inference lags (correct for a smoothly-updating score). Transcription
must NOT reuse that: overlap would transcribe the same audio 4× and dropping would lose words.

So this module keeps its own buffer of the raw, non-overlapping audio stream (fed via
``AudioChunker.add_tap`` — every pushed sample, exactly once, in order) and runs ASR on
coarse, non-overlapping segments (``segment_seconds``). It may lag behind the live risk score
if the engine is slow, but it processes every second of the call and never skips any.

Engines are pluggable (chosen via ``transcription.engine`` config / ``VG_TRANSCRIPTION__ENGINE``):
  * cloud  — Deepgram prerecorded REST (the chosen production engine; needs DEEPGRAM_API_KEY).
  * local  — faster-whisper, offline (optional dep; used for offline demos + verifying the
             pipeline against real audio without a cloud key).
  * stub   — deterministic, dependency-free (plumbing tests / safe fallback).
  * auto   — cloud if a key is set, else local if faster-whisper is importable, else stub.

Threading note: ``feed``/``take_segment`` are only ever called from the event-loop thread
(the WS receiver + transcription tasks); only ``engine.transcribe`` runs in a worker thread,
on a segment already copied out of the buffer. So the buffer needs no lock.
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.error
import urllib.request
import wave
from typing import Protocol

import numpy as np

logger = logging.getLogger("backend.transcriber")

TARGET_SR = 16000


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = TARGET_SR) -> bytes:
    """float32 mono [-1,1] -> 16-bit PCM WAV bytes (for cloud STT upload)."""
    clipped = np.clip(pcm, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- engines


class TranscriptionEngine(Protocol):
    name: str

    def transcribe(self, pcm: np.ndarray, language: str | None) -> str: ...


class StubEngine:
    """Deterministic, no dependencies. Proves the plumbing without any model or network."""

    name = "stub"

    def transcribe(self, pcm: np.ndarray, language: str | None) -> str:
        seconds = len(pcm) / TARGET_SR
        return f"[stub transcript: {seconds:.1f}s of speech]"


class DeepgramEngine:
    """Cloud STT via Deepgram's prerecorded REST API (stdlib urllib, no extra dependency).

    Multilingual: language="auto" asks Deepgram to detect the language (Indian languages
    included on nova-2); otherwise the given BCP-47 code is passed through.
    """

    name = "cloud:deepgram"
    _URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, api_key: str, model: str = "nova-2") -> None:
        self.api_key = api_key
        self.model = model

    def transcribe(self, pcm: np.ndarray, language: str | None) -> str:
        params = [f"model={self.model}", "smart_format=true", "punctuate=true"]
        if not language or language == "auto":
            params.append("detect_language=true")
        else:
            params.append(f"language={language}")
        url = f"{self._URL}?{'&'.join(params)}"
        req = urllib.request.Request(
            url,
            data=pcm_to_wav_bytes(pcm),
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("Deepgram request failed: %s", e)
            return ""
        try:
            return payload["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError):
            logger.warning("Deepgram response missing transcript: %s", str(payload)[:200])
            return ""


class LocalWhisperEngine:
    """Offline ASR via faster-whisper. Optional dependency; imported lazily on first use so the
    app runs without it unless this engine is selected."""

    name = "local:faster-whisper"

    def __init__(self, model_size: str = "base") -> None:
        from faster_whisper import WhisperModel

        # Default to CPU (portable, matches the no-GPU deployment target). ctranslate2's CUDA
        # support is separate from torch's and needs system cuBLAS DLLs present, so opt in
        # explicitly with VG_WHISPER_DEVICE=cuda only where those are installed.
        device = os.environ.get("VG_WHISPER_DEVICE", "cpu").strip().lower()
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("loading faster-whisper '%s' on %s (%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm: np.ndarray, language: str | None) -> str:
        lang = None if (not language or language == "auto") else language
        segments, _ = self.model.transcribe(pcm.astype(np.float32), language=lang)
        return " ".join(seg.text.strip() for seg in segments).strip()


def build_engine(cfg) -> TranscriptionEngine | None:
    """Resolve the configured engine to an instance (or None if transcription is off).

    cfg is a TranscriptionConfig (duck-typed). "auto" prefers cloud (key present), then local
    (faster-whisper installed), then stub.
    """
    engine = (cfg.engine or "auto").strip().lower()
    api_key = os.environ.get("DEEPGRAM_API_KEY") or os.environ.get("STT_API_KEY")

    if engine == "off" or not cfg.enabled:
        return None

    if engine == "auto":
        if api_key:
            engine = "cloud"
        elif _faster_whisper_available():
            engine = "local"
        else:
            engine = "stub"
            logger.info("transcription engine=auto -> stub (no STT key, faster-whisper absent)")

    if engine == "cloud":
        if not api_key:
            logger.warning("transcription engine=cloud but no DEEPGRAM_API_KEY set -> stub.")
            return StubEngine()
        return DeepgramEngine(api_key, model=getattr(cfg, "cloud_model", "nova-2"))
    if engine == "local":
        return LocalWhisperEngine(model_size=getattr(cfg, "local_model", "base"))
    return StubEngine()


def _faster_whisper_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


# --------------------------------------------------------------------------- streaming buffer


class StreamingTranscriber:
    """Accumulates the raw call audio and yields non-overlapping segments for ASR.

    ``feed`` (from the chunker tap) appends every sample once. ``take_segment`` pops a full
    ``segment_seconds`` block when available; ``take_tail`` returns whatever's left at end of
    call. Transcribed text is appended to ``full_text`` via ``record``.
    """

    def __init__(self, engine, segment_seconds: float = 5.0, language: str | None = "auto",
                 sample_rate: int = TARGET_SR) -> None:
        self.engine = engine
        self.language = language
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self._buf = np.zeros(0, dtype=np.float32)
        self.segments: list[str] = []
        self.full_text: str = ""

    # -- buffer (event-loop thread only) ------------------------------------
    def feed(self, wav: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(wav, dtype=np.float32)])

    def has_segment(self) -> bool:
        return self._buf.shape[0] >= self.segment_samples

    def take_segment(self) -> np.ndarray | None:
        if self._buf.shape[0] < self.segment_samples:
            return None
        seg = self._buf[: self.segment_samples].copy()
        self._buf = self._buf[self.segment_samples :]
        return seg

    def take_tail(self, min_seconds: float = 0.5) -> np.ndarray | None:
        """Whatever's left when the call ends — transcribed if it's long enough to be speech."""
        if self._buf.shape[0] < int(self.sample_rate * min_seconds):
            self._buf = np.zeros(0, dtype=np.float32)
            return None
        tail = self._buf.copy()
        self._buf = np.zeros(0, dtype=np.float32)
        return tail

    # -- text (worker result recorded back on the loop thread) --------------
    def transcribe_segment(self, seg: np.ndarray) -> str:
        """Blocking ASR call — run this via asyncio.to_thread so it never blocks the loop."""
        return self.engine.transcribe(seg, self.language)

    def record(self, text: str) -> str:
        text = (text or "").strip()
        if text:
            self.segments.append(text)
            self.full_text = (self.full_text + " " + text).strip()
        return text
