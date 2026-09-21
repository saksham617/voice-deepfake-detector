"""Unit tests for backend/audio/transcriber.py — the never-drop, non-overlapping streaming
buffer and engine selection. Uses the StubEngine so no model/network is needed."""

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.audio.transcriber import (  # noqa: E402
    StreamingTranscriber,
    StubEngine,
    build_engine,
    pcm_to_wav_bytes,
)

SR = 16000


def _tone(seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    return (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)


def test_segments_are_non_overlapping_and_lose_no_audio():
    t = StreamingTranscriber(StubEngine(), segment_seconds=5.0)
    # Feed 12 s in irregular chunks (like arbitrary-size WS frames).
    total = 0.0
    for secs in (0.3, 1.1, 2.0, 0.6, 3.0, 1.0, 2.5, 1.5):
        t.feed(_tone(secs))
        total += secs

    collected = 0
    seg_lengths = []
    while t.has_segment():
        seg = t.take_segment()
        seg_lengths.append(len(seg))
        collected += len(seg)
    tail = t.take_tail()
    if tail is not None:
        collected += len(tail)

    # Every full segment is exactly 5 s; nothing is dropped or double-counted.
    assert all(n == t.segment_samples for n in seg_lengths)
    assert len(seg_lengths) == 2  # 12 s -> two 5 s segments + a ~2 s tail
    assert collected == int(round(total * SR))  # total audio conserved exactly


def test_take_segment_none_until_full():
    t = StreamingTranscriber(StubEngine(), segment_seconds=5.0)
    t.feed(_tone(3.0))
    assert t.has_segment() is False
    assert t.take_segment() is None
    t.feed(_tone(2.5))
    assert t.take_segment() is not None  # now >= 5 s


def test_record_accumulates_full_text():
    t = StreamingTranscriber(StubEngine(), segment_seconds=5.0)
    t.record("hello there")
    t.record("this is a test")
    t.record("")  # empty ignored
    assert t.segments == ["hello there", "this is a test"]
    assert t.full_text == "hello there this is a test"


def test_stub_engine_and_transcribe_segment():
    t = StreamingTranscriber(StubEngine(), segment_seconds=5.0)
    text = t.transcribe_segment(_tone(5.0))
    assert "stub transcript" in text
    assert "5.0s" in text


def test_wav_encoding_roundtrips_length():
    import wave

    pcm = _tone(1.0)
    data = pcm_to_wav_bytes(pcm)
    import io

    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == SR
        assert w.getnchannels() == 1
        assert w.getnframes() == len(pcm)


def test_build_engine_stub_and_off():
    class Cfg:
        enabled = True
        engine = "stub"
        cloud_model = "nova-2"
        local_model = "base"

    assert isinstance(build_engine(Cfg()), StubEngine)

    class Off(Cfg):
        engine = "off"

    assert build_engine(Off()) is None

    class Disabled(Cfg):
        enabled = False

    assert build_engine(Disabled()) is None
