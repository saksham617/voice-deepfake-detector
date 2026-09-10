import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the test suite offline and fast: force the dummy SSL frontend unless a test or the
# caller explicitly asks for a real one. The app/scripts still default to wav2vec2 (XLS-R).
os.environ.setdefault("VG_FEATURE_EXTRACTOR__BACKEND", "dummy")

# Most WS-level tests exercise chunking/pacing/risk-engine plumbing with synthetic sine-wave
# PCM, which Silero VAD (correctly) classifies as non-speech -- that's irrelevant to what
# those tests check and would just make them hang waiting for score events that VAD is now
# skipping. Default VAD off; tests/test_vad.py and the dedicated WS VAD tests turn it back on.
os.environ.setdefault("VG_VAD__ENABLED", "false")
