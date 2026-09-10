"""Voice Deepfake Detector -- combined FastAPI application.

Serves two subsystems from one process:

  * the primary detection API at the app root (/predict,
    /enroll_speaker, /verify_speaker, /report(s), /health) -- CNN spoof
    detection + ECAPA-TDNN speaker verification + reporting. This is the
    contract the deployed React frontend depends on
    (frontend/src/types/prediction.ts); see backend/api/legacy.py.

  * VoiceGuard's live-call streaming subsystem under /live-call -- AASIST
    over a wav2vec2/IndicWav2Vec SSL frontend, scored per chunk over
    WebSocket with a rolling risk engine, plus its own static demo UI
    (frontend/live-call/{index,caller,receiver}.html). See
    backend/api/rest.py, backend/api/websocket.py, backend/inference/,
    backend/scoring/, backend/alerts/.

    uvicorn backend.main:app --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core import get_config
from src.models import message_detection as md
from src.models import reporting
from src.models import speaker_verification as sv
from src.models.inference import ensure_model_loaded

logger = logging.getLogger("backend")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Scoped to this one logger only -- deliberately not logging.basicConfig(),
    # which would reconfigure the root logger and affect every other
    # library's logging output too. Without an explicit handler here, INFO
    # records have nowhere to go: with no handler anywhere in this logger's
    # chain (uvicorn attaches handlers to its own "uvicorn.*" loggers, not
    # root), Python's logging module falls back to its "handler of last
    # resort", which itself only emits WARNING+ -- so an INFO call would be
    # silently dropped even with logger.setLevel(INFO) above.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_CALL_DIR = REPO_ROOT / "frontend" / "live-call"

# Comma-separated list of allowed origins, e.g. "https://myapp.vercel.app".
# Defaults to "*" so local dev (Vite on a different port) keeps working
# unconfigured; set explicitly in production instead of relying on the
# default. Deliberately independent of config/config.yaml's server.cors_origins
# (which only governs the live-call demo, not the deployed frontend's contract).
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = (
    ["*"] if _cors_origins_env == "*" else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)

_pipeline = None
_webhook = None
_vad = None
_VAD_UNSET = object()
_vad_loaded = _VAD_UNSET


def get_pipeline():
    """App-scoped singleton so the live-call AASIST + SSL frontend weights
    load once, not per WebSocket connection."""
    global _pipeline
    if _pipeline is None:
        from backend.inference import DetectionPipeline

        _pipeline = DetectionPipeline.from_config(get_config())
    return _pipeline


def get_vad():
    """App-scoped Silero VAD singleton gating the live-call stream, or ``None`` if
    ``vad.enabled`` is false in config."""
    global _vad, _vad_loaded
    if _vad_loaded is _VAD_UNSET:
        cfg = get_config().vad
        if cfg.enabled:
            from backend.audio import SileroVAD

            _vad = SileroVAD(
                frame_threshold=cfg.frame_threshold,
                min_speech_ratio=cfg.min_speech_ratio,
            )
        else:
            _vad = None
        _vad_loaded = True
    return _vad


def get_webhook():
    global _webhook
    if _webhook is None:
        from backend.alerts import WebhookDispatcher

        _webhook = WebhookDispatcher.from_config(get_config().webhook)
    return _webhook


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model once at startup so /health reflects real readiness
    instead of lazily loading (and possibly failing) on the first request.
    Each subsystem is independent: one failing to load doesn't block the
    others, and each is reported separately on /health."""
    try:
        device = ensure_model_loaded()
        app.state.model_ready = True
        logger.info("CNN spoof-detection model loaded successfully on device=%s", device)
    except Exception:
        app.state.model_ready = False
        logger.exception("CNN spoof-detection model failed to load at startup")

    try:
        sv.ensure_model_loaded()
        app.state.speaker_model_ready = True
        logger.info("speaker verification model loaded successfully")
    except Exception:
        app.state.speaker_model_ready = False
        logger.exception("speaker verification model failed to load at startup")

    try:
        md.ensure_model_loaded()
        app.state.message_model_ready = True
        logger.info("message detection model loaded successfully")
    except Exception:
        app.state.message_model_ready = False
        logger.exception("message detection model failed to load at startup")

    try:
        reporting.ensure_db_ready()
        app.state.reports_db_ready = True
        logger.info("reports database ready at %s", reporting.DB_PATH)
    except Exception:
        app.state.reports_db_ready = False
        logger.exception("reports database failed to initialize at startup")

    try:
        get_pipeline()  # warm the live-call AASIST + SSL frontend
        get_vad()  # warm the VAD gate (no-op if vad.enabled=false)
        get_webhook()
        app.state.live_call_ready = True
        logger.info("live-call detection pipeline loaded successfully")
    except Exception:
        app.state.live_call_ready = False
        logger.exception("live-call detection pipeline failed to load at startup")

    yield


app = FastAPI(title="Voice Deepfake Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api import legacy_router, rest_router, ws_router  # noqa: E402

app.include_router(legacy_router)
app.include_router(rest_router, prefix="/live-call")
app.include_router(ws_router, prefix="/live-call")

if LIVE_CALL_DIR.exists():
    app.mount("/live-call", StaticFiles(directory=str(LIVE_CALL_DIR), html=True), name="live-call")


if __name__ == "__main__":
    import uvicorn

    # Many hosting platforms (Render, Railway, etc.) inject PORT and require
    # the app to bind to it.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
