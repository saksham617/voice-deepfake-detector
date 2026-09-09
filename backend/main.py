"""Minimal FastAPI inference backend for the voice deepfake detector.

Exposes POST /predict (multipart/form-data, field "file") matching the
frontend's agreed contract (frontend/src/types/prediction.ts):
    {"prediction": "bonafide" | "spoof", "confidence": number}
plus the full bonafide/spoof probability breakdown as extra fields.

Reuses the exact trained-model inference pipeline used everywhere else in
this project (src.models.inference.predict_audio) rather than reimplementing
audio loading, spectrogram preprocessing, or model loading here. Does not
modify the trained model or the raw dataset.
"""

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import reporting
from src.models import speaker_verification as sv
from src.models.inference import ensure_model_loaded, predict_audio

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

# Matches what the CNN was trained/evaluated on (audio_io.py reads via
# soundfile, which natively supports WAV/FLAC). Other formats the frontend
# accepts (mp3, ogg, webm, m4a) are not yet supported server-side.
ALLOWED_EXTENSIONS = {".wav", ".flac"}

# Matches the frontend's client-side cap (frontend/src/config.ts
# MAX_FILE_SIZE_BYTES); enforced again here since the client-side check is
# only a UX convenience, not a security boundary.
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
_UPLOAD_CHUNK_SIZE = 1024 * 1024

# Exposed as a constant (rather than an inline literal) so tests can assert
# against it directly instead of duplicating the string.
GENERIC_PREDICTION_ERROR_DETAIL = "Internal error while processing the audio file."
GENERIC_SPEAKER_ERROR_DETAIL = "Internal error while processing the speaker verification request."
GENERIC_REPORT_ERROR_DETAIL = "Internal error while processing the report."
CORRUPT_AUDIO_ERROR_DETAIL = (
    "Could not read the uploaded audio file. It may be corrupted or in an unsupported format."
)

# Comma-separated list of allowed origins, e.g. "https://myapp.vercel.app".
# Defaults to "*" so local dev (Vite on a different port) keeps working
# unconfigured; set explicitly in production instead of relying on the default.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = (
    ["*"] if _cors_origins_env == "*" else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so /health reflects real readiness
    instead of lazily loading (and possibly failing) on the first request."""
    try:
        device = ensure_model_loaded()
        app.state.model_ready = True
        logger.info("model loaded successfully on device=%s", device)
    except Exception:
        app.state.model_ready = False
        logger.exception("model failed to load at startup")

    try:
        sv.ensure_model_loaded()
        app.state.speaker_model_ready = True
        logger.info("speaker verification model loaded successfully")
    except Exception:
        app.state.speaker_model_ready = False
        logger.exception("speaker verification model failed to load at startup")

    try:
        reporting.ensure_db_ready()
        app.state.reports_db_ready = True
        logger.info("reports database ready at %s", reporting.DB_PATH)
    except Exception:
        app.state.reports_db_ready = False
        logger.exception("reports database failed to initialize at startup")
    yield


app = FastAPI(title="Voice Deepfake Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    bonafide_probability: float
    spoof_probability: float


class EnrollSpeakerResponse(BaseModel):
    name: str
    duration_sec: float
    short_clip: bool


class VerifySpeakerResponse(BaseModel):
    name: str
    similarity: float
    is_match: bool
    threshold: float
    short_clip: bool


class ReportCreateRequest(BaseModel):
    type: str
    verdict: str
    confidence_score: float
    claimed_identity: str | None = None
    user_notes: str | None = None


class ReportResponse(BaseModel):
    id: int
    timestamp: str
    type: str
    verdict: str
    confidence_score: float
    claimed_identity: str | None = None
    user_notes: str | None = None


@app.get("/health")
def health() -> JSONResponse:
    """Reports whether the models actually loaded at startup, not just that
    the process is running -- so a load balancer/host treats a failed model
    load as unhealthy rather than routing traffic to a broken instance.

    Overall `status`/status-code tracks the primary spoof-detection model
    only (the app's core feature); `speaker_model_ready` and
    `reports_db_ready` are reported separately since the endpoints for those
    features fail with their own clear errors rather than a confusing 500."""
    model_ready = getattr(app.state, "model_ready", False)
    speaker_model_ready = getattr(app.state, "speaker_model_ready", False)
    reports_db_ready = getattr(app.state, "reports_db_ready", False)
    payload = {
        "status": "ok" if model_ready else "error",
        "model_ready": model_ready,
        "speaker_model_ready": speaker_model_ready,
        "reports_db_ready": reports_db_ready,
    }
    return JSONResponse(content=payload, status_code=200 if model_ready else 503)


def _require_speaker_model_ready() -> None:
    if not getattr(app.state, "speaker_model_ready", False):
        raise HTTPException(status_code=503, detail="Speaker verification model is not ready.")


def _save_upload_to_tempfile(file: UploadFile, suffix: str) -> str:
    """Stream the upload to a temp file, rejecting it (HTTP 413) if it
    exceeds MAX_UPLOAD_SIZE_BYTES partway through -- avoids buffering an
    arbitrarily large file just to reject it after the fact."""
    total_bytes = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        try:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large. Maximum allowed size is "
                            f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                tmp.write(chunk)
        except HTTPException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
    return tmp_path


# Note: a plain `def` route (not `async def`) so FastAPI/Starlette runs it in
# its worker thread pool automatically, instead of blocking the single event
# loop on the CPU-bound CNN inference inside predict_audio().
@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = _save_upload_to_tempfile(file, suffix)

    try:
        result = predict_audio(tmp_path)
    except Exception:
        logger.exception("prediction failed for uploaded file %r", file.filename)
        raise HTTPException(
            status_code=500,
            detail=GENERIC_PREDICTION_ERROR_DETAIL,
        ) from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PredictionResponse(
        prediction=result["predicted_label"],
        confidence=max(result["bonafide_probability"], result["spoof_probability"]),
        bonafide_probability=result["bonafide_probability"],
        spoof_probability=result["spoof_probability"],
    )


@app.post("/enroll_speaker", response_model=EnrollSpeakerResponse)
def enroll_speaker(name: str = Form(...), file: UploadFile = File(...)) -> EnrollSpeakerResponse:
    _require_speaker_model_ready()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = _save_upload_to_tempfile(file, suffix)
    try:
        result = sv.enroll_speaker(name, tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except sv.AudioDecodeError:
        raise HTTPException(status_code=400, detail=CORRUPT_AUDIO_ERROR_DETAIL) from None
    except Exception:
        logger.exception("speaker enrollment failed for name=%r file=%r", name, file.filename)
        raise HTTPException(status_code=500, detail=GENERIC_SPEAKER_ERROR_DETAIL) from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return EnrollSpeakerResponse(**result)


@app.post("/verify_speaker", response_model=VerifySpeakerResponse)
def verify_speaker(name: str = Form(...), file: UploadFile = File(...)) -> VerifySpeakerResponse:
    _require_speaker_model_ready()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = _save_upload_to_tempfile(file, suffix)
    try:
        result = sv.verify_speaker(name, tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except sv.SpeakerNotEnrolledError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except sv.AudioDecodeError:
        raise HTTPException(status_code=400, detail=CORRUPT_AUDIO_ERROR_DETAIL) from None
    except Exception:
        logger.exception("speaker verification failed for name=%r file=%r", name, file.filename)
        raise HTTPException(status_code=500, detail=GENERIC_SPEAKER_ERROR_DETAIL) from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return VerifySpeakerResponse(**result)


@app.post("/report", response_model=ReportResponse)
def create_report(payload: ReportCreateRequest) -> ReportResponse:
    try:
        result = reporting.create_report(
            report_type=payload.type,
            verdict=payload.verdict,
            confidence_score=payload.confidence_score,
            claimed_identity=payload.claimed_identity,
            user_notes=payload.user_notes,
        )
    except reporting.InvalidReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("report creation failed for payload=%r", payload)
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return ReportResponse(**result)


@app.get("/reports", response_model=list[ReportResponse])
def list_reports() -> list[ReportResponse]:
    try:
        results = reporting.list_reports()
    except Exception:
        logger.exception("listing reports failed")
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return [ReportResponse(**r) for r in results]


@app.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: int) -> ReportResponse:
    try:
        result = reporting.get_report(report_id)
    except reporting.ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception:
        logger.exception("fetching report id=%r failed", report_id)
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return ReportResponse(**result)


@app.delete("/reports/{report_id}")
def delete_report(report_id: int) -> JSONResponse:
    try:
        reporting.delete_report(report_id)
    except reporting.ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception:
        logger.exception("deleting report id=%r failed", report_id)
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return JSONResponse(content={"deleted": True, "id": report_id})


if __name__ == "__main__":
    import uvicorn

    # Many hosting platforms (Render, Railway, etc.) inject PORT and require
    # the app to bind to it.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
