"""Primary detection API: CNN spoof detection + ECAPA-TDNN speaker
verification + reporting.

Exposes (all unprefixed, at the app root -- this is the contract the
deployed React frontend depends on, per frontend/src/types/prediction.ts):

    POST   /predict          multipart "file" -> bonafide/spoof + confidence
    POST   /enroll_speaker    multipart "name" + "file" -> enroll a voiceprint
    POST   /verify_speaker    multipart "name" + "file" -> match against one
    POST   /check_message     json "text" -> safe/suspicious + confidence
    POST   /report            create a report
    GET    /reports           list reports
    GET    /reports/{id}      fetch one report
    DELETE /reports/{id}      delete one report
    GET    /health            liveness + per-subsystem readiness

Reuses the exact trained-model inference pipeline used everywhere else in
this project (src.models.inference.predict_audio) rather than reimplementing
audio loading, spectrogram preprocessing, or model loading here. Does not
modify the trained model or the raw dataset.

The live-call streaming subsystem (AASIST over WebRTC) is a separate API
under /live-call -- see backend/api/rest.py and backend/api/websocket.py.
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.models import message_detection as md
from src.models import reporting
from src.models import speaker_verification as sv
from src.models.inference import predict_audio

logger = logging.getLogger("backend")

router = APIRouter()

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
GENERIC_MESSAGE_ERROR_DETAIL = "Internal error while processing the message."
GENERIC_REPORT_ERROR_DETAIL = "Internal error while processing the report."
CORRUPT_AUDIO_ERROR_DETAIL = (
    "Could not read the uploaded audio file. It may be corrupted or in an unsupported format."
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


class CheckMessageRequest(BaseModel):
    text: str


class CheckMessageResponse(BaseModel):
    verdict: str
    confidence: float
    spam_probability: float


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


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Reports whether the models actually loaded at startup, not just that
    the process is running -- so a load balancer/host treats a failed model
    load as unhealthy rather than routing traffic to a broken instance.

    Overall `status`/status-code tracks the primary spoof-detection model
    only (the app's core feature); `speaker_model_ready`,
    `message_model_ready`, `reports_db_ready`, and `live_call_ready` are
    reported separately since the endpoints for those features fail with
    their own clear errors rather than a confusing 500."""
    state = request.app.state
    model_ready = getattr(state, "model_ready", False)
    payload = {
        "status": "ok" if model_ready else "error",
        "model_ready": model_ready,
        "speaker_model_ready": getattr(state, "speaker_model_ready", False),
        "message_model_ready": getattr(state, "message_model_ready", False),
        "reports_db_ready": getattr(state, "reports_db_ready", False),
        "live_call_ready": getattr(state, "live_call_ready", False),
    }
    return JSONResponse(content=payload, status_code=200 if model_ready else 503)


def _require_speaker_model_ready(request: Request) -> None:
    if not getattr(request.app.state, "speaker_model_ready", False):
        raise HTTPException(status_code=503, detail="Speaker verification model is not ready.")


def _require_message_model_ready(request: Request) -> None:
    if not getattr(request.app.state, "message_model_ready", False):
        raise HTTPException(status_code=503, detail="Message detection model is not ready.")


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
@router.post("/predict", response_model=PredictionResponse)
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


@router.post("/enroll_speaker", response_model=EnrollSpeakerResponse)
def enroll_speaker(
    request: Request, name: str = Form(...), file: UploadFile = File(...)
) -> EnrollSpeakerResponse:
    _require_speaker_model_ready(request)

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


@router.post("/verify_speaker", response_model=VerifySpeakerResponse)
def verify_speaker(
    request: Request, name: str = Form(...), file: UploadFile = File(...)
) -> VerifySpeakerResponse:
    _require_speaker_model_ready(request)

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


@router.post("/check_message", response_model=CheckMessageResponse)
def check_message(request: Request, payload: CheckMessageRequest) -> CheckMessageResponse:
    _require_message_model_ready(request)

    try:
        result = md.classify_message(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("message classification failed for text of length %d", len(payload.text))
        raise HTTPException(status_code=500, detail=GENERIC_MESSAGE_ERROR_DETAIL) from None

    return CheckMessageResponse(**result)


@router.post("/report", response_model=ReportResponse)
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


@router.get("/reports", response_model=list[ReportResponse])
def list_reports() -> list[ReportResponse]:
    try:
        results = reporting.list_reports()
    except Exception:
        logger.exception("listing reports failed")
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return [ReportResponse(**r) for r in results]


@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: int) -> ReportResponse:
    try:
        result = reporting.get_report(report_id)
    except reporting.ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception:
        logger.exception("fetching report id=%r failed", report_id)
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return ReportResponse(**result)


@router.delete("/reports/{report_id}")
def delete_report(report_id: int) -> JSONResponse:
    try:
        reporting.delete_report(report_id)
    except reporting.ReportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception:
        logger.exception("deleting report id=%r failed", report_id)
        raise HTTPException(status_code=500, detail=GENERIC_REPORT_ERROR_DETAIL) from None

    return JSONResponse(content={"deleted": True, "id": report_id})
