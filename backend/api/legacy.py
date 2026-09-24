"""Primary detection API: CNN spoof detection + ECAPA-TDNN speaker
verification + reporting.

Exposes (all unprefixed, at the app root -- this is the contract the
deployed React frontend depends on, per frontend/src/types/prediction.ts):

    POST   /predict          multipart "file" -> bonafide/spoof + confidence
    GET    /contacts          list enrolled contacts (id, name, enrolled_at)
    POST   /enroll_speaker    multipart "name" + "file" -> enroll a voiceprint
    POST   /verify_speaker    multipart "contact_id" + "file" -> match against one
    POST   /check_message     json "text" -> safe/suspicious + confidence
    POST   /report            create a report
    GET    /reports           list reports
    GET    /reports/number/{phone_number}   list reports against one number
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
import os
import subprocess
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

# What the audio loader (librosa/soundfile) can decode natively, no
# transcoding needed -- WAV always, FLAC and MP3 via libsndfile 1.1+.
NATIVE_EXTENSIONS = {".wav", ".flac", ".mp3"}

# Formats the frontend also offers (file picker + browser MediaRecorder
# output) that libsndfile can't decode directly -- see
# frontend/src/config.ts ACCEPTED_AUDIO_EXTENSIONS. Transcoded to WAV via
# ffmpeg (_transcode_to_wav) before being handed to the loader.
TRANSCODE_EXTENSIONS = {".ogg", ".webm", ".m4a", ".mp4"}

ALLOWED_EXTENSIONS = NATIVE_EXTENSIONS | TRANSCODE_EXTENSIONS

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
    contact_id: str
    name: str
    duration_sec: float
    short_clip: bool
    enrolled_at: str


class VerifySpeakerResponse(BaseModel):
    contact_id: str
    name: str
    match: bool
    similarity: float
    is_match: bool
    threshold: float
    short_clip: bool


class ContactResponse(BaseModel):
    contact_id: str
    name: str
    enrolled_at: str


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
    phone_number: str | None = None
    status: str = "open"


class ReportResponse(BaseModel):
    id: int
    timestamp: str
    type: str
    verdict: str
    confidence_score: float
    claimed_identity: str | None = None
    user_notes: str | None = None
    phone_number: str | None = None
    status: str = "open"


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


class TranscodeError(Exception):
    """The upload couldn't be converted to WAV -- corrupt/truncated input,
    not a server misconfiguration (see _transcode_to_wav)."""


def _transcode_to_wav(src_path: str) -> str:
    """Convert an upload in a non-natively-supported format (see
    TRANSCODE_EXTENSIONS) to a 16 kHz mono WAV via ffmpeg, so the rest of
    the pipeline never has to care what format the browser sent.

    Raises TranscodeError for bad/corrupt input (mapped to the same 400 as
    a native-format decode failure); lets FileNotFoundError propagate
    uncaught if the ffmpeg binary itself is missing, since that's a server
    deployment problem, not a bad upload.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as exc:
        Path(wav_path).unlink(missing_ok=True)
        raise TranscodeError(exc.stderr.decode("utf-8", errors="replace")) from exc
    except subprocess.TimeoutExpired as exc:
        Path(wav_path).unlink(missing_ok=True)
        raise TranscodeError("transcoding timed out") from exc
    return wav_path


def _receive_audio_upload(file: UploadFile) -> str:
    """Validate, save, and (if needed) transcode an uploaded audio file.

    Returns a single path the audio loader can read natively -- the caller
    only ever has one file to clean up, regardless of whether a transcode
    happened."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_path = _save_upload_to_tempfile(file, suffix)
    if suffix not in TRANSCODE_EXTENSIONS:
        return tmp_path

    try:
        return _transcode_to_wav(tmp_path)
    except TranscodeError:
        raise HTTPException(status_code=400, detail=CORRUPT_AUDIO_ERROR_DETAIL) from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Note: a plain `def` route (not `async def`) so FastAPI/Starlette runs it in
# its worker thread pool automatically, instead of blocking the single event
# loop on the CPU-bound CNN inference inside predict_audio().
@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)) -> PredictionResponse:
    tmp_path = _receive_audio_upload(file)

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


@router.get("/contacts", response_model=list[ContactResponse])
def list_contacts(request: Request) -> list[ContactResponse]:
    _require_speaker_model_ready(request)
    return [ContactResponse(**c) for c in sv.list_enrolled_speakers()]


@router.post("/enroll_speaker", response_model=EnrollSpeakerResponse)
def enroll_speaker(
    request: Request, name: str = Form(...), file: UploadFile = File(...)
) -> EnrollSpeakerResponse:
    _require_speaker_model_ready(request)

    tmp_path = _receive_audio_upload(file)
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


# Note: the form field is "contact_id" (not "name") to match the frontend's
# contract (frontend/src/services/api.ts realVerifySpeaker) -- a contact_id
# is just the sanitized-name slug enroll_speaker already keys storage by
# (see sv.contact_id_for_name), so passing it straight through to
# sv.verify_speaker as the lookup key works unchanged.
@router.post("/verify_speaker", response_model=VerifySpeakerResponse)
def verify_speaker(
    request: Request, contact_id: str = Form(...), file: UploadFile = File(...)
) -> VerifySpeakerResponse:
    _require_speaker_model_ready(request)

    tmp_path = _receive_audio_upload(file)
    try:
        result = sv.verify_speaker(contact_id, tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except sv.SpeakerNotEnrolledError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except sv.AudioDecodeError:
        raise HTTPException(status_code=400, detail=CORRUPT_AUDIO_ERROR_DETAIL) from None
    except Exception:
        logger.exception(
            "speaker verification failed for contact_id=%r file=%r", contact_id, file.filename
        )
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
            phone_number=payload.phone_number,
            status=payload.status,
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


# Registered before /reports/{report_id} so FastAPI's literal "number"
# segment isn't swallowed by that route's {report_id}: int converter --
# it wouldn't match a phone number anyway, but keeping this first is the
# same defensive ordering FastAPI's own docs recommend for path vs literal.
@router.get("/reports/number/{phone_number}", response_model=list[ReportResponse])
def list_reports_for_number(phone_number: str) -> list[ReportResponse]:
    try:
        results = reporting.list_reports_by_phone_number(phone_number)
    except Exception:
        logger.exception("listing reports for phone_number=%r failed", phone_number)
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
