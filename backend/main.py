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

import os
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.inference import predict_audio

# Matches what the CNN was trained/evaluated on (audio_io.py reads via
# soundfile, which natively supports WAV/FLAC). Other formats the frontend
# accepts (mp3, ogg, webm, m4a) are not yet supported server-side.
ALLOWED_EXTENSIONS = {".wav", ".flac"}

# Comma-separated list of allowed origins, e.g. "https://myapp.vercel.app".
# Defaults to "*" so local dev (Vite on a different port) keeps working
# unconfigured; set explicitly in production instead of relying on the default.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = (
    ["*"] if _cors_origins_env == "*" else [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
)

app = FastAPI(title="Voice Deepfake Detector API")

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_audio(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PredictionResponse(
        prediction=result["predicted_label"],
        confidence=max(result["bonafide_probability"], result["spoof_probability"]),
        bonafide_probability=result["bonafide_probability"],
        spoof_probability=result["spoof_probability"],
    )


if __name__ == "__main__":
    import uvicorn

    # Many hosting platforms (Render, Railway, etc.) inject PORT and require
    # the app to bind to it.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
