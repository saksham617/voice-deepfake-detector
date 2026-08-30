# Minimal production image for the FastAPI inference backend.
# Does NOT include the raw dataset, training scripts, notebooks, or the
# frontend -- those are excluded via .dockerignore. The frontend is deployed
# separately as a static build (see DEPLOYMENT.md).

FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first, from PyTorch's dedicated CPU wheel index --
# the default PyPI index bundles multi-GB CUDA runtime packages this app
# never uses (inference here is CPU-only, one file at a time).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

# Then the rest of the backend's runtime dependencies from the normal index.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Only the source actually needed to serve predictions -- not the full
# src/ tree (no training scripts, evaluation, or SVM/RF baselines).
COPY src/data/audio_io.py src/data/asvspoof_cm_loader.py src/data/
COPY src/features/spectrogram.py src/features/
COPY src/models/cnn_baseline.py src/models/inference.py src/models/
COPY backend/main.py backend/main.py

# The trained checkpoint (committed to git as a deliberate exception --
# see .gitignore).
COPY data/processed/models/cnn_baseline.pt data/processed/models/cnn_baseline.pt

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
