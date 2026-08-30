# Deployment

Two independent pieces: a FastAPI backend (Docker) serving the trained CNN,
and a static frontend (Vite build) that calls it. No database, no queue --
single-file inference, so this is deliberately the simplest architecture
that works.

## Backend (Docker)

```
docker build -t voice-deepfake-backend .
docker run -p 8000:8000 \
  -e CORS_ORIGINS=https://your-frontend-host.example.com \
  voice-deepfake-backend
```

Environment variables:
- `PORT` -- defaults to 8000. Most hosts (Render, Railway, etc.) inject this automatically.
- `CORS_ORIGINS` -- comma-separated allowed origins. Defaults to `*` (fine for local dev); set this explicitly to your deployed frontend's URL in production.

The image installs a CPU-only torch build (see Dockerfile) and only the
backend's runtime dependencies (`backend/requirements.txt`) -- not
scikit-learn, matplotlib, or the training/evaluation code, which the
inference path never imports.

The trained checkpoint (`data/processed/models/cnn_baseline.pt`, ~13MB) is
committed to git as a deliberate, narrow exception to the usual
"dataset-derived artifacts stay out of git" rule (see `.gitignore`), so a
plain `git clone` + `docker build` has everything it needs -- no external
storage or download step required.

Health check: `GET /health` -> `200 {"status": "ok", "model_ready": true}` if the
CNN checkpoint loaded successfully at startup, or `503 {"status": "error", "model_ready": false}`
if it didn't -- so a load balancer/host correctly treats a failed model load
as unhealthy rather than routing traffic to a broken instance.

## Frontend (static build)

```
cd frontend
cp .env.production.example .env.production   # set VITE_API_BASE_URL to your backend's public URL
npm run build
```

Deploy the resulting `dist/` folder to any static host (Vercel, Netlify,
GitHub Pages, S3+CloudFront, or serve it via nginx). Vite bakes
`VITE_API_BASE_URL`/`VITE_USE_MOCK` into the bundle at build time, so they
must be set before building, not after.

## Local production-style check (no Docker required)

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (or reuse an existing venv)
2. `CORS_ORIGINS=http://localhost:4173 uvicorn backend.main:app --host 0.0.0.0 --port 8000` (no `--reload`)
3. `cd frontend && npm run build && npm run preview -- --port 4173`
4. Open the preview URL and exercise upload -> analyze -> result/error.
