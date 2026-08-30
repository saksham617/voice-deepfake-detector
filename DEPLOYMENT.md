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

### Model artifacts

The project contains three model implementations: CNN, SVM, and Random
Forest. Only the CNN has a persisted model artifact:
`data/processed/models/cnn_baseline.pt`. The deployed FastAPI inference
service uses this CNN checkpoint.

The SVM and Random Forest models are not stored as standalone model files.
They are retrained on demand from the cached MFCC feature matrices in
`data/processed/features/*.npz`. This is intentional because training from
the cached features is fast enough that persistent SVM/RF artifacts are
unnecessary.

## Frontend (static build)

This is a monorepo: the only Vite/React project lives under `frontend/`,
the repo root has no `package.json`. Two ways to deploy:

### Vercel (git-connected)

A root-level `vercel.json` tells Vercel how to build the `frontend/`
subdirectory regardless of the project's "Root Directory" dashboard setting:
```json
{
  "installCommand": "cd frontend && npm install",
  "buildCommand": "cd frontend && npm run build",
  "outputDirectory": "frontend/dist"
}
```
(If the project's Root Directory is instead set to `frontend` in the Vercel
dashboard, this file is simply not read from there and Vercel's normal
zero-config Vite detection applies -- either configuration works.)

Vercel builds in its own cloud environment and never sees your local
`.env`/`.env.production` files (both gitignored, and irrelevant to a
git-connected build anyway). Set these in the Vercel project's
**Settings -> Environment Variables** (Production) instead:
- `VITE_API_BASE_URL` -- the deployed backend's public URL (e.g. the Render URL)
- `VITE_USE_MOCK` -- `false`

Vite bakes both into the bundle at build time, so they must be set before
triggering a build, not after.

### Manual build + any static host

```
cd frontend
cp .env.production.example .env.production   # set VITE_API_BASE_URL to your backend's public URL
npm run build
```

Deploy the resulting `dist/` folder to any static host (Netlify, GitHub
Pages, S3+CloudFront, or serve it via nginx).

## Local production-style check (no Docker required)

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (or reuse an existing venv)
2. `CORS_ORIGINS=http://localhost:4173 uvicorn backend.main:app --host 0.0.0.0 --port 8000` (no `--reload`)
3. `cd frontend && npm run build && npm run preview -- --port 4173`
4. Open the preview URL and exercise upload -> analyze -> result/error.
