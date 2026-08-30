# VoiceGuard — Frontend

The user interface for the Voice Deepfake Detector (SIH project). It lets a user
upload an audio recording and see whether the voice is predicted **bonafide**
(genuine) or **spoof** (AI-generated / cloned).

> This is the **frontend only**. No ML logic lives here. It talks to a FastAPI
> backend via a single agreed contract (see _API contract_ below).

## Tech stack

- **React 18** + **TypeScript**
- **Vite** (dev server / build)
- **Tailwind CSS** (styling)
- No UI or HTTP libraries — native `fetch`, hand-built components.

## Getting started

```bash
cd frontend
npm install
npm run dev
```

Then open the printed URL (default http://localhost:5173).

The app runs in **mock mode** by default, so it works fully without a backend.

### Available scripts

| Script            | What it does                          |
| ----------------- | ------------------------------------- |
| `npm run dev`     | Start the dev server                  |
| `npm run build`   | Type-check and build for production    |
| `npm run preview` | Preview the production build locally  |

## Configuration

Copy `.env.example` to `.env` and adjust:

```
VITE_API_BASE_URL=http://localhost:8000   # FastAPI base URL
VITE_USE_MOCK=true                        # set false to use the real backend
```

Defaults (no `.env` needed): mock mode ON, backend at `http://localhost:8000`.

## Switching from mock to the real backend

The mock lives **only** in `src/services/api.ts`. To go live:

1. Set `VITE_USE_MOCK=false` in `.env`.
2. Point `VITE_API_BASE_URL` at the running FastAPI server.

No UI changes are required. When the backend is stable, the mock block in
`src/services/api.ts` (marked with `TODO(remove-when-backend-ready)`) can be
deleted.

## API contract

Agreed with the backend team:

```
POST /predict
Content-Type: multipart/form-data
Body: file=<audio file>

200 OK
{ "prediction": "spoof" | "bonafide", "confidence": 0.92 }
```

`confidence` is a number in `[0, 1]`. The UI labels it simply as **Confidence**
— it does not claim to be the probability that the model is correct.

## Project structure

```
frontend/
├── index.html
├── src/
│   ├── components/      # Presentational UI pieces (upload, result, loading…)
│   ├── pages/
│   │   └── AnalyzerPage.tsx   # Composes the upload → analyze → result flow
│   ├── hooks/
│   │   └── useAnalyzer.ts     # State machine: idle→selected→analyzing→result/error
│   ├── services/
│   │   └── api.ts            # The ONLY place that talks to the backend
│   ├── types/
│   │   └── prediction.ts     # API contract types + runtime validation
│   ├── utils/
│   │   └── audioFile.ts      # Client-side file validation
│   ├── config.ts            # Env flags: USE_MOCK, API_BASE_URL, accepted types
│   ├── App.tsx
│   └── main.tsx
└── ...
```

Architecture: **UI → `useAnalyzer` hook → `api` service → FastAPI → ML pipeline.**
Components never call `fetch` directly.

## Roadmap (not yet implemented)

- Microphone recording (UI is scaffolded behind the "Record" tab).
- Richer analysis panel (model info, ensemble breakdown, MFCC/spectrogram
  visualizations, explainability) — the layout leaves room for these.
