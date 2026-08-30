/**
 * Central runtime configuration.
 *
 * These values are read from Vite env variables (see `.env.example`) with
 * safe defaults so the app runs out-of-the-box in mock mode.
 *
 * To connect the real backend:
 *   1. Copy `.env.example` to `.env`
 *   2. Set VITE_USE_MOCK=false
 *   3. Point VITE_API_BASE_URL at the FastAPI server
 */

/** Base URL of the FastAPI backend. */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * When true, the app uses a local mock prediction instead of the network.
 * Defaults to true so the frontend is fully usable before the backend exists.
 */
export const USE_MOCK: boolean =
  (import.meta.env.VITE_USE_MOCK ?? "true") !== "false";

/** The prediction endpoint path. */
export const PREDICT_PATH = "/predict";

/**
 * Audio formats the UI accepts. Kept in sync with what the backend supports.
 * Update this list when the backend's supported formats are confirmed.
 */
export const ACCEPTED_AUDIO_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/mpeg", // .mp3
  "audio/mp3",
  "audio/flac",
  "audio/x-flac",
  "audio/ogg",
  "audio/webm",
  "audio/mp4",
  "audio/x-m4a",
] as const;

/** File-extension fallback (some browsers report empty/odd MIME types). */
export const ACCEPTED_AUDIO_EXTENSIONS = [
  ".wav",
  ".mp3",
  ".flac",
  ".ogg",
  ".webm",
  ".m4a",
  ".mp4",
] as const;

/** Maximum upload size (bytes). 25 MB is generous for short clips. */
export const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024;
