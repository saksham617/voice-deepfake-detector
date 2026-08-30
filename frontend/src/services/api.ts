/**
 * API service — the ONLY place that talks to the backend.
 *
 * UI components and hooks call `predictAudio(file)` and receive a typed
 * `PredictionResponse`. They never touch `fetch`, URLs, or FormData directly.
 *
 * Swapping mock -> real backend is a single flag (`USE_MOCK` in config.ts);
 * no UI changes are required.
 */

import {
  API_BASE_URL,
  PREDICT_PATH,
  USE_MOCK,
} from "../config";
import {
  isPredictionResponse,
  type PredictionResponse,
} from "../types/prediction";

/** Categorised error so the UI can show friendly, non-technical messages. */
export type ApiErrorKind =
  | "network" // couldn't reach the server
  | "server" // server returned a non-2xx status
  | "invalid_response" // server replied but not in the agreed shape
  | "unknown";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  constructor(kind: ApiErrorKind, message: string) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
  }
}

/**
 * Submit an audio file for spoof/bonafide analysis.
 *
 * @param file  The user-selected (or recorded) audio file.
 * @param signal  Optional AbortSignal to cancel an in-flight request.
 */
export async function predictAudio(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  if (USE_MOCK) {
    return mockPredict(file, signal);
  }
  return realPredict(file, signal);
}

// ---------------------------------------------------------------------------
// Real backend call
// ---------------------------------------------------------------------------

async function realPredict(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${PREDICT_PATH}`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (err) {
    // Aborts are re-thrown so callers can distinguish user cancellation.
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(
      "network",
      "Could not reach the analysis server. Please check your connection and try again.",
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "server",
      `The analysis server responded with an error (${response.status}). Please try again.`,
    );
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      "invalid_response",
      "The server sent a response we couldn't read. Please try again.",
    );
  }

  if (!isPredictionResponse(data)) {
    throw new ApiError(
      "invalid_response",
      "The server sent an unexpected result. Please try again.",
    );
  }

  return data;
}

// ---------------------------------------------------------------------------
// Mock backend (development only)
//
// TODO(remove-when-backend-ready): Delete this block and the USE_MOCK branch
// above once the FastAPI /predict endpoint is live. The real path already
// exists and is exercised by flipping VITE_USE_MOCK=false.
// ---------------------------------------------------------------------------

async function mockPredict(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  // Simulate realistic processing latency.
  await delay(1600, signal);

  // Deterministic-ish but varied result so the demo isn't hardcoded:
  // derive a pseudo-random outcome from the file so repeated uploads of the
  // same file are stable, but different files can differ.
  const seed = file.size + file.name.length;
  const prediction = seed % 2 === 0 ? "spoof" : "bonafide";
  const confidence = 0.82 + ((seed % 17) / 100); // 0.82 .. 0.98

  return {
    prediction,
    confidence: Math.min(0.99, Number(confidence.toFixed(2))),
  };
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const id = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(id);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}
