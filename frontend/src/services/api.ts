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
  ENROLL_SPEAKER_PATH,
  MESSAGE_CHECK_PATH,
  PREDICT_PATH,
  USE_MOCK,
  VERIFY_SPEAKER_PATH,
} from "../config";
import {
  isMessageCheckResponse,
  type MessageCheckResponse,
} from "../types/message";
import {
  isPredictionResponse,
  type PredictionResponse,
} from "../types/prediction";
import {
  isEnrollSpeakerResponse,
  isVerifySpeakerResponse,
  type EnrollSpeakerResponse,
  type VerifySpeakerResponse,
} from "../types/speaker";

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

/**
 * Submit a text message for phishing/scam analysis.
 *
 * @param text  The message text to check.
 * @param signal  Optional AbortSignal to cancel an in-flight request.
 */
export async function checkMessage(
  text: string,
  signal?: AbortSignal,
): Promise<MessageCheckResponse> {
  if (USE_MOCK) {
    return mockCheckMessage(text, signal);
  }
  return realCheckMessage(text, signal);
}

/**
 * Enroll a new contact's voiceprint.
 *
 * @param name  The contact's display name.
 * @param file  An audio sample (upload or recording) of the contact's voice.
 * @param signal  Optional AbortSignal to cancel an in-flight request.
 */
export async function enrollSpeaker(
  name: string,
  file: File,
  signal?: AbortSignal,
): Promise<EnrollSpeakerResponse> {
  if (USE_MOCK) {
    return mockEnrollSpeaker(name, file, signal);
  }
  return realEnrollSpeaker(name, file, signal);
}

/**
 * Verify an audio sample against a previously enrolled contact's voiceprint.
 *
 * @param contactId  The enrolled contact to verify against.
 * @param file  An audio sample (upload or recording) to check.
 * @param signal  Optional AbortSignal to cancel an in-flight request.
 */
export async function verifySpeaker(
  contactId: string,
  file: File,
  signal?: AbortSignal,
): Promise<VerifySpeakerResponse> {
  if (USE_MOCK) {
    return mockVerifySpeaker(contactId, file, signal);
  }
  return realVerifySpeaker(contactId, file, signal);
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

async function realCheckMessage(
  text: string,
  signal?: AbortSignal,
): Promise<MessageCheckResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${MESSAGE_CHECK_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal,
    });
  } catch (err) {
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

  if (!isMessageCheckResponse(data)) {
    throw new ApiError(
      "invalid_response",
      "The server sent an unexpected result. Please try again.",
    );
  }

  return data;
}

async function realEnrollSpeaker(
  name: string,
  file: File,
  signal?: AbortSignal,
): Promise<EnrollSpeakerResponse> {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${ENROLL_SPEAKER_PATH}`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (err) {
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

  if (!isEnrollSpeakerResponse(data)) {
    throw new ApiError(
      "invalid_response",
      "The server sent an unexpected result. Please try again.",
    );
  }

  return data;
}

async function realVerifySpeaker(
  contactId: string,
  file: File,
  signal?: AbortSignal,
): Promise<VerifySpeakerResponse> {
  const formData = new FormData();
  formData.append("contact_id", contactId);
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${VERIFY_SPEAKER_PATH}`, {
      method: "POST",
      body: formData,
      signal,
    });
  } catch (err) {
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

  if (!isVerifySpeakerResponse(data)) {
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
  const confidence = Math.min(0.99, Number((0.82 + (seed % 17) / 100).toFixed(2))); // 0.82 .. 0.98

  const spoofProbability = prediction === "spoof" ? confidence : 1 - confidence;
  const bonafideProbability = 1 - spoofProbability;

  return {
    prediction,
    confidence,
    bonafide_probability: Number(bonafideProbability.toFixed(2)),
    spoof_probability: Number(spoofProbability.toFixed(2)),
  };
}

// TODO(remove-when-backend-ready): Delete this block and the USE_MOCK branch
// above once the /check_message endpoint is live.
const SUSPICIOUS_PHRASES = [
  "verify your account",
  "act now",
  "click here",
  "urgent",
  "suspended",
  "confirm your",
  "limited time",
  "wire transfer",
  "gift card",
  "social security",
  "bank details",
  "one-time password",
  "otp",
  "password",
  "prize",
  "you've won",
];

async function mockCheckMessage(
  text: string,
  signal?: AbortSignal,
): Promise<MessageCheckResponse> {
  await delay(1200, signal);

  const lower = text.toLowerCase();
  const flagged = SUSPICIOUS_PHRASES.filter((phrase) => lower.includes(phrase));

  // Deterministic-ish score: more flagged phrases -> higher probability, with
  // a little variation from the text itself so the demo isn't hardcoded.
  const seed = text.length % 15;
  const phishingProbability =
    flagged.length > 0
      ? Math.min(0.97, 0.55 + flagged.length * 0.14 + seed / 100)
      : Math.min(0.22, seed / 100);

  return {
    phishing_probability: Number(phishingProbability.toFixed(2)),
    flagged_phrases: flagged,
  };
}

// TODO(remove-when-backend-ready): Delete this block and the USE_MOCK branch
// above once the /enroll_speaker endpoint is live.
async function mockEnrollSpeaker(
  name: string,
  file: File,
  signal?: AbortSignal,
): Promise<EnrollSpeakerResponse> {
  await delay(1400, signal);

  const seed = file.size + file.name.length + name.length;
  const contact_id = `contact_${Date.now()}_${seed.toString(36)}`;

  return { contact_id, name };
}

// TODO(remove-when-backend-ready): Delete this block and the USE_MOCK branch
// above once the /verify_speaker endpoint is live.
async function mockVerifySpeaker(
  contactId: string,
  file: File,
  signal?: AbortSignal,
): Promise<VerifySpeakerResponse> {
  await delay(1500, signal);

  // Deterministic-ish but varied result, same approach as mockPredict.
  const seed = (file.size + file.name.length + contactId.length) % 100;
  const match = seed % 3 !== 0; // ~2/3 of samples match, for a believable demo
  const similarity = match
    ? Number((0.8 + (seed % 18) / 100).toFixed(2))
    : Number((0.2 + (seed % 30) / 100).toFixed(2));

  return { match, similarity };
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
