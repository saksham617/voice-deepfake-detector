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
  CONTACTS_PATH,
  ENROLL_SPEAKER_PATH,
  MESSAGE_CHECK_PATH,
  PREDICT_PATH,
  REPORTS_PATH,
  USE_MOCK,
  VERIFY_SPEAKER_PATH,
} from "../config";
import {
  isBackendReportArray,
  type BackendReport,
} from "../types/report";
import {
  isMessageCheckResponse,
  type MessageCheckResponse,
} from "../types/message";
import {
  isPredictionResponse,
  type PredictionResponse,
} from "../types/prediction";
import {
  isBackendContactArray,
  isEnrollSpeakerResponse,
  isVerifySpeakerResponse,
  type BackendContact,
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

// --------------------------------------------------------------------------- reports

/** Fetch all reports (GET /reports), most-recent first. */
export async function fetchReports(signal?: AbortSignal): Promise<BackendReport[]> {
  if (USE_MOCK) return mockFetchReports(signal);
  return realFetchReports(signal);
}

/** Delete one report by id (DELETE /reports/{id}). */
export async function deleteReport(id: number, signal?: AbortSignal): Promise<void> {
  if (USE_MOCK) return mockDeleteReport(id, signal);
  return realDeleteReport(id, signal);
}

async function realFetchReports(signal?: AbortSignal): Promise<BackendReport[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${REPORTS_PATH}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("network", "Could not reach the server to load reports.");
  }
  if (!response.ok) {
    throw new ApiError("server", `The server returned an error loading reports (${response.status}).`);
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("invalid_response", "The server sent a response we couldn't read.");
  }
  if (!isBackendReportArray(data)) {
    throw new ApiError("invalid_response", "The server sent reports in an unexpected shape.");
  }
  return data;
}

async function realDeleteReport(id: number, signal?: AbortSignal): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${REPORTS_PATH}/${id}`, { method: "DELETE", signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("network", "Could not reach the server to delete the report.");
  }
  // 404 = already gone; treat as success so the row leaves the UI either way.
  if (!response.ok && response.status !== 404) {
    throw new ApiError("server", `The server returned an error deleting the report (${response.status}).`);
  }
}

/** Fetch every report against one phone number (GET /reports/number/{phoneNumber}), most-recent first. */
export async function fetchReportsByPhoneNumber(
  phoneNumber: string,
  signal?: AbortSignal,
): Promise<BackendReport[]> {
  if (USE_MOCK) return mockFetchReportsByPhoneNumber(phoneNumber, signal);
  return realFetchReportsByPhoneNumber(phoneNumber, signal);
}

async function realFetchReportsByPhoneNumber(
  phoneNumber: string,
  signal?: AbortSignal,
): Promise<BackendReport[]> {
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE_URL}${REPORTS_PATH}/number/${encodeURIComponent(phoneNumber)}`,
      { signal },
    );
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("network", "Could not reach the server to load this number's history.");
  }
  if (!response.ok) {
    throw new ApiError(
      "server",
      `The server returned an error loading this number's history (${response.status}).`,
    );
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("invalid_response", "The server sent a response we couldn't read.");
  }
  if (!isBackendReportArray(data)) {
    throw new ApiError("invalid_response", "The server sent reports in an unexpected shape.");
  }
  return data;
}

// --------------------------------------------------------------------------- contacts

/** List every enrolled contact (GET /contacts). */
export async function fetchContacts(signal?: AbortSignal): Promise<BackendContact[]> {
  if (USE_MOCK) return mockFetchContacts(signal);
  return realFetchContacts(signal);
}

async function realFetchContacts(signal?: AbortSignal): Promise<BackendContact[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${CONTACTS_PATH}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("network", "Could not reach the server to load contacts.");
  }
  if (!response.ok) {
    throw new ApiError("server", `The server returned an error loading contacts (${response.status}).`);
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("invalid_response", "The server sent a response we couldn't read.");
  }
  if (!isBackendContactArray(data)) {
    throw new ApiError("invalid_response", "The server sent contacts in an unexpected shape.");
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

  mockContactsStore = [
    ...mockContactsStore,
    { contact_id, name, enrolled_at: new Date().toISOString() },
  ];

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

// TODO(remove-when-backend-ready): sample reports for USE_MOCK mode. A module-level array so
// mock deletes persist within a session (until reload), mirroring real backend behaviour.
let mockReportsStore: BackendReport[] = [
  {
    id: 1, timestamp: "2026-09-20T09:12:00.000Z", type: "voice", verdict: "spoof",
    confidence_score: 0.93, claimed_identity: "Bank representative", phone_number: "+1 202-555-0143",
    user_notes: "Live call: 42s, 11 windows scored, peak HIGH. Windows per level: LOW:2  MEDIUM:3  HIGH:6.",
    status: "open",
  },
  {
    id: 2, timestamp: "2026-09-19T18:40:00.000Z", type: "message", verdict: "suspicious",
    confidence_score: 0.71, claimed_identity: null, phone_number: "+1 202-555-0143",
    user_notes: "SMS phishing link disguised as a bank verification request.", status: "reviewed",
  },
  {
    id: 3, timestamp: "2026-09-18T11:05:00.000Z", type: "voice", verdict: "bonafide",
    confidence_score: 0.12, claimed_identity: null, phone_number: null,
    user_notes: "Live call: 30s, 8 windows scored, peak NONE. Windows per level: NONE:8.", status: "reviewed",
  },
];

async function mockFetchReports(signal?: AbortSignal): Promise<BackendReport[]> {
  await delay(500, signal);
  return mockReportsStore.slice();
}

async function mockFetchReportsByPhoneNumber(
  phoneNumber: string,
  signal?: AbortSignal,
): Promise<BackendReport[]> {
  await delay(500, signal);
  return mockReportsStore.filter((r) => r.phone_number === phoneNumber);
}

// TODO(remove-when-backend-ready): sample contacts for USE_MOCK mode, seeded the same as the
// real backend's demo state would be. A module-level array so mock enrollments persist within
// a session (until reload), mirroring mockReportsStore above.
let mockContactsStore: BackendContact[] = [
  { contact_id: "aditi_rao", name: "Aditi Rao", enrolled_at: "2026-09-02T10:00:00.000Z" },
  { contact_id: "rahul_mehta", name: "Rahul Mehta", enrolled_at: "2026-09-08T14:30:00.000Z" },
];

async function mockFetchContacts(signal?: AbortSignal): Promise<BackendContact[]> {
  await delay(400, signal);
  return mockContactsStore.slice();
}

async function mockDeleteReport(id: number, signal?: AbortSignal): Promise<void> {
  await delay(300, signal);
  mockReportsStore = mockReportsStore.filter((r) => r.id !== id);
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
