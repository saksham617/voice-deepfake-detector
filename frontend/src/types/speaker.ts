/**
 * Shared types for the speaker-enrollment/verification API contract.
 *
 * Agreed contract with the backend team (future endpoints):
 *   POST /enroll_speaker  (multipart/form-data, fields "name" + "file")
 *   -> { "contact_id": string, "name": string }
 *
 *   POST /verify_speaker  (multipart/form-data, fields "contact_id" + "file")
 *   -> { "match": boolean, "similarity": number }
 */

export interface EnrollSpeakerResponse {
  contact_id: string;
  name: string;
}

export interface VerifySpeakerResponse {
  /** Whether the sample is judged to belong to the enrolled contact. */
  match: boolean;
  /** Similarity score in [0, 1] between the sample and the enrolled voiceprint. */
  similarity: number;
}

function isProbability(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
  );
}

/** Type guard used to validate an untrusted backend/JSON response at runtime. */
export function isEnrollSpeakerResponse(value: unknown): value is EnrollSpeakerResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.contact_id === "string" && typeof v.name === "string";
}

/** Type guard used to validate an untrusted backend/JSON response at runtime. */
export function isVerifySpeakerResponse(value: unknown): value is VerifySpeakerResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return typeof v.match === "boolean" && isProbability(v.similarity);
}
