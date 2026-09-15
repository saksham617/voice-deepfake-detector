/**
 * Shared types for the message-check API contract.
 *
 * Agreed contract with the backend team (future endpoint):
 *   POST /check_message  (application/json, { "text": string })
 *   -> {
 *        "phishing_probability": number,
 *        "flagged_phrases": string[]
 *      }
 */

export interface MessageCheckResponse {
  /** Probability in [0, 1] that the message is a phishing/scam attempt. */
  phishing_probability: number;
  /** Substrings of the input text the model flagged as suspicious. */
  flagged_phrases: string[];
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
export function isMessageCheckResponse(value: unknown): value is MessageCheckResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    isProbability(v.phishing_probability) &&
    Array.isArray(v.flagged_phrases) &&
    v.flagged_phrases.every((phrase) => typeof phrase === "string")
  );
}
