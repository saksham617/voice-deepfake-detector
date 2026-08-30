/**
 * Shared types for the prediction API contract.
 *
 * Agreed contract with the backend team:
 *   POST /predict  (multipart/form-data, field name "file")
 *   -> { "prediction": "spoof" | "bonafide", "confidence": number }
 */

export type PredictionLabel = "bonafide" | "spoof";

export interface PredictionResponse {
  prediction: PredictionLabel;
  /** Model confidence in [0, 1]. Labeled simply as "Confidence" in the UI. */
  confidence: number;
}

/** Type guard used to validate an untrusted backend/JSON response at runtime. */
export function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  const validLabel = v.prediction === "bonafide" || v.prediction === "spoof";
  const validConfidence =
    typeof v.confidence === "number" &&
    Number.isFinite(v.confidence) &&
    v.confidence >= 0 &&
    v.confidence <= 1;
  return validLabel && validConfidence;
}
