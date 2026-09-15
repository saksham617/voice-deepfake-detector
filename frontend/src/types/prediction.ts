/**
 * Shared types for the prediction API contract.
 *
 * Agreed contract with the backend team (backend/api/legacy.py):
 *   POST /predict  (multipart/form-data, field name "file")
 *   -> {
 *        "prediction": "spoof" | "bonafide",
 *        "confidence": number,
 *        "bonafide_probability": number,
 *        "spoof_probability": number
 *      }
 */

export type PredictionLabel = "bonafide" | "spoof";

export interface PredictionResponse {
  prediction: PredictionLabel;
  /** Model confidence in [0, 1]. Labeled simply as "Confidence" in the UI. */
  confidence: number;
  /** Probability in [0, 1] that the audio is genuine. */
  bonafide_probability: number;
  /** Probability in [0, 1] that the audio is AI-generated/cloned. */
  spoof_probability: number;
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
export function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  const validLabel = v.prediction === "bonafide" || v.prediction === "spoof";
  return (
    validLabel &&
    isProbability(v.confidence) &&
    isProbability(v.bonafide_probability) &&
    isProbability(v.spoof_probability)
  );
}
