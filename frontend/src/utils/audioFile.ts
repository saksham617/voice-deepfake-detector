/**
 * Client-side audio file validation helpers.
 *
 * These give fast, friendly feedback before we ever hit the network. The
 * backend remains the source of truth for what it can actually process.
 */

import {
  ACCEPTED_AUDIO_EXTENSIONS,
  ACCEPTED_AUDIO_TYPES,
  MAX_FILE_SIZE_BYTES,
} from "../config";

export interface FileValidationResult {
  ok: boolean;
  /** User-facing message when `ok` is false. */
  error?: string;
}

export function validateAudioFile(file: File): FileValidationResult {
  const nameLower = file.name.toLowerCase();
  const extOk = ACCEPTED_AUDIO_EXTENSIONS.some((ext) => nameLower.endsWith(ext));
  const typeOk =
    file.type !== "" &&
    (ACCEPTED_AUDIO_TYPES as readonly string[]).includes(file.type);

  // Accept if either the MIME type OR the extension looks right — browsers are
  // inconsistent about MIME types, especially on Windows.
  if (!typeOk && !extOk) {
    return {
      ok: false,
      error:
        "Unsupported file type. Please choose an audio file (WAV, MP3, FLAC, OGG, M4A).",
    };
  }

  if (file.size === 0) {
    return { ok: false, error: "This file appears to be empty." };
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      ok: false,
      error: `File is too large (max ${formatBytes(MAX_FILE_SIZE_BYTES)}).`,
    };
  }

  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
