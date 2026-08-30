/**
 * useAnalyzer — the single source of truth for the upload -> analyze -> result
 * flow. Components read `status` and the current data; they call the returned
 * actions. All API access goes through the api service.
 *
 * Status machine:
 *   idle -> selected -> analyzing -> (result | error)
 *   result/error -> selected (choose another file) -> ...
 */

import { useCallback, useRef, useState } from "react";
import { ApiError, predictAudio } from "../services/api";
import type { PredictionResponse } from "../types/prediction";
import { validateAudioFile } from "../utils/audioFile";

export type AnalyzerStatus =
  | "idle"
  | "selected"
  | "analyzing"
  | "result"
  | "error";

export interface AnalyzerState {
  status: AnalyzerStatus;
  file: File | null;
  result: PredictionResponse | null;
  error: string | null;
}

const initialState: AnalyzerState = {
  status: "idle",
  file: null,
  result: null,
  error: null,
};

export function useAnalyzer() {
  const [state, setState] = useState<AnalyzerState>(initialState);
  const fileRef = useRef<File | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /** Select (or replace) the file to analyze; validates before accepting. */
  const selectFile = useCallback((file: File) => {
    const validation = validateAudioFile(file);
    if (!validation.ok) {
      fileRef.current = null;
      setState({
        status: "error",
        file: null,
        result: null,
        error: validation.error ?? "That file can't be used.",
      });
      return;
    }
    fileRef.current = file;
    setState({ status: "selected", file, result: null, error: null });
  }, []);

  /** Submit the currently selected file for analysis. */
  const analyze = useCallback(async () => {
    const file = fileRef.current;
    // Guard against no file or a double-submit while already analyzing.
    if (!file || abortRef.current) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setState((prev) => ({
      ...prev,
      status: "analyzing",
      error: null,
      result: null,
    }));

    try {
      const result = await predictAudio(file, controller.signal);
      if (controller.signal.aborted) return;
      setState((prev) => ({ ...prev, status: "result", result, error: null }));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message =
        err instanceof ApiError
          ? err.message
          : "Something went wrong while analyzing the audio. Please try again.";
      setState((prev) => ({ ...prev, status: "error", error: message }));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, []);

  /** Cancel any in-flight analysis and return to the selected state. */
  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState((prev) =>
      prev.file
        ? { ...prev, status: "selected", error: null }
        : { ...initialState },
    );
  }, []);

  /** Clear everything and go back to the start. */
  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    fileRef.current = null;
    setState({ ...initialState });
  }, []);

  return { ...state, selectFile, analyze, cancel, reset };
}
