/**
 * useMessageCheck — the single source of truth for the submit -> check ->
 * result flow on the Message check page. Mirrors useAnalyzer's shape so the
 * two pages read the same way.
 *
 * Status machine:
 *   idle -> checking -> (result | error)
 *   result/error -> checking (check another message) -> ...
 */

import { useCallback, useRef, useState } from "react";
import { ApiError, checkMessage } from "../services/api";
import type { MessageCheckResponse } from "../types/message";

export type MessageCheckStatus = "idle" | "checking" | "result" | "error";

interface MessageCheckState {
  status: MessageCheckStatus;
  /** The exact text that was submitted, snapshotted so edits to the input
   * afterwards don't desync from the displayed result/highlights. */
  submittedText: string | null;
  result: MessageCheckResponse | null;
  error: string | null;
}

const initialState: MessageCheckState = {
  status: "idle",
  submittedText: null,
  result: null,
  error: null,
};

export function useMessageCheck() {
  const [state, setState] = useState<MessageCheckState>(initialState);
  const abortRef = useRef<AbortController | null>(null);

  /** Submit a message's text for phishing analysis. */
  const check = useCallback(async (text: string) => {
    const trimmed = text.trim();
    // Guard against empty input or a double-submit while already checking.
    if (!trimmed || abortRef.current) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setState({
      status: "checking",
      submittedText: trimmed,
      result: null,
      error: null,
    });

    try {
      const result = await checkMessage(trimmed, controller.signal);
      if (controller.signal.aborted) return;
      setState((prev) => ({ ...prev, status: "result", result, error: null }));
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message =
        err instanceof ApiError
          ? err.message
          : "Something went wrong while checking the message. Please try again.";
      setState((prev) => ({ ...prev, status: "error", error: message }));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, []);

  /** Cancel any in-flight check. */
  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState(initialState);
  }, []);

  /** Clear everything and go back to the start. */
  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState(initialState);
  }, []);

  return { ...state, check, cancel, reset };
}
