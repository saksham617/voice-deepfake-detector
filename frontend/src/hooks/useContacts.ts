/**
 * useContacts — owns the enrolled-contacts list plus the enroll and verify
 * flows on the Contacts page. All API access goes through the api service.
 *
 * The roster is loaded from GET /contacts on mount and appended to locally
 * as new contacts are enrolled during the session (no need to refetch the
 * whole list after every enroll).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, enrollSpeaker, fetchContacts, verifySpeaker } from "../services/api";

export interface Contact {
  id: string;
  name: string;
  enrolledAt: string; // ISO timestamp
}

export type ContactsLoadStatus = "loading" | "ready" | "error";
export type EnrollStatus = "idle" | "enrolling" | "success" | "error";
export type VerifyStatus = "idle" | "verifying" | "result" | "error";

export interface VerifyResult {
  match: boolean;
  similarity: number;
}

export function useContacts() {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [contactsLoadStatus, setContactsLoadStatus] = useState<ContactsLoadStatus>("loading");
  const [contactsLoadError, setContactsLoadError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchContacts(controller.signal)
      .then((backendContacts) => {
        setContacts(
          backendContacts.map((c) => ({ id: c.contact_id, name: c.name, enrolledAt: c.enrolled_at })),
        );
        setContactsLoadStatus("ready");
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setContactsLoadError(
          err instanceof ApiError ? err.message : "Could not load your contacts. Please try again.",
        );
        setContactsLoadStatus("error");
      });
    return () => controller.abort();
  }, []);

  const [enrollStatus, setEnrollStatus] = useState<EnrollStatus>("idle");
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const enrollAbortRef = useRef<AbortController | null>(null);

  const [verifyStatus, setVerifyStatus] = useState<VerifyStatus>("idle");
  const [verifyResult, setVerifyResult] = useState<VerifyResult | null>(null);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const verifyAbortRef = useRef<AbortController | null>(null);

  /** Enroll a new contact's voiceprint from a name + audio sample. */
  const enroll = useCallback(async (name: string, file: File) => {
    if (enrollAbortRef.current) return;
    const controller = new AbortController();
    enrollAbortRef.current = controller;
    setEnrollStatus("enrolling");
    setEnrollError(null);

    try {
      const response = await enrollSpeaker(name, file, controller.signal);
      if (controller.signal.aborted) return;
      setContacts((prev) => [
        ...prev,
        { id: response.contact_id, name: response.name, enrolledAt: new Date().toISOString() },
      ]);
      setEnrollStatus("success");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setEnrollError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while enrolling this contact. Please try again.",
      );
      setEnrollStatus("error");
    } finally {
      if (enrollAbortRef.current === controller) enrollAbortRef.current = null;
    }
  }, []);

  const resetEnroll = useCallback(() => {
    enrollAbortRef.current?.abort();
    enrollAbortRef.current = null;
    setEnrollStatus("idle");
    setEnrollError(null);
  }, []);

  /** Verify an audio sample against a previously enrolled contact. */
  const verify = useCallback(async (contactId: string, file: File) => {
    if (verifyAbortRef.current) return;
    const controller = new AbortController();
    verifyAbortRef.current = controller;
    setVerifyStatus("verifying");
    setVerifyError(null);
    setVerifyResult(null);

    try {
      const response = await verifySpeaker(contactId, file, controller.signal);
      if (controller.signal.aborted) return;
      setVerifyResult({ match: response.match, similarity: response.similarity });
      setVerifyStatus("result");
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setVerifyError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while verifying this voice. Please try again.",
      );
      setVerifyStatus("error");
    } finally {
      if (verifyAbortRef.current === controller) verifyAbortRef.current = null;
    }
  }, []);

  const resetVerify = useCallback(() => {
    verifyAbortRef.current?.abort();
    verifyAbortRef.current = null;
    setVerifyStatus("idle");
    setVerifyResult(null);
    setVerifyError(null);
  }, []);

  return {
    contacts,
    contactsLoadStatus,
    contactsLoadError,
    enrollStatus,
    enrollError,
    enroll,
    resetEnroll,
    verifyStatus,
    verifyResult,
    verifyError,
    verify,
    resetVerify,
  };
}
