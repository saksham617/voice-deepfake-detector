/**
 * useMicRecorder — captures microphone audio via getUserMedia + MediaRecorder
 * and hands the finished recording back as a plain `File`, so it can be fed
 * straight into the same `selectFile`/`analyze` flow already used for
 * uploads (see useAnalyzer). This hook only produces the File; it knows
 * nothing about prediction, validation, or the analyze flow.
 *
 * Status machine: idle -> requesting -> recording -> idle (via onRecorded)
 *                              \-> idle (on permission/device error)
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderStatus = "idle" | "requesting" | "recording";

interface UseMicRecorderOptions {
  /** Called with the finished recording once the user stops. */
  onRecorded: (file: File) => void;
}

export function useMicRecorder({ onRecorded }: UseMicRecorderOptions) {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timerRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);

  const releaseResources = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    analyserRef.current = null;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const AudioContextCtor =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioCtx = new AudioContextCtor();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const extension = mimeType.includes("ogg")
          ? "ogg"
          : mimeType.includes("mp4")
            ? "m4a"
            : "webm";
        const file = new File([blob], `recording-${Date.now()}.${extension}`, {
          type: mimeType,
        });
        releaseResources();
        setStatus("idle");
        setElapsedMs(0);
        onRecorded(file);
      };
      recorderRef.current = recorder;

      startedAtRef.current = Date.now();
      timerRef.current = window.setInterval(() => {
        setElapsedMs(Date.now() - startedAtRef.current);
      }, 250);

      recorder.start();
      setStatus("recording");
    } catch (err) {
      releaseResources();
      setStatus("idle");
      const denied =
        err instanceof DOMException &&
        (err.name === "NotAllowedError" || err.name === "PermissionDeniedError");
      setError(
        denied
          ? "Microphone access was denied. Allow microphone access in your browser settings and try again."
          : "Could not access the microphone. Please check your device and try again.",
      );
    }
  }, [onRecorded, releaseResources]);

  const stop = useCallback(() => {
    recorderRef.current?.stop();
  }, []);

  const dismissError = useCallback(() => setError(null), []);

  // Release the mic/AudioContext if the component unmounts mid-recording.
  useEffect(() => releaseResources, [releaseResources]);

  return {
    status,
    error,
    elapsedMs,
    analyser: analyserRef.current,
    start,
    stop,
    dismissError,
  };
}
