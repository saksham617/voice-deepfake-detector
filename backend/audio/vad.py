"""Voice Activity Detection gate for the live-call pipeline.  [VAD filter]

AASIST/XLS-R was trained to distinguish real vs. fake *speech*; it has no reliable
behaviour on pure background noise (no speech at all), which has been observed to drift
into a sustained HIGH-risk score. ``SileroVAD`` runs ahead of the classifier so a
non-speech analysis window can be skipped entirely instead of scored.

Silero VAD (https://github.com/snakers4/silero-vad) ships its own small (~2 MB) JIT model
inside the ``silero-vad`` package -- no network access or extra runtime (e.g. onnxruntime)
needed beyond what's already installed (torch/torchaudio).
"""

from __future__ import annotations

import torch

VAD_FRAME_SAMPLES = 512  # Silero's fixed frame size at 16 kHz


class SileroVAD:
    """Frame-level speech/non-speech gate for a single analysis window.

    Splits the window into 512-sample (32 ms) frames, scores each independently, and
    classifies the whole window as speech if at least ``min_speech_ratio`` of its frames
    are above ``frame_threshold``. A ratio (rather than a mean probability) keeps the
    decision robust to where in the window speech happens to fall.
    """

    def __init__(
        self,
        frame_threshold: float = 0.5,
        min_speech_ratio: float = 0.2,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("SileroVAD only supports 16 kHz audio")
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=False)
        self.model.eval()
        self.frame_threshold = frame_threshold
        self.min_speech_ratio = min_speech_ratio
        self.sample_rate = sample_rate

    @torch.no_grad()
    def speech_ratio(self, wav: torch.Tensor) -> float:
        """Fraction of 512-sample frames in ``wav`` classified as speech."""
        wav = torch.as_tensor(wav, dtype=torch.float32).reshape(-1)
        n = wav.shape[0]
        if n < VAD_FRAME_SAMPLES:
            return 0.0
        n_frames = n // VAD_FRAME_SAMPLES  # trailing partial frame (<32 ms) dropped
        wav = wav[: n_frames * VAD_FRAME_SAMPLES]

        self.model.reset_states()  # each window is judged independently
        speech_frames = 0
        for i in range(n_frames):
            frame = wav[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            prob = float(self.model(frame, self.sample_rate).item())
            if prob >= self.frame_threshold:
                speech_frames += 1
        return speech_frames / n_frames

    def is_speech(self, wav: torch.Tensor) -> bool:
        return self.speech_ratio(wav) >= self.min_speech_ratio
