# Task Brief: Multi-Speaker + Long-Duration Detection

**Project:** VoiceGuard (SIH26104 — AI voice cloning/impersonation detection)
**Repo:** github.com/saksham617/voice-deepfake-detector
**Priority:** Medium — not blocking, but a good demo differentiator (handles realistic scam calls with two people talking, not just short single-speaker clips)
**Depends on training?** No. This feature works with the existing/current model checkpoint (`aasist_indicw2v.pt` or whichever is latest) and does not require any retraining. It can be built and tested independently of any training run in progress.

## The problem this solves

Right now, the detection model only scores a single ~4-second window of audio, implicitly assuming one speaker. Real scam calls involve two people (the victim and a possibly-cloned scammer voice). We want to:

1. Accept a longer recording (a full call, any duration).
2. Figure out who is speaking when (speaker diarization).
3. Score each speaker's speech separately through the existing model.
4. Show a per-speaker timeline of real/fake verdicts, instead of one blended score for the whole file.

## Why this doesn't need retraining

The core spoof-detection model (`wav2vec2-XLS-R-300M` frontend + `AASIST` classifier) already takes short, fixed-length audio windows (4 seconds, see `training/config_train.yaml`'s `crop_seconds`) and outputs a bonafide/spoof score. This feature just calls that *same* model multiple times — once per windowed segment per speaker — rather than changing anything about how the model itself works. Think of it as new orchestration/application logic wrapped around the existing, already-trained model.

## Suggested approach

### 1. Speaker diarization

Use [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) (pretrained pipeline `pyannote/speaker-diarization-3.1` or similar) to process the uploaded recording and get a list of `(speaker_label, start_time, end_time)` segments.

**Known friction points to watch for:**
- The pretrained pipeline is gated on HuggingFace — you'll need to accept the model's terms on huggingface.co and generate an access token (`HF_TOKEN`) before it will download.
- Dependency conflicts are common with `pyannote.audio` (it pins specific `torch`/`torchaudio` versions) — install it in a way that doesn't silently downgrade the project's existing PyTorch/CUDA setup. Test `torch.cuda.is_available()` again after installing, the same way we had to fix a CUDA/cuDNN mismatch earlier tonight.
- Start by testing on a real two-person recording if one exists (ask Arnav / the real-world test-data folder — a few 2-person phone call samples were requested from the team for exactly this purpose). If none exist yet, a quick two-person recording made on the spot works fine for testing.

### 2. Windowing each speaker's segments

For each diarized segment belonging to a speaker, slice it into non-overlapping 4-second windows (matching what the model was trained on). Reuse the windowing/VAD-gating logic already written in `scripts/prepare_realworld_manifest.py` (functions `process_file()`, `speech_ratio_silero()` / `speech_ratio_energy()`) as a reference — it already knows how to chop audio into windows and skip near-silent ones, which avoids wasting inference calls on silence or breath sounds.

### 3. Scoring each window

Run each surviving window through the existing trained model using the same inference code path the current single-clip `/predict` endpoint already uses (look at how the existing Voice Check feature loads the checkpoint and scores one clip — reuse that model-loading and inference logic rather than writing it from scratch).

### 4. New backend endpoint

Add a new endpoint, e.g. `POST /predict_multi_speaker`, that:
- Accepts an uploaded audio file (any duration).
- Runs diarization → windowing → scoring as above.
- Returns a structured result per speaker, e.g.:

```json
{
  "speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "segments": [
        {"start": 0.0, "end": 4.0, "verdict": "bonafide", "score": 0.92},
        {"start": 4.0, "end": 8.0, "verdict": "spoof", "score": 0.11}
      ],
      "overall_verdict": "mixed"
    },
    {
      "speaker_id": "SPEAKER_01",
      "segments": [...],
      "overall_verdict": "bonafide"
    }
  ]
}
```

### 5. Frontend: per-speaker timeline

Extend the existing timeline concept already used in the Live Call dashboard mockup so it can render multiple rows — one per detected speaker — each showing their own sequence of real/fake segments over time, instead of one blended timeline. Reuse the existing design system: dark background (`#0a0d11`), single accent color, IBM Plex Mono for scores/data, IBM Plex Sans for body text, semantic colors only (green = safe/bonafide, amber = medium confidence, red = high-confidence spoof).

This could live as a new mode on the existing Voice Check page (toggle: "single speaker" vs "call recording with multiple speakers") rather than a whole new page, if that's simpler to fit into the current routing.

## Testing checklist

- [ ] Diarization correctly separates two distinct speakers on a real test recording
- [ ] Windowing skips silent segments (reuse the VAD gating, don't waste inference on silence)
- [ ] Each window scores correctly against the existing model (sanity-check against a known bonafide and known spoof clip first, in isolation, before testing the full pipeline)
- [ ] Endpoint returns correctly structured per-speaker JSON
- [ ] Frontend renders multiple speaker rows correctly, including edge cases (only 1 speaker detected, 3+ speakers detected, very short recordings)

## Open questions to resolve before/during implementation

- Do we have any real two-person call recordings to test with yet, or does someone need to record a couple as placeholder test data?
- Should the endpoint cap max recording duration (to bound diarization + inference time for the demo)?
- Once the new real-world-accuracy-focused model checkpoint (currently training) is validated, this feature should be pointed at that checkpoint instead of the older one — no code changes needed, just swap which checkpoint file gets loaded.
