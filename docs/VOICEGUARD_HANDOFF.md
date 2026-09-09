# VoiceGuard — handoff

> **This project has been merged into `voice-deepfake-detector` as the primary repo going
> forward.** VoiceGuard's backend (`backend/inference/`, `backend/scoring/`, `backend/alerts/`,
> `backend/api/`), training pipeline (`training/`), and live-call demo UI now live there under
> `frontend/live-call/`, mounted alongside that repo's existing CNN spoof detector, ECAPA-TDNN
> speaker verification, and reporting system in one FastAPI app. This file is kept as the
> historical record of VoiceGuard's own development; going forward, track state in
> `voice-deepfake-detector`'s own docs instead of here.

Snapshot at commit `7094dbc` (2026-09-09). The ML half is done; what's left is the browser
demo and polish.

## Update — night of 2026-09-09/10: P6/P7 closed out, accuracy caveat found

- **`chunk_seconds` config fix.** `config/config.yaml` had `chunk_seconds: 1.0` /
  `hop_seconds: 0.5`, left over from before the serving checkpoint existed. The XLS-R frontend
  was fine-tuned on **4 s** crops (see `docs/RESULTS.md`), so 1 s analysis windows scored too
  unstably to ever latch a sustained HIGH alert. Fixed to `chunk_seconds: 4.0` /
  `hop_seconds: 1.0`, matching the trained crop size. Confirmed via `scripts/e2e_demo.py`
  against a synthetic AI-voice clip: HIGH alert + webhook now fire correctly.
- **WebSocket disconnect bug fix.** A client vanishing mid-send (tab closed, network drop)
  surfaced as `OSError`/`RuntimeError` (uvicorn's `ClientDisconnected`, a subclass of
  `OSError`), not `WebSocketDisconnect` — so the `except WebSocketDisconnect` handlers on
  `/ws/stream` and `/ws/signal/<room>` didn't catch it, and every normal tab close logged as an
  unhandled ASGI exception. Broadened both handlers to catch
  `(WebSocketDisconnect, OSError, RuntimeError)`. See `backend/api/websocket.py`.
- **Browser E2E verification (P6).** Confirmed the full caller → receiver path end to end in
  real Chrome (Playwright, caller mic fed via `--use-file-for-fake-audio-capture`): room-code
  signaling on `/ws/signal/<room>` → WebRTC connects → receiver captures the remote stream →
  PCM streams over `/ws/stream` → gauge updates live → a sustained deepfake trips the HIGH
  banner → webhook fires. This item is no longer "code written, never run in a real browser" —
  it's verified.
- **Accuracy testing finding: benchmark-accurate, not yet real-world-validated.** The model
  performs as `docs/RESULTS.md` describes on ASVspoof/In-the-Wild-style benchmark data (clean
  polarity, ~14% pooled EER, near-perfect on genuine Indian-language corpus speech). Informal
  testing tonight with real-world/external audio — clips outside the curated eval manifests —
  did **not** hold up to the same standard. Treat the `docs/RESULTS.md` numbers as in-domain
  benchmark performance only, not a guarantee for arbitrary real-world calls, until this gap is
  investigated further (likely the same training-data skew `docs/RESULTS.md` "Next" already
  flags — fakes skew heavily to MMS-TTS, so the model hasn't learned the breadth of real-world
  spoof signatures).
- `tests/test_websocket.py` updated: hardcoded window-count assumptions tied to the old
  1 s/0.5 s config were hanging on 10 s timeouts against the new default.
- P7 (partial): README got a browser demo walkthrough + the HIGH-alert screenshot
  (`docs/screenshots/receiver_high_alert.png`) before the merge.

## What's done

| Area | State |
|---|---|
| Detection pipeline — wav2vec2 SSL frontend → vendored AASIST → OC-Softmax → `fake_prob` | ✅ `backend/inference/` |
| Risk engine — rolling weighted avg over 10, LOW/MED/HIGH, 3-chunk HIGH latch, webhook + cooldown | ✅ `backend/scoring/`, `backend/alerts/` |
| FastAPI — `/health`, `/config`, `/score`, `WS /ws/stream`, `WS /ws/signal/<room>` | ✅ `backend/api/`, `backend/main.py` |
| Audio — PCM framing, resample, overlap ring buffer, stale-window drop | ✅ `backend/audio/` |
| Training + eval code — e2e fine-tune, RawBoost, OC-Softmax, resumable, `training.evaluate` | ✅ `training/` |
| **Trained checkpoint** — fine-tuned XLS-R-300m + AASIST, epoch 10 | ✅ **not in git** — see below |
| Datasets — ASVspoof 2019/2021 LA, In-the-Wild, IndicTTS + MMS-TTS fakes | acquired locally; scripts in `scripts/` |
| Backend E2E — deepfake clip → HIGH alert → webhook; genuine → silent | ✅ validated (`tests/test_e2e_alert.py`, `scripts/e2e_demo.py`) |
| Frontend — caller/receiver pages, AudioWorklet PCM, room signaling, live gauge | ✅ verified end-to-end in real Chrome (see night-of-09-09/10 update below) |
| README + screenshots + demo script | ✅ |

`pytest` → ~39 pass / 2 skip (the 2 skips are opt-in real-HF tests; the suite forces the
`dummy` feature extractor so it runs offline).

## Detection performance (epoch 10)

Pooled EER **14.09%** on a balanced 400/domain eval subset — see [`RESULTS.md`](RESULTS.md)
for the full table and the "epoch 15 overfit" write-up. Short version: it works (clean polarity,
~perfect on genuine Indian-language speech, decent on the unseen In-the-Wild domain) but ~14%
EER is mediocre for anti-spoofing. **More training epochs made it worse** — the lever is the
training fake mix (currently too MMS-TTS-heavy), not compute. Not a blocker for the demo.

## Getting the model

`backend/models/aasist_indicw2v.pt` is gitignored (1.2 GB). Without it the backend still runs
on `facebook/wav2vec2-base` (English-only, untrained head) or in `dummy` mode — fine for
wiring the frontend, not for real detection numbers.

The repo owner is sending `aasist_indicw2v.pt` directly (1.2 GB). Drop it at
`backend/models/aasist_indicw2v.pt`.

- size: `1263924365` bytes
- sha256: `dffc95d010631e0476ece6bdebcb0310502d261d822d3cf80840431fdf7e71ab`

```bash
python -c "import torch; c=torch.load('backend/models/aasist_indicw2v.pt',map_location='cpu',weights_only=False); print('epoch',c['epoch'],'dev_eer',c['dev_eer'],'oc_softmax' in c)"
# -> epoch 10 dev_eer 0.0033... True
```

(Alternative source: Kaggle notebook `srivarreddy77/notebook54095765c6` **version 5's Output
tab** — v5 is epoch 10, v6 is the overfit epoch 15, don't use v6.)

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Python 3.13, Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn backend.main:app        # http://localhost:8000/  → Caller + Receiver pages
# light, model-free:  set VG_FEATURE_EXTRACTOR__BACKEND=dummy first
```

Config is `config/config.yaml`, overridable by env (`VG_<SECTION>__<KEY>`). Note: with the
fine-tuned XLS-R checkpoint loaded, inference is ~0.6 s / 1 s chunk on CPU — set
`analysis.hop_seconds: 1.0` (base wav2vec2 keeps up at 0.5 s). Serving XLS-R needs ~3 GB peak
RAM.

## Next (the remaining 7-day-plan items)

- ~~**P6 — browser E2E.**~~ Done — see the night-of-09-09/10 update above.
- ~~**P7 — polish.**~~ README demo section + HIGH-alert screenshot done. `scripts/e2e_demo.py`
  walkthrough still outstanding.
- **Real-world audio validation (new, higher priority than it was).** Benchmark numbers in
  `docs/RESULTS.md` don't hold up on real-world/external audio per tonight's informal testing.
  Needs a proper eval pass against real-world clips (not just curated ASVspoof/In-the-Wild
  manifests) before this is trusted for arbitrary calls.
- **Detection quality (separate track).** Rebalance the training fakes: all ASVspoof
  2019-train spoof families (not a capped subset) + a second TTS engine so "fake" ≠ "MMS-TTS".
  Then retrain (fewer frontend-finetune epochs / lower frontend LR). See `docs/RESULTS.md` "Next".
  Likely the same root cause as the real-world-audio gap above.

## Key context

- **[VOICEGUARD_CLAUDE.md](VOICEGUARD_CLAUDE.md)** — architecture + full build plan, source of truth.
- **[TRAINING.md](TRAINING.md)** — training pipeline.
- **[RESULTS.md](RESULTS.md)** — eval numbers + the overfit finding.
- OC-Softmax scoring: the checkpoint is scored by **centre distance**, not the 2-logit head
  (which is untrained). `backend/inference/classifier.py::_oc_fake_prob`, auto-detected from
  an `oc_softmax` block in the checkpoint.
- `notebooks/train_kaggle.ipynb` resumes across Kaggle's 12 h wall by mounting a prior
  `.resume.pt` as a dataset input (Cell 2 seeds it, Cell 4 picks resume vs `--warm-start`).
