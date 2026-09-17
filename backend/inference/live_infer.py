"""Live-call inference boundary — THE SWAP POINT for the real model.

The whole live-call path calls exactly one function, ``infer(window)``, and gets back the
agreed contract:

    {"is_fake": bool, "confidence": float}   # confidence in [0, 1]

Right now that's a stub (random/heuristic) so the end-to-end call flow can be demoed before
the model is trained. To drop in the real model later, you only touch THIS file — nothing in
the Twilio/WebSocket/dashboard code needs to change:

  * If the real model is an in-process pipeline: implement ``_real_infer`` below to call it
    (a `DetectionPipeline` already exists — ``backend.main.get_pipeline().infer_chunk`` returns
    a result with ``.fake_prob``; map it to the {is_fake, confidence} contract).
  * If it's a separate HTTP endpoint: POST the audio to it inside ``_real_infer`` and return
    its {is_fake, confidence} JSON.

Select the backend with env ``VG_LIVE_INFERENCE`` = ``stub`` (default) | ``real``.
"""

from __future__ import annotations

import os
import random

import numpy as np

# Optional demo knob for the stub: 0.0 = always trend "real", 1.0 = always trend "AI-cloned",
# 0.5 = neutral random. Lets a live demo show a convincing slide either way without a model.
_STUB_BIAS = float(os.environ.get("VG_LIVE_STUB_BIAS", "0.5"))


def _stub_infer(window: np.ndarray) -> dict:
    """Random/heuristic verdict in the {is_fake, confidence} shape. No model involved.

    Draws a fake-probability around ``_STUB_BIAS``; the rolling RiskEngine downstream smooths
    per-chunk noise into a steady meter, so even random draws animate sensibly on the dashboard.
    """
    p_fake = min(1.0, max(0.0, random.gauss(_STUB_BIAS, 0.18)))
    is_fake = p_fake >= 0.5
    confidence = p_fake if is_fake else 1.0 - p_fake
    return {"is_fake": bool(is_fake), "confidence": round(float(confidence), 4)}


def _real_infer(window: np.ndarray) -> dict:
    """Real model. Swap this in when the trained model is ready (see module docstring)."""
    from backend.main import get_pipeline

    result = get_pipeline().infer_chunk(window)  # existing AASIST pipeline -> .fake_prob
    p_fake = float(result.fake_prob)
    is_fake = p_fake >= 0.5
    confidence = p_fake if is_fake else 1.0 - p_fake
    return {"is_fake": bool(is_fake), "confidence": round(confidence, 4)}


def infer(window: np.ndarray) -> dict:
    """Analyze one ~4 s audio window (float32 mono @16 kHz). Returns {is_fake, confidence}."""
    backend = os.environ.get("VG_LIVE_INFERENCE", "stub").strip().lower()
    if backend == "real":
        return _real_infer(window)
    return _stub_infer(window)


def fake_probability(verdict: dict) -> float:
    """Map a {is_fake, confidence} verdict to P(AI-cloned) in [0,1] for the risk meter."""
    conf = float(verdict["confidence"])
    return conf if verdict["is_fake"] else 1.0 - conf
