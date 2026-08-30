# Voice Deepfake Detector

## About

This is a Smart India Hackathon (SIH) project. The goal is to build a system
that can distinguish **real human speech** from **AI-generated / deepfake
speech**, helping combat the growing misuse of voice cloning and synthetic
audio.

## Goal

Given an audio clip, the system should predict whether the voice is genuine
or synthetically generated (e.g. via TTS or voice cloning models), and
provide a confidence score for that prediction.

## Planned Scope

This repository will eventually contain:

- **Machine learning models** for extracting audio features and classifying
  speech as real or fake.
- A **FastAPI backend** to serve predictions via an API.
- A **frontend** for uploading/recording audio and viewing detection
  results.

## Status

🚧 Early stage — project foundation only. No models, backend, or frontend
have been implemented yet.

## Project Structure

```
voice-deepfake-detector/
├── data/           # datasets (raw, processed, small samples)
├── notebooks/       # exploratory analysis / experiments
├── src/             # reusable source code (features, models, utils)
├── models/           # trained model artifacts (not committed)
├── tests/            # automated tests
├── backend/          # FastAPI backend (future)
└── frontend/          # frontend application (future)
```

## Setup

Not applicable yet — dependencies and environment setup instructions will be
added as the project develops.
