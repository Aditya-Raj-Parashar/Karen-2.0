# KAREN 2.0 — Project Context

## Product summary

KAREN is an existing multi-camera human detection, tracking, and re-identification system. KAREN 2.0 should evolve the current application, not replace it with an unrelated demo.

## User priorities

- Accuracy is the first priority.
- The system should remain simple to run and demonstrate.
- The demo laptop is relatively low-powered.
- Demo inputs are prerecorded CCTV clips or sample videos.
- Internet connectivity at demonstrations is not guaranteed.
- Existing normal input/output behavior should remain familiar.
- NVIDIA technology is the preferred direction for the vision pipeline.
- NVIDIA Nemotron via Nebius Token Factory is intended for conversational Q&A.
- Full-video upload to a hosted model should not be the default.

## Intended architecture direction (proposed, not yet verified or approved in implementation)

### Vision pipeline
Investigate suitable NVIDIA person-detection and tracking components. Potential candidates include PeopleNet and DeepStream tracker options, but no candidate is selected until licensing, model capability, hardware, deployment, and compatibility are verified.

### KAREN application logic
Retain and improve application-controlled tracking, identity matching, temporal validation, persistence, camera mapping, and dashboard behavior where appropriate. Confirm actual implementation before designing changes.

### Nemotron intelligence
Default path: retrieve only relevant records from the local database, package them into a compact structured context, and ask Nemotron to answer a natural-language question.

Optional path: when structured records cannot answer a question, retrieve a bounded number of relevant video frames and send them to a compatible image-capable model. The application controls frame selection and retrieval. Do not assume all Nemotron endpoints support image input.

### Online/offline behavior
Offline core: local video playback, locally available inference, local database, and dashboard functionality to the extent supported by the chosen deployment.
Online additions: hosted Nemotron Q&A and any remote inference or image analysis.
Precomputed replay must be clearly labelled and must not be represented as live inference.

## Important uncertainties

The current codebase must be audited to establish:
- Exact module responsibilities and runtime data flow.
- Actual database schema and available event data.
- Existing detector/tracker/re-identification behavior.
- Actual optimizations and tests.
- Current hardware and deployment compatibility.
- Current official hackathon requirements and model eligibility.

Do not treat this context file as proof that a proposed feature already exists.
