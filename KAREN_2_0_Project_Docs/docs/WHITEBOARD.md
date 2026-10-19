# KAREN 2.0 Whiteboard

Use this file to record open design questions and candidate approaches before committing to an architecture.

## Problem statement

Improve KAREN's vision accuracy and reduce dependence on powerful local hardware, while preserving a simple workflow and supporting an unreliable-internet demonstration environment.

## Constraints

- Accuracy is the primary objective.
- Demo inputs are prerecorded CCTV clips or sample videos.
- Demo internet connectivity is not guaranteed.
- Existing KAREN behavior and useful optimizations should be preserved where practical.
- NVIDIA vision components are preferred.
- Nemotron via Nebius Token Factory is intended for structured-data Q&A.
- Selective frame analysis is optional and must be bounded.
- No implementation until the plan is reviewed and approved.

## Questions to resolve

### Vision
- Which detection model is best supported and accurate for the target clips?
- Which tracker handles occlusion and identity stability best?
- What provides cross-camera re-identification?
- Which components run locally versus remotely?
- What hardware and licensing constraints apply?

### Intelligence
- Which questions can the existing schema answer?
- Which event records are missing?
- Which Nemotron model and API are appropriate?
- How should insufficient evidence be presented?
- Under what conditions should selected frames be analyzed?

### Deployment
- What remains functional offline?
- Is local inference feasible on the demo laptop?
- Is a remote GPU path available and affordable?
- How will replay be distinguished from live inference?

### Evaluation
- What representative clips exist?
- Is ground truth available?
- What metrics define success?
- What is the baseline?

## Candidate architectures

Record each candidate, its pros/cons, evidence, cost, compatibility, and decision status.

## Decisions made

Only add a decision after explicit review. Link to the relevant entry in `DECISIONS.md`.
