# KAREN 2.0 Implementation Plan

**Status:** Proposed — requires codebase audit and owner approval before implementation.

## Phase 0 — Full codebase audit and planning

**Goal:** Understand the current system before changing it.

Tasks:
- Inventory and inspect the project directory and relevant nested files.
- Trace actual runtime data flow.
- Verify existing detector, tracker, Re-ID, identity gallery, database, UI, and floor-map behavior.
- Document current optimizations and known challenges.
- Identify missing tests and unknowns.
- Research current official NVIDIA, Nebius Token Factory, Nemotron, and hackathon documentation.
- Produce a file-by-file implementation plan.

Exit criteria:
- Findings cite actual files/functions.
- Facts are distinguished from assumptions.
- Risks and unresolved questions are listed.
- No application code was modified.
- Owner has reviewed and approved the plan.

## Phase 1 — Establish baseline

Tasks:
- Select representative prerecorded clips.
- Record reproducible configuration and environment details.
- Measure detection/tracking quality where ground truth exists.
- Record ID switches, track fragmentation, cross-camera matching quality, FPS, latency, memory, and compute usage as applicable.
- Record failures qualitatively when ground truth is unavailable.

Exit criteria:
- Baseline results are reproducible.
- Metrics and limitations are documented in `docs/BASELINE.md`.

## Phase 2 — Validate NVIDIA vision candidates

Tasks:
- Compare viable NVIDIA detection and tracking candidates.
- Verify license, availability, input/output, hardware, OS, runtime, and deployment constraints.
- Determine whether inference is local, remote on Nebius AI Cloud, or hybrid.
- Create a small proof-of-compatibility only after approval.

Exit criteria:
- Candidate and fallback selected with documented reasons.
- No accuracy claim without comparable evaluation.

## Phase 3 — Integrate and evaluate vision improvements

Tasks:
- Integrate the approved vision components in a bounded, reviewable change.
- Preserve compatible KAREN identity logic and existing optimizations.
- Add configuration for the new pipeline where necessary.
- Compare against the baseline on the same clips.
- Test regressions in CLI, output, identity persistence, and dashboard.

Exit criteria:
- Measured results are recorded.
- Accuracy, latency, hardware, and deployment trade-offs are explicit.
- Owner approves moving forward.

## Phase 4 — Structured database Q&A

Tasks:
- Inspect the actual schema and define supported question categories.
- Implement bounded, parameterized retrieval in application code.
- Build compact context objects from relevant records only.
- Preserve timestamp, camera, identity, confidence, and evidence status.
- Define answer grounding and insufficient-evidence behavior.

Exit criteria:
- Tests cover valid, empty, ambiguous, and insufficient-evidence queries.
- No unrestricted model-generated SQL.
- No full database dump sent to the hosted model.

## Phase 5 — Token Factory / Nemotron integration

Tasks:
- Verify the current API and model capabilities.
- Configure endpoint/model/API key outside source control.
- Add timeouts, bounded retries, error handling, and usage logging without secrets.
- Keep language-model requests outside the real-time vision loop.
- Add the chat UI without breaking the existing workflow.

Exit criteria:
- Online Q&A works against known test records.
- API failures do not crash core KAREN functionality.
- Offline state is clearly represented.

## Phase 6 — Selective frame analysis (optional)

Tasks:
- Determine when structured records are insufficient.
- Select relevant frames using application-controlled timestamp/path retrieval.
- Bound frame count and image size.
- Verify image input support and API constraints for the selected model.
- Separate observed visual details from tracker-derived claims.

Exit criteria:
- Feature can be disabled cleanly.
- No whole-video uploads by default.
- Tests cover missing videos, invalid timestamps, API errors, and inconclusive images.

## Phase 7 — Online/offline resilience

Tasks:
- Define local-only, cloud-enabled, and preprocessed replay behavior.
- Handle unavailable internet and API errors.
- Make replay visibly distinct from live inference.
- Test restart and recovery behavior.

Exit criteria:
- Offline core capabilities are documented and tested.
- No feature falsely claims live cloud inference during replay.

## Phase 8 — Final regression and demonstration

Tasks:
- Run tests and benchmark scripts.
- Validate installation and setup instructions.
- Test the demo on the intended laptop.
- Verify hackathon requirements against official rules.
- Prepare an honest demo flow and known-limitations list.

Exit criteria:
- Reproducible test report.
- Known limitations documented.
- Submission claims supported by evidence.

## Approval gates

Do not start a phase with significant architectural changes until its prerequisites and owner approval are satisfied. Adjust phase order only with a written rationale.
