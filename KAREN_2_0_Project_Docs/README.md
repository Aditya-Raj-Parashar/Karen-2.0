# KAREN 2.0

KAREN is an existing multi-camera human detection, tracking, and re-identification project. KAREN 2.0 is intended to improve vision accuracy and add a grounded, Nemotron-powered conversational intelligence layer while preserving the existing application's familiar input/output workflow.

> **Important:** This repository already contains an implementation. Do not treat it as a greenfield project. The actual codebase is the source of truth; documentation must be verified against it.

## Project priorities

1. **Accuracy first:** evaluate improvements to person detection, multi-object tracking, and cross-camera re-identification using reproducible tests.
2. **Preserve working behavior:** retain existing inputs, outputs, CLI options, dashboard behavior, data, and optimizations unless a reviewed plan justifies a change.
3. **NVIDIA vision:** investigate suitable NVIDIA models and tracking/deployment components; validate licensing, compatibility, and hardware requirements before selecting them.
4. **Nemotron intelligence:** use Nebius Token Factory for natural-language Q&A grounded in KAREN's structured records.
5. **Selective image analysis:** only when structured data is insufficient, retrieve a small number of relevant frames and send them to a compatible image-capable model, if supported and approved.
6. **Online/offline modes:** keep core local functionality usable offline. Clearly distinguish live inference, cloud inference, and preprocessed replay.
7. **Evidence over assumptions:** do not claim improvements, completed work, API capabilities, or compliance without evidence.

## Documentation map

- `AGENTS.md` — instructions for coding agents.
- `GEMINI.md` — project guidance for Gemini-based tools, including Antigravity agents.
- `PROJECT_CONTEXT.md` — concise product and engineering context.
- `ARCHITECTURE.md` — verified architecture and data-flow record; update only after inspection.
- `PLAN.md` — staged implementation plan and approval gates.
- `TASKS.md` — actionable work items and status.
- `DECISIONS.md` — architecture decision records.
- `TESTING.md` — testing and benchmarking expectations.
- `SECURITY.md` — secrets, data, API, and model-safety rules.
- `CONTRIBUTING.md` — engineering workflow and change standards.
- `CHANGELOG.md` — user-visible and engineering changes.
- `docs/RESEARCH_NOTES.md` — sourced research notes and unresolved questions.
- `docs/BASELINE.md` — current measured performance baseline.
- `docs/DATA_DICTIONARY.md` — verified database schema and field meanings.

## First step

Before implementation, inspect the complete project directory and its relevant source, configuration, tests, and documentation. Produce a repository-specific audit and detailed plan. Do not edit application code until the owner has reviewed and approved the plan.
