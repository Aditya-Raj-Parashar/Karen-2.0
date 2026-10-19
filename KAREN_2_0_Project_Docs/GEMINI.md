# KAREN Project Guidance for Gemini / Antigravity

This file provides project-specific guidance to Gemini-based coding agents. Also follow `AGENTS.md`; where guidance overlaps, obey the more restrictive rule.

## Current assignment default

Unless the user explicitly approves implementation, treat work as **analysis and planning only**. For the initial KAREN 2.0 assignment, inspect the repository and produce a detailed plan without modifying application code.

## Understand before acting

- Read the entire relevant codebase, not only README files.
- Trace real function calls and data structures.
- Cite actual file paths, classes, functions, and configuration keys in findings.
- Identify which documentation is outdated or incomplete.
- Never assume an existing module behaves as a document says without verification.

## Preferred engineering style

- Make small, reviewable changes.
- Preserve the current workflow and interfaces.
- Avoid unnecessary abstractions and new dependencies.
- Prefer explicit configuration and bounded failure behavior.
- Keep visual inference independent from Nemotron response latency.
- Make database retrieval application-controlled and parameterized.
- Treat track IDs, persistent person IDs, and cross-camera matches as distinct concepts unless the code proves otherwise.

## External research

When researching NVIDIA, Nebius Token Factory, Nemotron, model licensing, or hackathon rules:
- Prefer official documentation and current rules.
- Record source URLs and access/review date in `docs/RESEARCH_NOTES.md`.
- Distinguish verified facts from assumptions.
- Verify supported input modalities, deployment requirements, API behavior, pricing/rate limits, and licensing before recommending a model.

## Stop conditions

Stop and ask for review before:
- Replacing the detector or tracker.
- Changing the database schema or stored identity format.
- Adding a new model provider or sending video/images externally.
- Adding a new dependency with significant runtime or hardware requirements.
- Removing an existing optimization.
- Changing the application's user-facing workflow.
