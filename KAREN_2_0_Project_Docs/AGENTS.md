# Agent Instructions

## Mission

Help evolve the existing KAREN project into KAREN 2.0 without unnecessary rewrites. Prioritize accuracy, reliability, maintainability, and a realistic demonstration on a low-powered laptop.

## Mandatory first-run workflow

Before changing code:

1. Inventory the project directory, including nested source and configuration directories.
2. Read the relevant source files, dependency manifests, tests, configuration, and existing documentation.
3. Trace startup, video ingestion, detection, tracking, re-identification, persistence, dashboard rendering, and shutdown.
4. Identify existing optimizations and the evidence for them.
5. Separate verified facts, documented intentions, hypotheses, and unknowns.
6. Write a repository-specific audit and detailed implementation plan.
7. Wait for explicit human approval before implementation.

Do not say the entire codebase was reviewed unless the review was actually comprehensive. Record files or areas not inspected and why.

## Non-negotiable rules

- The actual source code is the source of truth; documentation can be stale.
- Do not invent features, past bugs, fixes, benchmarks, schema fields, or successful tests.
- Do not perform broad rewrites when a small, isolated change will solve the problem.
- Preserve working interfaces and behavior unless a reviewed plan approves a breaking change.
- Do not delete data, enrolled identities, databases, model weights, or user files.
- Do not run destructive commands or modify databases during read-only audit.
- Do not install dependencies, change environments, or download large models without approval.
- Do not expose secrets in logs, reports, commits, or generated documentation.
- Do not put language-model calls in the frame-by-frame critical path.
- Do not send full videos or the complete database to a hosted model by default.
- Keep offline core behavior distinct from online/cloud features.
- Never claim a benchmark or test passed unless it was actually run and its result observed.
- When blocked, report the limitation instead of guessing.

## KAREN 2.0 priorities

1. Accuracy-first vision improvements.
2. Preserve and benchmark current optimizations before replacing them.
3. Investigate NVIDIA model/tracker options for compatibility, licensing, and deployment.
4. Add Nemotron via Nebius Token Factory for structured database Q&A.
5. Consider selective frame analysis only when structured records cannot answer the question.
6. Support honest online/offline behavior and failure handling.
7. Validate the hackathon's current official requirements before claiming compliance.

## Communication format

For each task, report:
- What you inspected.
- What you changed (if changes were approved).
- Why the change was necessary.
- Tests run and exact outcomes.
- Risks and unresolved issues.
- Files affected.

During planning-only tasks, do not modify application code. Present the plan for review.
