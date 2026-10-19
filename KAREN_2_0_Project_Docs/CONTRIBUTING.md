# Contributing to KAREN

## Before coding

1. Read `AGENTS.md`, `GEMINI.md`, `PROJECT_CONTEXT.md`, and the relevant architecture docs.
2. Understand the existing implementation and tests.
3. Check `PLAN.md`, `TASKS.md`, and `DECISIONS.md`.
4. Confirm that the task has been approved.
5. Identify compatibility and migration risks.

## Implementation standards

- Prefer small, focused changes.
- Preserve existing interfaces where practical.
- Avoid unrelated cleanup in feature changes.
- Avoid new dependencies unless the benefit is clear and documented.
- Use explicit configuration for model, endpoint, device, and mode choices where appropriate.
- Handle missing files, unavailable cameras, malformed records, and network failures gracefully.
- Keep inference, persistence, and UI responsibilities separated.
- Do not block real-time tracking on cloud language-model calls.

## Before reporting completion

- Run relevant tests and state exactly which ones ran.
- Report failures and skipped tests honestly.
- Update documentation when behavior or setup changes.
- Update `CHANGELOG.md` for meaningful user-visible changes.
- Update `DECISIONS.md` for consequential architecture decisions.
- Never claim that code was tested on hardware or a deployment environment unless it was.

## Pull request / change summary template

- Purpose:
- Files changed:
- Behavior changed:
- Compatibility impact:
- Tests run and results:
- Risks/limitations:
- Follow-up work:
