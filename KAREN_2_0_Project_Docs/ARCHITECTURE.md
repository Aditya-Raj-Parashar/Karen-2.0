# KAREN Architecture

> **Status: Audit required.** This document is a template for recording the verified architecture. Do not fill unknown details with assumptions.

## 1. System overview

- Application entry point: **To verify**
- Runtime/configuration loading: **To verify**
- Video ingestion: **To verify**
- Detector: **To verify against source**
- Tracker: **To verify against source**
- Re-identification: **To verify against source**
- Identity gallery and persistence: **To verify**
- Database schema: **To verify**
- Dashboard: **To verify**
- Floor-map projection: **To verify**
- Shutdown/error handling: **To verify**

## 2. Runtime data flow

Document the actual sequence from command invocation to shutdown. Cite file paths and functions for every major transition.

1. CLI/configuration → …
2. Video source initialization → …
3. Frame acquisition → …
4. Detection → …
5. Tracking → …
6. Re-identification and identity association → …
7. Persistence/events → …
8. Dashboard/floor-map output → …
9. Cleanup/shutdown → …

## 3. Module map

| Module/file | Verified responsibility | Inputs | Outputs | Dependencies | Notes |
|---|---|---|---|---|---|
| To fill after audit | | | | | |

## 4. Existing optimizations

| Optimization | Source location | Problem addressed | Evidence/test | Risk when changing |
|---|---|---|---|---|
| To verify | | | | |

## 5. Proposed KAREN 2.0 boundaries

These are intended boundaries to evaluate, not an implementation claim:

- Vision inference: detector and tracker deployment.
- KAREN core: identity association, confidence, temporal validation, event/data access.
- Persistence: local structured records.
- Intelligence: Nemotron-based answers grounded in retrieved records.
- Optional visual analysis: application-selected frames sent only to a compatible model.
- UI: familiar live dashboard plus chat and relevant evidence references.

## 6. Design constraints

- The real-time vision path must not block on language-model latency.
- Hosted model requests must be bounded and failure-tolerant.
- Local data access must be controlled by the application.
- A cross-camera match must preserve confidence/uncertainty.
- Online and offline behavior must be explicitly defined.
- All architecture changes require evidence and regression testing.
