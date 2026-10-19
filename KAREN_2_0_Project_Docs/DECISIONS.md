# Architecture Decision Records

Record consequential decisions here. Do not silently make irreversible or high-risk choices.

## ADR template

### ADR-NNN: Short title

- **Status:** Proposed / Accepted / Rejected / Superseded
- **Date:**
- **Decision owner:**
- **Context:** What problem or constraint prompted the decision?
- **Options considered:**
- **Decision:** What was selected?
- **Evidence:** Benchmarks, official documentation, code paths, or experiments.
- **Consequences:** Benefits, costs, risks, and migration impact.
- **Revisit when:** What evidence would cause this decision to be reconsidered?

---

## Proposed decision topics

Create a separate ADR when enough evidence exists for:
- NVIDIA detector selection.
- Tracker selection.
- Local versus remote vision inference.
- Cross-camera Re-ID strategy.
- Structured Q&A context format.
- Token Factory model and API integration.
- Optional frame-analysis model and policy.
- Online/offline behavior and replay strategy.
- Database schema changes, if required.

Do not mark any of these as accepted until reviewed.
