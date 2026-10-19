# KAREN Data Dictionary

> Populate only after inspecting the actual database schema and all relevant read/write paths.

## Database inventory

- Database location/configuration:
- Database engine/version:
- Initialization or migration code:
- Backup/reset behavior:
- Sensitive fields to avoid sending externally:

## Tables

### Table: [verified table name]

| Field | Type | Nullable | Meaning | Written by | Read by | Sent to hosted model? |
|---|---|---|---|---|---|---|
| To verify | | | | | | |

## Identity concepts

Document separately:
- Per-camera tracker ID.
- Persistent person identity.
- Identity-gallery entry.
- Cross-camera match and its confidence.
- Event/observation identifier.

Do not assume these are interchangeable.

## Evidence semantics

Define how the application distinguishes:
- Directly recorded observation.
- Tracker-derived association.
- Cross-camera match.
- Derived event.
- Language-model interpretation.
- Unknown or insufficient evidence.

## External context policy

Document the minimum fields allowed in Nemotron prompts, how records are filtered, and how sensitive or unnecessary data is excluded.
