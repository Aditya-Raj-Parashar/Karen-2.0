# KAREN 2.0 Project Memory

## Session 1: Initial Audit and Phase 1 Approval
**Context:**
- Conducted full codebase audit of KAREN 2.0 detecting pipeline (main.py, config.py, database.py, detect.py, track.py, match.py, reid.py).
- Designed KAREN 2.0 Implementation Plan:
  - Phase 1: Add `events` table to SQLite, log track sightings to DB, save one crop image to `/evidence/` per track.
  - Phase 2: Natural Language Q&A with Nemotron.
  - Phase 3: Selective VLM (Image Analysis).
  - Phase 4: Final Polish.
- User approved proceeding with Phase 1.
- Noted two new folders provided by user:
  - `feed/`: Contains camera videos for testing. Mapping available in `map.json`.
  - `people/`: Contains enrollment videos for testing/extraction.
- Noted requirement: Read instruction files in `KAREN_2_0_Project_Docs` every session and maintain this `memory.md`.
