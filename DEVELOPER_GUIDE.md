# Karen: Developer's Guide

This document outlines the architecture, internal modules, and future roadmap of the **Karen** Multi-Camera ReID Tracking System.

## 1. System Architecture
The codebase follows a single-responsibility modular architecture designed for high frames-per-second (FPS) optimization using DeepSORT caching.

- **`main.py`**: The central controller. Parses CLI arguments, manages the camera read loops, performs YOLO frame skipping when identities are tracked via DeepSORT, and coordinates output rendering.
- **`config.py`**: A centralized configuration manager holding thresholds, scaling factors, and paths. Responsible for parsing `map.json`.
- **`database.py`**: Handles SQLite interactions for the `karen_tracking.db`. Loads gallery target embeddings and stores new ones.
- **`detect.py`**: YOLOv8 Wrapper. Filters target frames for `person` class bounding boxes.
- **`reid.py`**: The feature extraction layer (currently utilizing MobileNetV2 from `deep_sort_realtime`). Transforms an isolated bounding box crop into a 128-float embedding signature.
- **`match.py`**: Performs embedding authentication utilizing cosine similarity against the database profile gallery.
- **`track.py`**: Initializes and configures the DeepSORT pure IoU logic (ReID occurs passively on new track creations or conditionally skipped interval frames).
- **`draw.py`**: Handles the logic for drawing bounding boxes, displaying labels, and selecting stable tracking colors for unique IDs.
- **`map_view.py`**: Contains the complex geometry calculations to project a 2D camera bounding box into a 2D pixel space on the top-down floor plan (`MAP.png`).
- **`rotate_cams.py`**: A visual debugging tool utilized to adjust FOV, tilt, range, and direction for `map.json` calibration using OpenCV trackbars.

## 2. Core Operational Logic
*   **Lazy ReID Evaluation:** ReID does not run on every frame. Pure IoU intersection tracking carries bounding boxes across frames. ReID verification only fires on newly detected identities or every 15 frames, vastly improving CPU/GPU performance.
*   **Temporal Stability Voting:** ReID inference suffers from momentary glitches (e.g. lighting spikes). A rolling sliding-window vote requires at least 3 positive ReID matches across the last 5 inferenced checks for an identity label to be formally locked to a track.

## 3. Current System Limitations
1.  **MobileNetV2 Capabilities:** The baseline uses MobileNetV2 which degrades when presented with heavy shadowing or nuanced similar clothing colors. Trading MobileNet for a dedicated surveillance network (e.g. `osnet_ain_x1_0`) improves match certainty inside dark corridors.
2.  **Occlusion Dropouts:** Dense crowds and cross-walking targets can cause DeepSORT's IoU tracker to merge bonding boxes and fail.

## 4. Extension Recommendations & Roadmap
We recommend four low-effort, high-impact features for subsequent sprints:

- **Geo-Fencing:** Incorporate polygonal alerts inside `map_view.py`. Issue warnings if coordinates encroach configured coordinate zones.
- **Path Heatmaps:** Create an export hook for `person_paths` inside `main.py` to write coordinates to CSV for spatial analytics.
- **Stale Track Cleanup:** Implement a timer inside `map_view.py`. Fade or remove trailing person markers after 60 seconds with no active camera sighting.
- **Live Gallery Updating:** Expand `match.py` to continuously insert `0.95+` (extremely confident) cosine similarity embeddings into the database. This acts as "auto-learning" and adapts to temporal lighting drift across the lifespan of a tracking session.
