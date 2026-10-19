# Karen: User Guide

Welcome to the **Karen** Multi-Camera Person Identity Tracking System! This guide describes how to install, set up, and operate the real-time CCTV tracking platform.

## 1. Prerequisites and Setup
Before running the tracker, ensure your Python environment is ready.

1. Install Python (tested on Python 3.8+).
2. Create and activate a Virtual Environment.
   *(The provided batch scripts assume a `.venv` directory exists in the main folder).*
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Enrolling Identities
The system uses Zero-Shot ReID feature extraction to identify people. Before tracking anyone, you must enroll their face/body features into the database.

1. Place high-quality, continuous videos containing the person into the `people/` folder (e.g., `people/aditya.mp4`).
2. You can use the automated `enroll.bat` script or run the Python script manually:
   ```bash
   python main.py --enroll people/john_doe.mp4 --label "John Doe"
   ```
3. To clear the existing gallery/database, run:
   ```bash
   python main.py --clear-gallery
   ```

*Tip: The `enroll.bat` script provides a batch enrollment reference that clears the gallery and enrolls multiple people sequentially.*

## 3. Starting the Tracking System
Once your gallery is populated, you can start the real-time Multi-CCTV tracking system.

1. Ensure your camera feeds (e.g., `.mp4` video files or live stream links) are accessible.
2. Run `run_cctv.bat` or use the command below:
   ```bash
   python main.py \
       --source feed/c1.mp4 feed/c2.mp4 feed/c3.mp4 \
       --map-config map.json \
       --map-image MAP.png \
       --threshold 0.7
   ```
3. A dashboard view will open showing all active camera feeds and a top-down projection map (`MAP.png`) tracking the movements of identified individuals across your space.
4. Press `Q` in the video window to stop tracking.

## 4. Key System Features
- **Live Floor Map Projection:** View real-time locations of tracked individuals on a 2D floor plan.
- **Dynamic Grid Dashboard:** Watch all camera feeds tiled smoothly in an easy-to-monitor visualization window.
- **Persistent Footprints:** Moving dots freeze at their last known location when someone steps out of view, reducing tracking loss on the map.
