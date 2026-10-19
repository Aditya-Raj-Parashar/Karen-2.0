# config.py — System settings and map configuration loader

import json

# ── Default paths ──────────────────────────────────────────────────────────────
DB_PATH          = "karen_tracking.db"
YOLO_MODEL       = "yolov8s.pt"
REID_MODEL       = "osnet_ibn_x1_0" # example models : osnet_x10, osnet_x1_0, osnet_ibn_x1_0, mobilenet
REID_DEVICE      = "cuda"          
DETECTOR_DEVICE  = "cuda"
SIMILARITY_THRESHOLD = 0.80     # cosine similarity cutoff for recognition
SIMILARITY_THRESHOLD_matching = 0.75 # lower threshold for matching during enrollment (to avoid false matches)
ENROLLMENT_FPS_SAMPLE = 1         # sample every Nth frame during enrollment

ENROLLMENT_SECONDS   = 16         # max seconds to read from an enrollment video


def load_camera_configs(map_json_path: str) -> list:
    """
    Load camera configurations from a JSON file.
    Returns a list of dicts, each with:
        name, position (x,y), orientation (degrees), fov (degrees), range_radius (px)
    """
    with open(map_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both {"cameras": [...]} and bare list formats
    cameras = data.get("cameras", data) if isinstance(data, dict) else data
    return cameras 
