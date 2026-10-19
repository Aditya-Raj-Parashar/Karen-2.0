# detect.py — YOLO-based person detection

import cv2
import numpy as np
from ultralytics import YOLO
import config


def load_detector(model_path: str = config.YOLO_MODEL):
    """Load the YOLO model once and reuse it."""
    model = YOLO(model_path)
    print("Detection Model:", model_path)
    return model


def detect_people(detector, frame: np.ndarray, conf: float = 0.35, device: str = "cpu") -> list:
    """
    Run YOLO on a frame and return a list of detected people.
    Each item: (x1, y1, x2, y2, confidence, crop)
    """
    results = detector(frame, classes=[0], conf=conf, device=device, verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        # Clamp to frame boundaries
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes

        confidence = float(box.conf[0])
        crop = frame[y1:y2, x1:x2].copy()

        if crop.size > 0:
            detections.append((x1, y1, x2, y2, confidence, crop))

    return detections
