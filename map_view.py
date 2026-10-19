# map_view.py — Render the 2D floor map with cameras and person trails

import math
import cv2
import numpy as np


def load_map_base(image_path: str, size: tuple = (785, 390)) -> np.ndarray:
    """Load a floor-map image and resize it. Creates a blank grid if not found."""
    img = cv2.imread(image_path) if image_path else None
    if img is not None:
        return cv2.resize(img, size)

    # Fallback: dark grid canvas
    w, h = size
    canvas = np.full((h, w, 3), (24, 28, 38), dtype=np.uint8)
    for x in range(0, w, 40):
        cv2.line(canvas, (x, 0), (x, h), (34, 40, 54), 1)
    for y in range(0, h, 40):
        cv2.line(canvas, (0, y), (w, y), (34, 40, 54), 1)
    return canvas


import math

def render_map(base: np.ndarray, camera_configs: list, person_paths: dict, show_cameras: bool = False) -> np.ndarray:
    """
    Draw person movement trails on the map.
    camera_configs: used for rendering cones if show_cameras is True
    person_paths:   dict { identity_str: [(x,y), (x,y), ...] }  (last N positions)
    """
    canvas = base.copy()

    if show_cameras:
        # Provide minimal visual if they just want to see where cameras are
        for cam in camera_configs:
            cx, cy = int(cam["position"][0]), int(cam["position"][1])
            cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4,  (95, 210, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, cam["name"], (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 239, 244), 2, cv2.LINE_AA)

    # ── Draw person trails ────────────────────────────────────────────────────
    for identity, data in person_paths.items():
        # data = (label, deque_of_positions)
        if isinstance(data, tuple):
            label, path = data
        else:
            label, path = str(identity), data  # backwards-compat

        if not path:
            continue

        # Pick a color from the same palette used in draw.py
        _PALETTE = [
            (63, 196, 255), (96, 174, 39),  (64, 126, 255), (218, 112, 214),
            (255, 191, 0),  (255, 105, 65), (0, 200, 150),  (180, 180, 60),
        ]
        color = _PALETTE[sum(ord(c) for c in str(identity)) % len(_PALETTE)]

        pts = list(path)

        # Current position dot + label
        px, py = pts[-1]
        cv2.circle(canvas, (px, py), 8, color, -1, cv2.LINE_AA)
        display_text = f"{label} ({identity})"
        cv2.putText(canvas, display_text, (px + 10, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 0), 2, cv2.LINE_AA)

    return canvas

