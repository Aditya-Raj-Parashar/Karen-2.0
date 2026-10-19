# draw.py — Draw bounding boxes and labels on video frames

import cv2
import numpy as np


# A fixed palette of colors for consistent identity coloring
_PALETTE = [
    (63, 196, 255), (96, 174, 39),  (64, 126, 255), (218, 112, 214),
    (255, 191, 0),  (255, 105, 65), (0, 200, 150),  (180, 180, 60),
]


def stable_color(identity: str) -> tuple:
    """Return a consistent BGR color for a given identity string or ID number."""
    idx = sum(ord(c) for c in str(identity)) % len(_PALETTE)
    return _PALETTE[idx]


def draw_track(frame: np.ndarray, bbox: tuple, label: str, color: tuple) -> None:
    """
    Draw a minimal UI on the frame (in-place).
    Instead of a heavy box, draws a center-bottom point and a text label.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    
    # Draw a distinct circle at the feet
    cv2.circle(frame, (cx, y2), 5, color, -1)
    cv2.circle(frame, (cx, y2), 5, (255, 255, 255), 1)

    # Draw a filled rectangle behind the text so it's easy to read
    # Place text above the top of the person
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bg_y1 = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)
