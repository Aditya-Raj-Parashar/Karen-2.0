import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4,
            nn_budget=None
        )

    def update(self, detections, frame):
        """
        detections format:
        [
            [([x1, y1, x2, y2], confidence, class_id)],
            ...
        ]
        """

        bbox_xywh = []
        confidences = []

        for det in detections:
            print(det)
            x1, y1, x2, y2, conf, cls = det
            w = x2 - x1
            h = y2 - y1
            bbox_xywh.append((
                [x1, y1, w, h],
                float(conf),
                int(cls)
            ))

        tracks = self.tracker.update_tracks(
            bbox_xywh,
            frame=frame
        )

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            results.append({
                "track_id": track_id,
                "bbox": [int(l), int(t), int(r), int(b)]
            })

        return results