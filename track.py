# track.py — DeepSORT-based within-camera tracker

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


def build_tracker(max_age: int = 30) -> DeepSort:
    """Create a fresh DeepSORT tracker for one camera."""
    return DeepSort(
        max_age=max_age,
        n_init=1,               # Instant confirmation (removes the 3-second lag)
        max_cosine_distance=0.4,
        embedder=None,          # we supply our own ReID embeddings
    )


def update_tracker(tracker: DeepSort, detections: list, embeddings: list) -> list:
    """
    Feed detections + embeddings into DeepSORT.

    detections: list of (x1, y1, x2, y2, conf, crop)
    embeddings: list of np.ndarray (one per detection, same order)

    Returns list of active confirmed tracks:
        [{ 'local_id': int, 'bbox': (x1,y1,x2,y2), 'emb_index': int }, ...]
    emb_index is the index into the original detections list, or None if
    there is no fresh detection for this track in this frame.
    """
    # DeepSORT expects detections as ([left, top, w, h], conf, "person")
    ds_dets = []
    for (x1, y1, x2, y2, conf, _crop) in detections:
        ds_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    tracks = tracker.update_tracks(ds_dets, embeds=embeddings)

    active = []
    for track in tracks:
        if track.is_deleted() or not track.is_confirmed():
            continue
        x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
        det_ref = track.get_det_supplementary()
        emb_idx = det_ref.get("idx") if det_ref else None
        active.append({
            "local_id": int(track.track_id),
            "bbox":     (x1, y1, x2, y2),
            "emb_idx":  emb_idx,
        })
    return active
