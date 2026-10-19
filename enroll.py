# enroll.py — Phase 1: Enroll a person from a short video

import cv2
import numpy as np

import config
import database
import detect
import reid


def enroll_person(video_path: str, label: str = None, db_path: str = config.DB_PATH) -> str:
    """
    Enroll a person from a ~12-second video.

    Steps:
    1. Load YOLO + ReID models
    2. Read the first 12 seconds of the video (sample every Nth frame)
    3. Detect the person, crop them, extract a feature embedding per frame
    4. Build a profile: averaged (prototype) + best-quality embedding
    5. Save to the gallery table and print the assigned ID

    Returns the assigned person_id string e.g. "ID1"
    """
    # ── Load models ───────────────────────────────────────────────────────────
    detector  = detect.load_detector(config.YOLO_MODEL)
    reid_model = reid.load_reid(config.REID_MODEL, config.REID_DEVICE)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * config.ENROLLMENT_SECONDS)

    all_embeddings = []
    quality_scores = []
    frame_idx = 0

    print(f"  Reading video (up to {config.ENROLLMENT_SECONDS}s, every {config.ENROLLMENT_FPS_SAMPLE} frames)...")

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= max_frames:
            break

        frame_idx += 1

        # Sample every Nth frame to get diverse viewpoints
        if frame_idx % config.ENROLLMENT_FPS_SAMPLE != 0:
            continue

        # Detect people in this frame
        detections = detect.detect_people(detector, frame,
                                          conf=0.35, device=config.DETECTOR_DEVICE)
        if not detections:
            continue

        # Take the largest/most-confident detection (assume one person in enrollment video)
        best = max(detections, key=lambda d: d[4])  # highest confidence
        x1, y1, x2, y2, conf, crop = best

        # Extract ReID embedding
        emb = reid.get_embedding(reid_model, crop)
        all_embeddings.append(emb)
        quality_scores.append(conf)

    cap.release()

    if not all_embeddings:
        print("  ⚠  No person detected in the video. Enrollment skipped.")
        return ""

    print(f"  Collected {len(all_embeddings)} embeddings from {frame_idx} frames.")

    # ── Build identity profile ────────────────────────────────────────────────
    stacked    = np.stack(all_embeddings)                        # shape (N, D)
    prototype  = stacked.mean(axis=0)                           # averaged embedding
    norm = float(np.linalg.norm(prototype))
    prototype  = prototype / norm if norm > 1e-12 else prototype # normalize

    best_idx   = int(np.argmax(quality_scores))
    best_emb   = all_embeddings[best_idx]

    # ── Assign ID and save ────────────────────────────────────────────────────
    person_id = database.next_person_id(db_path)
    label     = label or person_id

    database.save_person(
        db_path       = db_path,
        person_id     = person_id,
        label         = label,
        embeddings    = all_embeddings,
        prototype     = prototype,
        best          = best_emb,
        quality_scores= quality_scores,
        source_video  = video_path,
    )

    print(f"  ✅ Enrolled '{label}' as {person_id}")
    return person_id
