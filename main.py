# main.py — Entry point for the Multi-Camera Person Identification System
#
# USAGE:
#   Enroll a person:
#       python main.py --enroll people/person1.mp4 --label "Lakshay"
#
#   Run tracking on camera feeds:
#       python main.py --source feed/cam7.mp4 feed/cam7C.mp4 \
#                      --map-config map.json --map-image MAP.png
#
#   Both modes share the same database (karen_tracking.db by default).

import argparse
import sys
import time
from collections import deque
import numpy as np

import cv2

import config
import database
import detect
import reid
import match
import track
import draw
import map_view


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(description="Karen — Multi-Camera Person ID System")
    p.add_argument("--enroll",      help="Path to enrollment video (≈12 s of one person)")
    p.add_argument("--label",       help="Name/label for the enrolled person")
    p.add_argument("--source",      nargs="+", help="CCTV video paths for live tracking")
    p.add_argument("--db",          default=config.DB_PATH, help="SQLite database path")
    p.add_argument("--map-config",  default="map.json",     help="Camera map JSON")
    p.add_argument("--map-image",   default="MAP.png",       help="Floor-map image path")
    p.add_argument("--threshold",   type=float, default=config.SIMILARITY_THRESHOLD,
                   help="Cosine similarity threshold for recognition")
    p.add_argument("--no-display",   action="store_true",    help="Disable OpenCV windows")
    p.add_argument("--clear-gallery", action="store_true",   help="Delete all gallery entries and exit")
    p.add_argument("--detect-every",  type=int, default=3,   help="Run YOLO every N frames (default 3)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — ENROLLMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_enrollment(args):
    """Enroll a single person from a video file."""
    import enroll
    print(f"\n── Enrollment ──────────────────────────────")
    print(f"  Video : {args.enroll}")
    print(f"  Label : {args.label or '(auto)'}")
    enroll.enroll_person(args.enroll, label=args.label, db_path=args.db)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2–6 — LIVE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(args):
    """
    Main tracking loop across all provided camera feeds.

    For each frame of every camera:
      1. Detect people (YOLO)
      2. Extract ReID embeddings
      3. Update DeepSORT tracker → local IDs
      4. Match each track against the enrolled gallery
      5. Draw boxes and labels
      6. Render the 2D floor map
      7. Show everything in an OpenCV window
    """
    # ── Load models (once) ────────────────────────────────────────────────────
    print("\n── Loading models...")
    detector   = detect.load_detector(config.YOLO_MODEL)
    reid_model = reid.load_reid(config.REID_MODEL, config.REID_DEVICE)

    # ── Load enrolled gallery ─────────────────────────────────────────────────
    gallery = database.load_gallery(args.db)
    print(f"  Gallery: {len(gallery)} enrolled {'person' if len(gallery)==1 else 'people'}")
    for pid, data in gallery.items():
        print(f"    {pid} → {data['label']}")

    # ── Load map and camera configs ───────────────────────────────────────────
    camera_configs = []
    try:
        camera_configs = config.load_camera_configs(args.map_config)
        print(f"  Map config: {len(camera_configs)} cameras loaded from {args.map_config}")
    except Exception as e:
        print(f"  ⚠  Could not load map config: {e}")

    map_base = map_view.load_map_base(args.map_image)

    # ── Open video captures ───────────────────────────────────────────────────
    sources = args.source
    captures = []
    trackers = []
    cam_names = []

    for src in sources:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"  ⚠  Cannot open: {src} — skipping")
            continue
        captures.append(cap)
        trackers.append(track.build_tracker())
        # Use the filename stem as the camera name (or map to c1/c2/... if configured)
        name = _source_to_name(src, camera_configs)
        cam_names.append(name)
        print(f"  Camera '{name}' → {src}")

    if not captures:
        print("  No valid sources. Exiting.")
        return

    # person_paths: { person_id: (label, deque_of_map_positions) }
    person_paths = {pid: (data['label'], deque(maxlen=20)) for pid, data in gallery.items()}
    local_id_identity = [{} for _ in captures]   # local_id → (person_id, label)
    track_votes = [{} for _ in captures]         # local_id → deque of recent matches
    logged_tracks = [set() for _ in captures]    # local_id → tracked if already logged to DB
    frame_count = 0
    frame_count = 0

    print("\n── Tracking started. Press Q to quit. ──────")

    # Per-camera detection cache for frame-skip
    cam_detections_cache: dict = {}

    if not args.no_display:
        cv2.namedWindow("Karen Dashboard", cv2.WINDOW_NORMAL)

    while True:
        annotated_frames = []
        frame_count += 1

        for i, (cap, trk, cam_name) in enumerate(zip(captures, trackers, cam_names)):
            ok, frame = cap.read()
            if not ok:
                # Loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    continue

            # ── Step 1: Detect people (every N frames, otherwise reuse) ──────
            DETECT_EVERY = args.detect_every
            if frame_count % DETECT_EVERY == 0:
                detections = detect.detect_people(detector, frame,
                                                  conf=0.35, device=config.DETECTOR_DEVICE)
                cam_detections_cache[i] = detections
            else:
                detections = cam_detections_cache.get(i, [])

            # ── Step 2: Skip dense ReID, prepare for DeepSORT ─────────────
            ds_dets = []
            others_info = []
            for det in detections:
                x1, y1, x2, y2, conf, crop = det
                ds_dets.append(([x1, y1, x2-x1, y2-y1], conf, "person"))
                others_info.append({"crop": crop})

            # ── Step 3: Update DeepSORT tracker (Pure IoU) ────────────────
            # Pass dummy constant embeddings to force DeepSORT into spatial-only (IoU) mode
            dummy_embeds = [np.ones(512, dtype=np.float32) for _ in ds_dets]
            
            raw_tracks = trk.update_tracks(
                ds_dets,
                embeds=dummy_embeds,
                others=others_info
            )

            used_pids_this_frame = set()
            
            # Sort tracks by time-since-update to prioritize established tracks
            raw_tracks = sorted(raw_tracks, key=lambda t: t.time_since_update)

            # ── Step 4 & 5: Match identity + draw ─────────────────────────
            for t in raw_tracks:
                if t.is_deleted() or not t.is_confirmed():
                    continue

                local_id = int(t.track_id)
                x1, y1, x2, y2 = [int(v) for v in t.to_ltrb()]
                bbox = (x1, y1, x2, y2)

                # Expand voting window to 25 frames (much more stable)
                votes = track_votes[i].setdefault(local_id, deque(maxlen=25))

                # Lazy ReID: compute continuously while track is new, then slow down
                if len(votes) < 10 or frame_count % 8 == 0:
                    det_ref = t.get_det_supplementary()
                    if det_ref is not None:
                        crop = det_ref.get("crop")
                        if crop is not None:
                            emb = reid.get_embedding(reid_model, crop)
                            pid, label, sim = match.match_identity(emb, gallery, args.threshold)
                            if pid != "unknown":
                                votes.append((pid, label, sim))

                # Smoothing / Voting logic over a rolling window
                if len(votes) > 0:
                    from collections import Counter
                    c = Counter([v[0] for v in votes])
                    best_pid = c.most_common(1)[0][0]
                    
                    matching_votes = [v for v in votes if v[0] == best_pid]
                    best_label = matching_votes[0][1]
                    best_sim = max([v[2] for v in matching_votes])
                    
                    # Prevent twin glitch: Never have two "Lakshay"s in the same exact camera frame
                    if best_pid in used_pids_this_frame:
                        best_pid, best_label, best_sim = "unknown", "Unknown", 0.0
                    else:
                        used_pids_this_frame.add(best_pid)

                    final_pid, final_label, final_sim = best_pid, best_label, best_sim
                    local_id_identity[i][local_id] = (final_pid, final_label, final_sim)
                else:
                    final_pid, final_label, final_sim = local_id_identity[i].get(
                        local_id, ("unknown", "Unknown", 0.0))

                # Build display label
                if final_pid == "unknown":
                    display_label = f"Unknown | T{local_id}"
                    color = (120, 120, 120)
                else:
                    display_label = f"{final_label} | {final_pid} | Sim:{final_sim:.2f}"
                    color = draw.stable_color(final_pid)
                    # Update map trail under (label, deque) tuple
                    if final_pid not in person_paths:
                        person_paths[final_pid] = (final_label, deque(maxlen=20))
                    
                    if local_id not in logged_tracks[i]:
                        # Save event to DB
                        database.log_event(args.db, final_pid, cam_name, final_sim)
                        
                        # Save evidence crop
                        import os
                        os.makedirs("evidence", exist_ok=True)
                        det_sup = t.get_det_supplementary()
                        if det_sup is not None:
                            crop = det_sup.get("crop")
                            if crop is not None and crop.size > 0:
                                stamp = time.strftime("%Y%m%d_%H%M%S")
                                path = f"evidence/{final_pid}_{cam_name}_{stamp}.jpg"
                                cv2.imwrite(path, crop)
                        
                        logged_tracks[i].add(local_id)
                        
                    # Exclude C3 from adding points to the map trail
                    if cam_name.lower() != "c3":
                        map_pt = _project_to_map(cam_name, camera_configs, bbox, frame.shape)
                        if map_pt:
                            person_paths[final_pid][1].append(map_pt)

                draw.draw_track(frame, bbox, display_label, color)

            # Add camera name overlay
            cv2.putText(frame, cam_name, (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            annotated_frames.append(frame)

        # ── Step 6: Show Dashboard (2x2 Grid) ──────────────────────────────────
        if not args.no_display:
            # Resize everything using letterbox to prevent distortion
            tile_w, tile_h = 640, 360
            tiles = [letterbox_resize(f, (tile_w, tile_h)) for f in annotated_frames]
            
            # Map tile
            map_img = map_view.render_map(map_base, camera_configs, person_paths)
            tiles.append(cv2.resize(map_img, (tile_w, tile_h)))
            
            # Fill remaining slots with black if needed (for 2x2 = 4 slots)
            while len(tiles) < 4:
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            
            # Create the 2x2 grid
            row1 = cv2.hconcat([tiles[0], tiles[1]])
            row2 = cv2.hconcat([tiles[2], tiles[3]])
            dashboard = cv2.vconcat([row1, row2])
            
            cv2.imshow("Karen Dashboard", dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    # ── Clean up ──────────────────────────────────────────────────────────────
    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")


# ── Utility functions ─────────────────────────────────────────────────────────

def letterbox_resize(img: np.ndarray, new_shape: tuple = (640, 360), color: tuple = (0, 0, 0)) -> np.ndarray:
    """Resize maintaining aspect ratio by padding with color bars."""
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def _source_to_name(source_path: str, camera_configs: list) -> str:
    """
    Map a source file path to a camera name.
    Tries to match by stem (filename without extension) against config names.
    Falls back to the stem itself.
    """
    from pathlib import Path
    stem = Path(source_path).stem.lower()

    for cam in camera_configs:
        if cam["name"].lower() == stem:
            return cam["name"]

    # Common pattern: feed/lakshay.mp4 → we don't have that in config,
    # so return the stem (e.g. "lakshay")
    return stem


def _project_to_map(cam_name: str, camera_configs: list,
                    bbox: tuple, frame_shape: tuple):
    """
    Project a bounding-box bottom-center from camera-space to map-space using Trigonometry.
    Returns (map_x, map_y) or None if camera not in config.
    """
    cam = next((c for c in camera_configs if c["name"].lower() == cam_name.lower()), None)
    if cam is None:
        return None

    import math
    cx_px, cy_px = cam["position"][0], cam["position"][1]
    orientation = float(cam["orientation"])
    fov = float(cam["fov"])
    range_radius = float(cam.get("range_radius", 120))
    tilt_power = float(cam.get("tilt", 15)) / 10.0
    
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_cx = (x1 + x2) / 2.0
    
    # Depth based on bottom of bounding box (feet). 
    # y2 = frame_h means depth = 0 (standing right at the camera).
    # y2 = 0 means depth = range_radius
    # Tilt adjustment: power < 1 pushes towards camera, power > 1 pushes towards door
    raw_depth_ratio = max(0.0, min(1.0, (frame_h - y2) / max(frame_h, 1)))
    depth_ratio = math.pow(raw_depth_ratio, tilt_power)
    dist = depth_ratio * range_radius
    
    # Angle offset: horizontal center relative to frame width
    # 0 = center mapping to orientation angle.
    rel_x = (box_cx / max(frame_w, 1)) - 0.5
    angle_offset = rel_x * fov
    theta = orientation + angle_offset
    
    # Calculate map coordinates
    map_x = cx_px + dist * math.cos(math.radians(theta))
    map_y = cy_px + dist * math.sin(math.radians(theta))
    
    # Clamp to map bounds (785x390)
    map_x = max(0, min(785, int(map_x)))
    map_y = max(0, min(390, int(map_y)))
    
    return (map_x, map_y)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_args()

    if args.clear_gallery:
        print("── Clearing gallery... ─────────────────────")
        database.clear_gallery(args.db)
        sys.exit(0)
    elif args.enroll:
        # Phase 1 mode — enroll one person and exit
        run_enrollment(args)
    elif args.source:
        # Phase 2–6 mode — live CCTV tracking
        run_tracking(args)
    else:
        print("Usage:")
        print("  Clear:   python main.py --clear-gallery")
        print("  Enroll:  python main.py --enroll people/person1.mp4 --label 'Name'")
        print("  Track:   python main.py --source feed/cam7.mp4 [feed/cam7C.mp4 ...]")
        sys.exit(1)
