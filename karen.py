from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

IMPORT_ERRORS: Dict[str, str] = {}

try:
    import torch
except Exception as exc:  # pragma: no cover - dependency check
    torch = None
    IMPORT_ERRORS["torch"] = str(exc)

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - dependency check
    YOLO = None
    IMPORT_ERRORS["ultralytics"] = str(exc)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as exc:  # pragma: no cover - dependency check
    DeepSort = None
    IMPORT_ERRORS["deep_sort_realtime"] = str(exc)

try:
    from torchreid.reid.utils import FeatureExtractor
except Exception as exc:  # pragma: no cover - dependency check
    FeatureExtractor = None
    IMPORT_ERRORS["torchreid"] = str(exc)

try:
    from flask import Flask, jsonify
except Exception as exc:  # pragma: no cover - dependency check
    Flask = None
    jsonify = None
    IMPORT_ERRORS["flask"] = str(exc)


PathLikeSource = Union[str, int]


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def stable_color(identifier: Union[int, str]) -> Tuple[int, int, int]:
    if isinstance(identifier, str):
        identifier = sum(ord(c) for c in identifier)
    palette = [
        (63, 196, 255),
        (96, 174, 39),
        (64, 126, 255),
        (218, 112, 214),
        (71, 99, 255),
        (255, 191, 0),
        (255, 105, 65),
        (0, 200, 150),
        (180, 180, 60),
        (200, 90, 170),
    ]
    return palette[identifier % len(palette)]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def bbox_iou_ltrb(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    left_area = float(max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1))
    right_area = float(max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1))
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def clamp_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_source_value(raw_value: str) -> PathLikeSource:
    cleaned = raw_value.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def is_live_source(source: PathLikeSource) -> bool:
    if isinstance(source, int):
        return True
    normalized = str(source).strip().lower()
    return normalized.startswith(
        (
            "rtsp://",
            "rtmp://",
            "http://",
            "https://",
            "udp://",
            "tcp://",
        )
    )


def discover_default_sources(base_dir: Path) -> List[str]:
    feed_dir = base_dir / "feed"
    video_exts = {".avi", ".mp4", ".mov", ".mkv"}

    if feed_dir.exists():
        feed_sources = sorted(
            path for path in feed_dir.iterdir() if path.suffix.lower() in video_exts
        )
        if feed_sources:
            return [f"{path.stem}={path.as_posix()}" for path in feed_sources]

    raise FileNotFoundError(
        "No camera sources were provided and no default videos were found in ./feed."
    )


def ensure_runtime_dependencies(need_api: bool) -> None:
    required = ["torch", "ultralytics", "deep_sort_realtime", "torchreid"]
    if need_api:
        required.append("flask")

    missing = [name for name in required if name in IMPORT_ERRORS]
    if not missing:
        return

    install_hint = (
        "Run the app with the project virtualenv interpreter:\n"
        "  .\\.venv\\Scripts\\python.exe karen.py\n"
        "If the venv is missing packages, install them with:\n"
        "  .\\.venv\\Scripts\\pip.exe install ultralytics opencv-python torch torchvision numpy scipy flask deep_sort_realtime torchreid gdown tensorboard"
    )
    details = "\n".join(f"- {name}: {IMPORT_ERRORS[name]}" for name in missing)
    raise RuntimeError(f"Missing runtime dependencies:\n{details}\n{install_hint}")


@dataclass
class CameraConfig:
    name: str
    source: PathLikeSource
    position: Optional[Tuple[int, int]] = None
    orientation: float = 0.0
    fov: float = 90.0
    range_radius: float = 100.0


@dataclass
class GlobalIdentity:
    global_id: int
    embedding: np.ndarray
    created_at: str
    updated_at: str
    samples: int = 1
    sightings_count: int = 0
    best_embedding: Optional[np.ndarray] = None
    best_quality: float = 0.0
    last_quality: float = 0.0


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self.lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id INTEGER PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    samples INTEGER NOT NULL DEFAULT 1,
                    sightings_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    camera TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stream_time_ms REAL,
                    local_track_id INTEGER,
                    confidence REAL,
                    similarity REAL,
                    frame_index INTEGER,
                    bbox TEXT NOT NULL,
                    map_point TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(person_id)
                );
                CREATE TABLE IF NOT EXISTS gallery (
                    person_id TEXT PRIMARY KEY,
                    label TEXT,
                    embeddings TEXT NOT NULL,
                    prototype_embedding TEXT NOT NULL,
                    best_embedding TEXT NOT NULL,
                    quality_scores TEXT NOT NULL,
                    source_video TEXT
                );
                """
            )
            self.connection.commit()

    def load_gallery(self) -> Dict[str, Any]:
        with self.lock:
            rows = self.connection.execute("SELECT * FROM gallery").fetchall()
        gallery = {}
        for row in rows:
            gallery[row["person_id"]] = {
                "person_id": row["person_id"],
                "label": row["label"],
                "embeddings": [np.array(e, dtype=np.float32) for e in json.loads(row["embeddings"])],
                "prototype_embedding": np.array(json.loads(row["prototype_embedding"]), dtype=np.float32),
                "best_embedding": np.array(json.loads(row["best_embedding"]), dtype=np.float32),
                "quality_scores": json.loads(row["quality_scores"]),
                "source_video": row["source_video"]
            }
        return gallery

    def insert_gallery_person(self, person: Dict[str, Any]) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO gallery (
                    person_id, label, embeddings, prototype_embedding, best_embedding, quality_scores, source_video
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    label = excluded.label,
                    embeddings = excluded.embeddings,
                    prototype_embedding = excluded.prototype_embedding,
                    best_embedding = excluded.best_embedding,
                    quality_scores = excluded.quality_scores,
                    source_video = excluded.source_video
                """,
                (
                    person["person_id"],
                    person.get("label"),
                    json.dumps([e.tolist() for e in person["embeddings"]]),
                    json.dumps(person["prototype_embedding"].tolist()),
                    json.dumps(person["best_embedding"].tolist()),
                    json.dumps(person["quality_scores"]),
                    person.get("source_video")
                )
            )
            self.connection.commit()

    def load_people(self) -> Dict[int, GlobalIdentity]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT person_id, embedding, created_at, updated_at, samples, sightings_count
                FROM people
                ORDER BY person_id
                """
            ).fetchall()

        identities: Dict[int, GlobalIdentity] = {}
        for row in rows:
            embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)
            identities[int(row["person_id"])] = GlobalIdentity(
                global_id=int(row["person_id"]),
                embedding=embedding,
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                samples=int(row["samples"]),
                sightings_count=int(row["sightings_count"]),
                best_embedding=embedding.copy(),
                best_quality=0.5,
                last_quality=0.5,
            )
        return identities

    def upsert_person(self, identity: GlobalIdentity) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO people (person_id, embedding, created_at, updated_at, samples, sightings_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at,
                    samples = excluded.samples,
                    sightings_count = excluded.sightings_count
                """,
                (
                    identity.global_id,
                    json.dumps(identity.embedding.tolist()),
                    identity.created_at,
                    identity.updated_at,
                    identity.samples,
                    identity.sightings_count,
                ),
            )
            self.connection.commit()

    def record_sighting(self, sighting: Dict[str, Any]) -> None:
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO sightings (
                    person_id, camera, timestamp, stream_time_ms, local_track_id,
                    confidence, similarity, frame_index, bbox, map_point
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sighting["person_id"],
                    sighting["camera"],
                    sighting["timestamp"],
                    sighting["stream_time_ms"],
                    sighting["local_track_id"],
                    sighting["confidence"],
                    sighting["similarity"],
                    sighting["frame_index"],
                    json.dumps(sighting["bbox"]),
                    json.dumps(sighting["map_point"]),
                ),
            )
            self.connection.execute(
                """
                UPDATE people
                SET sightings_count = sightings_count + 1,
                    updated_at = ?
                WHERE person_id = ?
                """,
                (sighting["timestamp"], sighting["person_id"]),
            )
            self.connection.commit()

    def people_summary(self) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                """
                SELECT
                    p.person_id,
                    p.created_at,
                    p.updated_at,
                    p.samples,
                    p.sightings_count,
                    (
                        SELECT s.camera
                        FROM sightings s
                        WHERE s.person_id = p.person_id
                        ORDER BY s.id DESC
                        LIMIT 1
                    ) AS last_camera,
                    (
                        SELECT s.timestamp
                        FROM sightings s
                        WHERE s.person_id = p.person_id
                        ORDER BY s.id DESC
                        LIMIT 1
                    ) AS last_seen
                FROM people p
                ORDER BY p.person_id
                """
            ).fetchall()

        return [
            {
                "person_id": int(row["person_id"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "samples": int(row["samples"]),
                "sightings_count": int(row["sightings_count"]),
                "last_camera": row["last_camera"],
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def person_history(self, person_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            person_row = self.connection.execute(
                """
                SELECT person_id, created_at, updated_at, samples, sightings_count
                FROM people
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()

            if person_row is None:
                return None

            sighting_rows = self.connection.execute(
                """
                SELECT camera, timestamp, stream_time_ms, local_track_id,
                       confidence, similarity, frame_index, bbox, map_point
                FROM sightings
                WHERE person_id = ?
                ORDER BY id
                """,
                (person_id,),
            ).fetchall()

        sightings = []
        for row in sighting_rows:
            sightings.append(
                {
                    "camera": row["camera"],
                    "timestamp": row["timestamp"],
                    "stream_time_ms": row["stream_time_ms"],
                    "local_track_id": row["local_track_id"],
                    "confidence": row["confidence"],
                    "similarity": row["similarity"],
                    "frame_index": row["frame_index"],
                    "bbox": json.loads(row["bbox"]),
                    "map_point": json.loads(row["map_point"]),
                }
            )

        return {
            "person_id": int(person_row["person_id"]),
            "created_at": person_row["created_at"],
            "updated_at": person_row["updated_at"],
            "samples": int(person_row["samples"]),
            "sightings_count": int(person_row["sightings_count"]),
            "sightings": sightings,
        }

    def export_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": iso_now(),
            "people": self.people_summary(),
        }
        people_with_history = []
        for person in payload["people"]:
            history = self.person_history(int(person["person_id"]))
            if history is not None:
                people_with_history.append(history)
        payload["people"] = people_with_history
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        with self.lock:
            self.connection.close()


class GlobalIdentityManager:
    def __init__(self, database: DatabaseManager, similarity_threshold: float, allow_new: bool = False):
        self.database = database
        self.similarity_threshold = similarity_threshold
        self.allow_new = allow_new
        self.identities = self.database.load_people()
        self.gallery = self.database.load_gallery()
        self.next_global_id = max([k for k in self.identities.keys() if isinstance(k, int)], default=0) + 1

    def _identity_similarity(self, identity: GlobalIdentity, embedding: np.ndarray) -> float:
        similarity = cosine_similarity(embedding, identity.embedding)
        if identity.best_embedding is not None:
            similarity = max(similarity, cosine_similarity(embedding, identity.best_embedding))
        return similarity

    def assign(
        self,
        embedding: np.ndarray,
        sample_quality: float = 0.5,
        preferred_id: Optional[Union[int, str]] = None,
        blocked_ids: Optional[set[Union[int, str]]] = None,
    ) -> Tuple[Union[int, str], float, bool]:
        blocked_ids = blocked_ids or set()
        preferred_similarity: Optional[float] = None
        
        best_gallery_id = None
        best_gallery_sim = -1.0
        
        for g_id, g_data in self.gallery.items():
            if g_id in blocked_ids:
                continue
            sim = cosine_similarity(embedding, g_data["best_embedding"])
            sim = max(sim, cosine_similarity(embedding, g_data["prototype_embedding"]))
            for emb in g_data["embeddings"]:
                sim = max(sim, cosine_similarity(embedding, emb))
            if sim > best_gallery_sim:
                best_gallery_sim = sim
                best_gallery_id = g_id
                
        if best_gallery_id is not None and best_gallery_sim >= self.similarity_threshold:
            return best_gallery_id, best_gallery_sim, False
            
        if not self.allow_new:
            return "unknown", max(best_gallery_sim, 0.0), False

        if preferred_id is not None and isinstance(preferred_id, int) and preferred_id in self.identities:
            preferred_identity = self.identities[int(preferred_id)]
            preferred_similarity = self._identity_similarity(preferred_identity, embedding)
            relaxed_keep_threshold = max(self.similarity_threshold - 0.18, 0.45)
            if preferred_similarity >= relaxed_keep_threshold:
                self._update_identity(
                    preferred_id,
                    embedding,
                    observed_similarity=preferred_similarity,
                    sample_quality=sample_quality,
                )
                return preferred_id, preferred_similarity, False

        best_id: Optional[int] = None
        best_similarity = -1.0

        for global_id, identity in self.identities.items():
            if global_id in blocked_ids:
                continue
            similarity = self._identity_similarity(identity, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = global_id

        if best_id is not None and best_similarity >= self.similarity_threshold:
            self._update_identity(
                best_id,
                embedding,
                observed_similarity=best_similarity,
                sample_quality=sample_quality,
            )
            return best_id, best_similarity, False

        if preferred_id is not None and isinstance(preferred_id, int) and preferred_id in self.identities and preferred_similarity is not None:
            self._update_identity(
                int(preferred_id),
                embedding,
                observed_similarity=preferred_similarity,
                sample_quality=sample_quality,
            )
            return preferred_id, float(preferred_similarity or 0.0), False

        timestamp = iso_now()
        global_id = self.next_global_id
        self.next_global_id += 1

        normalized_embedding = normalize_vector(embedding)
        identity = GlobalIdentity(
            global_id=global_id,
            embedding=normalized_embedding,
            created_at=timestamp,
            updated_at=timestamp,
            best_embedding=normalized_embedding.copy(),
            best_quality=sample_quality,
            last_quality=sample_quality,
        )
        self.identities[global_id] = identity
        self.database.upsert_person(identity)
        return global_id, 1.0, True

    def increment_sightings(self, global_id: Union[int, str], timestamp: str) -> None:
        if isinstance(global_id, int) and global_id in self.identities:
            identity = self.identities[global_id]
            identity.sightings_count += 1
            identity.updated_at = timestamp
            self.database.upsert_person(identity)

    def _update_identity(
        self,
        global_id: Union[int, str],
        embedding: np.ndarray,
        observed_similarity: float,
        sample_quality: float,
    ) -> None:
        if not isinstance(global_id, int) or global_id not in self.identities:
            return
        identity = self.identities[global_id]
        normalized_embedding = normalize_vector(embedding)
        identity.last_quality = sample_quality

        if identity.best_embedding is None:
            identity.best_embedding = normalized_embedding.copy()
            identity.best_quality = sample_quality
        elif sample_quality >= max(identity.best_quality + 0.05, 0.72):
            identity.best_embedding = normalized_embedding.copy()
            identity.best_quality = sample_quality
        elif sample_quality >= 0.58 and observed_similarity >= self.similarity_threshold:
            refreshed = 0.7 * identity.best_embedding + 0.3 * normalized_embedding
            identity.best_embedding = normalize_vector(refreshed)
            identity.best_quality = max(identity.best_quality, sample_quality)

        relaxed_update_threshold = max(self.similarity_threshold - 0.18, 0.45)
        if observed_similarity >= relaxed_update_threshold and sample_quality >= 0.18:
            quality_strength = min(max(sample_quality, 0.0), 1.0)
            similarity_strength = min(
                max((observed_similarity - relaxed_update_threshold) / max(1e-6, 1.0 - relaxed_update_threshold), 0.0),
                1.0,
            )
            blend = 0.05 + 0.25 * max(quality_strength, similarity_strength)
            if sample_quality >= 0.6 and observed_similarity >= self.similarity_threshold:
                blend = max(blend, 0.2)

            identity.embedding = normalize_vector(
                (1.0 - blend) * identity.embedding + blend * normalized_embedding
            )
            identity.samples += 1

        identity.updated_at = iso_now()
        self.database.upsert_person(identity)


class ReIDExtractor:
    def __init__(self, model_name: str, device: str):
        if FeatureExtractor is None:
            raise RuntimeError(f"torchreid is unavailable: {IMPORT_ERRORS.get('torchreid', 'unknown error')}")

        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device,
            verbose=False,
        )

    def extract(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        if not crops:
            return []

        rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]
        features = self.extractor(rgb_crops)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        return [normalize_vector(np.asarray(feature, dtype=np.float32)) for feature in features]


class CameraStream:
    def __init__(
        self,
        config: CameraConfig,
        max_age: int,
        nn_budget: Optional[int],
        loop_video: bool,
        record_interval_seconds: float,
        low_latency: bool,
        max_buffer_drain: int,
    ):
        if DeepSort is None:
            raise RuntimeError(
                f"deep_sort_realtime is unavailable: {IMPORT_ERRORS.get('deep_sort_realtime', 'unknown error')}"
            )

        self.config = config
        self.loop_video = loop_video
        self.record_interval_seconds = record_interval_seconds
        self.low_latency = low_latency
        self.max_buffer_drain = max(0, max_buffer_drain)
        self.capture = self._open_capture()
        self.source_fps = self._read_source_fps()
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            max_cosine_distance=0.35,
            nn_budget=nn_budget,
            embedder=None,
        )
        self.frame_index = 0
        self.local_to_global: Dict[int, Union[int, str]] = {}
        self.local_last_recorded_at: Dict[int, float] = {}
        self.local_last_recorded_gid: Dict[int, Union[int, str]] = {}
        self.display_queue: Deque[Tuple[np.ndarray, List[Dict[str, Any]]]] = deque()
        self.pending_frame: Optional[np.ndarray] = None
        self.previous_processed_tracks: Optional[List[Dict[str, Any]]] = None
        self.local_embedding_cache: Dict[int, np.ndarray] = {}
        self.local_embedding_quality: Dict[int, float] = {}
        self.local_embedding_frame: Dict[int, int] = {}
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_tracks: List[Dict[str, Any]] = []

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.config.source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source for camera '{self.config.name}': {self.config.source}")
        if self.low_latency:
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return capture

    def _read_source_fps(self) -> Optional[float]:
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps < 1.0:
            return None
        return fps

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.low_latency:
            ok, frame = self._read_latest_frame()
        else:
            ok, frame = self.capture.read()
        if ok:
            self.frame_index += 1
            return True, frame

        if not self.loop_video:
            return False, None

        self.capture.release()
        self.capture = self._open_capture()
        ok, frame = self.capture.read()
        if ok:
            self.frame_index += 1
            return True, frame

        return False, None

    def _read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.capture.grab():
            return False, None

        ok, frame = self.capture.retrieve()
        if not ok:
            return False, None

        for _ in range(self.max_buffer_drain):
            if not self.capture.grab():
                break
            next_ok, next_frame = self.capture.retrieve()
            if not next_ok:
                break
            frame = next_frame

        return True, frame

    def current_stream_time_ms(self) -> float:
        return float(self.capture.get(cv2.CAP_PROP_POS_MSEC))

    def should_record(
        self,
        local_track_id: int,
        global_id: Union[int, str],
        stream_time_ms: float,
        timestamp_monotonic: float,
    ) -> bool:
        last_at = self.local_last_recorded_at.get(local_track_id)
        last_gid = self.local_last_recorded_gid.get(local_track_id)
        current_marker = stream_time_ms if stream_time_ms > 0 else timestamp_monotonic * 1000.0
        minimum_gap = self.record_interval_seconds * 1000.0
        if last_at is None or last_gid != global_id:
            self.local_last_recorded_at[local_track_id] = current_marker
            self.local_last_recorded_gid[local_track_id] = global_id
            return True

        if current_marker - last_at >= minimum_gap:
            self.local_last_recorded_at[local_track_id] = current_marker
            self.local_last_recorded_gid[local_track_id] = global_id
            return True

        return False

    def cleanup_inactive(self, active_track_ids: Iterable[int]) -> None:
        active_set = set(active_track_ids)
        stale = [track_id for track_id in self.local_to_global if track_id not in active_set]
        for track_id in stale:
            self.local_to_global.pop(track_id, None)
            self.local_last_recorded_at.pop(track_id, None)
            self.local_last_recorded_gid.pop(track_id, None)
            self.local_embedding_cache.pop(track_id, None)
            self.local_embedding_quality.pop(track_id, None)
            self.local_embedding_frame.pop(track_id, None)

    def release(self) -> None:
        self.capture.release()


class TrackingCoordinator:
    def __init__(
        self,
        cameras: List[CameraConfig],
        model_path: str,
        detector_device: str,
        detector_imgsz: int,
        detector_half: bool,
        process_every: int,
        reid_refresh_every: int,
        reid_model_name: str,
        reid_device: str,
        detection_confidence: float,
        similarity_threshold: float,
        min_detection_quality: float,
        cross_camera_similarity_threshold: float,
        low_latency: bool,
        max_buffer_drain: int,
        db_path: Path,
        json_path: Optional[Path],
        loop_video: bool,
        max_age: int,
        nn_budget: Optional[int],
        record_interval_seconds: float,
        map_image_path: Optional[Path],
        map_size: Tuple[int, int],
    ):
        if YOLO is None:
            raise RuntimeError(f"ultralytics is unavailable: {IMPORT_ERRORS.get('ultralytics', 'unknown error')}")

        self.lock = threading.RLock()
        self.detector_device = detector_device
        self.detector_imgsz = detector_imgsz
        self.detector_half = detector_half
        self.process_every = max(1, process_every)
        self.reid_refresh_every = max(1, reid_refresh_every)
        self.detection_confidence = detection_confidence
        self.min_detection_quality = clamp01(min_detection_quality)
        self.cross_camera_similarity_threshold = clamp01(
            max(cross_camera_similarity_threshold, similarity_threshold - 0.12)
        )
        self.low_latency = low_latency
        self.max_buffer_drain = max(0, max_buffer_drain)
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.reid_extractor = ReIDExtractor(reid_model_name, reid_device)
        self.database = DatabaseManager(db_path)
        self.global_ids = GlobalIdentityManager(self.database, similarity_threshold)
        self.json_path = json_path
        self.json_dirty = False
        self.last_json_export_at = 0.0
        self.camera_positions = self._resolve_camera_positions(cameras, map_size)
        self.map_size = map_size
        self.map_base = self._build_map_base(map_image_path, map_size)
        self.map_paths: Dict[Union[int, str], Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=12))
        self.cameras = [
            CameraStream(
                config=CameraConfig(
                    name=config.name,
                    source=config.source,
                    position=self.camera_positions[config.name],
                ),
                max_age=max_age,
                nn_budget=nn_budget,
                loop_video=loop_video,
                record_interval_seconds=record_interval_seconds,
                low_latency=low_latency and is_live_source(config.source),
                max_buffer_drain=max_buffer_drain,
            )
            for config in cameras
        ]
        self.display_frame_period_s = self._resolve_display_frame_period_s()
        self.display_wait_ms = max(1, int(round(self.display_frame_period_s * 1000.0))) if self.display_frame_period_s > 0.0 else 1

    def _resolve_camera_positions(
        self,
        cameras: List[CameraConfig],
        map_size: Tuple[int, int],
    ) -> Dict[str, Tuple[int, int]]:
        width, height = map_size
        explicit = {camera.name: camera.position for camera in cameras if camera.position is not None}
        unresolved = [camera for camera in cameras if camera.position is None]

        if not unresolved:
            return {camera.name: camera.position for camera in cameras if camera.position is not None}

        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 3

        resolved = dict(explicit)
        count = len(unresolved)
        for index, camera in enumerate(unresolved):
            angle = -math.pi / 2 + (2 * math.pi * index / max(count, 1))
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            resolved[camera.name] = (x, y)

        return resolved

    def _build_map_base(self, map_image_path: Optional[Path], map_size: Tuple[int, int]) -> np.ndarray:
        width, height = map_size
        if map_image_path is not None and map_image_path.exists():
            image = cv2.imread(str(map_image_path))
            if image is not None:
                return cv2.resize(image, (width, height))

        canvas = np.full((height, width, 3), (24, 28, 38), dtype=np.uint8)
        for x in range(0, width, 40):
            cv2.line(canvas, (x, 0), (x, height), (34, 40, 54), 1)
        for y in range(0, height, 40):
            cv2.line(canvas, (0, y), (width, y), (34, 40, 54), 1)
        cv2.putText(
            canvas,
            "Global Person Map",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 235, 240),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _resolve_display_frame_period_s(self) -> float:
        if any(camera.low_latency for camera in self.cameras):
            return 0.0

        valid_fps = [camera.source_fps for camera in self.cameras if camera.source_fps is not None]
        if not valid_fps:
            return 0.0

        target_fps = min(valid_fps)
        return 1.0 / target_fps

    def process_camera(self, camera: CameraStream) -> bool:
        ok, frame = camera.read()
        if not ok or frame is None:
            return False

        detection_records = self._extract_detections_from_result(self._run_detector([frame])[0], frame)
        detections, embeddings, qualities = self._build_embeddings_for_camera(camera, detection_records, frame.shape)
        self._update_camera_tracks(camera, frame, detections, embeddings, qualities, round_identity_hints={})
        self._maybe_export_json()
        return True

    def process_round(self) -> bool:
        round_payload: List[Tuple[CameraStream, np.ndarray]] = []
        read_any = False
        for camera in self.cameras:
            ok, frame = camera.read()
            if ok and frame is not None:
                read_any = True
                if self._should_process_camera_frame(camera):
                    round_payload.append((camera, frame))
                else:
                    latest_tracks = [dict(track) for track in (camera.previous_processed_tracks or [])]
                    camera.latest_frame = self._annotate_frame_from_tracks(
                        frame,
                        camera.config.name,
                        latest_tracks,
                    )
                    camera.latest_tracks = latest_tracks
                    camera.pending_frame = None
                    camera.display_queue.clear()

        if not read_any:
            return False

        if round_payload:
            frames = [frame for _, frame in round_payload]
            detection_results = self._run_detector(frames)

            all_crops: List[np.ndarray] = []
            per_camera_records: List[List[Dict[str, Any]]] = []
            round_identity_hints: Dict[Union[int, str], Dict[str, Any]] = {}

            for (camera, frame), result in zip(round_payload, detection_results):
                detection_records = self._extract_detections_from_result(result, frame)
                self._mark_reusable_embeddings(camera, detection_records)
                per_camera_records.append(detection_records)
                for record in detection_records:
                    if record["needs_extract"]:
                        all_crops.append(record["crop"])

            extracted_embeddings = self.reid_extractor.extract(all_crops)
            extracted_iter = iter(extracted_embeddings)

            for (camera, frame), detection_records in zip(
                round_payload,
                per_camera_records,
            ):
                detections, embeddings, qualities = self._finalize_detection_records(
                    camera,
                    detection_records,
                    extracted_iter,
                )
                self._update_camera_tracks(
                    camera,
                    frame,
                    detections,
                    embeddings,
                    qualities,
                    round_identity_hints=round_identity_hints,
                )

        self._maybe_export_json()
        return True

    def _should_process_camera_frame(self, camera: CameraStream) -> bool:
        return ((camera.frame_index - 1) % self.process_every) == 0

    def _run_detector(self, frames: List[np.ndarray]) -> List[Any]:
        return self.model(
            frames,
            verbose=False,
            classes=[0],
            conf=self.detection_confidence,
            device=self.detector_device,
            imgsz=self.detector_imgsz,
            half=self.detector_half,
        )

    def _extract_detections_from_result(
        self,
        result: Any,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        detection_records: List[Dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            clamped = clamp_bbox((x1, y1, x2, y2), frame.shape)
            if clamped is None:
                continue
            x1, y1, x2, y2 = clamped
            confidence = float(box.conf[0].item())
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            quality = self._compute_sample_quality(frame.shape, (x1, y1, x2, y2), confidence)
            if not self._accept_detection(frame.shape, (x1, y1, x2, y2), confidence, quality):
                continue
            detection_records.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "ltwh": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "confidence": confidence,
                    "crop": crop,
                    "quality": quality,
                    "needs_extract": True,
                    "embedding": None,
                }
            )

        return detection_records

    def _build_embeddings_for_camera(
        self,
        camera: CameraStream,
        detection_records: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[List[Tuple[List[float], float, str]], List[np.ndarray], List[float]]:
        self._mark_reusable_embeddings(camera, detection_records)
        crops_to_extract = [record["crop"] for record in detection_records if record["needs_extract"]]
        extracted_embeddings = iter(self.reid_extractor.extract(crops_to_extract))
        return self._finalize_detection_records(camera, detection_records, extracted_embeddings)

    def _mark_reusable_embeddings(
        self,
        camera: CameraStream,
        detection_records: List[Dict[str, Any]],
    ) -> None:
        previous_tracks = camera.previous_processed_tracks or []
        if not previous_tracks:
            return

        used_track_ids: set[int] = set()
        for record in detection_records:
            best_track: Optional[Dict[str, Any]] = None
            best_iou = 0.0
            for track in previous_tracks:
                local_track_id = int(track["local_track_id"])
                if local_track_id in used_track_ids:
                    continue
                overlap = bbox_iou_ltrb(record["bbox"], tuple(track["bbox"]))
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track

            if best_track is None or best_iou < 0.65:
                continue

            local_track_id = int(best_track["local_track_id"])
            cached_embedding = camera.local_embedding_cache.get(local_track_id)
            cached_quality = camera.local_embedding_quality.get(local_track_id, 0.0)
            cached_frame = camera.local_embedding_frame.get(local_track_id, -10**9)
            frame_delta = camera.frame_index - cached_frame
            quality_gain = record["quality"] - cached_quality

            should_reuse = (
                cached_embedding is not None
                and frame_delta <= self.reid_refresh_every * self.process_every
                and quality_gain < 0.08
            )
            if should_reuse:
                record["needs_extract"] = False
                record["embedding"] = cached_embedding
                used_track_ids.add(local_track_id)

    def _finalize_detection_records(
        self,
        camera: CameraStream,
        detection_records: List[Dict[str, Any]],
        extracted_embeddings: Iterable[np.ndarray],
    ) -> Tuple[List[Tuple[List[float], float, str]], List[np.ndarray], List[float]]:
        detections: List[Tuple[List[float], float, str]] = []
        embeddings: List[np.ndarray] = []
        qualities: List[float] = []
        extracted_iter = iter(extracted_embeddings)

        for record in detection_records:
            embedding = record["embedding"]
            if embedding is None:
                embedding = next(extracted_iter)
            detections.append((record["ltwh"], record["confidence"], "person"))
            embeddings.append(embedding)
            qualities.append(float(record["quality"]))

        return detections, embeddings, qualities

    def _update_camera_tracks(
        self,
        camera: CameraStream,
        frame: np.ndarray,
        detections: List[Tuple[List[float], float, str]],
        embeddings: List[np.ndarray],
        qualities: List[float],
        round_identity_hints: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    ) -> None:
        if round_identity_hints is None:
            round_identity_hints = {}
        others = [{"det_index": index} for index in range(len(detections))]
        tracks = camera.tracker.update_tracks(detections, embeds=embeddings, others=others)

        active_track_ids: List[int] = []
        display_tracks: List[Dict[str, Any]] = []
        annotated = frame.copy()
        reserved_global_ids: set[Union[int, str]] = set()

        for track in tracks:
            if track.is_deleted():
                continue

            local_track_id = int(track.track_id)
            detection_ref = track.get_det_supplementary()
            if detection_ref is None and getattr(track, "time_since_update", 0) > 0:
                continue

            display_bbox = self._track_bbox(track)
            confidence = float(track.get_det_conf() or 0.0)

            if not track.is_confirmed():
                if detection_ref is not None:
                    self._draw_tentative_track(
                        annotated,
                        camera.config.name,
                        local_track_id,
                        display_bbox,
                        confidence,
                    )
                continue

            active_track_ids.append(local_track_id)
            global_id = camera.local_to_global.get(local_track_id)
            similarity = None
            sample_quality = self._compute_sample_quality(frame.shape, display_bbox, confidence)

            if detection_ref is not None:
                det_index = int(detection_ref["det_index"])
                embedding = embeddings[det_index]
                sample_quality = qualities[det_index]
                blocked_global_ids = reserved_global_ids.copy()
                if global_id is not None:
                    blocked_global_ids.discard(global_id)
                hinted_global_id = self._find_cross_camera_hint(
                    embedding=embedding,
                    sample_quality=sample_quality,
                    camera_name=camera.config.name,
                    round_identity_hints=round_identity_hints,
                    blocked_ids=blocked_global_ids,
                    current_global_id=global_id,
                )
                if hinted_global_id is not None:
                    global_id = hinted_global_id
                global_id, similarity, _ = self.global_ids.assign(
                    embedding,
                    sample_quality=sample_quality,
                    preferred_id=global_id,
                    blocked_ids=blocked_global_ids,
                )
                camera.local_to_global[local_track_id] = global_id
                camera.local_embedding_cache[local_track_id] = embedding
                camera.local_embedding_quality[local_track_id] = sample_quality
                camera.local_embedding_frame[local_track_id] = camera.frame_index
                round_identity_hints[global_id] = {
                    "camera_name": camera.config.name,
                    "embedding": embedding,
                    "quality": sample_quality,
                }
                stream_time_ms = camera.current_stream_time_ms()

                if camera.should_record(
                    local_track_id,
                    global_id,
                    stream_time_ms,
                    time.monotonic(),
                ):
                    self._record_sighting(
                        camera=camera,
                        global_id=global_id,
                        local_track_id=local_track_id,
                        confidence=confidence,
                        similarity=similarity,
                        frame=frame,
                        frame_index=camera.frame_index,
                        bbox=display_bbox,
                        stream_time_ms=stream_time_ms,
                    )

            display_tracks.append(
                {
                    "local_track_id": local_track_id,
                    "global_id": global_id,
                    "bbox": display_bbox,
                    "confidence": confidence,
                    "similarity": similarity,
                    "quality": round(sample_quality, 3),
                }
            )
            self._draw_track(annotated, camera.config.name, display_tracks[-1])
            if global_id is not None:
                reserved_global_ids.add(global_id)

        camera.cleanup_inactive(active_track_ids)
        camera.pending_frame = None
        camera.display_queue.clear()
        camera.latest_frame = annotated
        camera.latest_tracks = [dict(track) for track in display_tracks]
        camera.previous_processed_tracks = [dict(track) for track in display_tracks]

    def _accept_detection(
        self,
        frame_shape: Tuple[int, int, int],
        bbox: Tuple[int, int, int, int],
        confidence: float,
        quality: float,
    ) -> bool:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        area_ratio = (bbox_w * bbox_h) / max(frame_w * frame_h, 1)
        height_ratio = bbox_h / max(frame_h, 1)
        aspect_ratio = bbox_w / max(bbox_h, 1)

        if quality < self.min_detection_quality:
            return False
        if area_ratio < 0.003 and confidence < 0.65:
            return False
        if height_ratio < 0.10 and confidence < 0.70:
            return False
        if (aspect_ratio < 0.18 or aspect_ratio > 1.05) and confidence < 0.80:
            return False
        return True

    def _find_cross_camera_hint(
        self,
        embedding: np.ndarray,
        sample_quality: float,
        camera_name: str,
        round_identity_hints: Dict[Union[int, str], Dict[str, Any]],
        blocked_ids: set[Union[int, str]],
        current_global_id: Optional[Union[int, str]],
    ) -> Optional[Union[int, str]]:
        if sample_quality < 0.32:
            return current_global_id

        best_id = current_global_id
        best_similarity = -1.0
        for global_id, hint in round_identity_hints.items():
            if global_id in blocked_ids:
                continue
            if hint["camera_name"] == camera_name:
                continue
            similarity = cosine_similarity(embedding, hint["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = global_id

        if best_id is None or best_similarity < self.cross_camera_similarity_threshold:
            return current_global_id
        return best_id

    def _track_bbox(self, track: Any) -> Tuple[int, int, int, int]:
        original = track.to_ltrb(orig=True, orig_strict=True)
        if original is None:
            original = track.to_ltrb()
        x1, y1, x2, y2 = [int(round(value)) for value in original]
        return x1, y1, x2, y2

    def _compute_sample_quality(
        self,
        frame_shape: Tuple[int, int, int],
        bbox: Tuple[int, int, int, int],
        confidence: float,
    ) -> float:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)

        area_ratio = (bbox_w * bbox_h) / max(frame_w * frame_h, 1)
        height_ratio = bbox_h / max(frame_h, 1)
        aspect_ratio = bbox_w / max(bbox_h, 1)

        conf_score = clamp01(
            (confidence - self.detection_confidence) / max(1e-6, 1.0 - self.detection_confidence)
        )
        size_score = clamp01(area_ratio / 0.08)
        height_score = clamp01(height_ratio / 0.45)
        aspect_score = 1.0 - clamp01(abs(aspect_ratio - 0.45) / 0.45)

        return (
            0.35 * conf_score
            + 0.30 * height_score
            + 0.20 * size_score
            + 0.15 * aspect_score
        )

    def _project_map_point(
        self,
        camera_name: str,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int]:
        camera_x, camera_y = self.camera_positions[camera_name]
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        offset_x = int((((center_x / max(frame_w, 1)) - 0.5) * 60.0))
        offset_y = int((((center_y / max(frame_h, 1)) - 0.5) * 60.0))
        return camera_x + offset_x, camera_y + offset_y

    def _record_sighting(
        self,
        camera: CameraStream,
        global_id: Union[int, str],
        local_track_id: int,
        confidence: float,
        similarity: Optional[float],
        frame: np.ndarray,
        frame_index: int,
        bbox: Tuple[int, int, int, int],
        stream_time_ms: float,
    ) -> None:
        timestamp = iso_now()
        map_point = self._project_map_point(camera.config.name, bbox, frame.shape)
        sighting = {
            "person_id": global_id,
            "camera": camera.config.name,
            "timestamp": timestamp,
            "stream_time_ms": stream_time_ms,
            "local_track_id": local_track_id,
            "confidence": round(confidence, 4),
            "similarity": None if similarity is None else round(similarity, 4),
            "frame_index": frame_index,
            "bbox": list(bbox),
            "map_point": list(map_point),
        }

        with self.lock:
            self.database.record_sighting(sighting)
            if isinstance(global_id, int):
                self.global_ids.increment_sightings(global_id, timestamp)
            self.map_paths[global_id].append(map_point)
            self.json_dirty = True

    def _maybe_export_json(self, force: bool = False) -> None:
        if self.json_path is None or not self.json_dirty:
            return

        now = time.monotonic()
        if not force and now - self.last_json_export_at < 1.0:
            return

        with self.lock:
            self.database.export_json(self.json_path)
            self.last_json_export_at = now
            self.json_dirty = False

    def _draw_track(self, frame: np.ndarray, camera_name: str, track_info: Dict[str, Any]) -> None:
        x1, y1, x2, y2 = track_info["bbox"]
        global_id = track_info["global_id"]
        local_track_id = track_info["local_track_id"]
        color = stable_color(global_id or local_track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_parts = [camera_name, f"L{local_track_id}"]
        if global_id is not None:
            label_parts.append(f"G{global_id}")
        if track_info["similarity"] is not None:
            label_parts.append(f"{track_info['similarity']:.2f}")
        label = " ".join(label_parts)

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - label_size[1] - 10)),
            (x1 + label_size[0] + 8, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (18, 18, 18),
            2,
            cv2.LINE_AA,
        )

    def _draw_tentative_track(
        self,
        frame: np.ndarray,
        camera_name: str,
        local_track_id: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        x1, y1, x2, y2 = bbox
        color = (80, 220, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"{camera_name} T{local_track_id} DET {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - label_size[1] - 10)),
            (x1 + label_size[0] + 8, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (18, 18, 18),
            1,
            cv2.LINE_AA,
        )

    def _annotate_frame_from_tracks(
        self,
        frame: np.ndarray,
        camera_name: str,
        tracks: List[Dict[str, Any]],
    ) -> np.ndarray:
        annotated = frame.copy()
        for track_info in tracks:
            self._draw_track(annotated, camera_name, track_info)
        return annotated

    def _track_key(self, track_info: Dict[str, Any]) -> Tuple[str, Union[int, str]]:
        if track_info.get("global_id") is not None:
            return ("g", track_info["global_id"])
        return ("l", int(track_info["local_track_id"]))

    def _interpolate_tracks(
        self,
        previous_tracks: List[Dict[str, Any]],
        current_tracks: List[Dict[str, Any]],
        alpha: float,
    ) -> List[Dict[str, Any]]:
        previous_map = {self._track_key(track): track for track in previous_tracks}
        current_map = {self._track_key(track): track for track in current_tracks}
        keys = list(dict.fromkeys(list(previous_map.keys()) + list(current_map.keys())))
        interpolated: List[Dict[str, Any]] = []

        for key in keys:
            previous = previous_map.get(key)
            current = current_map.get(key)

            if previous is not None and current is not None:
                prev_bbox = previous["bbox"]
                curr_bbox = current["bbox"]
                bbox = [
                    int(round((1.0 - alpha) * prev_bbox[index] + alpha * curr_bbox[index]))
                    for index in range(4)
                ]
                interpolated.append(
                    {
                        "local_track_id": current["local_track_id"],
                        "global_id": current["global_id"],
                        "bbox": bbox,
                        "confidence": float(((previous.get("confidence") or 0.0) + (current.get("confidence") or 0.0)) / 2.0),
                        "similarity": self._average_optional_numeric(previous.get("similarity"), current.get("similarity")),
                        "quality": float(((previous.get("quality") or 0.0) + (current.get("quality") or 0.0)) / 2.0),
                    }
                )
            elif current is not None:
                interpolated.append(dict(current))
            elif previous is not None:
                interpolated.append(dict(previous))

        return interpolated

    def _average_optional_numeric(
        self,
        left: Optional[float],
        right: Optional[float],
    ) -> Optional[float]:
        if left is None and right is None:
            return None
        if left is None:
            return float(right)
        if right is None:
            return float(left)
        return float((left + right) / 2.0)

    def _decorate_tile(
        self,
        frame: np.ndarray,
        title: str,
        subtitle: str,
        tile_size: Tuple[int, int],
    ) -> np.ndarray:
        tile_width, tile_height = tile_size
        tile = cv2.resize(frame, (tile_width, tile_height))
        cv2.rectangle(tile, (0, 0), (tile_width, 38), (16, 20, 28), -1)
        cv2.putText(
            tile,
            title,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 244, 248),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            subtitle,
            (14, tile_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (222, 228, 235),
            1,
            cv2.LINE_AA,
        )
        return tile

    def render_camera_grid(self, tile_size: Tuple[int, int] = (480, 270)) -> Optional[np.ndarray]:
        available = [camera for camera in self.cameras if camera.latest_frame is not None]
        if not available:
            return None

        tile_width, tile_height = tile_size
        prepared: List[np.ndarray] = []

        for index, camera in enumerate(self.cameras[:3], start=1):
            if camera.latest_frame is None:
                blank = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
                prepared.append(self._decorate_tile(blank, f"CAM {index}", "No frame", tile_size))
                continue

            subtitle = f"{camera.config.name} | tracks: {len(camera.latest_tracks)} | frame: {camera.frame_index}"
            prepared.append(
                self._decorate_tile(camera.latest_frame, f"CAM {index}", subtitle, tile_size)
            )

        map_tile = self._decorate_tile(
            self.render_map(),
            "GLOBAL MAP",
            f"people: {len(self.map_paths)}",
            tile_size,
        )
        prepared.append(map_tile)

        while len(prepared) < 6:
            blank = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
            prepared.append(self._decorate_tile(blank, "EMPTY", "", tile_size))

        row1 = cv2.hconcat(prepared[:2])
        row2 = cv2.hconcat(prepared[2:4])
        dashboard = cv2.vconcat([row1, row2])
        cv2.putText(
            dashboard,
            "Karen CCTV Wall",
            (18, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return dashboard

    def render_map(self) -> np.ndarray:
        canvas = self.map_base.copy()

        for global_id, points in self.map_paths.items():
            if not points:
                continue
            color = stable_color(global_id)
            point_array = np.array(list(points), dtype=np.int32)
            if len(point_array) > 1:
                cv2.polylines(canvas, [point_array], False, color, 2, cv2.LINE_AA)
            current_x, current_y = point_array[-1]
            cv2.circle(canvas, (current_x, current_y), 8, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"G{global_id}",
                (current_x + 10, current_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

        for camera in self.cameras:
            x, y = self.camera_positions[camera.config.name]
            config = camera.config
            
            orientation = config.orientation
            fov = config.fov
            radius = int(config.range_radius)
            start_angle = orientation - fov / 2.0
            end_angle = orientation + fov / 2.0
            axes = (radius, radius)
            cv2.ellipse(canvas, (x, y), axes, 0, start_angle, end_angle, (60, 70, 90), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (x, y), axes, 0, start_angle, end_angle, (100, 150, 250), 1, cv2.LINE_AA)
            
            cv2.circle(canvas, (x, y), 14, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 5, (95, 210, 255), -1, cv2.LINE_AA)
            
            arrow_len = 20
            end_x = int(x + arrow_len * math.cos(math.radians(orientation)))
            end_y = int(y + arrow_len * math.sin(math.radians(orientation)))
            cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.3)
            
            cv2.putText(
                canvas,
                camera.config.name,
                (x + 12, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (235, 239, 244),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"Updated {iso_now()}",
            (24, canvas.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 205, 212),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def camera_payload(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "generated_at": iso_now(),
                "cameras": [
                    {
                        "name": camera.config.name,
                        "source": camera.config.source,
                        "position": list(self.camera_positions[camera.config.name]),
                        "frame_index": camera.frame_index,
                        "active_tracks": camera.latest_tracks,
                    }
                    for camera in self.cameras
                ],
            }

    def people_payload(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "generated_at": iso_now(),
                "people": self.database.people_summary(),
            }

    def person_payload(self, person_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.database.person_history(person_id)

    def close(self) -> None:
        self._maybe_export_json(force=True)
        for camera in self.cameras:
            camera.release()
        self.database.close()


def create_app(coordinator: TrackingCoordinator) -> Any:
    if Flask is None or jsonify is None:
        raise RuntimeError(f"flask is unavailable: {IMPORT_ERRORS.get('flask', 'unknown error')}")

    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "generated_at": iso_now(),
                "camera_count": len(coordinator.cameras),
            }
        )

    @app.get("/people")
    def people() -> Any:
        return jsonify(coordinator.people_payload())

    @app.get("/person/<int:person_id>")
    def person(person_id: int) -> Any:
        payload = coordinator.person_payload(person_id)
        if payload is None:
            return jsonify({"error": f"person {person_id} not found"}), 404
        return jsonify(payload)

    @app.get("/cameras")
    def cameras() -> Any:
        return jsonify(coordinator.camera_payload())

    return app


def start_api_server(coordinator: TrackingCoordinator, host: str, port: int) -> threading.Thread:
    app = create_app(coordinator)
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread


def parse_camera_positions(raw_positions: Optional[List[str]]) -> Dict[str, Tuple[int, int]]:
    resolved: Dict[str, Tuple[int, int]] = {}
    if not raw_positions:
        return resolved

    for raw_position in raw_positions:
        if "=" not in raw_position:
            raise ValueError(f"Invalid camera position '{raw_position}'. Expected name=x,y")
        name, coords = raw_position.split("=", 1)
        x_raw, y_raw = coords.split(",", 1)
        resolved[name.strip()] = (int(x_raw.strip()), int(y_raw.strip()))
    return resolved


def parse_camera_configs(
    raw_sources: Optional[List[str]],
    position_overrides: Dict[str, Tuple[int, int]],
    working_dir: Path,
) -> List[CameraConfig]:
    source_entries = raw_sources or discover_default_sources(working_dir)
    configs: List[CameraConfig] = []

    for index, entry in enumerate(source_entries, start=1):
        if "=" in entry:
            name, source_value = entry.split("=", 1)
            camera_name = name.strip()
            source = parse_source_value(source_value)
        else:
            source = parse_source_value(entry)
            camera_name = f"cam{index}"

        if isinstance(source, str):
            source_path = Path(source)
            if not source_path.is_absolute():
                source = str((working_dir / source_path).resolve())
            else:
                source = str(source_path)

        configs.append(
            CameraConfig(
                name=camera_name,
                source=source,
                position=position_overrides.get(camera_name),
            )
        )

    return configs


def default_reid_device() -> str:
    if torch is not None and bool(torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def default_detector_device() -> str:
    if torch is not None and bool(torch.cuda.is_available()):
        return "0"
    return "cpu"


def resolve_detector_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized in {"cuda", "gpu"}:
        return "0"
    return str(device)


def default_detector_half() -> bool:
    return torch is not None and bool(torch.cuda.is_available())


def print_status(message: str) -> None:
    print(message, flush=True)


def print_cuda_status(stage: str) -> None:
    if torch is None or not torch.cuda.is_available():
        return

    device_name = torch.cuda.get_device_name(0)
    allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    print_status(
        f"[CUDA:{stage}] {device_name} | allocated={allocated_mb:.1f} MB | reserved={reserved_mb:.1f} MB"
    )


def run_enrollment(
    video_path: str,
    label: Optional[str],
    db_path: Path,
    model_path: str,
    detector_device: str,
    detector_imgsz: int,
    detector_half: bool,
    reid_model_name: str,
    reid_device: str,
    det_conf: float,
) -> None:
    if YOLO is None:
        raise RuntimeError("ultralytics is unavailable")
    
    print_status(f"Starting enrollment for {video_path}...")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open enrollment video: {video_path}")
        
    db = DatabaseManager(db_path)
    gallery = db.load_gallery()
    used_ids = set()
    for gid in gallery.keys():
        if gid.startswith("ID"):
            try:
                used_ids.add(int(gid[2:]))
            except ValueError:
                pass
                
    next_id_num = max(used_ids) + 1 if used_ids else 1
    new_person_id = f"ID{next_id_num}"
    
    detector = YOLO(model_path)
    try:
        detector.fuse()
    except Exception:
        pass
        
    extractor = ReIDExtractor(reid_model_name, reid_device)
    all_embeddings = []
    quality_scores = []
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    max_frames = int(fps * 12)
    
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= max_frames:
            break
        frame_index += 1
        if frame_index % 3 != 0:
            continue
            
        results = detector(
            frame,
            verbose=False,
            classes=[0],
            conf=det_conf,
            device=detector_device,
            imgsz=detector_imgsz,
            half=detector_half,
        )
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                clamped = clamp_bbox((x1, y1, x2, y2), frame.shape)
                if not clamped:
                    continue
                x1, y1, x2, y2 = clamped
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    quality = float(conf)
                    emb = extractor.extract([crop])[0]
                    all_embeddings.append(emb)
                    quality_scores.append(quality)
                break
            if len(all_embeddings) == len(quality_scores):
                break

    capture.release()
    if not all_embeddings:
        print_status("No person detected in the enrollment video.")
        return
        
    print_status(f"Extracted {len(all_embeddings)} embeddings from {frame_index} frames.")
    embeddings_np = np.stack(all_embeddings)
    mean_emb = np.mean(embeddings_np, axis=0)
    prototype_emb = normalize_vector(mean_emb)
    best_idx = int(np.argmax(quality_scores))
    best_emb = embeddings_np[best_idx]
    
    person_record = {
        "person_id": new_person_id,
        "label": label or new_person_id,
        "embeddings": [emb for emb in embeddings_np],
        "prototype_embedding": prototype_emb,
        "best_embedding": best_emb,
        "quality_scores": quality_scores,
        "source_video": video_path,
    }
    db.insert_gallery_person(person_record)
    print_status(f"Enrollment successful! Person assigned ID: {new_person_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-camera person tracking with YOLOv8, DeepSORT, OSNet ReID, SQLite logging, map view, and Flask API."
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Camera source, optionally named, for example --source cam1=feed/video1_1.avi",
    )
    parser.add_argument(
        "--camera-position",
        action="append",
        help="Camera map position as name=x,y, for example --camera-position cam1=120,80",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--detector-device", default=default_detector_device(), help="YOLO device, for example cpu or 0")
    parser.add_argument("--detector-imgsz", type=int, default=512, help="YOLO inference size")
    parser.add_argument("--detector-half", action="store_true", default=default_detector_half(), help="Use FP16 for YOLO when running on CUDA")
    parser.add_argument("--detector-full", action="store_false", dest="detector_half", help="Disable FP16 YOLO inference")
    parser.add_argument("--reid-model", default="osnet_x0_25", help="Torchreid model name")
    parser.add_argument("--reid-device", default=default_reid_device(), help="Torchreid device")
    parser.add_argument("--reid-refresh-every", type=int, default=2, help="Reuse local ReID embeddings for stable tracks and refresh every N processed frames")
    parser.add_argument("--det-conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--min-det-quality", type=float, default=0.18, help="Secondary person-box quality filter to suppress ghost detections")
    parser.add_argument("--global-threshold", type=float, default=0.7, help="Cosine similarity threshold for global ID reuse")
    parser.add_argument("--cross-camera-threshold", type=float, default=0.62, help="Relaxed similarity threshold used only for same-round cross-camera matching")
    parser.add_argument("--low-latency", action="store_true", default=True, help="Prefer freshest live-camera frames instead of processing old buffered frames")
    parser.add_argument("--ordered-playback", action="store_false", dest="low_latency", help="Preserve buffered frame order instead of low-latency live behavior")
    parser.add_argument("--buffer-drain", type=int, default=2, help="How many extra buffered frames to discard per live-camera read")
    parser.add_argument("--db-path", default="karen_tracking.db", help="SQLite database file")
    parser.add_argument("--json-path", default="karen_tracking.json", help="Optional JSON export path")
    parser.add_argument("--disable-json", action="store_true", help="Disable JSON snapshot export")
    parser.add_argument("--serve-api", action="store_true", help="Start the Flask API server")
    parser.add_argument("--api-host", default="127.0.0.1", help="Flask host")
    parser.add_argument("--api-port", type=int, default=5000, help="Flask port")
    parser.add_argument("--loop", action="store_true", help="Loop video files when they end")
    parser.add_argument("--max-age", type=int, default=30, help="DeepSORT max age")
    parser.add_argument("--nn-budget", type=int, default=100, help="DeepSORT appearance budget")
    parser.add_argument("--process-every", type=int, default=2, help="Run detection/ReID every Nth frame while reusing the latest known tracks in-between")
    parser.add_argument("--record-interval", type=float, default=1.0, help="Minimum seconds between logged sightings for the same local track")
    parser.add_argument("--map-image", help="Optional floor-map image path")
    parser.add_argument("--map-width", type=int, default=785, help="Map canvas width")
    parser.add_argument("--map-height", type=int, default=390, help="Map canvas height")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processing rounds")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display windows")
    parser.add_argument("--enroll", help="Path to enrollment video")
    parser.add_argument("--enroll-label", help="Optional label for enrolled person")
    parser.add_argument("--map-config", help="JSON map config for cameras")
    parser.add_argument("--allow-new", action="store_true", help="Allow new global IDs during live tracking")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.process_every < 1:
        raise ValueError("--process-every must be at least 1.")
    if args.buffer_drain < 0:
        raise ValueError("--buffer-drain must be 0 or greater.")
    args.detector_device = resolve_detector_device(args.detector_device)
    ensure_runtime_dependencies(need_api=args.serve_api)
    if torch is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if str(args.detector_device).lower() == "cpu":
        args.detector_half = False

    working_dir = Path.cwd()
    
    if args.enroll:
        run_enrollment(
            video_path=args.enroll,
            label=args.enroll_label,
            db_path=Path(args.db_path).resolve(),
            model_path=args.model,
            detector_device=args.detector_device,
            detector_imgsz=args.detector_imgsz,
            detector_half=args.detector_half,
            reid_model_name=args.reid_model,
            reid_device=args.reid_device,
            det_conf=args.det_conf,
        )
        return
        
    position_overrides = parse_camera_positions(args.camera_position)
    camera_configs = parse_camera_configs(args.source, position_overrides, working_dir)
    
    if args.map_config:
        try:
            with open(args.map_config, "r", encoding="utf-8") as f:
                map_config_data = json.load(f)
            if "cameras" in map_config_data:
                conf_list = map_config_data["cameras"]
            else:
                conf_list = [map_config_data] if isinstance(map_config_data, dict) else map_config_data
            
            for cam_conf in conf_list:
                for camera in camera_configs:
                    if camera.name == cam_conf.get("name"):
                        if "position" in cam_conf:
                            camera.position = tuple(cam_conf["position"])
                        if "orientation" in cam_conf:
                            camera.orientation = float(cam_conf["orientation"])
                        if "fov" in cam_conf:
                            camera.fov = float(cam_conf["fov"])
                        if "range_radius" in cam_conf:
                            camera.range_radius = float(cam_conf["range_radius"])
        except Exception as e:
            print_status(f"Warning: Failed to load map config: {e}")
            
    json_path = None if args.disable_json else Path(args.json_path).resolve()

    print_status(f"Python: {Path(sys.executable)}")
    print_status(f"Detector device: {args.detector_device}")
    print_status(f"Detector image size: {args.detector_imgsz}")
    print_status(f"Detector FP16: {args.detector_half}")
    print_status(f"Process every Nth frame: {args.process_every}")
    print_status(f"ReID refresh every N processed frames: {args.reid_refresh_every}")
    print_status(f"ReID device: {args.reid_device}")
    print_status(f"Min detection quality: {args.min_det_quality}")
    print_status(f"Global similarity threshold: {args.global_threshold}")
    print_status(f"Cross-camera threshold: {args.cross_camera_threshold}")
    print_status(f"Low latency mode: {args.low_latency}")
    print_status(f"Buffer drain: {args.buffer_drain}")
    print_status("Preparing camera sources...")
    for camera in camera_configs:
        print_status(f"  - {camera.name}: {camera.source}")

    if len(camera_configs) > 1 and str(args.detector_device).lower() == "cpu":
        print_status(
            "Running multiple cameras on CPU can take a while before the first window updates."
        )

    print_status("Loading YOLO and OSNet models. The first startup can take a bit...")

    coordinator = TrackingCoordinator(
        cameras=camera_configs,
        model_path=args.model,
        detector_device=args.detector_device,
        detector_imgsz=args.detector_imgsz,
        detector_half=args.detector_half,
        process_every=args.process_every,
        reid_refresh_every=args.reid_refresh_every,
        reid_model_name=args.reid_model,
        reid_device=args.reid_device,
        detection_confidence=args.det_conf,
        similarity_threshold=args.global_threshold,
        min_detection_quality=args.min_det_quality,
        cross_camera_similarity_threshold=args.cross_camera_threshold,
        low_latency=args.low_latency,
        max_buffer_drain=args.buffer_drain,
        db_path=Path(args.db_path).resolve(),
        json_path=json_path,
        loop_video=args.loop,
        max_age=args.max_age,
        nn_budget=args.nn_budget,
        record_interval_seconds=args.record_interval,
        map_image_path=Path(args.map_image).resolve() if args.map_image else None,
        map_size=(args.map_width, args.map_height),
    )
    coordinator.global_ids.allow_new = args.allow_new
    print_cuda_status("after-init")
    if not args.no_display:
        print_status(f"Display wait: {coordinator.display_wait_ms} ms")

    api_thread: Optional[threading.Thread] = None
    try:
        if args.serve_api:
            api_thread = start_api_server(coordinator, args.api_host, args.api_port)
            print_status(f"Flask API running on http://{args.api_host}:{args.api_port}")

        if args.no_display:
            print_status("Display disabled. Processing video without OpenCV windows.")
        else:
            cv2.namedWindow("Karen CCTV Dashboard", cv2.WINDOW_NORMAL)
            print_status("OpenCV windows should appear after the first processed frame.")
            print_status("Press 'q' or ESC in the display window to stop.")

        rounds = 0
        while True:
            loop_started_at = time.monotonic()
            processed_any = coordinator.process_round()

            if not processed_any:
                break

            if not args.no_display:
                grid = coordinator.render_camera_grid()
                if grid is not None:
                    cv2.imshow("Karen CCTV Dashboard", grid)
                wait_ms = 1
                if coordinator.display_frame_period_s > 0.0:
                    remaining_s = coordinator.display_frame_period_s - (time.monotonic() - loop_started_at)
                    wait_ms = max(1, int(round(max(0.0, remaining_s) * 1000.0)))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (27, ord("q")):
                    break

            rounds += 1
            if rounds == 1:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                print_cuda_status("after-first-frame")
                print_status("First frame batch processed.")
            if args.max_frames is not None and rounds >= args.max_frames:
                break

        if args.no_display:
            print_status("Tracking finished without display.")
        else:
            cv2.destroyAllWindows()
            print_status("Tracking stopped.")

        if api_thread is not None:
            print_status("API thread will stop automatically when the process exits.")
    finally:
        coordinator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
