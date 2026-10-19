# database.py — Save and load person gallery from SQLite

import json
import sqlite3
import numpy as np


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Create the gallery table if it doesn't already exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gallery (
            person_id TEXT PRIMARY KEY,
            label TEXT,
            embeddings TEXT NOT NULL,
            prototype_embedding TEXT NOT NULL,
            best_embedding TEXT NOT NULL,
            quality_scores TEXT NOT NULL,
            source_video TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            camera_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
            confidence REAL
        )
    """)
    conn.commit()
    return conn


def save_person(db_path: str, person_id: str, label: str,
                embeddings: list, prototype: np.ndarray,
                best: np.ndarray, quality_scores: list,
                source_video: str) -> None:
    """Save (or update) an enrolled person in the gallery table."""
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO gallery (person_id, label, embeddings, prototype_embedding,
                             best_embedding, quality_scores, source_video)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(person_id) DO UPDATE SET
            label = excluded.label,
            embeddings = excluded.embeddings,
            prototype_embedding = excluded.prototype_embedding,
            best_embedding = excluded.best_embedding,
            quality_scores = excluded.quality_scores,
            source_video = excluded.source_video
    """, (
        person_id,
        label,
        json.dumps([e.tolist() for e in embeddings]),
        json.dumps(prototype.tolist()),
        json.dumps(best.tolist()),
        json.dumps(quality_scores),
        source_video,
    ))
    conn.commit()
    conn.close()


def load_gallery(db_path: str) -> dict:
    """
    Load all enrolled people from the gallery.
    Returns a dict: { person_id: { label, embeddings, prototype, best } }
    """
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM gallery").fetchall()
    conn.close()

    gallery = {}
    for row in rows:
        gallery[row["person_id"]] = {
            "label": row["label"] or row["person_id"],
            "embeddings": [np.array(e, dtype=np.float32) for e in json.loads(row["embeddings"])],
            "prototype":  np.array(json.loads(row["prototype_embedding"]), dtype=np.float32),
            "best":       np.array(json.loads(row["best_embedding"]),      dtype=np.float32),
        }
    return gallery


def next_person_id(db_path: str) -> str:
    """Return the next available ID string like ID1, ID2, ID3..."""
    conn = _connect(db_path)
    rows = conn.execute("SELECT person_id FROM gallery").fetchall()
    conn.close()

    used = set()
    for row in rows:
        pid = row["person_id"]
        if pid.startswith("ID"):
            try:
                used.add(int(pid[2:]))
            except ValueError:
                pass

    n = 1
    while n in used:
        n += 1
    return f"ID{n}"


def clear_gallery(db_path: str) -> None:
    """Delete all enrolled people from the gallery."""
    conn = _connect(db_path)
    conn.execute("DELETE FROM gallery")
    conn.commit()
    conn.close()
    print("  Gallery cleared.")

def log_event(db_path: str, person_id: str, camera_name: str, confidence: float) -> int:
    """Log an identity sighting event. Returns the event_id."""
    conn = _connect(db_path)
    cursor = conn.execute("""
        INSERT INTO events (person_id, camera_name, confidence)
        VALUES (?, ?, ?)
    """, (person_id, camera_name, confidence))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id

