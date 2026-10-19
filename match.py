# match.py — Compare an embedding against the enrolled gallery

import numpy as np
import config

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns a value in [-1, 1]."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_identity(embedding: np.ndarray, gallery: dict, threshold: float = config.SIMILARITY_THRESHOLD_matching):
    """
    Find the best matching person in the gallery.

    Returns:
        (person_id, label, similarity)  — if a match is found above threshold
        ("unknown", "Unknown", score)   — otherwise
    """
    best_id    = None
    best_label = "Unknown"
    best_sim   = -1.0

    for person_id, data in gallery.items():
        sims = []
        # Compare current detection against EVERY embedding stored for this person
        # This is much more robust than just checking an average (prototype)
        for enrolled_emb in data["embeddings"]:
            sim = cosine_similarity(embedding, enrolled_emb)
            sims.append(sim)

        if len(sims) == 0:
            continue

        # maxing out -> person_sim = max(sims)    
        sims.sort(reverse=True)
        person_sim = sum(sims[:3]) / min(3, len(sims))

        
        if person_sim > best_sim:
            best_sim   = person_sim
            best_id    = person_id
            best_label = data["label"]

    if best_sim >= threshold:
        return best_id, best_label, round(best_sim, 4)

    return "unknown", "Unknown", round(max(best_sim, 0.0), 4)
