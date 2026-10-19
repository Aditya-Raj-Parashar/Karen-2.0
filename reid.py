# reid.py — Re-Identification: extract a feature embedding from a person crop

import numpy as np
import torch
import torchreid

class OSNetEmbedder:
    def __init__(self, model_name="osnet_ibn_x1_0", device="cuda"):
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=device,
            image_size=(256, 128)
        )

    def predict(self, crops):
        if not crops:
            return []
        
        # torchreid FeatureExtractor natively accepts list of numpy images
        features = self.extractor(crops)
        
        # Returns a pytorch tensor of shape (B, D)
        if isinstance(features, torch.Tensor):
            features = features.cpu().detach().numpy()
            
        return features

def load_reid(model_name: str = "osnet_ibn_x1_0", device: str = "cuda"):
    """Load the OSNet ReID model. Extractor handles weights download automatically."""
    print(f"Loading state-of-the-art ReID model: {model_name}")
    return OSNetEmbedder(model_name=model_name, device=device)

def get_embedding(reid_model, crop: np.ndarray) -> np.ndarray:
    """
    Extract a normalized feature vector from a person crop (BGR image).
    Returns a 1-D float32 numpy array.
    """
    features = reid_model.predict([crop])
    vec = np.asarray(features[0], dtype=np.float32)

    # Normalize to unit length so cosine similarity = dot product
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        vec = vec / norm
    return vec
