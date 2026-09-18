"""
Local embedding model (no external API calls, runs fully offline after the
first download of the model weights). Cached with Streamlit's resource cache
by the caller so it only loads once per session.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")