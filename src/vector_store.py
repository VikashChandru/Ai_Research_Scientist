"""
Simple in-process vector store (brute-force cosine similarity via numpy).
Good enough for the tens-to-low-hundreds of passages this agent handles per
session; avoids the extra native-dependency headache of installing faiss on
Windows.
"""
import numpy as np


class VectorStore:
    def __init__(self):
        self._vecs = np.zeros((0, 384), dtype="float32")
        self._texts = []
        self._metadatas = []

    def add(self, vecs: np.ndarray, texts: list, metadatas: list):
        if vecs.shape[0] == 0:
            return
        self._vecs = np.vstack([self._vecs, vecs]) if self._vecs.shape[0] else vecs
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list:
        if self._vecs.shape[0] == 0:
            return []
        query_vec = query_vec.reshape(1, -1)
        sims = (self._vecs @ query_vec.T).flatten()  # vectors are already normalized
        top_idx = np.argsort(-sims)[:k]
        return [
            {"text": self._texts[i], "metadata": self._metadatas[i], "score": float(sims[i])}
            for i in top_idx
        ]

    def __len__(self):
        return len(self._texts)