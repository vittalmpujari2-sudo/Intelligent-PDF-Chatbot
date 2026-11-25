"""
vector_database.py - Turns chunks into embeddings and searches them with FAISS.

- The embedding model (all-MiniLM-L6-v2) is loaded ONLY ONCE and reused.
- Vectors are normalised, so FAISS inner-product search = cosine similarity.
"""

from functools import lru_cache
from typing import Dict, List

import faiss
import numpy as np

import config


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model one time (cached). First run downloads ~90 MB."""
    from sentence_transformers import SentenceTransformer  # imported here so the app starts fast

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")


def _embed(texts: List[str]) -> np.ndarray:
    """Convert a list of texts into a float32 numpy matrix of normalised vectors."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


class VectorDatabase:
    """A tiny in-memory vector database for ONE uploaded document."""

    def __init__(self):
        self.index = None
        self.chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        """Create embeddings for all chunks and store them in a FAISS index."""
        if not chunks:
            raise ValueError("There is no text to index.")
        vectors = _embed([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = chunks

    def search(self, query: str, top_k: int = config.TOP_K) -> List[Dict]:
        """Return the top_k most similar chunks (each with a 'score')."""
        if self.index is None or not self.chunks:
            return []
        query_vector = _embed([query])
        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)
        return results
