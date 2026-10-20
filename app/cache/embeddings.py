from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from app.config import settings

logger = logging.getLogger("smartroute.embeddings")


class BaseEmbeddingModel(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a 1D normalized float32 embedding vector for the input text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate a 2D normalized float32 array (N, D) for a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        pass

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize a vector or batch of vectors to unit L2 norm."""
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            return (vec / norm).astype(np.float32)
        else:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (vec / norms).astype(np.float32)


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Embedding implementation using the sentence-transformers library.
    Generates dense semantic representations using a pretrained model (e.g. all-MiniLM-L6-v2).
    """
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._dim = 384
        try:
            from sentence_transformers import SentenceTransformer
            try:
                # Attempt fast load from local cache first to avoid huggingface network retries
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer model: {model_name} (dim: {self._dim})")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer ({model_name}): {e}", exc_info=True)

            raise

    def embed_text(self, text: str) -> np.ndarray:
        cleaned = text.strip()
        if not cleaned:
            return np.zeros(self._dim, dtype=np.float32)
        raw = self.model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
        return self.normalize(raw)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        cleaned = [t.strip() for t in texts]
        raw = self.model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
        return self.normalize(raw)

    @property
    def dimension(self) -> int:
        return self._dim


class LocalHashingEmbedding(BaseEmbeddingModel):
    """
    High-speed, zero-dependency, local subword hashing embedding model.
    Uses character and word n-gram feature hashing to produce dense vectors
    with zero external downloads or network access.
    """
    def __init__(self, dim: int = 256):
        self._dim = dim
        logger.info(f"Initialized LocalHashingEmbedding (dim: {self._dim})")

    def _hash_token(self, token: str) -> int:
        # FNV-1a 32-bit hash
        h = 2166136261
        for ch in token.encode("utf-8"):
            h = (h ^ ch) * 16777619 & 0xFFFFFFFF
        return h % self._dim

    def embed_text(self, text: str) -> np.ndarray:
        cleaned = text.strip().lower()
        if not cleaned:
            return np.zeros(self._dim, dtype=np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        words = cleaned.split()

        # Word unigrams and bigrams
        for i, w in enumerate(words):
            idx = self._hash_token(w)
            vec[idx] += 1.0
            if i < len(words) - 1:
                bigram = f"{w}_{words[i+1]}"
                vec[self._hash_token(bigram)] += 1.5

        # Character trigrams for morphological and typo tolerance
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i:i+3]
            vec[self._hash_token(trigram)] += 0.5

        return self.normalize(vec)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed_text(t) for t in texts], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


class EmbeddingFactory:
    """Factory to instantiate and cache the active embedding model."""
    _instance: Optional[BaseEmbeddingModel] = None

    @classmethod
    def get_embedding_model(cls) -> BaseEmbeddingModel:
        if cls._instance is None:
            try:
                cls._instance = SentenceTransformerEmbedding()
            except Exception as e:
                logger.warning(f"Could not initialize SentenceTransformer ({e}). Falling back to LocalHashingEmbedding.")
                cls._instance = LocalHashingEmbedding()
        return cls._instance
