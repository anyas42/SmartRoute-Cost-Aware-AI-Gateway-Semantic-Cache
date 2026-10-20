from __future__ import annotations

import hashlib
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

from app.config import settings
from app.schemas import CacheType, ModelTier
from app.cache.embeddings import EmbeddingFactory, BaseEmbeddingModel
from app.cache.safety import CacheSafetyGuard
from app.database.repository import repository, Repository

logger = logging.getLogger("smartroute.cache")


@dataclass
class CacheLookupResult:
    hit: bool
    cache_type: CacheType
    similarity_score: Optional[float]
    cached_query: Optional[str]
    answer: str
    model_name: str
    model_tier: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float
    cache_id: Optional[int] = None


class SemanticCache:
    """
    Production-grade Semantic Cache combining O(1) hash exact matching
    with normalized vector cosine similarity search over SQLite persistent storage.
    """
    def __init__(
        self,
        repo: Repository = repository,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        similarity_threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        ttl_seconds: int = settings.CACHE_TTL_SECONDS
    ):
        self.repo = repo
        self.embedding_model = embedding_model or EmbeddingFactory.get_embedding_model()
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def compute_query_hash(query: str) -> str:
        """Create a deterministic SHA-256 hash of the normalized query string."""
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def lookup(
        self,
        query: str,
        namespace: str = "default",
        threshold: Optional[float] = None
    ) -> Optional[CacheLookupResult]:
        """
        Search for a cached response for the given query.
        1. Fast exact-match check (O(1)).
        2. Normalized vector cosine similarity scan over active namespace records.
        3. Polarity/intent safety validation to reject false positives.
        """
        if not settings.CACHE_ENABLED:
            return None

        now = time.time()
        q_hash = self.compute_query_hash(query)

        # 1. Exact Match Check
        exact_entry = self.repo.get_exact_cache_entry(namespace=namespace, query_hash=q_hash, now=now)
        if exact_entry:
            self.repo.record_cache_hit(exact_entry["id"], now)
            logger.info(f"Cache EXACT hit for query: '{query[:50]}...'")
            return CacheLookupResult(
                hit=True,
                cache_type=CacheType.EXACT,
                similarity_score=1.0,
                cached_query=exact_entry["query_text"],
                answer=exact_entry["answer"],
                model_name=exact_entry["model_name"],
                model_tier=exact_entry["model_tier"],
                prompt_tokens=exact_entry["prompt_tokens"],
                completion_tokens=exact_entry["completion_tokens"],
                estimated_cost=exact_entry["estimated_cost"],
                cache_id=exact_entry["id"]
            )

        # 2. Semantic Similarity Check
        active_entries = self.repo.get_all_active_cache_entries(namespace=namespace, now=now)
        if not active_entries:
            return None

        # Generate query embedding
        query_vec = self.embedding_model.embed_text(query)
        if np.linalg.norm(query_vec) == 0:
            return None

        # Build matrix of active cached embeddings
        embedding_matrix = np.vstack([e["embedding_vec"] for e in active_entries])

        # Compute cosine similarities via dot product (since vectors are L2-normalized)
        similarities = np.dot(embedding_matrix, query_vec)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_entry = active_entries[best_idx]

        active_threshold = threshold if threshold is not None else self.similarity_threshold

        logger.debug(
            f"Semantic similarity check: query='{query[:40]}' matched='{best_entry['query_text'][:40]}' "
            f"score={best_score:.4f} (threshold={active_threshold:.4f})"
        )

        if best_score >= active_threshold:
            # 3. Cache Safety Guard: Validate negation and intent to block false positives
            if settings.ENABLE_NEGATION_DETECTION:
                is_safe, reason = CacheSafetyGuard.validate_semantic_match(
                    new_query=query,
                    cached_query=best_entry["query_text"],
                    similarity_score=best_score,
                    threshold=active_threshold
                )
                if not is_safe:
                    logger.warning(
                        f"Semantic match REJECTED by safety guard: {reason} "
                        f"(query='{query}', cached='{best_entry['query_text']}')"
                    )
                    return None

            # Hit confirmed!
            self.repo.record_cache_hit(best_entry["id"], now)
            logger.info(
                f"Cache SEMANTIC hit (score={best_score:.4f}) for query='{query[:50]}...' "
                f"matched with '{best_entry['query_text'][:50]}...'"
            )
            return CacheLookupResult(
                hit=True,
                cache_type=CacheType.SEMANTIC,
                similarity_score=round(best_score, 4),
                cached_query=best_entry["query_text"],
                answer=best_entry["answer"],
                model_name=best_entry["model_name"],
                model_tier=best_entry["model_tier"],
                prompt_tokens=best_entry["prompt_tokens"],
                completion_tokens=best_entry["completion_tokens"],
                estimated_cost=best_entry["estimated_cost"],
                cache_id=best_entry["id"]
            )

        return None

    def store(
        self,
        namespace: str,
        query: str,
        answer: str,
        model_name: str,
        model_tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost: float,
        ttl_seconds: Optional[int] = None
    ) -> Optional[int]:
        """Validate query eligibility, embed, and store in the persistent cache."""
        is_cacheable, reason = CacheSafetyGuard.is_cacheable_query(query)
        if not is_cacheable:
            logger.info(f"Skipping cache storage: {reason}")
            return None

        # Check if identical exact hash already exists to avoid redundant duplicate entries
        q_hash = self.compute_query_hash(query)
        now = time.time()
        existing = self.repo.get_exact_cache_entry(namespace=namespace, query_hash=q_hash, now=now)
        if existing:
            # Already cached
            return existing["id"]

        embedding = self.embedding_model.embed_text(query)
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        entry_id = self.repo.add_cache_entry(
            namespace=namespace,
            query_text=query.strip(),
            query_hash=q_hash,
            embedding=embedding,
            answer=answer,
            model_name=model_name,
            model_tier=model_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
            ttl_seconds=ttl
        )
        logger.info(f"Stored response in cache (id={entry_id}, namespace='{namespace}')")
        return entry_id

    def invalidate(self, cache_id: int) -> bool:
        """Invalidate a specific cache record."""
        return self.repo.invalidate_cache_entry(cache_id)

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries globally or for a specific namespace."""
        return self.repo.clear_cache(namespace)

    def get_stats(self) -> Dict[str, Any]:
        """Fetch cache statistics."""
        now = time.time()
        stats = self.repo.get_cache_stats(now)
        stats["similarity_threshold"] = self.similarity_threshold
        stats["ttl_seconds"] = self.ttl_seconds
        return stats


# Global semantic cache instance
semantic_cache = SemanticCache()
