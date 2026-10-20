from __future__ import annotations

import time
import pytest
from app.cache.semantic_cache import SemanticCache
from app.cache.safety import CacheSafetyGuard
from app.cache.embeddings import LocalHashingEmbedding, EmbeddingFactory
from app.database.db import DatabaseManager
from app.database.repository import Repository
from app.schemas import CacheType


@pytest.fixture
def test_repo(tmp_path):
    db_file = str(tmp_path / "test_smartroute.db")
    db_mgr = DatabaseManager(db_file)
    return Repository(db_mgr)


@pytest.fixture
def cache(test_repo):
    # Use the fast, deterministic local embedding model for consistent unit testing
    embedder = EmbeddingFactory.get_embedding_model()
    return SemanticCache(
        repo=test_repo,
        embedding_model=embedder,
        similarity_threshold=0.80,
        ttl_seconds=3600
    )


def test_exact_cache_hit(cache):
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    
    # Initial store
    entry_id = cache.store(
        namespace="default",
        query=query,
        answer=answer,
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=10,
        completion_tokens=8,
        estimated_cost=0.000005
    )
    assert entry_id is not None

    # Exact lookup
    result = cache.lookup(query, namespace="default")
    assert result is not None
    assert result.hit is True
    assert result.cache_type == CacheType.EXACT
    assert result.similarity_score == 1.0
    assert result.answer == answer


def test_semantic_cache_hit_paraphrase(cache):
    q1 = "What is the capital of France?"
    q2 = "Which city is France's capital?"
    answer = "The capital of France is Paris."

    cache.store(
        namespace="default",
        query=q1,
        answer=answer,
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=10,
        completion_tokens=8,
        estimated_cost=0.000005
    )

    # Lookup with paraphrased query at a reasonable similarity threshold
    result = cache.lookup(q2, namespace="default", threshold=0.70)
    assert result is not None
    assert result.hit is True
    assert result.cache_type == CacheType.SEMANTIC
    assert result.similarity_score is not None
    assert result.similarity_score >= 0.70
    assert result.answer == answer


def test_cache_miss_dissimilar_query(cache):
    q1 = "What is gradient descent?"
    q2 = "How do you cook chocolate chip cookies?"

    cache.store(
        namespace="default",
        query=q1,
        answer="Optimization algorithm.",
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=10,
        completion_tokens=5,
        estimated_cost=0.000005
    )

    result = cache.lookup(q2, namespace="default")
    assert result is None


def test_cache_expiration_ttl(cache, test_repo):
    query = "Temporary ephemeral query"
    # Store with TTL of -10 seconds (already expired)
    now = time.time()
    emb = cache.embedding_model.embed_text(query)
    q_hash = cache.compute_query_hash(query)
    test_repo.add_cache_entry(
        namespace="default",
        query_text=query,
        query_hash=q_hash,
        embedding=emb,
        answer="Old answer",
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=5,
        completion_tokens=5,
        estimated_cost=0.000001,
        ttl_seconds=-10
    )

    result = cache.lookup(query, namespace="default")
    assert result is None


def test_namespace_isolation(cache):
    query = "Shared question text"
    
    # Store in tenant_alpha
    cache.store(
        namespace="tenant_alpha",
        query=query,
        answer="Answer Alpha",
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=5,
        completion_tokens=5,
        estimated_cost=0.000001
    )

    # Lookup in tenant_beta should MISS
    res_beta = cache.lookup(query, namespace="tenant_beta")
    assert res_beta is None

    # Lookup in tenant_alpha should HIT
    res_alpha = cache.lookup(query, namespace="tenant_alpha")
    assert res_alpha is not None
    assert res_alpha.answer == "Answer Alpha"


def test_cache_invalidation_and_clear(cache):
    query = "Query to invalidate"
    entry_id = cache.store(
        namespace="default",
        query=query,
        answer="Answer to invalidate",
        model_name="gpt-4o-mini",
        model_tier="small",
        prompt_tokens=5,
        completion_tokens=5,
        estimated_cost=0.000001
    )
    assert entry_id is not None

    # Invalidate by id
    deleted = cache.invalidate(entry_id)
    assert deleted is True

    # Should now miss
    assert cache.lookup(query, namespace="default") is None


def test_negation_safety_guard():
    q_pos = "Why is Python good for machine learning?"
    q_neg = "Why is Python not good for machine learning?"

    # Even with high similarity, safety guard must reject false-positive semantic match
    is_safe, reason = CacheSafetyGuard.validate_semantic_match(
        new_query=q_neg,
        cached_query=q_pos,
        similarity_score=0.92,
        threshold=0.85
    )
    assert is_safe is False
    assert "Polarity mismatch" in reason
