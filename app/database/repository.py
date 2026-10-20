from __future__ import annotations

import time
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from app.database.db import db_manager, DatabaseManager


class Repository:
    """
    Data access layer for reading and writing request logs and cache entries in SQLite.
    """
    def __init__(self, db: DatabaseManager = db_manager):
        self.db = db

    def log_request(self, data: Dict[str, Any]) -> None:
        """Record an incoming request and its processing telemetry."""
        query = """
        INSERT INTO requests (
            request_id, timestamp, query_text, answer, cache_hit, cache_type,
            similarity_score, selected_model, model_tier, routing_strategy,
            routing_reason, prompt_tokens, completion_tokens, total_tokens,
            estimated_cost, actual_cost, cost_saved, latency_ms, namespace,
            provider, is_mock, error
        ) VALUES (
            :request_id, :timestamp, :query_text, :answer, :cache_hit, :cache_type,
            :similarity_score, :selected_model, :model_tier, :routing_strategy,
            :routing_reason, :prompt_tokens, :completion_tokens, :total_tokens,
            :estimated_cost, :actual_cost, :cost_saved, :latency_ms, :namespace,
            :provider, :is_mock, :error
        )
        """
        with self.db.session() as conn:
            conn.execute(query, {
                "request_id": data["request_id"],
                "timestamp": data["timestamp"],
                "query_text": data["query_text"],
                "answer": data.get("answer", ""),
                "cache_hit": 1 if data.get("cache_hit") else 0,
                "cache_type": data.get("cache_type", "none"),
                "similarity_score": data.get("similarity_score"),
                "selected_model": data.get("selected_model", "unknown"),
                "model_tier": data.get("model_tier", "small"),
                "routing_strategy": data.get("routing_strategy", "default"),
                "routing_reason": data.get("routing_reason", ""),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
                "total_tokens": data.get("total_tokens", 0),
                "estimated_cost": data.get("estimated_cost", 0.0),
                "actual_cost": data.get("actual_cost"),
                "cost_saved": data.get("cost_saved", 0.0),
                "latency_ms": data.get("latency_ms", 0.0),
                "namespace": data.get("namespace", "default"),
                "provider": data.get("provider", "mock"),
                "is_mock": 1 if data.get("is_mock") else 0,
                "error": data.get("error")
            })

    def get_request_history(
        self,
        limit: int = 50,
        offset: int = 0,
        model: Optional[str] = None,
        cache_hit: Optional[bool] = None,
        namespace: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Fetch paginated request history with optional filtering."""
        where_clauses = []
        params: Dict[str, Any] = {}

        if model:
            where_clauses.append("selected_model = :model")
            params["model"] = model
        if cache_hit is not None:
            where_clauses.append("cache_hit = :cache_hit")
            params["cache_hit"] = 1 if cache_hit else 0
        if namespace:
            where_clauses.append("namespace = :namespace")
            params["namespace"] = namespace

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        count_sql = f"SELECT COUNT(*) AS total FROM requests {where_str}"
        select_sql = f"""
        SELECT * FROM requests
        {where_str}
        ORDER BY id DESC
        LIMIT :limit OFFSET :offset
        """
        params["limit"] = limit
        params["offset"] = offset

        with self.db.session() as conn:
            cursor = conn.execute(count_sql, params)
            total = cursor.fetchone()["total"]

            cursor = conn.execute(select_sql, params)
            rows = [dict(row) for row in cursor.fetchall()]

            # Convert 0/1 back to booleans
            for r in rows:
                r["cache_hit"] = bool(r["cache_hit"])
                r["is_mock"] = bool(r["is_mock"])

            return rows, total

    def add_cache_entry(
        self,
        namespace: str,
        query_text: str,
        query_hash: str,
        embedding: np.ndarray,
        answer: str,
        model_name: str,
        model_tier: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost: float,
        ttl_seconds: int
    ) -> int:
        """Store a newly generated query, its normalized embedding, and response in the cache."""
        now = time.time()
        expires_at = now + ttl_seconds
        blob = embedding.astype(np.float32).tobytes()

        sql = """
        INSERT INTO semantic_cache (
            namespace, query_text, query_hash, embedding, embedding_dim,
            answer, model_name, model_tier, prompt_tokens, completion_tokens,
            estimated_cost, created_at, expires_at, hit_count, last_hit_at
        ) VALUES (
            :namespace, :query_text, :query_hash, :embedding, :embedding_dim,
            :answer, :model_name, :model_tier, :prompt_tokens, :completion_tokens,
            :estimated_cost, :created_at, :expires_at, 0, NULL
        )
        """
        with self.db.session() as conn:
            cursor = conn.execute(sql, {
                "namespace": namespace,
                "query_text": query_text,
                "query_hash": query_hash,
                "embedding": blob,
                "embedding_dim": len(embedding),
                "answer": answer,
                "model_name": model_name,
                "model_tier": model_tier,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost": estimated_cost,
                "created_at": now,
                "expires_at": expires_at
            })
            return cursor.lastrowid

    def get_exact_cache_entry(self, namespace: str, query_hash: str, now: float) -> Optional[Dict[str, Any]]:
        """Find an unexpired exact match by hash and namespace."""
        sql = """
        SELECT * FROM semantic_cache
        WHERE namespace = :namespace AND query_hash = :query_hash AND expires_at > :now
        ORDER BY id DESC LIMIT 1
        """
        with self.db.session() as conn:
            cursor = conn.execute(sql, {"namespace": namespace, "query_hash": query_hash, "now": now})
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_all_active_cache_entries(self, namespace: str, now: float) -> List[Dict[str, Any]]:
        """Retrieve all non-expired cache records for vector similarity scanning."""
        sql = """
        SELECT id, namespace, query_text, query_hash, embedding, embedding_dim,
               answer, model_name, model_tier, prompt_tokens, completion_tokens,
               estimated_cost, created_at, expires_at, hit_count
        FROM semantic_cache
        WHERE namespace = :namespace AND expires_at > :now
        """
        entries = []
        with self.db.session() as conn:
            cursor = conn.execute(sql, {"namespace": namespace, "now": now})
            for row in cursor.fetchall():
                d = dict(row)
                # Reconstruct float32 numpy vector
                d["embedding_vec"] = np.frombuffer(d["embedding"], dtype=np.float32)
                entries.append(d)
        return entries

    def record_cache_hit(self, cache_id: int, now: float) -> None:
        """Increment hit count and update last hit timestamp on cache access."""
        sql = """
        UPDATE semantic_cache
        SET hit_count = hit_count + 1, last_hit_at = :now
        WHERE id = :id
        """
        with self.db.session() as conn:
            conn.execute(sql, {"id": cache_id, "now": now})

    def invalidate_cache_entry(self, cache_id: int) -> bool:
        """Invalidate a specific cache entry by ID."""
        sql = "DELETE FROM semantic_cache WHERE id = :id"
        with self.db.session() as conn:
            cursor = conn.execute(sql, {"id": cache_id})
            return cursor.rowcount > 0

    def clear_cache(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries globally or for a specific namespace."""
        if namespace:
            sql = "DELETE FROM semantic_cache WHERE namespace = :namespace"
            params = {"namespace": namespace}
        else:
            sql = "DELETE FROM semantic_cache"
            params = {}

        with self.db.session() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def get_cache_stats(self, now: float) -> Dict[str, Any]:
        """Compute live cache statistics from database tables."""
        with self.db.session() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM semantic_cache")
            total = cursor.fetchone()["total"]

            cursor = conn.execute("SELECT COUNT(*) as active FROM semantic_cache WHERE expires_at > :now", {"now": now})
            active = cursor.fetchone()["active"]

            cursor = conn.execute("SELECT DISTINCT namespace FROM semantic_cache")
            namespaces = [r["namespace"] for r in cursor.fetchall()]

            # Also query hits/misses from requests table
            cursor = conn.execute("""
                SELECT 
                    SUM(CASE WHEN cache_hit = 1 AND cache_type = 'exact' THEN 1 ELSE 0 END) as exact_hits,
                    SUM(CASE WHEN cache_hit = 1 AND cache_type = 'semantic' THEN 1 ELSE 0 END) as semantic_hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as misses
                FROM requests
            """)
            hit_data = cursor.fetchone()
            exact_hits = hit_data["exact_hits"] or 0
            semantic_hits = hit_data["semantic_hits"] or 0
            misses = hit_data["misses"] or 0
            total_lookups = exact_hits + semantic_hits + misses
            hit_rate_pct = round(((exact_hits + semantic_hits) / total_lookups * 100), 2) if total_lookups > 0 else 0.0

            return {
                "total_entries": total,
                "active_entries": active,
                "expired_entries": max(0, total - active),
                "exact_hits": exact_hits,
                "semantic_hits": semantic_hits,
                "misses": misses,
                "hit_rate_pct": hit_rate_pct,
                "namespaces": namespaces
            }

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Compute production gateway metrics from historical request records."""
        with self.db.session() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as cache_misses,
                    SUM(CASE WHEN cache_hit = 1 AND cache_type = 'exact' THEN 1 ELSE 0 END) as exact_cache_hits,
                    SUM(CASE WHEN cache_hit = 1 AND cache_type = 'semantic' THEN 1 ELSE 0 END) as semantic_cache_hits,
                    SUM(estimated_cost) as total_estimated_cost,
                    SUM(actual_cost) as total_actual_cost,
                    SUM(cost_saved) as total_cost_saved,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                FROM requests
            """)
            summary = cursor.fetchone()
            total_requests = summary["total_requests"] or 0
            cache_hits = summary["cache_hits"] or 0
            cache_misses = summary["cache_misses"] or 0
            exact_hits = summary["exact_cache_hits"] or 0
            semantic_hits = summary["semantic_cache_hits"] or 0
            total_estimated_cost = round(summary["total_estimated_cost"] or 0.0, 6)
            total_actual_cost = round(summary["total_actual_cost"], 6) if summary["total_actual_cost"] is not None else None
            total_cost_saved = round(summary["total_cost_saved"] or 0.0, 6)
            avg_latency = round(summary["avg_latency"] or 0.0, 2)
            error_count = summary["error_count"] or 0

            # Calculate p50 and p95 latency
            cursor = conn.execute("SELECT latency_ms FROM requests ORDER BY latency_ms ASC")
            latencies = [r["latency_ms"] for r in cursor.fetchall()]
            if latencies:
                p50_latency = round(float(np.percentile(latencies, 50)), 2)
                p95_latency = round(float(np.percentile(latencies, 95)), 2)
            else:
                p50_latency = 0.0
                p95_latency = 0.0

            # Model distribution
            cursor = conn.execute("""
                SELECT selected_model, COUNT(*) as count 
                FROM requests 
                GROUP BY selected_model
            """)
            model_dist = {r["selected_model"]: r["count"] for r in cursor.fetchall()}

            # Tier distribution
            cursor = conn.execute("""
                SELECT model_tier, COUNT(*) as count 
                FROM requests 
                GROUP BY model_tier
            """)
            tier_dist = {r["model_tier"]: r["count"] for r in cursor.fetchall()}

            hit_rate = round((cache_hits / total_requests * 100), 2) if total_requests > 0 else 0.0
            error_rate = round((error_count / total_requests * 100), 2) if total_requests > 0 else 0.0
            
            # Gross cost without SmartRoute (estimated cost + cost saved)
            baseline_gross_cost = total_estimated_cost + total_cost_saved
            cost_reduction_pct = round((total_cost_saved / baseline_gross_cost * 100), 2) if baseline_gross_cost > 0 else 0.0

            return {
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "exact_cache_hits": exact_hits,
                "semantic_cache_hits": semantic_hits,
                "cache_hit_rate_pct": hit_rate,
                "total_estimated_cost_usd": total_estimated_cost,
                "total_actual_cost_usd": total_actual_cost,
                "total_cost_saved_usd": total_cost_saved,
                "net_cost_with_smartroute_usd": total_estimated_cost,
                "cost_reduction_pct": cost_reduction_pct,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "model_distribution": model_dist,
                "tier_distribution": tier_dist,
                "error_count": error_count,
                "error_rate_pct": error_rate
            }


# Global repository instance
repository = Repository()
