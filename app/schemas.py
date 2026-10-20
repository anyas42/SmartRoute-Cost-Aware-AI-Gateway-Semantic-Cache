from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    SMALL = "small"
    LARGE = "large"


class RoutingStrategy(str, Enum):
    RULE_BASED = "rule_based"
    CLASSIFIER = "classifier"
    DIRECT_SMALL = "direct_small"
    DIRECT_LARGE = "direct_large"


class CacheType(str, Enum):
    NONE = "none"
    EXACT = "exact"
    SEMANTIC = "semantic"


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User query string to be processed by the gateway"
    )
    namespace: str = Field(
        default="default",
        max_length=64,
        description="Logical isolation scope for cached responses"
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, bypasses cache lookup and forces fresh LLM generation"
    )
    routing_strategy: Optional[RoutingStrategy] = Field(
        default=None,
        description="Override the default routing strategy for this request"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation"
    )
    max_tokens: Optional[int] = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate"
    )


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    cache_hit: bool
    cache_type: CacheType
    similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score (0.0 to 1.0) if a semantic match was evaluated"
    )
    matched_query: Optional[str] = Field(
        default=None,
        description="Original query text that matched in the semantic cache"
    )
    selected_model: str
    model_tier: ModelTier
    routing_reason: str
    routing_strategy: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    cost_saved: float
    latency_ms: float
    provider: str
    is_mock: bool
    namespace: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_env: str
    provider: str
    is_mock: bool
    embedding_model: str
    cache_enabled: bool
    similarity_threshold: float
    cache_size: int
    total_requests_processed: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    exact_cache_hits: int
    semantic_cache_hits: int
    cache_hit_rate_pct: float
    total_estimated_cost_usd: float
    total_actual_cost_usd: Optional[float] = None
    total_cost_saved_usd: float
    net_cost_with_smartroute_usd: float
    cost_reduction_pct: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    model_distribution: Dict[str, int]
    tier_distribution: Dict[str, int]
    error_count: int
    error_rate_pct: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    active_entries: int
    expired_entries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    hit_rate_pct: float
    similarity_threshold: float
    ttl_seconds: int
    namespaces: List[str]


class CacheClearRequest(BaseModel):
    namespace: Optional[str] = Field(
        default=None,
        description="If provided, only entries within this namespace will be deleted"
    )


class CacheClearResponse(BaseModel):
    cleared_entries: int
    namespace: Optional[str]
    message: str


class RequestHistoryItem(BaseModel):
    request_id: str
    timestamp: str
    query_text: str
    answer: str
    cache_hit: bool
    cache_type: str
    similarity_score: Optional[float]
    selected_model: str
    model_tier: str
    routing_strategy: str
    routing_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    actual_cost: Optional[float]
    cost_saved: float
    latency_ms: float
    namespace: str
    error: Optional[str] = None


class RequestHistoryResponse(BaseModel):
    total: int
    items: List[RequestHistoryItem]
    page: int
    page_size: int
