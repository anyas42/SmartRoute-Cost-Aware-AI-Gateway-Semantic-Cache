from __future__ import annotations

import time
import uuid
import logging
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, status

from app.config import settings
from app.schemas import (
    QueryRequest, QueryResponse, CacheType, ModelTier,
    HealthResponse, MetricsResponse, CacheStatsResponse,
    CacheClearRequest, CacheClearResponse, RequestHistoryResponse,
    RequestHistoryItem
)
from app.cache.semantic_cache import semantic_cache
from app.routing.router import RouterFactory
from app.routing.pricing import calculate_savings_vs_baseline
from app.providers.factory import ProviderFactory
from app.database.repository import repository

logger = logging.getLogger("smartroute.api")
router = APIRouter()
START_TIME = time.time()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process query with semantic caching and cost-aware routing",
    status_code=status.HTTP_200_OK
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main Gateway Pipeline:
    1. Validate input size and namespace constraints.
    2. Check semantic cache (O(1) exact match -> cosine vector similarity).
    3. If cache hit, return immediately (zero LLM cost, minimal latency).
    4. If cache miss, determine optimal model tier via router.
    5. Call configured LLM provider (or mock simulator).
    6. Record telemetry and update persistent cache.
    7. Return full response with cost and latency breakdown.
    """
    req_start = time.time()
    request_id = str(uuid.uuid4())
    clean_query = request.query.strip()
    namespace = request.namespace.strip() or "default"
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    # Safeguard 1: Query length constraint
    if len(clean_query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query length ({len(clean_query)}) exceeds maximum limit of {settings.MAX_QUERY_LENGTH} characters."
        )

    # Step 1: Semantic Cache Lookup (unless force_refresh is requested)
    if not request.force_refresh and settings.CACHE_ENABLED:
        cache_result = semantic_cache.lookup(query=clean_query, namespace=namespace)
        if cache_result and cache_result.hit:
            latency_ms = round((time.time() - req_start) * 1000.0, 2)
            # Cost saved = what this query would have cost if run on baseline Large model
            cost_saved = calculate_savings_vs_baseline(
                prompt_tokens=cache_result.prompt_tokens,
                completion_tokens=cache_result.completion_tokens,
                served_model=cache_result.model_name,
                is_cache_hit=True
            )

            response_data = {
                "request_id": request_id,
                "timestamp": iso_timestamp,
                "query_text": clean_query,
                "answer": cache_result.answer,
                "cache_hit": True,
                "cache_type": cache_result.cache_type.value,
                "similarity_score": cache_result.similarity_score,
                "selected_model": cache_result.model_name,
                "model_tier": cache_result.model_tier,
                "routing_strategy": "cache",
                "routing_reason": f"Cached answer reused via {cache_result.cache_type.value} match (id={cache_result.cache_id})",
                "prompt_tokens": cache_result.prompt_tokens,
                "completion_tokens": cache_result.completion_tokens,
                "total_tokens": cache_result.prompt_tokens + cache_result.completion_tokens,
                "estimated_cost": 0.0,  # Zero cost incurred for cache hit
                "actual_cost": 0.0,
                "cost_saved": cost_saved,
                "latency_ms": latency_ms,
                "namespace": namespace,
                "provider": "cache",
                "is_mock": False,
                "error": None
            }
            repository.log_request(response_data)

            return QueryResponse(
                request_id=request_id,
                query=clean_query,
                answer=cache_result.answer,
                cache_hit=True,
                cache_type=cache_result.cache_type,
                similarity_score=cache_result.similarity_score,
                matched_query=cache_result.cached_query,
                selected_model=cache_result.model_name,
                model_tier=ModelTier(cache_result.model_tier),
                routing_reason=response_data["routing_reason"],
                routing_strategy="cache",
                prompt_tokens=cache_result.prompt_tokens,
                completion_tokens=cache_result.completion_tokens,
                total_tokens=response_data["total_tokens"],
                estimated_cost=0.0,
                actual_cost=0.0,
                cost_saved=cost_saved,
                latency_ms=latency_ms,
                provider="cache",
                is_mock=False,
                namespace=namespace,
                timestamp=iso_timestamp
            )

    # Step 2: Cache Miss -> Cost-Aware Model Routing
    decision = RouterFactory.route_query(
        query=clean_query,
        override_strategy=request.routing_strategy
    )

    # Step 3: LLM Provider Execution
    provider = ProviderFactory.get_provider()
    try:
        llm_response = await provider.generate(
            prompt=clean_query,
            model=decision.model_name,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 512
        )
    except Exception as exc:
        latency_ms = round((time.time() - req_start) * 1000.0, 2)
        err_msg = str(exc)
        logger.error(f"Provider invocation error for request {request_id}: {err_msg}")
        # Log failure event
        repository.log_request({
            "request_id": request_id,
            "timestamp": iso_timestamp,
            "query_text": clean_query,
            "answer": "",
            "cache_hit": False,
            "cache_type": CacheType.NONE.value,
            "similarity_score": None,
            "selected_model": decision.model_name,
            "model_tier": decision.tier.value,
            "routing_strategy": decision.strategy,
            "routing_reason": decision.reason,
            "prompt_tokens": decision.estimated_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": decision.estimated_prompt_tokens,
            "estimated_cost": 0.0,
            "actual_cost": 0.0,
            "cost_saved": 0.0,
            "latency_ms": latency_ms,
            "namespace": namespace,
            "provider": getattr(provider, "provider", "unknown"),
            "is_mock": getattr(provider, "is_mock", True),
            "error": err_msg
        })
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Provider failed: {err_msg}"
        )

    latency_ms = round((time.time() - req_start) * 1000.0, 2)

    # Step 4: Calculate Cost Savings vs Baseline
    cost_saved = calculate_savings_vs_baseline(
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        served_model=llm_response.model,
        is_cache_hit=False
    )

    # Step 5: Store Eligible Response in Semantic Cache
    if settings.CACHE_ENABLED:
        semantic_cache.store(
            namespace=namespace,
            query=clean_query,
            answer=llm_response.content,
            model_name=llm_response.model,
            model_tier=decision.tier.value,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            estimated_cost=llm_response.estimated_cost
        )

    # Step 6: Log Telemetry to Database
    repository.log_request({
        "request_id": request_id,
        "timestamp": iso_timestamp,
        "query_text": clean_query,
        "answer": llm_response.content,
        "cache_hit": False,
        "cache_type": CacheType.NONE.value,
        "similarity_score": None,
        "selected_model": llm_response.model,
        "model_tier": decision.tier.value,
        "routing_strategy": decision.strategy,
        "routing_reason": decision.reason,
        "prompt_tokens": llm_response.prompt_tokens,
        "completion_tokens": llm_response.completion_tokens,
        "total_tokens": llm_response.total_tokens,
        "estimated_cost": llm_response.estimated_cost,
        "actual_cost": llm_response.actual_cost,
        "cost_saved": cost_saved,
        "latency_ms": latency_ms,
        "namespace": namespace,
        "provider": llm_response.provider,
        "is_mock": llm_response.is_mock,
        "error": None
    })

    return QueryResponse(
        request_id=request_id,
        query=clean_query,
        answer=llm_response.content,
        cache_hit=False,
        cache_type=CacheType.NONE,
        similarity_score=None,
        matched_query=None,
        selected_model=llm_response.model,
        model_tier=decision.tier,
        routing_reason=decision.reason,
        routing_strategy=decision.strategy,
        prompt_tokens=llm_response.prompt_tokens,
        completion_tokens=llm_response.completion_tokens,
        total_tokens=llm_response.total_tokens,
        estimated_cost=llm_response.estimated_cost,
        actual_cost=llm_response.actual_cost,
        cost_saved=cost_saved,
        latency_ms=latency_ms,
        provider=llm_response.provider,
        is_mock=llm_response.is_mock,
        namespace=namespace,
        timestamp=iso_timestamp
    )


@router.get("/health", response_model=HealthResponse, summary="Gateway health and configuration status")
def health_check() -> HealthResponse:
    stats = semantic_cache.get_stats()
    metrics = repository.get_aggregated_metrics()
    uptime = round(time.time() - START_TIME, 2)
    provider_inst = ProviderFactory.get_provider()
    is_mock = getattr(provider_inst, "is_mock", True) if hasattr(provider_inst, "is_mock") else (settings.LLM_PROVIDER == "mock")

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        app_env=settings.APP_ENV,
        provider=settings.LLM_PROVIDER,
        is_mock=is_mock,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        cache_enabled=settings.CACHE_ENABLED,
        similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
        cache_size=stats["active_entries"],
        total_requests_processed=metrics["total_requests"],
        uptime_seconds=uptime
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Aggregated gateway performance metrics")
def get_metrics() -> MetricsResponse:
    metrics = repository.get_aggregated_metrics()
    return MetricsResponse(**metrics)


@router.get("/cache/stats", response_model=CacheStatsResponse, summary="Detailed cache statistics")
def get_cache_stats() -> CacheStatsResponse:
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@router.post("/cache/clear", response_model=CacheClearResponse, summary="Clear cache entries")
def clear_cache(req: Optional[CacheClearRequest] = None) -> CacheClearResponse:
    ns = req.namespace if req else None
    count = semantic_cache.clear(namespace=ns)
    scope = f"in namespace '{ns}'" if ns else "globally"
    return CacheClearResponse(
        cleared_entries=count,
        namespace=ns,
        message=f"Successfully cleared {count} cache entries {scope}."
    )


@router.delete("/cache/{cache_id}", summary="Invalidate a specific cache entry by ID")
def invalidate_cache_entry(cache_id: int):
    success = semantic_cache.invalidate(cache_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cache entry ID {cache_id} not found."
        )
    return {"status": "success", "message": f"Cache entry {cache_id} invalidated."}


@router.get("/history", response_model=RequestHistoryResponse, summary="Query paginated request history")
def get_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    model: Optional[str] = None,
    cache_hit: Optional[bool] = None,
    namespace: Optional[str] = None
) -> RequestHistoryResponse:
    offset = (page - 1) * page_size
    rows, total = repository.get_request_history(
        limit=page_size,
        offset=offset,
        model=model,
        cache_hit=cache_hit,
        namespace=namespace
    )
    items = [RequestHistoryItem(**r) for r in rows]
    return RequestHistoryResponse(
        total=total,
        items=items,
        page=page,
        page_size=page_size
    )
