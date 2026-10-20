from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database.repository import repository

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_cache():
    repository.clear_cache()
    yield
    repository.clear_cache()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "provider" in data


def test_query_pipeline_and_cache_hits():
    prompt = "What is Docker and how do containers work?"

    # 1. First invocation: Cache miss -> model generation
    r1 = client.post("/query", json={"query": prompt})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["cache_hit"] is False
    assert d1["cache_type"] == "none"
    assert len(d1["answer"]) > 0
    assert d1["estimated_cost"] > 0
    assert d1["prompt_tokens"] > 0

    # 2. Second invocation: Exact cache hit
    r2 = client.post("/query", json={"query": prompt})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["cache_hit"] is True
    assert d2["cache_type"] == "exact"
    assert d2["estimated_cost"] == 0.0  # Zero API cost
    assert d2["cost_saved"] > 0.0

    # 3. Third invocation with force_refresh: Cache bypassed
    r3 = client.post("/query", json={"query": prompt, "force_refresh": True})
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["cache_hit"] is False
    assert d3["cache_type"] == "none"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "cache_hit_rate_pct" in data
    assert "total_cost_saved_usd" in data


def test_cache_stats_and_clear():
    # Store an entry via query
    client.post("/query", json={"query": "Test entry query for stats"})
    
    stats_r = client.get("/cache/stats")
    assert stats_r.status_code == 200
    stats = stats_r.json()
    assert stats["total_entries"] >= 1

    # Clear cache
    clear_r = client.post("/cache/clear")
    assert clear_r.status_code == 200
    clear_data = clear_r.json()
    assert clear_data["cleared_entries"] >= 1

    # Verify cache is now empty
    stats_after = client.get("/cache/stats").json()
    assert stats_after["active_entries"] == 0


def test_history_endpoint():
    client.post("/query", json={"query": "History check prompt"})
    response = client.get("/history?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1


def test_max_query_length_validation():
    long_query = "x" * 5000
    response = client.post("/query", json={"query": long_query})
    # Pydantic schema validation returns 422 Unprocessable Entity
    assert response.status_code in (400, 422)
