from __future__ import annotations

import pytest
from app.routing.router import RuleBasedRouter, ClassifierRouter, RouterFactory
from app.routing.pricing import count_tokens, estimate_cost, calculate_savings_vs_baseline
from app.schemas import ModelTier, RoutingStrategy


def test_token_counting():
    text = "Hello world! This is a test prompt for token estimation."
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_cost_estimation():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out
    # 1,000,000 in + 1,000,000 out = $0.75
    cost = estimate_cost(1_000_000, 1_000_000, "gpt-4o-mini")
    assert round(cost, 2) == 0.75

    # Zero tokens = $0.00
    assert estimate_cost(0, 0, "gpt-4o-mini") == 0.0


def test_rule_based_router_simple_query():
    router = RuleBasedRouter(small_model="gpt-4o-mini", large_model="gpt-4o", token_threshold=50)
    query = "What is the capital of Italy?"
    decision = router.route(query)

    assert decision.tier == ModelTier.SMALL
    assert decision.model_name == "gpt-4o-mini"
    assert "Tier 1 (Small)" in decision.reason


def test_rule_based_router_complex_reasoning():
    router = RuleBasedRouter(small_model="gpt-4o-mini", large_model="gpt-4o", token_threshold=50)
    query = "Explain step by step how to derive the backpropagation equations and analyze the trade-offs of different optimizers."
    decision = router.route(query)

    assert decision.tier == ModelTier.LARGE
    assert decision.model_name == "gpt-4o"
    assert "Tier 2 (Large)" in decision.reason


def test_rule_based_router_length_threshold():
    router = RuleBasedRouter(small_model="gpt-4o-mini", large_model="gpt-4o", token_threshold=20)
    # Prompt longer than 20 tokens
    query = (
        "This is an unusually detailed and extended prompt with many words explaining a specific business scenario "
        "and requiring deep contextual reasoning across multiple sentences and paragraphs to solve."
    )
    decision = router.route(query)

    assert decision.tier == ModelTier.LARGE
    assert decision.model_name == "gpt-4o"
    assert "exceeds complexity threshold" in decision.reason


def test_classifier_router():
    router = ClassifierRouter(small_model="gpt-4o-mini", large_model="gpt-4o", threshold=0.50)
    
    # Simple query
    simple_dec = router.route("Define recursion in programming.")
    assert simple_dec.strategy == RoutingStrategy.CLASSIFIER.value
    assert simple_dec.complexity_score is not None

    # Complex query with code block
    complex_query = "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```\nOptimize this function and analyze the time complexity."
    complex_dec = router.route(complex_query)
    assert complex_dec.tier == ModelTier.LARGE
    assert complex_dec.complexity_score > 0.50


def test_savings_calculation():
    prompt_tokens = 100
    completion_tokens = 200

    # Savings from cache hit = full large model cost
    savings_cache = calculate_savings_vs_baseline(
        prompt_tokens, completion_tokens, "gpt-4o", is_cache_hit=True
    )
    large_cost = estimate_cost(prompt_tokens, completion_tokens, "gpt-4o")
    assert savings_cache == large_cost

    # Savings from routing to small model = large_cost - small_cost
    savings_routing = calculate_savings_vs_baseline(
        prompt_tokens, completion_tokens, "gpt-4o-mini", is_cache_hit=False
    )
    small_cost = estimate_cost(prompt_tokens, completion_tokens, "gpt-4o-mini")
    assert round(savings_routing, 6) == round(large_cost - small_cost, 6)
