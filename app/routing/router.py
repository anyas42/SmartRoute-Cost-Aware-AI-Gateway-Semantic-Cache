from __future__ import annotations

import re
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from app.config import settings
from app.schemas import ModelTier, RoutingStrategy
from app.routing.pricing import count_tokens, estimate_cost

logger = logging.getLogger("smartroute.routing")

# Complex task keywords and patterns that warrant the Large tier
COMPLEX_PATTERNS = [
    (r"\b(step[- ]by[- ]step|prove|derivation|derive|formal proof)\b", "multi-step reasoning"),
    (r"\b(trade[- ]?offs?|pros and cons|compare and contrast|architect(ure)?)\b", "system architecture/trade-off analysis"),
    (r"\b(debug|refactor|optimize algorithm|time complexity|space complexity|big[- ]o)\b", "software engineering/algorithmic analysis"),
    (r"\b(root cause|failure mode|threat model|security vulnerability)\b", "deep diagnostic/security analysis"),
    (r"```|def\s+\w+|class\s+\w+|SELECT\s+.*FROM|CREATE\s+TABLE", "source code / query syntax"),
]

# Simple task keywords typically well-handled by the Small tier
SIMPLE_PATTERNS = [
    (r"^(what is|define|who is|where is|when was|translate|summarize)\b", "factual lookup/definition"),
    (r"\b(synonym|antonym|spell check|format as json)\b", "simple text transformation"),
]


@dataclass
class RoutingDecision:
    tier: ModelTier
    model_name: str
    strategy: str
    reason: str
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_cost: float
    complexity_score: Optional[float] = None
    fallback_triggered: bool = False


class BaseRouter(ABC):
    """Abstract interface for model routers."""

    @abstractmethod
    def route(self, query: str) -> RoutingDecision:
        pass


class RuleBasedRouter(BaseRouter):
    """
    Transparent, deterministic rule-based router.
    Evaluates token length, presence of code, and multi-step reasoning indicators.
    """
    def __init__(
        self,
        small_model: str = settings.SMALL_MODEL_NAME,
        large_model: str = settings.LARGE_MODEL_NAME,
        token_threshold: int = settings.COMPLEXITY_TOKEN_THRESHOLD
    ):
        self.small_model = small_model
        self.large_model = large_model
        self.token_threshold = token_threshold

    def route(self, query: str) -> RoutingDecision:
        tokens = count_tokens(query)
        estimated_output_tokens = min(512, max(50, tokens * 2))
        lower_query = query.lower()

        # Rule 1: High query token length indicates complex prompt context
        if tokens > self.token_threshold:
            model = self.large_model
            tier = ModelTier.LARGE
            reason = f"Query length ({tokens} tokens) exceeds complexity threshold ({self.token_threshold}) -> Tier 2 (Large)"
            cost = estimate_cost(tokens, estimated_output_tokens, model)
            return RoutingDecision(
                tier=tier,
                model_name=model,
                strategy=RoutingStrategy.RULE_BASED.value,
                reason=reason,
                estimated_prompt_tokens=tokens,
                estimated_completion_tokens=estimated_output_tokens,
                estimated_cost=cost,
                complexity_score=1.0
            )

        # Rule 2: Pattern match on complex reasoning, system design, or code blocks
        matched_complex = []
        for regex, desc in COMPLEX_PATTERNS:
            if re.search(regex, query, re.IGNORECASE):
                matched_complex.append(desc)

        if matched_complex:
            model = self.large_model
            tier = ModelTier.LARGE
            reason = f"Detected complex patterns: [{', '.join(matched_complex)}] -> Tier 2 (Large)"
            cost = estimate_cost(tokens, estimated_output_tokens, model)
            return RoutingDecision(
                tier=tier,
                model_name=model,
                strategy=RoutingStrategy.RULE_BASED.value,
                reason=reason,
                estimated_prompt_tokens=tokens,
                estimated_completion_tokens=estimated_output_tokens,
                estimated_cost=cost,
                complexity_score=0.85
            )

        # Rule 3: Pattern match on simple tasks
        matched_simple = []
        for regex, desc in SIMPLE_PATTERNS:
            if re.search(regex, lower_query):
                matched_simple.append(desc)

        if matched_simple:
            model = self.small_model
            tier = ModelTier.SMALL
            reason = f"Detected simple task pattern: [{', '.join(matched_simple)}] -> Tier 1 (Small)"
            cost = estimate_cost(tokens, estimated_output_tokens, model)
            return RoutingDecision(
                tier=tier,
                model_name=model,
                strategy=RoutingStrategy.RULE_BASED.value,
                reason=reason,
                estimated_prompt_tokens=tokens,
                estimated_completion_tokens=estimated_output_tokens,
                estimated_cost=cost,
                complexity_score=0.20
            )

        # Default fallback for short general queries
        model = self.small_model
        tier = ModelTier.SMALL
        reason = f"Standard query ({tokens} tokens, no complex reasoning markers detected) -> Tier 1 (Small)"
        cost = estimate_cost(tokens, estimated_output_tokens, model)
        return RoutingDecision(
            tier=tier,
            model_name=model,
            strategy=RoutingStrategy.RULE_BASED.value,
            reason=reason,
            estimated_prompt_tokens=tokens,
            estimated_completion_tokens=estimated_output_tokens,
            estimated_cost=cost,
            complexity_score=0.35
        )


class ClassifierRouter(BaseRouter):
    """
    Feature-based scoring router.
    Extracts syntactic, lexical, and structural features to generate a continuous
    complexity score in [0.0, 1.0], routing to Large if score >= threshold.
    """
    def __init__(
        self,
        small_model: str = settings.SMALL_MODEL_NAME,
        large_model: str = settings.LARGE_MODEL_NAME,
        threshold: float = settings.CLASSIFIER_THRESHOLD
    ):
        self.small_model = small_model
        self.large_model = large_model
        self.threshold = threshold

    def _extract_features(self, query: str) -> Dict[str, float]:
        tokens = count_tokens(query)
        words = query.strip().split()
        num_words = len(words)
        unique_words = len(set(words)) if words else 0
        lexical_diversity = (unique_words / num_words) if num_words > 0 else 0.0

        has_code = 1.0 if re.search(r"```|def\s+|class\s+|SELECT\s+", query) else 0.0
        reasoning_hits = sum(1 for reg, _ in COMPLEX_PATTERNS if re.search(reg, query, re.IGNORECASE))
        has_multiple_clauses = 1.0 if query.count("?") > 1 or query.count(";") > 0 or " and " in query else 0.0
        length_normalized = min(1.0, tokens / 100.0)

        return {
            "tokens": float(tokens),
            "length_norm": length_normalized,
            "has_code": has_code,
            "reasoning_hits": float(reasoning_hits),
            "has_multiple_clauses": has_multiple_clauses,
            "lexical_diversity": lexical_diversity,
        }

    def route(self, query: str) -> RoutingDecision:
        feats = self._extract_features(query)
        tokens = int(feats["tokens"])
        estimated_output_tokens = min(512, max(50, tokens * 2))

        # Weighted logistic scoring
        # Weights calibrated for semantic complexity discrimination
        raw_score = (
            0.35 * feats["length_norm"] +
            0.30 * min(1.0, feats["reasoning_hits"] * 0.5) +
            0.25 * feats["has_code"] +
            0.10 * feats["has_multiple_clauses"]
        )
        # Apply sigmoid shaping: 1 / (1 + exp(-4 * (score - 0.4)))
        score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.35)))
        score = round(score, 4)

        if score >= self.threshold:
            model = self.large_model
            tier = ModelTier.LARGE
            reason = (
                f"Classifier score ({score:.2f} >= {self.threshold:.2f}) indicates complex task "
                f"[tokens={tokens}, code={feats['has_code'] > 0}, reasoning_hits={int(feats['reasoning_hits'])}] -> Tier 2 (Large)"
            )
        else:
            model = self.small_model
            tier = ModelTier.SMALL
            reason = (
                f"Classifier score ({score:.2f} < {self.threshold:.2f}) within standard capacity -> Tier 1 (Small)"
            )

        cost = estimate_cost(tokens, estimated_output_tokens, model)
        return RoutingDecision(
            tier=tier,
            model_name=model,
            strategy=RoutingStrategy.CLASSIFIER.value,
            reason=reason,
            estimated_prompt_tokens=tokens,
            estimated_completion_tokens=estimated_output_tokens,
            estimated_cost=cost,
            complexity_score=score
        )


class RouterFactory:
    """Dispatches queries to appropriate routing strategy and handles direct tier overrides."""
    _rule_router = RuleBasedRouter()
    _classifier_router = ClassifierRouter()

    @classmethod
    def get_router(cls, strategy: Optional[RoutingStrategy] = None) -> BaseRouter:
        active_strategy = strategy.value if strategy else settings.DEFAULT_ROUTING_STRATEGY
        if active_strategy == RoutingStrategy.CLASSIFIER.value:
            return cls._classifier_router
        return cls._rule_router

    @classmethod
    def route_query(
        cls,
        query: str,
        override_strategy: Optional[RoutingStrategy] = None
    ) -> RoutingDecision:
        # Check direct override modes
        if override_strategy == RoutingStrategy.DIRECT_SMALL:
            tokens = count_tokens(query)
            est_out = min(512, max(50, tokens * 2))
            cost = estimate_cost(tokens, est_out, settings.SMALL_MODEL_NAME)
            return RoutingDecision(
                tier=ModelTier.SMALL,
                model_name=settings.SMALL_MODEL_NAME,
                strategy=RoutingStrategy.DIRECT_SMALL.value,
                reason="Direct override: forced Tier 1 (Small)",
                estimated_prompt_tokens=tokens,
                estimated_completion_tokens=est_out,
                estimated_cost=cost,
                complexity_score=0.0
            )
        elif override_strategy == RoutingStrategy.DIRECT_LARGE:
            tokens = count_tokens(query)
            est_out = min(512, max(50, tokens * 2))
            cost = estimate_cost(tokens, est_out, settings.LARGE_MODEL_NAME)
            return RoutingDecision(
                tier=ModelTier.LARGE,
                model_name=settings.LARGE_MODEL_NAME,
                strategy=RoutingStrategy.DIRECT_LARGE.value,
                reason="Direct override: forced Tier 2 (Large)",
                estimated_prompt_tokens=tokens,
                estimated_completion_tokens=est_out,
                estimated_cost=cost,
                complexity_score=1.0
            )

        router = cls.get_router(override_strategy)
        return router.route(query)
