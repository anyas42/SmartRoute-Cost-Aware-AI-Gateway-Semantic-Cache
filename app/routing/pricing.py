from __future__ import annotations

import re
import logging
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger("smartroute.pricing")

# Tiktoken tokenizer cache
_ENCODER = None


def get_tokenizer():
    """Lazily load tiktoken tokenizer with safe fallback."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding ({e}). Using heuristic tokenizer fallback.")
            _ENCODER = "FALLBACK"
    return _ENCODER


def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Count tokens using tiktoken (OpenAI BPE tokenizer) when available,
    falling back to a calibrated word/character count ratio (~4 chars per token).
    """
    if not text:
        return 0

    encoder = get_tokenizer()
    if encoder != "FALLBACK":
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Heuristic fallback: Calibrated rule of thumb for English text
    # Standard approximation: ~0.75 words per token or ~4 characters per token
    words = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(words) * 1.3))


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str
) -> float:
    """
    Calculate estimated LLM API cost in USD from token counts and configured pricing.
    Formula: (prompt_tokens * input_rate + completion_tokens * output_rate) / 1,000,000
    """
    pricing = settings.get_pricing_for_model(model_name)
    input_rate = pricing.get("input_cost_per_million", 0.15)
    output_rate = pricing.get("output_cost_per_million", 0.60)

    input_cost = (prompt_tokens * input_rate) / 1_000_000.0
    output_cost = (completion_tokens * output_rate) / 1_000_000.0
    return round(input_cost + output_cost, 6)


def calculate_savings_vs_baseline(
    prompt_tokens: int,
    completion_tokens: int,
    served_model: str,
    baseline_model: Optional[str] = None,
    is_cache_hit: bool = False
) -> float:
    """
    Calculate cost saved against an unoptimized baseline configuration.
    
    Baseline assumption:
    Without SmartRoute, every request would hit the un-cached, premium Large model (e.g. gpt-4o).
    
    If request is a cache hit:
        Savings = full cost of generating the answer on the Large model.
    If request is routed to the Small model:
        Savings = Large model cost - Small model cost.
    If request is routed to the Large model:
        Savings = $0.00.
    """
    baseline = baseline_model or settings.LARGE_MODEL_NAME
    baseline_cost = estimate_cost(prompt_tokens, completion_tokens, baseline)

    if is_cache_hit:
        return baseline_cost

    served_cost = estimate_cost(prompt_tokens, completion_tokens, served_model)
    savings = max(0.0, baseline_cost - served_cost)
    return round(savings, 6)
