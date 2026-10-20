from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost: float
    actual_cost: Optional[float]
    provider: str
    is_mock: bool
    fallback_used: bool = False


class BaseLLMProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> LLMResponse:
        """Generate a completion for the given prompt using the specified model."""
        pass
