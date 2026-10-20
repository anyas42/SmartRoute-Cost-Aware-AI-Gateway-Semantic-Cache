from __future__ import annotations

import os
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseSettings):
    """
    Centralized pricing configuration per 1 Million tokens in USD.
    These are illustrative standard prices (based on public provider rate cards).
    Can be updated or overridden via environment variables.
    """
    input_cost_per_million: float
    output_cost_per_million: float


# Default illustrative model pricing dictionary
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # Tier 1: Small / Economy models
    "gpt-4o-mini": {
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
    },
    "claude-3-haiku": {
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 1.25,
    },
    "llama-3.1-8b-instant": {
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.08,
    },
    # Tier 2: Large / Reasoning / Flagship models
    "gpt-4o": {
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
    },
    "claude-3-5-sonnet": {
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
    },
    "llama-3.3-70b-versatile": {
        "input_cost_per_million": 0.59,
        "output_cost_per_million": 0.79,
    },
}


class Settings(BaseSettings):
    """
    Application settings for SmartRoute AI Gateway.
    Loads configurations from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    APP_NAME: str = "SmartRoute AI Gateway"
    APP_ENV: str = Field(default="development", description="development | production | test")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Provider Configuration
    LLM_PROVIDER: str = Field(
        default="mock",
        description="Active provider adapter: 'mock' (offline demo) or 'openai' (live OpenAI / Groq / Ollama)"
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="API key for live LLM provider")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="Provider API base URL")
    REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, description="Network timeout for LLM provider calls")

    # Configurable Model Tiers
    SMALL_MODEL_NAME: str = Field(default="gpt-4o-mini", description="Model name for Tier 1 (Economy/Simple)")
    LARGE_MODEL_NAME: str = Field(default="gpt-4o", description="Model name for Tier 2 (Complex/Reasoning)")

    # Semantic Cache Configuration
    CACHE_ENABLED: bool = Field(default=True, description="Master switch for semantic caching")
    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold (0.0 - 1.0) required to trigger a semantic cache hit"
    )
    CACHE_TTL_SECONDS: int = Field(default=86400, description="Cache entry time-to-live in seconds (default 24h)")
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers embedding model name or path"
    )
    MAX_CACHE_ENTRIES: int = Field(default=10000, description="Maximum stored cache items")

    # Cost-Aware Routing Configuration
    DEFAULT_ROUTING_STRATEGY: str = Field(
        default="rule_based",
        description="Routing strategy: 'rule_based' (deterministic rules) or 'classifier' (feature scoring)"
    )
    COMPLEXITY_TOKEN_THRESHOLD: int = Field(
        default=60,
        description="Input token count threshold above which requests automatically route to the large model tier"
    )
    CLASSIFIER_THRESHOLD: float = Field(
        default=0.50,
        description="Complexity score cutoff (0.0 to 1.0) for the classifier-based router"
    )

    # Database
    DATABASE_PATH: str = Field(default="smartroute.db", description="SQLite database file path")

    # Safeguards
    MAX_QUERY_LENGTH: int = Field(default=4000, description="Maximum characters allowed in an incoming query")
    ENABLE_NEGATION_DETECTION: bool = Field(
        default=True,
        description="Prevent cache reuse if query polarity or negation markers differ significantly"
    )

    def get_pricing_for_model(self, model_name: str) -> Dict[str, float]:
        """
        Retrieve pricing for a given model, with safe fallbacks.
        """
        if model_name in DEFAULT_PRICING:
            return DEFAULT_PRICING[model_name]
        # Fallback to small model pricing if unknown
        return DEFAULT_PRICING.get(self.SMALL_MODEL_NAME, {"input_cost_per_million": 0.15, "output_cost_per_million": 0.60})


# Global singleton instance
settings = Settings()
