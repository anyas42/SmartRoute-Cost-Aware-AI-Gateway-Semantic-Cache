from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.providers.base import BaseLLMProvider
from app.providers.mock_provider import MockLLMProvider
from app.providers.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger("smartroute.providers.factory")


class ProviderFactory:
    """Factory to instantiate and manage LLM providers."""
    _provider_instance: Optional[BaseLLMProvider] = None

    @classmethod
    def get_provider(cls, force_reload: bool = False) -> BaseLLMProvider:
        if cls._provider_instance is None or force_reload:
            provider_type = settings.LLM_PROVIDER.lower().strip()
            if provider_type == "mock":
                logger.info("Using MockLLMProvider (Offline Simulation Mode)")
                cls._provider_instance = MockLLMProvider()
            else:
                logger.info(f"Using OpenAICompatibleProvider (Base URL: {settings.OPENAI_BASE_URL})")
                cls._provider_instance = OpenAICompatibleProvider()
        return cls._provider_instance
