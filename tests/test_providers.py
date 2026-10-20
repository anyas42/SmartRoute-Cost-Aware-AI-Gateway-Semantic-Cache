from __future__ import annotations

import pytest
from app.providers.mock_provider import MockLLMProvider
from app.providers.openai_provider import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_mock_llm_provider_generation():
    provider = MockLLMProvider(simulate_latency=False)
    response = await provider.generate(
        prompt="What is gradient descent?",
        model="gpt-4o-mini"
    )

    assert response.is_mock is True
    assert response.provider == "mock"
    assert response.model == "gpt-4o-mini"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert "gradient descent" in response.content.lower()
    assert "[Simulated Response via gpt-4o-mini]" in response.content


@pytest.mark.asyncio
async def test_openai_provider_missing_key_error():
    provider = OpenAICompatibleProvider(api_key=None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        await provider.generate(prompt="Hello", model="gpt-4o")
