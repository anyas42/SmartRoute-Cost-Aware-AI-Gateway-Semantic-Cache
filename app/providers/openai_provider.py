from __future__ import annotations

import time
import logging
from typing import Optional, Dict, Any
import httpx

from app.config import settings
from app.providers.base import BaseLLMProvider, LLMResponse
from app.routing.pricing import estimate_cost

logger = logging.getLogger("smartroute.provider.openai")


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Adapter for standard OpenAI-compatible Chat Completions endpoints
    (OpenAI, Groq, Together, DeepSeek, Local Ollama, vLLM).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = settings.OPENAI_BASE_URL,
        timeout: float = settings.REQUEST_TIMEOUT_SECONDS
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError(
                "Cannot call live provider: OPENAI_API_KEY is not set in environment or .env file. "
                "Switch to LLM_PROVIDER=mock or set your API key."
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.time()
        fallback_used = False
        actual_model_used = model

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
                
                # Check for rate limit or model unavailability that could trigger fallback
                if response.status_code in (429, 503) and model == settings.LARGE_MODEL_NAME:
                    logger.warning(f"Large model {model} returned {response.status_code}. Attempting fallback to {settings.SMALL_MODEL_NAME}...")
                    payload["model"] = settings.SMALL_MODEL_NAME
                    actual_model_used = settings.SMALL_MODEL_NAME
                    fallback_used = True
                    response = await client.post(endpoint, json=payload, headers=headers)

                response.raise_for_status()
                data = response.json()

            except httpx.TimeoutException:
                logger.error(f"Provider request timed out after {self.timeout}s calling {model}")
                raise RuntimeError(f"LLM Provider timeout after {self.timeout}s")
            except httpx.HTTPStatusError as e:
                # Mask sensitive key in logs
                clean_err = f"Provider HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.error(clean_err)
                raise RuntimeError(clean_err)
            except Exception as e:
                logger.error(f"Provider connection error: {str(e)}")
                raise RuntimeError(f"Provider network failure: {str(e)}")

        latency_ms = round((time.time() - start_time) * 1000.0, 2)

        # Parse usage
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""

        # Cost calculation
        estimated = estimate_cost(prompt_tokens, completion_tokens, actual_model_used)

        return LLMResponse(
            content=content,
            model=actual_model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            estimated_cost=estimated,
            actual_cost=None,  # Standard OpenAI response doesn't provide dollar amounts directly
            provider="openai_compatible",
            is_mock=False,
            fallback_used=fallback_used
        )
