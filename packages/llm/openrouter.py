"""OpenRouter LLM provider implementation."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Final

import httpx
import structlog

from packages.common.exceptions import ProviderError

from .base import BaseLLMProvider, LLMResponse

logger = structlog.get_logger(__name__)

_DEFAULT_BASE_URL: Final = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT: Final = 180.0
_MAX_RETRIES: Final = 3
_RETRYABLE_STATUS_CODES: Final = frozenset({429, 502, 503, 504})


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible chat completions API.

    Requires an API key from https://openrouter.ai/ (or OPENROUTER_API_KEY env var).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_tokens: int = 2048,
        site_url: str | None = None,
        site_name: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not resolved_key:
            msg = "OpenRouter API key required (pass api_key or set OPENROUTER_API_KEY)"
            raise ProviderError(msg)

        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens

        headers: dict[str, str] = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )
        logger.debug("openrouter_provider_init", base_url=self.base_url)

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
                if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "openrouter_retryable_error",
                        status=resp.status_code,
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return self._parse_response(resp.json(), model)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                    msg = f"OpenRouter HTTP {e.response.status_code}: {e.response.text[:200]}"
                    raise ProviderError(msg) from e
            except httpx.RequestError as e:
                msg = f"OpenRouter request failed: {e}"
                raise ProviderError(msg) from e

        msg = f"OpenRouter request failed after {_MAX_RETRIES} retries"
        raise ProviderError(msg) from last_error

    @staticmethod
    def _parse_response(data: dict[str, Any], fallback_model: str) -> LLMResponse:
        choices = data.get("choices", [])
        if not choices:
            msg = "OpenRouter returned empty choices"
            raise ProviderError(msg, context={"response": data})

        choice = choices[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return LLMResponse(
            text=message.get("content", ""),
            model=data.get("model", fallback_model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def check_connection(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("openrouter_list_models_failed", error=str(e))
            return []

    async def aclose(self) -> None:
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"OpenRouterProvider(base_url={self.base_url!r})"
