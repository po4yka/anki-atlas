"""Ollama LLM provider implementation."""

from __future__ import annotations

from typing import Any, Final

import httpx

from packages.common.exceptions import ProviderError
from packages.common.logging import get_logger
from packages.llm.base import BaseLLMProvider, LLMResponse

logger = get_logger(module=__name__)

_DEFAULT_BASE_URL: Final = "http://localhost:11434"
_DEFAULT_TIMEOUT: Final = 300.0


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local and cloud deployments.

    Uses the Ollama HTTP API (/api/generate, /api/tags).
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )
        logger.debug("ollama_provider_init", base_url=self.base_url)

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            msg = f"Ollama HTTP {e.response.status_code}: {e.response.text[:200]}"
            raise ProviderError(msg) from e
        except httpx.RequestError as e:
            msg = f"Ollama request failed: {e}"
            raise ProviderError(msg) from e

        data = resp.json()
        return LLMResponse(
            text=data.get("response", ""),
            model=data.get("model", model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
            raw=data,
        )

    async def check_connection(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("ollama_list_models_failed", error=str(e))
            return []

    async def aclose(self) -> None:
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"OllamaProvider(base_url={self.base_url!r})"
