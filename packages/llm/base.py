"""Base LLM provider interface and response types."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from packages.common.exceptions import ProviderError
from packages.common.logging import get_logger

logger = get_logger(module=__name__)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from an LLM provider."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement generate(), check_connection(), and list_models().
    Providers should use httpx.AsyncClient for HTTP and raise ProviderError on failure.
    """

    @abstractmethod
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
        """Generate a completion from the LLM.

        Args:
            model: Model identifier.
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature (0.0-1.0).
            json_mode: Request JSON-formatted output.
            json_schema: JSON schema for structured output.

        Returns:
            LLMResponse with generated text and metadata.

        Raises:
            ProviderError: If the request fails.
        """

    async def generate_json(
        self,
        model: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response and parse it.

        Args:
            model: Model identifier.
            prompt: User prompt (should request JSON output).
            system: System prompt.
            temperature: Sampling temperature.
            json_schema: JSON schema for structured output.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ProviderError: If the request fails or response is not valid JSON.
        """
        response = await self.generate(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            json_mode=True,
            json_schema=json_schema,
        )
        try:
            parsed = json.loads(response.text)
        except json.JSONDecodeError as e:
            msg = f"LLM returned invalid JSON: {e}"
            raise ProviderError(msg, context={"response_text": response.text[:500]}) from e
        if not isinstance(parsed, dict):
            msg = f"Expected JSON object, got {type(parsed).__name__}"
            raise ProviderError(msg, context={"response_text": response.text[:500]})
        return parsed

    @abstractmethod
    async def check_connection(self) -> bool:
        """Check if the provider is accessible.

        Returns:
            True if accessible, False otherwise.
        """

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model identifiers.
        """

    async def aclose(self) -> None:  # noqa: B027
        """Close provider resources. Override in subclasses if needed."""

    async def __aenter__(self) -> BaseLLMProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
