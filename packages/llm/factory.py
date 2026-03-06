"""Provider factory for creating LLM provider instances."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from packages.common.exceptions import ProviderError

if TYPE_CHECKING:
    from .base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class ProviderType(StrEnum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class ProviderFactory:
    """Factory for creating LLM provider instances by type."""

    @staticmethod
    def create(provider_type: ProviderType | str, **kwargs: Any) -> BaseLLMProvider:
        """Create a provider instance.

        Args:
            provider_type: Provider type (ProviderType enum or string).
            **kwargs: Provider-specific configuration.

        Returns:
            Initialized provider instance.

        Raises:
            ProviderError: If provider_type is unsupported or creation fails.
        """
        try:
            ptype = ProviderType(str(provider_type).lower())
        except ValueError:
            available = ", ".join(t.value for t in ProviderType)
            msg = f"Unsupported provider: {provider_type}. Available: {available}"
            raise ProviderError(msg) from None

        if ptype == ProviderType.OLLAMA:
            from .ollama import OllamaProvider

            return OllamaProvider(**kwargs)

        if ptype == ProviderType.OPENROUTER:
            from .openrouter import OpenRouterProvider

            return OpenRouterProvider(**kwargs)

        msg = f"Unhandled provider type: {ptype}"  # pragma: no cover
        raise ProviderError(msg)  # pragma: no cover
