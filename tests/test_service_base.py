"""Tests for packages/indexer/service_base.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from packages.indexer.service_base import ServiceBase

if TYPE_CHECKING:
    import pytest


class TestServiceBase:
    """Tests for ServiceBase initialization and lazy accessors."""

    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_settings = MagicMock()
        monkeypatch.setattr(
            "packages.indexer.service_base.get_settings", lambda: mock_settings
        )
        base = ServiceBase()
        assert base.settings is mock_settings

    def test_injected_providers(self) -> None:
        mock_embed = MagicMock()
        mock_qdrant = MagicMock()
        base = ServiceBase(
            settings=MagicMock(),
            embedding_provider=mock_embed,
            qdrant_repository=mock_qdrant,
        )
        assert base.get_embedding_provider() is mock_embed

    async def test_injected_qdrant_provider(self) -> None:
        mock_qdrant = MagicMock()
        base = ServiceBase(
            settings=MagicMock(),
            qdrant_repository=mock_qdrant,
        )
        assert await base.get_qdrant_repository() is mock_qdrant

    def test_lazy_init_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_provider = MagicMock()
        monkeypatch.setattr(
            "packages.indexer.service_base.get_embedding_provider",
            lambda _settings: mock_provider,
        )
        base = ServiceBase(settings=MagicMock())
        result = base.get_embedding_provider()
        assert result is mock_provider

    async def test_lazy_init_qdrant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_repo = MagicMock()
        monkeypatch.setattr(
            "packages.indexer.service_base.get_qdrant_repository",
            AsyncMock(return_value=mock_repo),
        )
        base = ServiceBase(settings=MagicMock())
        result = await base.get_qdrant_repository()
        assert result is mock_repo
