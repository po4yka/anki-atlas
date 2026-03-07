"""Tests for packages/anki/sync/core.py."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from packages.anki.sync.core import SyncService, SyncStats

if TYPE_CHECKING:
    import pytest


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_defaults(self) -> None:
        stats = SyncStats()
        assert stats.decks_upserted == 0
        assert stats.models_upserted == 0
        assert stats.notes_upserted == 0
        assert stats.notes_deleted == 0
        assert stats.cards_upserted == 0
        assert stats.card_stats_upserted == 0
        assert stats.duration_ms == 0

    def test_custom_values(self) -> None:
        stats = SyncStats(decks_upserted=2, notes_upserted=10, duration_ms=500)
        assert stats.decks_upserted == 2
        assert stats.notes_upserted == 10
        assert stats.duration_ms == 500


class TestSyncService:
    """Tests for SyncService."""

    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_settings = MagicMock()
        monkeypatch.setattr(
            "packages.anki.sync.core.get_settings", lambda: mock_settings
        )
        service = SyncService()
        assert service.settings is mock_settings

    def test_custom_settings(self) -> None:
        mock_settings = MagicMock()
        service = SyncService(settings=mock_settings)
        assert service.settings is mock_settings

    async def test_sync_collection_reads_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_collection = self._make_mock_collection()
        mock_reader = MagicMock(return_value=mock_collection)
        self._patch_dependencies(monkeypatch, mock_reader)

        service = SyncService(settings=MagicMock())
        await service.sync_collection("/fake/collection.anki2")

        mock_reader.assert_called_once_with("/fake/collection.anki2")

    async def test_sync_collection_calls_all_methods(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_collection = self._make_mock_collection()
        self._patch_dependencies(
            monkeypatch,
            MagicMock(return_value=mock_collection),
        )

        service = SyncService(settings=MagicMock())
        service._sync_decks = AsyncMock(return_value=2)  # type: ignore[method-assign]
        service._sync_models = AsyncMock(return_value=1)  # type: ignore[method-assign]
        service._sync_notes = AsyncMock(return_value=3)  # type: ignore[method-assign]
        service._delete_missing_notes = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._sync_cards = AsyncMock(return_value=3)  # type: ignore[method-assign]
        service._sync_card_stats = AsyncMock(return_value=2)  # type: ignore[method-assign]
        service._update_sync_metadata = AsyncMock()  # type: ignore[method-assign]

        await service.sync_collection("/fake/path")

        service._sync_decks.assert_awaited_once()
        service._sync_models.assert_awaited_once()
        service._sync_notes.assert_awaited_once()
        service._delete_missing_notes.assert_awaited_once()
        service._sync_cards.assert_awaited_once()
        service._sync_card_stats.assert_awaited_once()
        service._update_sync_metadata.assert_awaited_once()

    async def test_sync_collection_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_collection = self._make_mock_collection()
        mock_conn = AsyncMock()
        self._patch_dependencies(
            monkeypatch,
            MagicMock(return_value=mock_collection),
            mock_conn=mock_conn,
        )

        service = SyncService(settings=MagicMock())
        service._sync_decks = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._sync_models = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._sync_notes = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._delete_missing_notes = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._sync_cards = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._sync_card_stats = AsyncMock(return_value=0)  # type: ignore[method-assign]
        service._update_sync_metadata = AsyncMock()  # type: ignore[method-assign]

        await service.sync_collection("/fake/path")
        mock_conn.commit.assert_awaited_once()

    async def test_sync_collection_returns_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_collection = self._make_mock_collection()
        self._patch_dependencies(
            monkeypatch,
            MagicMock(return_value=mock_collection),
        )

        service = SyncService(settings=MagicMock())
        service._sync_decks = AsyncMock(return_value=2)  # type: ignore[method-assign]
        service._sync_models = AsyncMock(return_value=1)  # type: ignore[method-assign]
        service._sync_notes = AsyncMock(return_value=5)  # type: ignore[method-assign]
        service._delete_missing_notes = AsyncMock(return_value=1)  # type: ignore[method-assign]
        service._sync_cards = AsyncMock(return_value=5)  # type: ignore[method-assign]
        service._sync_card_stats = AsyncMock(return_value=4)  # type: ignore[method-assign]
        service._update_sync_metadata = AsyncMock()  # type: ignore[method-assign]

        stats = await service.sync_collection("/fake/path")

        assert stats.decks_upserted == 2
        assert stats.models_upserted == 1
        assert stats.notes_upserted == 5
        assert stats.notes_deleted == 1
        assert stats.cards_upserted == 5
        assert stats.card_stats_upserted == 4
        assert stats.duration_ms >= 0

    async def test_sync_anki_collection_convenience(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from packages.anki.sync.core import sync_anki_collection

        mock_stats = SyncStats(notes_upserted=3)
        mock_sync = AsyncMock(return_value=mock_stats)
        monkeypatch.setattr(
            "packages.anki.sync.core.SyncService.sync_collection", mock_sync
        )
        result = await sync_anki_collection("/fake/path")
        assert result.notes_upserted == 3

    @staticmethod
    def _make_mock_collection() -> MagicMock:
        collection = MagicMock()
        collection.decks = []
        collection.models = []
        collection.notes = []
        collection.cards = []
        collection.card_stats = []
        collection.collection_path = "/fake/path"
        return collection

    @staticmethod
    def _patch_dependencies(
        monkeypatch: pytest.MonkeyPatch,
        mock_reader: Any,
        mock_conn: AsyncMock | None = None,
    ) -> None:
        monkeypatch.setattr(
            "packages.anki.sync.core.read_anki_collection", mock_reader
        )
        monkeypatch.setattr(
            "packages.anki.sync.core.build_deck_map", MagicMock(return_value={})
        )
        monkeypatch.setattr(
            "packages.anki.sync.core.build_card_deck_map",
            MagicMock(return_value={}),
        )
        monkeypatch.setattr(
            "packages.anki.sync.core.normalize_notes", MagicMock()
        )

        if mock_conn is None:
            mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_get_connection(_settings: Any) -> Any:
            yield mock_conn

        monkeypatch.setattr(
            "packages.anki.sync.core.get_connection", mock_get_connection
        )
