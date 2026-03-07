"""Tests for packages/analytics/service.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.analytics.service import AnalyticsResult, AnalyticsService


class TestAnalyticsResult:
    """Tests for the frozen AnalyticsResult dataclass."""

    def test_defaults(self) -> None:
        r = AnalyticsResult()
        assert r.coverage is None
        assert r.gaps is None
        assert r.weak_notes is None

    def test_frozen(self) -> None:
        r = AnalyticsResult()
        with pytest.raises(FrozenInstanceError):
            r.coverage = "x"  # type: ignore[misc]


class TestAnalyticsServiceDelegation:
    """Tests that AnalyticsService delegates to underlying functions."""

    async def test_get_coverage_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cov = MagicMock()
        mock_fn = AsyncMock(return_value=mock_cov)
        monkeypatch.setattr(
            "packages.analytics.service.get_topic_coverage", mock_fn
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.get_coverage("math", include_subtree=False)
        assert result is mock_cov
        mock_fn.assert_awaited_once_with("math", False, service.settings)

    async def test_get_gaps_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_gaps = [MagicMock()]
        mock_fn = AsyncMock(return_value=mock_gaps)
        monkeypatch.setattr(
            "packages.analytics.service.get_topic_gaps", mock_fn
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.get_gaps("math", min_coverage=2)
        assert result is mock_gaps
        mock_fn.assert_awaited_once_with("math", 2, service.settings)

    async def test_get_weak_notes_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_weak = [MagicMock()]
        mock_fn = AsyncMock(return_value=mock_weak)
        monkeypatch.setattr(
            "packages.analytics.service.get_weak_notes", mock_fn
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.get_weak_notes("math", max_results=5)
        assert result is mock_weak
        mock_fn.assert_awaited_once_with("math", 5, settings=service.settings)

    async def test_get_full_analysis_combines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_cov = MagicMock()
        mock_gaps = [MagicMock()]
        mock_weak = [MagicMock()]
        monkeypatch.setattr(
            "packages.analytics.service.get_topic_coverage",
            AsyncMock(return_value=mock_cov),
        )
        monkeypatch.setattr(
            "packages.analytics.service.get_topic_gaps",
            AsyncMock(return_value=mock_gaps),
        )
        monkeypatch.setattr(
            "packages.analytics.service.get_weak_notes",
            AsyncMock(return_value=mock_weak),
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.get_full_analysis("math")
        assert result.coverage is mock_cov
        assert result.gaps is mock_gaps
        assert result.weak_notes is mock_weak

    async def test_get_taxonomy_tree_delegates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_tree = [{"path": "math", "label": "Math"}]
        monkeypatch.setattr(
            "packages.analytics.service.get_coverage_tree",
            AsyncMock(return_value=mock_tree),
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.get_taxonomy_tree("math")
        assert result == mock_tree


class TestAnalyticsServiceLoadTaxonomy:
    """Tests for taxonomy loading paths."""

    async def test_load_from_db(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_tax = MagicMock()
        monkeypatch.setattr(
            "packages.analytics.service.load_taxonomy_from_database",
            AsyncMock(return_value=mock_tax),
        )
        service = AnalyticsService(settings=MagicMock())
        result = await service.load_taxonomy()
        assert result is mock_tax

    async def test_load_from_yaml_syncs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_yaml_tax = MagicMock()
        mock_db_tax = MagicMock()
        mock_load_yaml = MagicMock(return_value=mock_yaml_tax)
        mock_sync = AsyncMock()
        mock_load_db = AsyncMock(return_value=mock_db_tax)

        monkeypatch.setattr(
            "packages.analytics.service.load_taxonomy_from_yaml", mock_load_yaml
        )
        monkeypatch.setattr(
            "packages.analytics.service.sync_taxonomy_to_database", mock_sync
        )
        monkeypatch.setattr(
            "packages.analytics.service.load_taxonomy_from_database", mock_load_db
        )

        service = AnalyticsService(settings=MagicMock())
        result = await service.load_taxonomy(yaml_path=MagicMock())

        mock_load_yaml.assert_called_once()
        mock_sync.assert_awaited_once()
        mock_load_db.assert_awaited_once()
        assert result is mock_db_tax
