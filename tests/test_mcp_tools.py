"""Tests for apps/mcp/tools.py and apps/mcp/server.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.mcp.tools import _format_error
from packages.common.exceptions import (
    DatabaseConnectionError,
    VectorStoreConnectionError,
)


class TestFormatError:
    """Tests for _format_error helper."""

    def test_database_error(self) -> None:
        err = DatabaseConnectionError("conn refused")
        result = _format_error(err, "search")
        assert "Database unavailable" in result
        assert "PostgreSQL" in result

    def test_vector_store_error(self) -> None:
        err = VectorStoreConnectionError("timeout")
        result = _format_error(err, "indexing")
        assert "Vector database unavailable" in result
        assert "Qdrant" in result

    def test_timeout_error(self) -> None:
        err = TimeoutError()
        result = _format_error(err, "search")
        assert "timed out" in result

    def test_generic_error(self) -> None:
        err = RuntimeError("something broke")
        result = _format_error(err, "operation")
        assert "RuntimeError" in result
        assert "something broke" in result


class TestAnkiatlasSearch:
    """Tests for ankiatlas_search tool."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_service(self) -> None:
        mock_result = MagicMock()
        mock_result.results = [MagicMock(note_id=1)]
        mock_service = MagicMock()
        mock_service.search = AsyncMock(return_value=mock_result)
        mock_service.get_notes_details = AsyncMock(return_value={})

        with (
            patch(
                "packages.search.service.SearchService",
                return_value=mock_service,
            ),
            patch(
                "apps.mcp.formatters.format_search_result",
                return_value="formatted",
            ),
        ):
            from apps.mcp.tools import ankiatlas_search

            result = await ankiatlas_search(query="python lists")

        assert result == "formatted"
        mock_service.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_handles_error(self) -> None:
        with patch(
            "packages.search.service.SearchService",
            side_effect=DatabaseConnectionError("db down"),
        ):
            from apps.mcp.tools import ankiatlas_search

            result = await ankiatlas_search(query="test")

        assert "Database unavailable" in result


class TestAnkiatlasCoverage:
    """Tests for ankiatlas_topic_coverage tool."""

    @pytest.mark.asyncio
    async def test_coverage_not_found(self) -> None:
        mock_service = MagicMock()
        mock_service.get_coverage = AsyncMock(return_value=None)

        with patch(
            "packages.analytics.service.AnalyticsService",
            return_value=mock_service,
        ):
            from apps.mcp.tools import ankiatlas_topic_coverage

            result = await ankiatlas_topic_coverage(topic_path="nonexistent")

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_coverage_success(self) -> None:
        mock_cov = MagicMock()
        mock_service = MagicMock()
        mock_service.get_coverage = AsyncMock(return_value=mock_cov)

        with (
            patch(
                "packages.analytics.service.AnalyticsService",
                return_value=mock_service,
            ),
            patch(
                "apps.mcp.formatters.format_coverage_result",
                return_value="coverage output",
            ),
        ):
            from apps.mcp.tools import ankiatlas_topic_coverage

            result = await ankiatlas_topic_coverage(topic_path="math")

        assert result == "coverage output"


class TestAnkiatlasGaps:
    """Tests for ankiatlas_topic_gaps tool."""

    @pytest.mark.asyncio
    async def test_gaps_success(self) -> None:
        mock_gaps = [MagicMock()]
        mock_service = MagicMock()
        mock_service.get_gaps = AsyncMock(return_value=mock_gaps)

        with (
            patch(
                "packages.analytics.service.AnalyticsService",
                return_value=mock_service,
            ),
            patch(
                "apps.mcp.formatters.format_gaps_result",
                return_value="gaps output",
            ),
        ):
            from apps.mcp.tools import ankiatlas_topic_gaps

            result = await ankiatlas_topic_gaps(topic_path="math")

        assert result == "gaps output"


class TestAnkiatlasDuplicates:
    """Tests for ankiatlas_duplicates tool."""

    @pytest.mark.asyncio
    async def test_duplicates_success(self) -> None:
        mock_clusters = [MagicMock()]
        mock_stats = MagicMock()
        mock_detector = MagicMock()
        mock_detector.find_duplicates = AsyncMock(
            return_value=(mock_clusters, mock_stats)
        )

        with (
            patch(
                "packages.analytics.duplicates.DuplicateDetector",
                return_value=mock_detector,
            ),
            patch(
                "apps.mcp.formatters.format_duplicates_result",
                return_value="dups output",
            ),
        ):
            from apps.mcp.tools import ankiatlas_duplicates

            result = await ankiatlas_duplicates()

        assert result == "dups output"


class TestAnkiatlasSync:
    """Tests for ankiatlas_sync tool."""

    @pytest.mark.asyncio
    async def test_sync_file_not_found(self) -> None:
        from apps.mcp.tools import ankiatlas_sync

        result = await ankiatlas_sync(
            collection_path="/nonexistent/collection.anki2"
        )
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_sync_wrong_extension(self, tmp_path: object) -> None:
        from pathlib import Path

        from apps.mcp.tools import ankiatlas_sync

        bad_file = Path(str(tmp_path)) / "data.db"
        bad_file.touch()
        result = await ankiatlas_sync(collection_path=str(bad_file))
        assert ".anki2" in result


class TestMCPServer:
    """Tests for apps/mcp/server.py."""

    def test_logging_idempotency(self) -> None:
        from apps.mcp.server import _ensure_logging, _logging_state

        _logging_state.clear()
        with patch("apps.mcp.server.configure_logging") as mock_log:
            _ensure_logging()
            _ensure_logging()
        mock_log.assert_called_once()

    def test_mcp_re_exported(self) -> None:
        from apps.mcp.server import mcp

        assert mcp.name == "anki-atlas"
