"""Tests for CLI commands: sync, search, index, coverage, gaps, duplicates."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from apps.cli import app

runner = CliRunner()


class TestSyncCommand:
    """Tests for the sync CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["sync", "--help"])
        assert result.exit_code == 0
        assert "source" in result.output.lower()

    def test_missing_source(self) -> None:
        result = runner.invoke(
            app,
            ["sync", "--source", "/nonexistent/path.anki2", "--no-migrate", "--no-index"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Error" in result.output


class TestSearchCommand:
    """Tests for the search CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()


class TestIndexCommand:
    """Tests for the index CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
        assert "force" in result.output.lower()


class TestCoverageCommand:
    """Tests for the coverage CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["coverage", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output.lower()


class TestGapsCommand:
    """Tests for the gaps CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["gaps", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output.lower()


class TestDuplicatesCommand:
    """Tests for the duplicates CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["duplicates", "--help"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower()


class TestMigrateCommand:
    """Tests for the migrate CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["migrate", "--help"])
        assert result.exit_code == 0


class TestTopicsCommand:
    """Tests for the topics CLI command."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["topics", "--help"])
        assert result.exit_code == 0
        assert "taxonomy" in result.output.lower() or "file" in result.output.lower()


class TestCliErrorHandler:
    """Tests for _cli_error_handler context manager."""

    def test_error_handler_catches_exception(self) -> None:
        from click.exceptions import Exit

        from apps.cli import _cli_error_handler

        with pytest.raises(Exit), _cli_error_handler("cli_test_failed"):
            raise RuntimeError("boom")
