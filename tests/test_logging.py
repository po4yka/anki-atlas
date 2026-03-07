"""Tests for packages/common/logging.py."""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import structlog

from packages.common.logging import (
    clear_correlation_id,
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


class TestCorrelationId:
    """Tests for correlation ID context management."""

    def setup_method(self) -> None:
        clear_correlation_id()

    def teardown_method(self) -> None:
        clear_correlation_id()

    def test_default_none(self) -> None:
        assert get_correlation_id() is None

    def test_set_generates_uuid(self) -> None:
        result = set_correlation_id()
        assert isinstance(result, str)
        assert len(result) == 36  # UUID format
        assert get_correlation_id() == result

    def test_set_custom(self) -> None:
        result = set_correlation_id("abc-123")
        assert result == "abc-123"
        assert get_correlation_id() == "abc-123"

    def test_clear(self) -> None:
        set_correlation_id("test")
        clear_correlation_id()
        assert get_correlation_id() is None


class TestConfigureLogging:
    """Tests for configure_logging."""

    def teardown_method(self) -> None:
        structlog.reset_defaults()

    def test_debug_mode_sets_root_level(self) -> None:
        configure_logging(debug=True, log_stream=StringIO())
        assert logging.getLogger().level == logging.DEBUG

    def test_info_mode_default(self) -> None:
        configure_logging(debug=False, log_stream=StringIO())
        assert logging.getLogger().level == logging.INFO

    def test_json_output_produces_json(self) -> None:
        stream = StringIO()
        configure_logging(json_output=True, log_stream=stream)
        logger: Any = structlog.get_logger()
        logger.info("test_event", key="value")
        output = stream.getvalue()
        assert "{" in output


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_bound_logger(self) -> None:
        logger = get_logger(module="test")
        # Before configure, returns lazy proxy; after, returns BoundLogger
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")
