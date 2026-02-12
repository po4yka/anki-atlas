"""Structured logging configuration for Anki Atlas."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import TextIO
from uuid import uuid4

import structlog

# Correlation ID for request tracing across async operations
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context.

    Returns:
        Current correlation ID or None if not set.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set correlation ID in context, generating one if not provided.

    Args:
        correlation_id: Optional ID to set. If None, generates a new UUID.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID from context."""
    correlation_id_var.set(None)


def configure_logging(
    *,
    debug: bool = False,
    json_output: bool = False,
    log_stream: TextIO | None = None,
) -> None:
    """Configure structlog as the centralized logging layer.

    Call once per entry point (MCP server, API, CLI).

    Args:
        debug: Enable DEBUG level and verbose output.
        json_output: Use JSON renderer (for machine consumption) vs console renderer.
        log_stream: Output stream; defaults to ``sys.stderr``.
    """
    if log_stream is None:
        log_stream = sys.stderr

    level = logging.DEBUG if debug else logging.INFO

    def add_correlation_id(
        _logger: structlog.types.WrappedLogger,
        _method_name: str,
        event_dict: structlog.types.EventDict,
    ) -> structlog.types.EventDict:
        """Add correlation ID to log event if present in context."""
        corr_id = correlation_id_var.get()
        if corr_id is not None:
            event_dict["correlation_id"] = corr_id
        return event_dict

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.PrintLoggerFactory(file=log_stream),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers through structlog formatting so third-party
    # libraries (uvicorn, httpx, etc.) produce consistent output.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(**initial_context: object) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        **initial_context: Key-value pairs bound to every log entry from this logger.

    Returns:
        A ``BoundLogger`` instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(**initial_context)
    return logger
