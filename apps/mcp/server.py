"""FastMCP server for Anki Atlas tools."""

from __future__ import annotations

import sys

from apps.mcp.instance import mcp
from packages.common.config import get_settings
from packages.common.logging import configure_logging

# Re-export mcp for backwards compatibility
__all__ = ["mcp"]

_logging_configured = False


def _ensure_logging() -> None:
    """Configure logging on first use (avoids import-time side effects)."""
    global _logging_configured
    if not _logging_configured:
        settings = get_settings()
        configure_logging(debug=settings.debug, json_output=True, log_stream=sys.stderr)
        _logging_configured = True


# Import tools to register them with the server
import apps.mcp.tools  # noqa: E402, F401
