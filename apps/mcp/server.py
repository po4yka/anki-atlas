"""FastMCP server for Anki Atlas tools."""

from __future__ import annotations

import sys

from apps.mcp.instance import mcp
from packages.common.config import get_settings
from packages.common.logging import configure_logging

# Re-export mcp for backwards compatibility
__all__ = ["mcp"]

_logging_state: dict[str, bool] = {}


def _ensure_logging() -> None:
    """Configure logging on first use (avoids import-time side effects)."""
    if "configured" not in _logging_state:
        settings = get_settings()
        configure_logging(debug=settings.debug, json_output=True, log_stream=sys.stderr)
        _logging_state["configured"] = True


# Import tools to register them with the server
import apps.mcp.tools  # noqa: E402, F401
