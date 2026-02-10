"""FastMCP server for Anki Atlas tools."""

import sys

from mcp.server.fastmcp import FastMCP

from packages.common.config import get_settings
from packages.common.logging import configure_logging

# Configure structured logging to stderr (required for stdio transport)
settings = get_settings()
configure_logging(debug=settings.debug, json_output=True, log_stream=sys.stderr)

# Initialize FastMCP server
mcp = FastMCP(
    name="anki-atlas",
    instructions="""
Anki Atlas provides tools for searching and analyzing Anki flashcard collections.

Available tools:
- ankiatlas_search: Hybrid semantic + full-text search across notes
- ankiatlas_topic_coverage: Get coverage metrics for a topic
- ankiatlas_topic_gaps: Find missing or undercovered topics
- ankiatlas_duplicates: Detect near-duplicate notes
- ankiatlas_sync: Sync Anki collection to the index

Use these tools to help users understand their learning progress,
find specific cards, identify knowledge gaps, and maintain their collections.
""",
)

# Import tools to register them with the server
# This is done after mcp initialization to avoid circular imports
import apps.mcp.tools  # noqa: E402, F401
