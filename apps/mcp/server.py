"""FastMCP server for Anki Atlas tools."""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP

from packages.common.config import get_settings
from packages.common.logging import configure_logging

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
- ankiatlas_generate: Parse a note and preview card generation
- ankiatlas_validate: Validate card content and score quality
- ankiatlas_obsidian_sync: Scan an Obsidian vault and preview notes
- ankiatlas_tag_audit: Audit tags for convention violations

Use these tools to help users understand their learning progress,
find specific cards, identify knowledge gaps, generate and validate
flashcards, and maintain their collections.
""",
)

_logging_configured = False


def _ensure_logging() -> None:
    """Configure logging on first use (avoids import-time side effects)."""
    global _logging_configured
    if not _logging_configured:
        settings = get_settings()
        configure_logging(debug=settings.debug, json_output=True, log_stream=sys.stderr)
        _logging_configured = True


# Import tools to register them with the server
# This is done after mcp initialization to avoid circular imports
import apps.mcp.tools  # noqa: E402, F401
