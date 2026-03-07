"""FastMCP server instance for Anki Atlas."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

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
