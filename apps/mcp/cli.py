"""CLI entry point for Anki Atlas MCP server."""

from __future__ import annotations


def main() -> None:
    """Run the MCP server with stdio transport."""
    from apps.mcp.server import _ensure_logging, mcp

    _ensure_logging()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
