"""CLI entry point for Anki Atlas MCP server."""


def main() -> None:
    """Run the MCP server with stdio transport."""
    from apps.mcp.server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
