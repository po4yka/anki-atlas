"""Tests for apps/mcp/instance.py."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from apps.mcp.instance import mcp


class TestMCPInstance:
    """Tests for the FastMCP server instance."""

    def test_name(self) -> None:
        assert mcp.name == "anki-atlas"

    def test_type(self) -> None:
        assert isinstance(mcp, FastMCP)
