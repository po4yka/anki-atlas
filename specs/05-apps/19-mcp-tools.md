# Historical Note: MCP App Plan

This document described an earlier Python-era plan for extending `apps/mcp/`.

It is no longer the current implementation reference.

Use these documents instead:

- [MCP Spec](/Users/po4yka/GitRep/anki-atlas/specs/18-mcp.md)
- [MCP Tools Guide](/Users/po4yka/GitRep/anki-atlas/docs/MCP_TOOLS.md)
- [Architecture](/Users/po4yka/GitRep/anki-atlas/docs/ARCHITECTURE.md)

Current reality:

- the MCP server lives in `bins/mcp/`
- it uses `rmcp` tool routing
- it exposes 14 typed tools
- sync/index remain async-only job tools
- every tool supports `markdown` and `json` output modes
