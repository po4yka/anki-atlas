# Anki Atlas MCP Tools

MCP (Model Context Protocol) support in `main` is intentionally narrow. The server should advertise only tools that have real handler implementations and do not depend on unwired search or analytics surfaces.

## Running the Server

```bash
cargo run --bin anki-atlas-mcp
```

### With MCP Inspector

```bash
npx @anthropic-ai/mcp-inspector cargo run --bin anki-atlas-mcp
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "anki-atlas": {
      "command": "cargo",
      "args": ["run", "--bin", "anki-atlas-mcp"],
      "env": {
        "ANKIATLAS_POSTGRES_URL": "postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas",
        "ANKIATLAS_QDRANT_URL": "http://localhost:6333",
        "ANKIATLAS_EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "servers": {
    "anki-atlas": {
      "command": "cargo",
      "args": ["run", "--bin", "anki-atlas-mcp"],
      "cwd": "/path/to/anki-atlas",
      "env": {
        "ANKIATLAS_POSTGRES_URL": "postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas",
        "ANKIATLAS_QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Available Tools

### ankiatlas_generate

Parse markdown text and return a generation preview.

### ankiatlas_obsidian_sync

Scan an Obsidian vault directory and summarize discovered markdown notes.

### ankiatlas_tag_audit

Inspect tags for normalization issues such as uppercase characters or `/` separators.

## Example Agent Workflows

### Local Note Review

```
1. Preview cards from this markdown note
2. Scan this Obsidian vault for candidate notes
3. Audit these tags before importing them
```

## Output Format

Current tools return markdown-formatted responses optimized for LLM readability:

- **Tables** for search results and gap lists
- **Headings** for section organization
- **Bold** for key metrics
- **Code blocks** for paths and technical identifiers
- **Truncated previews** to keep responses concise

## Error Handling

Tools return user-friendly error messages prefixed with `**Error**:` instead of raising exceptions. Common errors:

- Vault path not found
- Invalid parameters

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANKIATLAS_POSTGRES_URL` | No | Reserved for future data-backed MCP tools |
| `ANKIATLAS_QDRANT_URL` | No | Reserved for future data-backed MCP tools |
