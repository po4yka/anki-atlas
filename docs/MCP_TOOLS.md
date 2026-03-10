# Anki Atlas MCP Tools

The MCP server on `main` is now service-aligned rather than intentionally narrow. It shares the same runtime wiring as the HTTP API and CLI through `crates/surface-runtime`.

## Running the Server

```bash
cargo run --bin anki-atlas-mcp
```

### With MCP Inspector

```bash
npx @anthropic-ai/mcp-inspector cargo run --bin anki-atlas-mcp
```

## Configuration

The server reads the same environment variables as the API and CLI:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANKIATLAS_POSTGRES_URL` | Yes | PostgreSQL connection used by search and analytics |
| `ANKIATLAS_QDRANT_URL` | Yes | Qdrant connection used by semantic search and duplicates |
| `ANKIATLAS_REDIS_URL` | Yes | Redis connection for async job tools |
| `ANKIATLAS_EMBEDDING_PROVIDER` | Yes | `mock`, `openai`, or `google` |
| `ANKIATLAS_EMBEDDING_MODEL` | Yes | embedding model name |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Sometimes | provider-specific embedding credentials |
| `ANKIATLAS_RERANK_ENABLED` | No | enable reranking for search |
| `ANKIATLAS_RERANK_ENDPOINT` | When reranking | endpoint for the reranker |

## Output Modes

Every tool accepts:

```json
{ "output_mode": "markdown" }
```

Supported values:

- `markdown`: human-readable default
- `json`: structured output for programmatic consumers

Both modes are backed by the same canonical result object. Markdown changes only the text rendering.

## Tool Catalog

### Read tools

- `ankiatlas_search`
- `ankiatlas_topics`
- `ankiatlas_topic_coverage`
- `ankiatlas_topic_gaps`
- `ankiatlas_topic_weak_notes`
- `ankiatlas_duplicates`

### Async job tools

- `ankiatlas_sync_job`
- `ankiatlas_index_job`
- `ankiatlas_job_status`
- `ankiatlas_job_cancel`

### Local workflow tools

- `ankiatlas_generate`
- `ankiatlas_validate`
- `ankiatlas_obsidian_sync`
- `ankiatlas_tag_audit`

## Behavioral Rules

- Read tools call the same shared search and analytics services as the API and CLI.
- MCP does not run sync or index directly. Those flows are exposed only as async job tools.
- Local workflow tools are real wrappers over Obsidian parsing, validation, and taxonomy logic.
- Unsupported paths fail explicitly. For example, `ankiatlas_obsidian_sync` rejects non-dry-run execution until a persistence sink exists.
