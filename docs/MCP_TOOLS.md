# Anki Atlas MCP Tools

The MCP server shares the same surface boundary as the API and CLI through [surface-runtime](/Users/po4yka/GitRep/anki-atlas/crates/surface-runtime/src/services.rs) and [surface-contracts](/Users/po4yka/GitRep/anki-atlas/crates/surface-contracts/src/lib.rs). That means its tools should describe real service behavior, not synthetic agent-only wrappers.

## Running the Server

```bash
cargo run --bin anki-atlas-mcp
```

With MCP Inspector:

```bash
npx @anthropic-ai/mcp-inspector cargo run --bin anki-atlas-mcp
```

## Configuration

The server reads the same environment variables as the rest of the workspace:

- `ANKIATLAS_POSTGRES_URL`
- `ANKIATLAS_QDRANT_URL`
- `ANKIATLAS_REDIS_URL`
- `ANKIATLAS_EMBEDDING_PROVIDER`
- `ANKIATLAS_EMBEDDING_MODEL`
- `ANKIATLAS_EMBEDDING_DIMENSION`
- `OPENAI_API_KEY` or `GEMINI_API_KEY` when using those providers (`GOOGLE_API_KEY` is still accepted for Google compatibility)
- `ANKIATLAS_ANKI_MEDIA_ROOT` when multimodal Anki indexing should use an explicit media directory
- `ANKIATLAS_RERANK_ENABLED`
- `ANKIATLAS_RERANK_ENDPOINT` when reranking is enabled

## Output Modes

Every tool accepts:

```json
{ "output_mode": "markdown" }
```

Supported values:

- `markdown`: default, optimized for human or agent reading
- `json`: machine-readable payload using the same canonical result object

Markdown does not invent a different schema. It is only a rendering of the same result data.

## Tool Catalog

### Read tools

| Tool | Purpose | Key inputs |
|---|---|---|
| `ankiatlas_search` | Hybrid note retrieval | `query`, `deck_names[]`, `tags[]`, `limit`, `semantic_only`, `fts_only` |
| `ankiatlas_search_chunks` | Semantic-only raw chunk retrieval | `query`, `deck_names[]`, `tags[]`, `limit` |
| `ankiatlas_topics` | Taxonomy tree inspection | `root_path` |
| `ankiatlas_topic_coverage` | Topic coverage metrics | `topic_path`, `include_subtree` |
| `ankiatlas_topic_gaps` | Missing or undercovered topics | `topic_path`, `min_coverage` |
| `ankiatlas_topic_weak_notes` | Weak-note listing | `topic_path`, `max_results` |
| `ankiatlas_duplicates` | Duplicate-note clustering | `threshold`, `max_clusters`, `deck_filter[]`, `tag_filter[]` |

### Async job tools

| Tool | Purpose | Key inputs |
|---|---|---|
| `ankiatlas_sync_job` | Enqueue sync work | `source`, `run_migrations`, `index`, `force_reindex` |
| `ankiatlas_index_job` | Enqueue indexing work | `force_reindex` |
| `ankiatlas_job_status` | Poll queued work | `job_id` |
| `ankiatlas_job_cancel` | Request cancellation | `job_id` |

Write-side behavior is async-only here. MCP does not run sync or index directly.

### Local workflow tools

| Tool | Purpose | Key inputs |
|---|---|---|
| `ankiatlas_generate` | Parse a note and preview card generation | `file_path` |
| `ankiatlas_validate` | Run validation pipeline on a file | `file_path`, `quality` |
| `ankiatlas_obsidian_sync` | Scan an Obsidian vault in preview mode | `vault_path`, `source_dirs[]`, `dry_run` |
| `ankiatlas_tag_audit` | Validate and normalize tags | `file_path`, `fix` |

## Behavioral Rules

- Read tools use the same search and analytics facades as the API and CLI.
- `ankiatlas_search` returns note results and includes best semantic chunk metadata when semantic retrieval contributes.
- `ankiatlas_search_chunks` is semantic-only in phase 1 and returns raw chunk hits with `chunk_id`, `chunk_kind`, `modality`, `source_field`, `asset_rel_path`, `mime_type`, and `preview_label`.
- Job tools enqueue Redis-backed work and return job-oriented results with poll and cancel hints.
- `ankiatlas_generate` is preview-only.
- `ankiatlas_obsidian_sync` rejects non-dry-run requests until a persistence sink exists.
- Tool errors are explicit and typed; unsupported paths should not masquerade as success.

## Example Calls

### Search in markdown mode

```json
{
  "query": "ownership",
  "limit": 5,
  "output_mode": "markdown"
}
```

### Search in json mode

```json
{
  "query": "ownership",
  "deck_names": ["Rust"],
  "limit": 5,
  "output_mode": "json"
}
```

### Chunk search in markdown mode

```json
{
  "query": "diagram",
  "deck_names": ["Rust"],
  "limit": 5,
  "output_mode": "markdown"
}
```

### Enqueue a sync job

```json
{
  "source": "/path/to/collection.anki2",
  "run_migrations": true,
  "index": true,
  "force_reindex": false,
  "output_mode": "markdown"
}
```

### Scan an Obsidian vault

```json
{
  "vault_path": "/path/to/vault",
  "source_dirs": ["notes", "cards"],
  "dry_run": true,
  "output_mode": "json"
}
```

## Source Files

- [server.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/server.rs)
- [tools.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/tools.rs)
- [formatters.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/formatters.rs)
- [handlers.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/handlers.rs)
