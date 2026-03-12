# Troubleshooting

This guide covers the current Rust runtime behavior on `main`.

## Configuration Errors

### `postgres_url must start with postgresql:// or postgres://`

`Settings::load()` validates connection strings up front.

Fix:

```bash
export ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas
```

### `qdrant_url must start with http:// or https://`

Fix:

```bash
export ANKIATLAS_QDRANT_URL=http://localhost:6333
```

### `redis_url must start with redis:// or rediss://`

Fix:

```bash
export ANKIATLAS_REDIS_URL=redis://localhost:6379/0
```

### Invalid embedding dimension

Most non-mock providers accept a fixed dimension set. Gemini Embedding 2 is the exception: it accepts any positive value up to `3072`, though `3072`, `1536`, and `768` are the recommended sizes.

If you want a lightweight local setup, switch to mock embeddings:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384
```

For Gemini Embedding 2:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=google
export ANKIATLAS_EMBEDDING_MODEL=gemini-embedding-2-preview
export ANKIATLAS_EMBEDDING_DIMENSION=3072
```

## Provider and Search Issues

### `OPENAI_API_KEY must be set for the OpenAI embedding provider`

Fix either the credential or the provider mode:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

Or:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384
```

### `GEMINI_API_KEY or GOOGLE_API_KEY must be set for the Google embedding provider`

Fix either the credential or the provider mode:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=google
export ANKIATLAS_EMBEDDING_MODEL=gemini-embedding-2-preview
export GEMINI_API_KEY=...
```

Or:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384
```

### Reranking never applies

If `ANKIATLAS_RERANK_ENABLED=true` but `ANKIATLAS_RERANK_ENDPOINT` is missing, reranking is disabled with a warning.

Fix:

```bash
export ANKIATLAS_RERANK_ENABLED=true
export ANKIATLAS_RERANK_ENDPOINT=http://localhost:8080/rerank
```

## API Issues

### Protected routes return `401 unauthorized`

If `ANKIATLAS_API_KEY` is set, every route except `/health` and `/ready` requires `X-API-Key`.

Fix:

```bash
curl -H "X-API-Key: $ANKIATLAS_API_KEY" http://localhost:8000/topics
```

### `/ready` says `ready` even when dependencies are broken

That is current behavior. `/ready` is not a deep dependency check.

Check dependencies directly instead:

```bash
psql "$ANKIATLAS_POSTGRES_URL" -c "SELECT 1"
curl "$ANKIATLAS_QDRANT_URL/healthz"
redis-cli -u "$ANKIATLAS_REDIS_URL" ping
```

### API startup fails with `address already in use`

Fix:

```bash
lsof -i :8000
ANKIATLAS_API_PORT=8001 cargo run --bin anki-atlas-api
```

## Worker and Job Issues

### Worker exits immediately with a message about being disabled

That is intentional until worker execution is fully stabilized.

Enable it explicitly:

```bash
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

### Jobs enqueue but never complete

Check all of the following:

- Redis is reachable
- the worker is actually running
- the worker was started with `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1`

Useful checks:

```bash
redis-cli -u "$ANKIATLAS_REDIS_URL" ping
curl http://localhost:8000/jobs/<job-id>
```

## Sync and Index Issues

### Collection file not found

Verify the path:

```bash
ls -la /path/to/collection.anki2
```

Common locations:

```bash
# macOS
ls ~/Library/Application\\ Support/Anki2/*/collection.anki2

# Linux
ls ~/.local/share/Anki2/*/collection.anki2
```

### Collection dimension mismatch

This happens when your Qdrant collection was created for a different embedding dimension. With multimodal indexing enabled, the same class of failure also happens when the stored embedding model or stored vector schema no longer matches the runtime.

Fix options:

```bash
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/path/to/collection.anki2","force_reindex":true}'
```

Or run an explicit CLI reindex:

```bash
cargo run --bin anki-atlas -- index --force
```

Or recreate the Qdrant collection:

```bash
curl -X DELETE "$ANKIATLAS_QDRANT_URL/collections/anki_notes"
```

### API or MCP startup fails with `reindex required: ...`

That is expected read-only behavior when the current runtime does not match the stored vector collection.

Fix by running an explicit indexing path that is allowed to mutate storage:

```bash
cargo run --bin anki-atlas -- index --force
```

or:

```bash
curl -X POST http://localhost:8000/jobs/index \
  -H "Content-Type: application/json" \
  -d '{"force_reindex":true}'
```

### Search, duplicates, or analytics return empty results

Verify both storage layers contain data:

```bash
psql "$ANKIATLAS_POSTGRES_URL" -c "SELECT COUNT(*) FROM notes"
curl "$ANKIATLAS_QDRANT_URL/collections/anki_notes" | jq .result.points_count
```

If they are empty, rerun sync/index work through CLI or jobs.

## CLI and MCP Workflow Issues

### `generate` completed but nothing was written

That is current behavior. `generate` is a preview workflow.

### `obsidian-sync` fails unless `--dry-run` is set

That is current behavior too. Obsidian persistence is not implemented yet.

Use:

```bash
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
```

or MCP:

```json
{
  "vault_path": "/path/to/vault",
  "dry_run": true
}
```

### `--fts is not supported with --chunks`

Chunk search is semantic-only in phase 1.

Use:

```bash
cargo run --bin anki-atlas -- search "diagram" --chunks -n 10
```

### Validation fails on your input file

The validation workflow expects:

```text
front
---
back
---
optional-tag-list
```

If the second section is missing, validation fails with an explicit input error.

## Connectivity Checks

### PostgreSQL

```bash
psql "$ANKIATLAS_POSTGRES_URL" -c "SELECT 1"
```

### Qdrant

```bash
curl "$ANKIATLAS_QDRANT_URL/healthz"
```

### Redis

```bash
redis-cli -u "$ANKIATLAS_REDIS_URL" ping
```

## When in Doubt

Start from a minimal local config:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384
export ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas
export ANKIATLAS_QDRANT_URL=http://localhost:6333
export ANKIATLAS_REDIS_URL=redis://localhost:6379/0
```

Then run:

```bash
cargo run --bin anki-atlas -- migrate
cargo run --bin anki-atlas-api
```
