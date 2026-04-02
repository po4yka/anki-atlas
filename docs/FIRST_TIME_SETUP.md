# First Time Setup

This guide is for the current Rust workspace on `main`.

## Prerequisites

- Rust `1.88+`
- Docker and Docker Compose
- An Anki collection (for sync)
- An embedding API key (Gemini or OpenAI)

## 1. Clone and Build

```bash
git clone https://github.com/po4yka/anki-atlas.git
cd anki-atlas

cargo build
cargo run --bin anki-atlas -- version
```

## 2. Start Infrastructure

```bash
docker compose up -d
docker compose ps   # wait until all services are healthy
```

This starts PostgreSQL 16, Qdrant v1.16.3, and Redis 7 with persistent volumes.

Quick health checks:

```bash
psql postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas -c "SELECT 1"
curl http://localhost:6333/healthz
redis-cli -u redis://localhost:6379/0 ping
```

## 3. Configure Environment

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` (get one at https://aistudio.google.com/apikey).

All binaries (CLI, API, MCP, worker) load `.env` automatically via `dotenvy`.

The `.env.example` ships pre-configured for **Gemini Embedding 2** (`gemini-embedding-2-preview`, dimension 3072). To use a different provider, see the alternatives below.

<details>
<summary>Alternative: OpenAI embeddings</summary>

Set these in your `.env`:

```bash
ANKIATLAS_EMBEDDING_PROVIDER=openai
ANKIATLAS_EMBEDDING_MODEL=text-embedding-3-small
ANKIATLAS_EMBEDDING_DIMENSION=1536
OPENAI_API_KEY=sk-...
```

</details>

<details>
<summary>Alternative: Mock provider (no API key needed)</summary>

For a quick smoke test without external calls:

```bash
ANKIATLAS_EMBEDDING_PROVIDER=mock
ANKIATLAS_EMBEDDING_DIMENSION=384
```

</details>

## 4. Run Migrations

```bash
cargo run --bin anki-atlas -- migrate
```

## 5. Find Your Anki Collection

```bash
# macOS
ls ~/Library/Application\ Support/Anki2/*/collection.anki2

# Linux
ls ~/.local/share/Anki2/*/collection.anki2

# Windows (PowerShell)
ls "$env:APPDATA\Anki2\*\collection.anki2"
```

Close Anki before using the SQLite collection file directly.

Update `ANKIATLAS_ANKI_COLLECTION_PATH` in your `.env` with the path you found.

Media files resolve in this order:

1. `ANKIATLAS_ANKI_MEDIA_ROOT` (explicit override in `.env`)
2. `last_collection_path` saved in `sync_metadata`
3. `ANKIATLAS_ANKI_COLLECTION_PATH`
4. sibling `collection.media` next to the collection file

## 6. Sync and Index

Sync imports cards from Anki SQLite into Postgres and indexes them into Qdrant:

```bash
cargo run --bin anki-atlas -- sync "$ANKIATLAS_ANKI_COLLECTION_PATH"
```

This runs migrations, imports cards, and creates vector embeddings in one step.

To re-index only (e.g. after changing embedding provider/dimension):

```bash
cargo run --bin anki-atlas -- index --force
```

## 7. Start the API

```bash
cargo run --bin anki-atlas-api
```

Check liveness:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Note: `/ready` currently reports process readiness only. It does not verify PostgreSQL, Qdrant, or Redis connectivity.

If you set `ANKIATLAS_API_KEY`, include it on protected routes:

```bash
curl -H "X-API-Key: $ANKIATLAS_API_KEY" http://localhost:8000/topics
```

## 8. Start the Worker

```bash
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

Without that env var, the worker exits intentionally.

## 9. Enqueue a Sync Job

```bash
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/path/to/collection.anki2"}'
```

Poll the job:

```bash
curl http://localhost:8000/jobs/<job-id>
```

## 10. Verify the Read API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"ownership","limit":5}'

curl -X POST http://localhost:8000/search/chunks \
  -H "Content-Type: application/json" \
  -d '{"query":"diagram","limit":5}'

curl http://localhost:8000/topics
curl "http://localhost:8000/topic-coverage?topic_path=rust/ownership"
curl "http://localhost:8000/topic-gaps?topic_path=rust&min_coverage=1"
curl "http://localhost:8000/topic-weak-notes?topic_path=rust&max_results=20"
curl "http://localhost:8000/duplicates?threshold=0.92&max_clusters=10&deck_filter[]=Rust"
```

Direct `/sync` and `/index` HTTP mutations are intentionally not exposed.

## 11. CLI Reference

```bash
cargo run --bin anki-atlas -- sync /path/to/collection.anki2 --force-reindex
cargo run --bin anki-atlas -- index --force
cargo run --bin anki-atlas -- search "ownership" --deck Rust -n 5
cargo run --bin anki-atlas -- topics tree --root-path rust
cargo run --bin anki-atlas -- coverage rust/ownership
cargo run --bin anki-atlas -- gaps rust --min-coverage 1
cargo run --bin anki-atlas -- weak-notes rust/ownership -n 10
cargo run --bin anki-atlas -- duplicates --threshold 0.92 --max 10
cargo run --bin anki-atlas -- generate /path/to/note.md --dry-run
cargo run --bin anki-atlas -- validate /path/to/cards.txt --quality
cargo run --bin anki-atlas -- search "diagram" --chunks -n 10
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
cargo run --bin anki-atlas -- tag-audit /path/to/tags.txt --fix
```

Behavioral notes:

- `generate` previews parsed cards; it does not persist them.
- `obsidian-sync` requires `--dry-run` today.
- CLI sync/index need PostgreSQL and Qdrant available.
- `search --chunks` is semantic-only raw chunk search. `--fts` is not supported with `--chunks`.
- Explicit CLI `index --force` or sync/index work recreates an incompatible vector collection automatically. API and MCP startup stay read-only and return `reindex required` until you reindex.

## 12. Set Up MCP

Example MCP configuration:

```json
{
  "mcpServers": {
    "anki-atlas": {
      "command": "cargo",
      "args": ["run", "--bin", "anki-atlas-mcp"],
      "cwd": "/path/to/anki-atlas"
    }
  }
}
```

Current tool set:

- `ankiatlas_search`
- `ankiatlas_search_chunks`
- `ankiatlas_topics`
- `ankiatlas_topic_coverage`
- `ankiatlas_topic_gaps`
- `ankiatlas_topic_weak_notes`
- `ankiatlas_duplicates`
- `ankiatlas_sync_job`
- `ankiatlas_index_job`
- `ankiatlas_job_status`
- `ankiatlas_job_cancel`
- `ankiatlas_generate`
- `ankiatlas_validate`
- `ankiatlas_obsidian_sync`
- `ankiatlas_tag_audit`

Every tool supports `output_mode = "markdown" | "json"`.

## Next Steps

- Use API or MCP job tools for async sync/index orchestration.
- Use the CLI for direct sync/index execution and local preview workflows.
- Use [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) as the source of truth for exposed surfaces.
- Use [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) when setup fails.
