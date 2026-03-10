# First Time Setup Guide

Step-by-step setup for the current Rust workspace on `main`.

## Prerequisites

- Rust 1.88+ with Cargo
- Docker and Docker Compose
- PostgreSQL, Qdrant, and Redis available locally or remotely
- An Anki collection if you plan to enqueue sync jobs
- Optional: `OPENAI_API_KEY` if your embedding provider requires it

## Step 1: Clone and Build

```bash
git clone https://github.com/po4yka/anki-atlas.git
cd anki-atlas

cargo build
cargo run --bin anki-atlas -- version
```

## Step 2: Start Infrastructure

Use the workspace Compose file:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Verify the dependencies:

```bash
docker compose -f infra/docker-compose.yml ps
psql postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas -c "SELECT 1"
curl http://localhost:6333/healthz
```

## Step 3: Configure Environment

Create a `.env` file or export the required variables:

```bash
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas
ANKIATLAS_QDRANT_URL=http://localhost:6333
ANKIATLAS_REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-your-api-key-here
ANKIATLAS_DEBUG=true
```

## Step 4: Initialize PostgreSQL

Run migrations:

```bash
cargo run --bin anki-atlas -- migrate
```

Verify the schema:

```bash
psql $ANKIATLAS_POSTGRES_URL -c "\dt"
```

## Step 5: Find Your Anki Collection

```bash
# macOS
ls ~/Library/Application\ Support/Anki2/*/collection.anki2

# Linux
ls ~/.local/share/Anki2/*/collection.anki2

# Windows (PowerShell)
ls "$env:APPDATA\Anki2\*\collection.anki2"
```

Close Anki before using the collection file directly.

## Step 6: Start the API

```bash
cargo run --bin anki-atlas-api
```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Step 7: Start the Worker

The worker remains gated while background job execution is still being completed:

```bash
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

## Step 8: Enqueue a Sync Job

```bash
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/path/to/collection.anki2"}'
```

Poll the job:

```bash
curl http://localhost:8000/jobs/<job-id>
```

## Step 9: Verify the Read API

The v2 read surface is synchronous and typed. Direct `/sync` and `/index` mutations are not exposed.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"ownership","limit":5}'

curl "http://localhost:8000/topics"
curl "http://localhost:8000/topic-coverage?topic_path=rust/ownership"
curl "http://localhost:8000/topic-gaps?topic_path=rust&min_coverage=1"
curl "http://localhost:8000/topic-weak-notes?topic_path=rust&max_results=20"
curl "http://localhost:8000/duplicates?threshold=0.92&max_clusters=10&deck_filter[]=Rust"
```

## Step 10: Use the Wired CLI Surface

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
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
cargo run --bin anki-atlas -- tag-audit /path/to/tags.txt
```

## Step 11: Set Up MCP

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

The MCP server now exposes typed read tools plus async-only job tools. Examples:

- `ankiatlas_search`
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

## Next Steps

- Use `/jobs/sync` and `/jobs/index` for data ingestion work
- Use `/search`, `/topics`, `/topic-coverage`, `/topic-gaps`, `/topic-weak-notes`, and `/duplicates` for read-only API access
- Use the CLI for direct sync/index, search, analytics, and local preview workflows
- Keep [docs/ARCHITECTURE.md](./ARCHITECTURE.md) as the source of truth for which surfaces are intentionally exposed on `main`

## Common Follow-Up

### "Collection dimension mismatch"

If you changed embedding models or dimensions, rerun sync/index work after updating configuration:

```bash
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/path/to/collection.anki2","force_reindex":true}'
```

For more troubleshooting, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
