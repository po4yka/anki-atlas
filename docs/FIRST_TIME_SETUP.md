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

## Step 9: Use the Wired CLI Workflows

Current stable CLI commands focus on local content workflows:

```bash
cargo run --bin anki-atlas -- generate /path/to/note.md --dry-run
cargo run --bin anki-atlas -- validate /path/to/cards.txt --quality
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
cargo run --bin anki-atlas -- tag-audit /path/to/tags.txt
```

## Step 10: Set Up MCP

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

## Next Steps

- Use `/jobs/sync` and `/jobs/index` for data ingestion work
- Use the CLI for generation, validation, and Obsidian workflows
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
