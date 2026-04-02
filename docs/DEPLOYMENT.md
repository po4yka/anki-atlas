# Deployment Guide

This guide documents the current deployable runtime on `main`.

## Scope

Today:

- the root [Dockerfile](Dockerfile) builds `anki-atlas-api`
- the API can serve read traffic and enqueue jobs
- full async execution also requires a worker process with `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1`
- CLI and MCP are usually run as host processes or from custom images

If you need containerized worker or MCP deployment, build custom images that include those binaries. The root Dockerfile does not currently publish them.

## Required Services

- PostgreSQL
- Qdrant
- Redis
- `anki-atlas-api`
- optional `anki-atlas-worker` for executing queued jobs

## Build the API Image

```bash
docker build -t anki-atlas-api:latest .
```

## Runtime Configuration

Settings are loaded from [config.rs](crates/common/src/config.rs).

### Core variables

| Variable | Default | Notes |
|---|---|---|
| `ANKIATLAS_POSTGRES_URL` | `postgresql://localhost:5432/ankiatlas` | PostgreSQL DSN |
| `ANKIATLAS_QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `ANKIATLAS_REDIS_URL` | `redis://localhost:6379/0` | Redis queue backend |
| `ANKIATLAS_JOB_QUEUE_NAME` | `ankiatlas_jobs` | Queue key |
| `ANKIATLAS_JOB_RESULT_TTL_SECONDS` | `86400` | Job retention |
| `ANKIATLAS_JOB_MAX_RETRIES` | `3` | Retry limit |
| `ANKIATLAS_API_HOST` | `0.0.0.0` | API bind host |
| `ANKIATLAS_API_PORT` | `8000` | API bind port |
| `ANKIATLAS_API_KEY` | unset | Optional API auth |
| `ANKIATLAS_DEBUG` | `false` | Logging verbosity |

### Search and model variables

| Variable | Default | Notes |
|---|---|---|
| `ANKIATLAS_EMBEDDING_PROVIDER` | `openai` | `openai`, `google`, or `mock` |
| `ANKIATLAS_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `ANKIATLAS_EMBEDDING_DIMENSION` | `1536` | Must match provider |
| `OPENAI_API_KEY` | unset | Required for OpenAI embeddings |
| `GEMINI_API_KEY` | unset | Preferred for Google embeddings |
| `GOOGLE_API_KEY` | unset | Backward-compatible fallback for Google embeddings |
| `ANKIATLAS_ANKI_MEDIA_ROOT` | unset | Optional explicit Anki media directory for multimodal indexing |
| `ANKIATLAS_RERANK_ENABLED` | `false` | Enables reranking |
| `ANKIATLAS_RERANK_ENDPOINT` | unset | Required when reranking is enabled |
| `ANKIATLAS_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker label |
| `ANKIATLAS_RERANK_TOP_N` | `50` | Candidate count |
| `ANKIATLAS_RERANK_BATCH_SIZE` | `32` | Batch size |

Gemini Embedding 2 deployment notes:

- Set `ANKIATLAS_EMBEDDING_PROVIDER=google` and `ANKIATLAS_EMBEDDING_MODEL=gemini-embedding-2-preview`.
- `ANKIATLAS_EMBEDDING_DIMENSION` accepts any positive value up to `3072`; `3072`, `1536`, and `768` are the recommended sizes.
- Anki indexing stores one text chunk plus supported local media chunks for images, audio, video, and PDFs referenced from note content.

## Example Compose Stack

This example deploys the API and its backing stores. It does not include the worker image because the root Dockerfile does not currently build it.

```yaml
services:
  api:
    image: anki-atlas-api:latest
    ports:
      - "8000:8000"
    environment:
      ANKIATLAS_POSTGRES_URL: postgresql://ankiatlas:secret@postgres:5432/ankiatlas
      ANKIATLAS_QDRANT_URL: http://qdrant:6333
      ANKIATLAS_REDIS_URL: redis://redis:6379/0
      ANKIATLAS_EMBEDDING_PROVIDER: mock
      ANKIATLAS_EMBEDDING_DIMENSION: "384"
      ANKIATLAS_DEBUG: "false"
    depends_on:
      - postgres
      - qdrant
      - redis

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ankiatlas
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: ankiatlas
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
  qdrant_data:
```

## Running the API

```bash
docker run --rm \
  -p 8000:8000 \
  -e ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:secret@postgres:5432/ankiatlas \
  -e ANKIATLAS_QDRANT_URL=http://qdrant:6333 \
  -e ANKIATLAS_REDIS_URL=redis://redis:6379/0 \
  anki-atlas-api:latest
```

## Running the Worker

There is no published worker container recipe in-repo yet. For now, the supported path is running the worker from a source checkout or from your own custom image:

```bash
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

Treat that worker runtime as development or controlled-use only until the worker contract is fully stabilized.

## Health and Monitoring

### API endpoints

- `GET /health` returns liveness and version
- `GET /ready` returns local process readiness only

Example:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Do not use `/ready` as a deep dependency probe. Monitor PostgreSQL, Qdrant, and Redis separately.

### Dependency checks

```bash
psql "$ANKIATLAS_POSTGRES_URL" -c "SELECT 1"
curl "$ANKIATLAS_QDRANT_URL/healthz"
redis-cli -u "$ANKIATLAS_REDIS_URL" ping
```

## Migrations and Bootstrap

Run migrations before serving production traffic:

```bash
cargo run --bin anki-atlas -- migrate
```

Then bootstrap data via CLI direct execution or by enqueuing jobs:

```bash
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/data/collection.anki2","force_reindex":true}'
```

Explicit indexing runs may recreate the Qdrant collection automatically when the stored embedding model, embedding dimension, or vector schema is incompatible with the current runtime.

## Security Notes

- Set `ANKIATLAS_API_KEY` if the API is reachable outside a trusted network.
- Put the API behind TLS termination.
- Keep PostgreSQL, Qdrant, and Redis on private networks.
- Use provider-specific secrets only for the embedding mode you actually run.

## Operational Caveats

- API write-side work is job-based only.
- API and MCP read-only bootstrap validate the vector collection but do not mutate it. If the collection dimension, stored model, or stored vector schema is incompatible, startup fails with `reindex required`.
- MCP is not covered by the root Dockerfile.
- Worker execution is intentionally gated.
- Reranking silently disables itself if `ANKIATLAS_RERANK_ENABLED=true` but `ANKIATLAS_RERANK_ENDPOINT` is missing; watch logs for that warning.

## Backups

### PostgreSQL

```bash
pg_dump "$ANKIATLAS_POSTGRES_URL" > backup.sql
psql "$ANKIATLAS_POSTGRES_URL" < backup.sql
```

### Qdrant

```bash
curl -X POST "$ANKIATLAS_QDRANT_URL/collections/anki_notes/snapshots"
curl "$ANKIATLAS_QDRANT_URL/collections/anki_notes/snapshots"
```
