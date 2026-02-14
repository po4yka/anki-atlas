# Anki Atlas

Searchable hybrid index for Anki collections with agent-friendly tools.

## Features

- **Hybrid Search**: Combines semantic (vector) and keyword (FTS) search with RRF fusion
- **CrossEncoder Reranking**: Optional second-stage reranking of top hybrid candidates
- **Typo-tolerant Lexical Search**: `pg_trgm` fuzzy matching with autocomplete fallback and suggestions
- **Topic Coverage**: Analyze what topics your cards cover and identify gaps
- **Duplicate Detection**: Find near-duplicate cards using embedding similarity
- **Agent Tools**: MCP server for integration with AI agents (Claude Code, Claude Desktop)

## Quick Start

### Prerequisites

- Python 3.13+
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone and enter directory
git clone https://github.com/po4yka/anki-atlas.git
cd anki-atlas

# Start dependencies (PostgreSQL + Qdrant + Redis)
make up

# Install Python dependencies
make install

# Run database migrations
anki-atlas migrate

# Sync your Anki collection
anki-atlas sync --source /path/to/collection.anki2

# Run the API server
make dev

# In another terminal, run background worker
make worker
```

The API will be available at http://localhost:8000

### Verify Installation

```bash
curl http://localhost:8000/health
```

## Configuration

Copy `config/env.example` to `.env` and adjust as needed:

```bash
cp config/env.example .env
```

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANKIATLAS_POSTGRES_URL` | PostgreSQL connection URL | `postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas` |
| `ANKIATLAS_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `ANKIATLAS_REDIS_URL` | Redis URL for async jobs | `redis://localhost:6379/0` |
| `ANKIATLAS_EMBEDDING_PROVIDER` | `openai` or `local` | `openai` |
| `ANKIATLAS_EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `ANKIATLAS_RERANK_ENABLED` | Enable CrossEncoder reranking | `false` |
| `ANKIATLAS_RERANK_MODEL` | CrossEncoder model name | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `ANKIATLAS_RERANK_TOP_N` | Candidates reranked per query | `50` |
| `ANKIATLAS_ANKI_COLLECTION_PATH` | Path to collection.anki2 | - |
| `OPENAI_API_KEY` | OpenAI API key (if using openai provider) | - |

Reranking requires `sentence-transformers` (install with `uv sync --extra embeddings-local`).

## CLI Usage

```bash
# Sync your Anki collection (with indexing)
anki-atlas sync --source /path/to/collection.anki2

# Sync without indexing
anki-atlas sync --source /path/to/collection.anki2 --no-index

# Index notes to vector database
anki-atlas index

# Search cards
anki-atlas search "compose recomposition" --deck Android --top 20

# Load and view topic taxonomy
anki-atlas topics --file topics.yml

# Label notes with topics
anki-atlas topics --file topics.yml --label

# Check topic coverage
anki-atlas coverage programming/python

# Find gaps in coverage
anki-atlas gaps programming --min-coverage 5

# Detect duplicates
anki-atlas duplicates --threshold 0.92

# Show version
anki-atlas version
```

## MCP Agent Tools

Anki Atlas provides an MCP (Model Context Protocol) server for AI agent integration:

```bash
# Run the MCP server
anki-atlas-mcp
```

Available tools:
- `ankiatlas_search` - Hybrid semantic + keyword search
- `ankiatlas_topic_coverage` - Topic coverage metrics
- `ankiatlas_topic_gaps` - Find knowledge gaps
- `ankiatlas_duplicates` - Near-duplicate detection
- `ankiatlas_sync` - Sync Anki collection

## Async Jobs API

Long-running operations can be queued and tracked asynchronously:

```bash
# Enqueue a sync job
curl -X POST http://localhost:8000/jobs/sync \
  -H "Content-Type: application/json" \
  -d '{"source":"/path/to/collection.anki2","run_migrations":true,"index":true}'

# Enqueue index-only job
curl -X POST http://localhost:8000/jobs/index \
  -H "Content-Type: application/json" \
  -d '{"force_reindex":false}'

# Poll job status/progress
curl http://localhost:8000/jobs/<job_id>

# Cancel job
curl -X POST http://localhost:8000/jobs/<job_id>/cancel
```

See [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md) for detailed documentation and example prompts.

## Development

```bash
# Run linter
make lint

# Format code
make format

# Run type checker
make typecheck

# Run tests
make test

# Run all checks
make check
```

## Project Structure

```
apps/
  api/           # FastAPI application
  cli/           # Typer CLI application
  mcp/           # MCP server for AI agents
packages/
  anki/          # Anki collection reader and sync
  indexer/       # Embedding and vector indexing
  analytics/     # Topic coverage, gaps, duplicates
  search/        # Hybrid search with RRF fusion
  common/        # Shared config and database
tests/           # Test suite
docs/            # Documentation
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

## License

MIT
