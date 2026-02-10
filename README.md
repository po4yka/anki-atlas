# Anki Atlas

Searchable hybrid index for Anki collections with agent-friendly tools.

## Features

- **Hybrid Search**: Combines semantic (vector) and keyword (FTS) search with RRF fusion
- **Topic Coverage**: Analyze what topics your cards cover and identify gaps
- **Duplicate Detection**: Find near-duplicate cards using embedding similarity
- **Agent Tools**: MCP server for integration with coding agents (Claude Code, etc.)

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

# Start dependencies
make up

# Install Python dependencies
make install

# Run the API server
make dev
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
| `ANKIATLAS_EMBEDDING_PROVIDER` | `openai` or `local` | `openai` |
| `ANKIATLAS_ANKI_COLLECTION_PATH` | Path to collection.anki2 | - |

## CLI Usage

```bash
# Sync your Anki collection
anki-atlas sync --source /path/to/collection.anki2

# Search cards
anki-atlas search "compose recomposition" --deck Android --top 20

# Check topic coverage
anki-atlas coverage android/compose/state

# Find gaps in coverage
anki-atlas gaps android --min-coverage 5

# Detect duplicates
anki-atlas duplicates --threshold 0.92
```

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

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

## License

MIT
