# First Time Setup Guide

Step-by-step guide to set up Anki Atlas from scratch.

## Prerequisites

Before starting, ensure you have:

- Python 3.11 or later
- Docker and Docker Compose (for databases)
- An Anki collection with flashcards
- OpenAI API key (for embeddings)

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/anki-atlas.git
cd anki-atlas

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

**Verify:** Check installation works:
```bash
uv run anki-atlas version
# Should print: anki-atlas 0.1.0
```

## Step 2: Start Databases

Create a `docker-compose.yml` file in the project root:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=ankiatlas
      - POSTGRES_PASSWORD=ankiatlas
      - POSTGRES_DB=ankiatlas
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ankiatlas"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  qdrant_data:
```

Start the databases:
```bash
docker compose up -d
```

**Verify:** Check databases are running:
```bash
docker compose ps
# Both should show "Up"

# Test PostgreSQL connection
psql postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas -c "SELECT 1"

# Test Qdrant
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant - vector search engine","version":"..."}
```

## Step 3: Configure Environment

Create a `.env` file:

```bash
# Database
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas

# Vector store
ANKIATLAS_QDRANT_URL=http://localhost:6333

# Embeddings (get your key from https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Debug mode
ANKIATLAS_DEBUG=true
```

**Verify:** Check environment is loaded:
```bash
source .env  # or use direnv
echo $ANKIATLAS_POSTGRES_URL
```

## Step 4: Initialize Database

Run migrations to create the database schema:

```bash
uv run anki-atlas migrate
```

**Verify:** Check tables were created:
```bash
psql $ANKIATLAS_POSTGRES_URL -c "\dt"
# Should list: cards, card_stats, decks, models, notes, migrations
```

## Step 5: Find Your Anki Collection

Locate your Anki collection file:

```bash
# macOS
ls ~/Library/Application\ Support/Anki2/*/collection.anki2

# Linux
ls ~/.local/share/Anki2/*/collection.anki2

# Windows (PowerShell)
ls "$env:APPDATA\Anki2\*\collection.anki2"
```

**Important:** Close Anki before proceeding - it locks the database.

## Step 6: Sync Your Collection

Sync your Anki collection to the database and create embeddings:

```bash
uv run anki-atlas sync --source /path/to/collection.anki2
```

This will:
1. Read all decks, models, notes, and cards from Anki
2. Store them in PostgreSQL
3. Generate embeddings for semantic search
4. Store embeddings in Qdrant

**Verify:** Check sync completed:
```bash
# Check notes in database
psql $ANKIATLAS_POSTGRES_URL -c "SELECT COUNT(*) FROM notes"

# Check vectors in Qdrant
curl http://localhost:6333/collections/anki_notes | jq .result.points_count
```

## Step 7: Test Search

Try a search:

```bash
uv run anki-atlas search "your query here"
```

**Verify:** Results should show:
- Note IDs
- Relevance scores
- Preview text
- Source (semantic, FTS, or both)

## Step 8: Start the API (Optional)

If you want to use the REST API:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

**Verify:** Check API is running:
```bash
# Health check
curl http://localhost:8000/health

# Search via API
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "top_k": 10}'
```

## Step 9: Set Up MCP Server (Optional)

For AI assistant integration, configure your MCP client:

```json
{
  "mcpServers": {
    "anki-atlas": {
      "command": "uv",
      "args": ["run", "python", "-m", "apps.mcp.server"],
      "cwd": "/path/to/anki-atlas"
    }
  }
}
```

## Next Steps

- **Regular sync:** Run `anki-atlas sync` after updating your Anki collection
- **Topics taxonomy:** Set up topic labeling with `anki-atlas topics`
- **Find duplicates:** Use `anki-atlas duplicates` to clean up your collection
- **API integration:** See [MCP_TOOLS.md](./MCP_TOOLS.md) for AI agent integration

## Common Issues

### "Cannot connect to PostgreSQL"

```bash
# Check Docker is running
docker compose ps

# Restart if needed
docker compose restart postgres
```

### "OpenAI API error"

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### "Collection dimension mismatch"

If you changed embedding models:
```bash
uv run anki-atlas sync --source /path/to/collection.anki2 --force-reindex
```

For more troubleshooting, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
