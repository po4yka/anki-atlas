# Anki Atlas Architecture

## Mission

Build a self-hostable utility that ingests an existing **Anki collection** (decks, notes, cards, tags, and learning/review metadata), produces a **searchable hybrid index** (semantic + keyword), and exposes **agent-friendly tools** to:

- Search across all cards/notes with metadata filters
- Compute **topic coverage** and highlight **missing topics / weak areas**
- Detect **redundant / near-duplicate** cards

Primary users are coding agents (Claude Code / Codex) and humans via CLI/API.

## Success Criteria

- Given a local `collection.anki2`, the system can ingest and continuously re-sync changes.
- Retrieval works cross-deck with filters (deck, tags, note type, maturity, lapses, etc.).
- Coverage analytics can answer:
  - "How well is topic X covered?"
  - "What subtopics under X are missing or underrepresented?"
  - "What do I have too many of / duplicates?"
- The agent can call documented tools (via MCP or a thin tool API) without direct DB access.

## Stack

| Layer | Technology |
|-------|------------|
| Runtime | Python 3.13+ |
| Canonical Store | PostgreSQL (+ FTS) |
| Vector Search | Qdrant |
| API | FastAPI |
| Agent Interface | MCP server (stdio) or HTTP tool adapter |
| Embeddings | Pluggable: OpenAI or local sentence-transformers/BGE |

## Repository Layout

```
apps/
  api/           # FastAPI app (search + analytics + health)
  cli/           # Typer CLI application
  mcp/           # MCP server exposing tool calls
packages/
  anki/          # Anki extractors (SQLite reader, sync)
  indexer/       # embedding providers, Qdrant upsert, incremental sync
  analytics/     # topic labeling, coverage scoring, duplicate detection
  search/        # hybrid search with RRF fusion
  common/        # config, database, text normalization
infra/
  docker-compose.yml
config/          # example configuration files
docs/            # architecture, API, agent-tool docs
tests/
  fixtures/      # test fixtures for Anki parsing
```

## Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Interfaces                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ FastAPI │  │   CLI   │  │   MCP   │                 │
│  │ (HTTP)  │  │ (typer) │  │ (stdio) │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
└───────┼────────────┼────────────┼──────────────────────┘
        │            │            │
        └────────────┴─────┬──────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Domain Services                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SyncService  │  │SearchService │  │ Analytics    │  │
│  │ - ingest     │  │ - hybrid     │  │ - coverage   │  │
│  │ - diff       │  │ - filters    │  │ - gaps       │  │
│  │ - normalize  │  │ - RRF fusion │  │ - labeling   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │ IndexService │  │ DuplicateDet │                    │
│  │ - embed      │  │ - cluster    │                    │
│  │ - upsert     │  │ - similarity │                    │
│  └──────────────┘  └──────────────┘                    │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Infrastructure                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Database   │  │  QdrantRepo  │  │EmbeddingProv │  │
│  │ - PostgreSQL │  │  - vectors   │  │ - OpenAI     │  │
│  │ - migrations │  │  - payloads  │  │ - Local      │  │
│  │ - FTS        │  │  - ANN       │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐                                      │
│  │  AnkiReader  │                                      │
│  │  (SQLite)    │                                      │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

**Key principles:**
- API is thin routing layer, all logic in domain services
- Embedding provider is explicit, pluggable interface
- Job runner handles async sync/indexing (not blocking HTTP)
- Services are importable modules - CLI and MCP can call them directly

## Data Model (Postgres)

### Canonical Entities

```sql
-- Decks
CREATE TABLE decks (
    deck_id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_name TEXT,
    config JSONB
);

-- Notes (primary unit)
CREATE TABLE notes (
    note_id BIGINT PRIMARY KEY,
    model_id BIGINT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    fields_json JSONB NOT NULL,
    normalized_text TEXT NOT NULL,
    normalized_text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', normalized_text)) STORED,
    mtime BIGINT NOT NULL,
    usn INTEGER NOT NULL,
    deleted_at TIMESTAMPTZ
);

-- Cards (children of notes)
CREATE TABLE cards (
    card_id BIGINT PRIMARY KEY,
    note_id BIGINT NOT NULL REFERENCES notes(note_id),
    deck_id BIGINT NOT NULL REFERENCES decks(deck_id),
    ord INTEGER NOT NULL,
    due INTEGER,
    ivl INTEGER,
    ease INTEGER,
    lapses INTEGER DEFAULT 0,
    reps INTEGER DEFAULT 0,
    queue INTEGER,
    type INTEGER,
    mtime BIGINT NOT NULL,
    usn INTEGER NOT NULL
);

-- Card stats (derived from revlog)
CREATE TABLE card_stats (
    card_id BIGINT PRIMARY KEY REFERENCES cards(card_id),
    reviews INTEGER DEFAULT 0,
    avg_ease REAL,
    fail_rate REAL,
    last_review_at TIMESTAMPTZ,
    total_time_ms BIGINT DEFAULT 0
);

-- Topics (user-defined taxonomy)
CREATE TABLE topics (
    topic_id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,  -- e.g., 'android/compose/state'
    label TEXT NOT NULL,
    description TEXT
);

-- Note-topic assignments
CREATE TABLE note_topics (
    note_id BIGINT REFERENCES notes(note_id),
    topic_id INTEGER REFERENCES topics(topic_id),
    confidence REAL NOT NULL,
    method TEXT NOT NULL,  -- 'embedding', 'llm', 'manual'
    PRIMARY KEY (note_id, topic_id)
);

-- Sync metadata
CREATE TABLE sync_metadata (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL
);
-- Stores: normalization_version, embedding_model, embedding_dim, last_sync_at
```

### Qdrant Payload (per vector point)

```json
{
  "note_id": 1234567890,
  "deck_names": ["Android::Compose", "Android::Kotlin"],
  "tags": ["kotlin", "compose", "state"],
  "model_id": 1234,
  "mature": true,
  "lapses": 2,
  "reps": 15,
  "fail_rate": 0.13
}
```

## Sync Flow

```
┌─────────────────┐
│  Anki Source    │
│ collection.anki2│
└────────┬────────┘
         │ copy to temp (avoid lock)
         ▼
┌─────────────────┐
│  AnkiReader     │
│ parse SQLite    │
└────────┬────────┘
         │ decks, notes, cards, revlog
         ▼
┌─────────────────┐     ┌─────────────────┐
│  SyncService    │────▶│  Postgres       │
│ - diff by mtime │     │ upsert entities │
│ - normalize     │     │ soft-delete old │
│ - track seen_ids│     └─────────────────┘
└────────┬────────┘
         │ changed note_ids
         ▼
┌─────────────────┐     ┌─────────────────┐
│  IndexService   │────▶│  Qdrant         │
│ - embed texts   │     │ upsert vectors  │
│ - skip unchanged│     │ delete removed  │
└─────────────────┘     └─────────────────┘
```

### Sync Design Decisions

| Concern | Approach |
|---------|----------|
| **Deletions** | Track `note_ids` seen in source. After sync, soft-delete (`deleted_at`) notes not seen. Qdrant: delete vectors for removed note_ids. |
| **Revlog aggregation** | Compute on sync: `reviews`, `avg_ease`, `fail_rate`, `last_review_at`, `total_time_ms` per card. Store in `card_stats` table. |
| **SQLite locking** | Copy `collection.anki2` to temp file before reading (Anki locks aggressively). AnkiConnect path avoids this entirely. |
| **Versioning** | Store in `sync_metadata`: `normalization_version`, `embedding_model`, `embedding_dim`. On mismatch, mark all notes for re-processing. |

## Hybrid Search

```
          ┌─────────────────────────────────┐
          │  search(query, filters, top_k)  │
          └───────────────┬─────────────────┘
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
   ┌───────────────┐             ┌───────────────┐
   │ Postgres FTS  │             │ Qdrant ANN    │
   │ ts_rank_cd()  │             │ cosine sim    │
   └───────┬───────┘             └───────┬───────┘
           │ ranked list                 │ ranked list
           └──────────────┬──────────────┘
                          ▼
              ┌───────────────────────┐
              │   RRF Fusion          │
              │   score = Σ 1/(k+rank)│
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Enrich with metadata │
              │  (cards, stats, deck) │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  Response with        │
              │  source attribution   │
              └───────────────────────┘
```

### Search Design Decisions

| Concern | Approach |
|---------|----------|
| **Over-fetch** | Fetch more than `top_k` from each path before fusing (e.g., 3x) to avoid missing good results |
| **Filter alignment** | Ensure filter semantics match across Postgres and Qdrant payloads |
| **Score semantics** | RRF produces a rank-based score, not a calibrated probability. Document this clearly. |

### Filters (applied to both paths)

- `deck` (include/exclude)
- `tags` (include/exclude)
- `model_id` (note type)
- `mature` (ivl > threshold, e.g., 21 days)
- `lapses_gte` / `reps_gte`

### Response Shape

```json
{
  "items": [
    {
      "note_id": 1234567890,
      "normalized_text": "Front: ...\nBack: ...",
      "decks": ["Android::Compose"],
      "tags": ["kotlin", "compose"],
      "scores": {
        "semantic": 0.82,
        "keyword": 0.65,
        "fused": 0.74
      },
      "card_stats": {
        "mature": true,
        "lapses": 2,
        "last_review": "2026-01-15T10:30:00Z"
      }
    }
  ],
  "filters_applied": {...},
  "explanations": {
    "fusion_method": "rrf",
    "k_constant": 60
  }
}
```

## Topic Coverage & Gap Detection

### 3-Tier Approach

1. **Seed taxonomy**: Load from `topics.yml` (user-editable)
2. **Auto-suggest expansion**: Clustering + LLM draft (optional), stored as proposals
3. **Label notes**: Assign topics to notes via embedding similarity with confidence scores

### Coverage Metrics

For a given topic (and optional filters):
- `coverage_count`: notes labeled to the topic/subtree
- `breadth`: number of covered children in subtree
- `maturity_weighted_coverage`: count weighted by maturity (ivl/review outcomes)
- `weak_spots`: notes/cards in topic with high lapse/fail signals

### Gap Detection

- `coverage_count == 0` → missing
- Coverage below threshold or only shallow → undercovered
- Output includes nearest existing notes (semantic similarity to topic description)

### Risk Mitigations (APPROVED)

| Risk | Mitigation |
|------|------------|
| **Taxonomy drift** | Explicit versioning of taxonomy + labeling runs. Store `taxonomy_version` and `labeling_run_id` with each assignment. |
| **False confidence from embeddings** | Threshold calibration + sampling eval. Periodically sample labeled notes and validate. Support multi-label + "unknown" category. |
| **Deck/tag ≠ topic** | Don't assume deck structure maps to topics. Use embeddings as primary signal, deck/tags as secondary features. |

## Duplicate Detection

- Embedding-based near-duplicate detection (cosine similarity > threshold)
- Output clusters with representative note + duplicates
- Batch similarity search via Qdrant ANN

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/sync` | POST | Ingest/resync from Anki source |
| `/search` | POST | Hybrid search with filters |
| `/notes/{note_id}` | GET | Canonical note + cards + stats |
| `/topics` | GET | Taxonomy tree |
| `/topics/{topic_id}/coverage` | GET | Coverage metrics |
| `/topics/{topic_id}/gaps` | GET | Missing/undercovered nodes |
| `/duplicates` | GET | Duplicate clusters |

## MCP Tools

The MCP server exposes the following tools for AI agent integration:

```
ankiatlas_search(query, filters, top_k)
ankiatlas_topic_coverage(topic_path, filters)
ankiatlas_topic_gaps(topic_path, filters, min_coverage)
ankiatlas_duplicates(filters, similarity_threshold)
ankiatlas_sync(source)
```

See [MCP_TOOLS.md](MCP_TOOLS.md) for detailed tool documentation, parameters, and example agent prompts.

## CLI

```bash
# Sync collection and index
anki-atlas sync --source /path/to/collection.anki2

# Search with filters
anki-atlas search "compose recomposition" --deck Android --tag kotlin --top 20

# Manage topic taxonomy
anki-atlas topics --file topics.yml --label

# View topic coverage
anki-atlas coverage android/compose/state

# Find coverage gaps
anki-atlas gaps android --min-coverage 5

# Find duplicates
anki-atlas duplicates --threshold 0.92

# Run MCP server
anki-atlas-mcp
```

## Implementation Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| 0 - Bootstrap | Repo setup, tooling, docker-compose, FastAPI `/health` | Complete |
| 1 - Anki Extractor | SQLite reader, normalization, Postgres schema, incremental sync | Complete |
| 2 - Indexing | Embedding provider, Qdrant upsert, version tracking | Complete |
| 3 - Hybrid Search | FTS + ANN + RRF fusion, `/search` endpoint | Complete |
| 4 - Topics + Coverage | Taxonomy loader, labeling, coverage/gaps endpoints | Complete |
| 5 - Duplicates | Duplicate finder, `/duplicates` endpoint | Complete |
| 6 - Agent Tools | MCP server, tool documentation | Complete |

## Text Normalization

- Strip HTML tags (keep code blocks; preserve inline code meaning)
- Normalize whitespace
- Keep `raw_fields` copy for debugging
- Produce `normalized_text` with deterministic template:
  ```
  Front: ...
  Back: ...
  Extra: ...
  Tags: tag1, tag2
  Decks: Deck::Subdeck
  ```

## Configuration

Configuration via environment variables (with `ANKIATLAS_` prefix) or `.env` file:

```bash
# Database
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas

# Vector store
ANKIATLAS_QDRANT_URL=http://localhost:6333

# Embedding configuration
ANKIATLAS_EMBEDDING_PROVIDER=openai  # or 'local'
ANKIATLAS_EMBEDDING_MODEL=text-embedding-3-small
ANKIATLAS_EMBEDDING_DIMENSION=1536

# OpenAI API key (required if using openai provider)
OPENAI_API_KEY=sk-...

# Anki source (optional, can be passed via CLI)
ANKIATLAS_ANKI_COLLECTION_PATH=/path/to/collection.anki2

# API configuration
ANKIATLAS_API_HOST=0.0.0.0
ANKIATLAS_API_PORT=8000
ANKIATLAS_DEBUG=false
```
