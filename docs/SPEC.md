# Anki Atlas - Original Specification

This is the original specification provided at project inception.

## Mission

Build a self-hostable utility that ingests an existing **Anki collection** (decks, notes, cards, tags, and learning/review metadata), produces a **searchable hybrid index** (semantic + keyword), and exposes **agent-friendly tools** to:

* search across all cards/notes with metadata filters
* compute **topic coverage** and highlight **missing topics / weak areas**
* detect **redundant / near-duplicate** cards

Primary users are coding agents (Claude Code / Codex) and humans via CLI/API.

## Success criteria

* Given a local `collection.anki2`, the system can ingest and continuously re-sync changes.
* Retrieval works cross-deck with filters (deck, tags, note type, maturity, lapses, etc.).
* Coverage analytics can answer:

  * "How well is topic X covered?"
  * "What subtopics under X are missing or underrepresented?"
  * "What do I have too many of / duplicates?"
* The agent can call documented tools (via MCP or a thin tool API) without direct DB access.

## Recommended stack (MVP -> v1)

### Runtime & languages

* **Python 3.13+** for ingestion, embeddings, analytics, API, and agent tooling.

### Storage

* **PostgreSQL**: canonical store + analytics + keyword search (FTS).
* **Qdrant**: dense vector search (and optional sparse later).

### API & tooling

* **FastAPI**: HTTP API for human/agent access.
* **MCP server wrapper** (stdio) or HTTP-tool adapter: exposes "tools" to coding agents.

### Embeddings

* Pluggable embedding provider:

  * hosted: OpenAI embeddings
  * local: sentence-transformers / BGE-class model

## Repository layout (target)

* `apps/api/` - FastAPI app (search + analytics + health)
* `apps/mcp/` - MCP server exposing tool calls (or tool adapter)
* `packages/anki/` - Anki extractors (SQLite + optional AnkiConnect)
* `packages/indexer/` - embedding, qdrant upsert, incremental sync
* `packages/analytics/` - topic labeling, coverage scoring, dedupe
* `packages/common/` - config, logging, text normalization
* `infra/docker-compose.yml` - postgres + qdrant + api
* `scripts/` - one-shot utilities (migrate, reset, backfill)
* `docs/` - architecture, API, agent-tool docs

## Data model

### Canonical entities (Postgres)

Store **notes as the primary unit**, with cards as children.

* `decks`
  * `deck_id`, `name`, `parent_name`, `json`

* `notes`
  * `note_id`, `model_id`, `tags[]`, `fields_json`, `normalized_text`, `mtime`, `usn`

* `cards`
  * `card_id`, `note_id`, `deck_id`, `ord`, `due`, `ivl`, `ease`, `lapses`, `reps`, `queue`, `type`, `mtime`, `usn`

* `revlog_agg` (derived)
  * `card_id`, `reviews`, `avg_grade`, `last_review_at`, `fail_rate`, `time_spent_ms`

* `topics`
  * `topic_id`, `path` (e.g., `android/compose/state`), `label`, `description`

* `note_topics`
  * `note_id`, `topic_id`, `confidence`, `method` (embedding/llm/manual)

### Search fields

* Postgres FTS: `notes.normalized_text_tsv` generated from `normalized_text`.
* Qdrant: per-note vector embedding of `normalized_text`.

### Qdrant payload (per vector point)

* `note_id`
* `deck_names[]`
* `tags[]`
* `model_id`
* key learning stats (optional): `mature`, `lapses`, `reps`, `fail_rate`

## Ingestion requirements (Anki)

### Sources

* MVP: direct read of `collection.anki2` (SQLite)
* Optional: AnkiConnect for "live" sync when Anki is open

### Extraction must include

* Deck structure and names
* Notes (fields, tags, note type/model)
* Cards (per note) with scheduling metadata
* Review log aggregates (per card) to support "weak areas" metrics

### Text normalization rules

* Strip HTML tags (keep code blocks if present; preserve inline code meaning)
* Normalize whitespace
* Optionally keep a `raw_fields` copy for debugging
* Produce `normalized_text` with a deterministic template, e.g.:
  * `Front: ...\nBack: ...\nExtra: ...\nTags: ...\nDecks: ...`

## Retrieval requirements

### MVP hybrid (recommended)

* **Semantic**: Qdrant dense vector search
* **Keyword**: Postgres full-text search (FTS)
* **Fusion**: implement Reciprocal Rank Fusion (RRF) in the API:
  * run both searches, fuse ranked lists, return merged results with per-source scores

### Filters

All endpoints must support:

* deck include/exclude
* tag include/exclude
* note type/model
* maturity (e.g., ivl threshold)
* lapses/reps thresholds

## Analytics requirements

### 1) Topic inventory

Implement 3-tier approach:

1. **Seed taxonomy**: load from `topics.yml` (user-editable).
2. **Auto-suggest taxonomy expansion**: clustering + LLM draft (optional), but store as proposals.
3. **Label notes**: assign topics to notes with confidence.

### 2) Coverage scoring

For a given topic (and optional filters):

* `coverage_count`: notes labeled to the topic/subtree
* `breadth`: number of covered children in subtree
* `maturity_weighted_coverage`: count weighted by maturity (ivl/review outcomes)
* `weak_spots`: notes/cards in topic with high lapse/fail signals

### 3) Gap detection

For each topic node:

* if `coverage_count == 0` -> missing
* if coverage below threshold or only shallow coverage -> undercovered

Output must include:

* missing/undercovered nodes
* nearest existing notes (top-k) that are semantically closest to that topic label/description (to validate true gap)
* suggested prompts/questions to create (optional)

### 4) Duplicate detection

* embedding-based near-duplicate detection (cosine similarity > threshold)
* output clusters with representative note + duplicates

## API design (FastAPI)

### Core endpoints

* `GET /health`
* `POST /sync` - ingest/resync from Anki source
* `POST /search` - hybrid search with filters
* `GET /notes/{note_id}` - canonical + cards + stats
* `GET /topics` - taxonomy tree
* `GET /topics/{topic_id}/coverage` - coverage metrics
* `GET /topics/{topic_id}/gaps` - missing/undercovered nodes
* `GET /duplicates` - duplicate clusters (filterable)

### Response shapes

Prefer stable, agent-friendly JSON with:

* `items[]` results
* `explanations` (which signals contributed: semantic vs keyword)
* `filters_applied`

## MCP tool interface (agent-facing)

Expose tools (names are important; keep them stable):

* `ankiatlas_search(query, filters, top_k)`
* `ankiatlas_topic_coverage(topic_path, filters)`
* `ankiatlas_topic_gaps(topic_path, filters, min_coverage)`
* `ankiatlas_duplicates(filters, similarity_threshold)`
* `ankiatlas_sync(source)`

If MCP SDK is available, use it. Otherwise implement a minimal stdio JSON-RPC adapter that maps tool calls to HTTP endpoints.

## Configuration

Support `.env` + `config.yml`:

* anki source path (or AnkiConnect URL)
* postgres url
* qdrant url
* embedding provider (openai/local)
* embedding model name
* vector dimension
* topic taxonomy path

Do not hardcode secrets; read from env.

## CLI

Provide a CLI for humans and CI usage:

* `anki-atlas sync --source /path/to/collection.anki2`
* `anki-atlas search "compose recomposition" --deck Android --tag kotlin --top 20`
* `anki-atlas coverage android/compose/state`
* `anki-atlas gaps android --min-coverage 5`
* `anki-atlas duplicates --threshold 0.92`

## Infra (self-host)

* `docker-compose.yml` for:
  * postgres
  * qdrant
  * api

* make targets:
  * `make dev` (run local)
  * `make up` (compose)
  * `make test`

## Testing requirements

* Unit tests for:
  * Anki parsing and normalization
  * RRF fusion correctness
  * filtering logic
  * coverage scoring

* Integration tests with:
  * a tiny fixture `collection.anki2` (or generated sqlite) committed under `tests/fixtures/`
  * qdrant + postgres via docker in CI

## Step-by-step implementation plan (what to build, in order)

### Milestone 0 - Bootstrap

* Initialize repo, tooling (lint/format/typecheck), docker-compose, CI pipeline.
* Create minimal FastAPI app with `/health`.

### Milestone 1 - Anki extractor

* Implement SQLite reader for `collection.anki2`.
* Extract decks, notes, cards.
* Build robust HTML -> text normalization.
* Write to Postgres canonical schema.
* Implement incremental sync using `mtime/usn` where available.

### Milestone 2 - Indexing

* Implement embedding provider interface:
  * `embed(texts[]) -> vectors[]`
* Upsert vectors to Qdrant with payload.
* Store vector version/hash to skip re-embedding unchanged notes.

### Milestone 3 - Hybrid search

* Implement Postgres FTS query.
* Implement Qdrant semantic query.
* Implement RRF fusion.
* Add `/search` endpoint + CLI command.

### Milestone 4 - Topics + coverage

* Implement `topics.yml` loader.
* Implement topic labeling:
  * embedding similarity between topic descriptions and notes
  * store `note_topics` with confidence
* Implement `/topics/*` endpoints for coverage and gaps.

### Milestone 5 - Duplicates

* Implement duplicate finder (batch similarity search / ANN + threshold).
* Expose `/duplicates` endpoint.

### Milestone 6 - Agent tools

* Implement MCP server (or tool adapter) exposing stable tools.
* Document example agent prompts for using tools.

## Documentation to include

* `README.md` with quickstart, docker setup, and example queries.
* `docs/ARCHITECTURE.md` with data flow diagram (text) and key tradeoffs.
* `docs/AGENT_GUIDE.md` describing tools + best prompting patterns.

## Definition of done

* `docker compose up` starts everything.
* `anki-atlas sync` ingests a real collection.
* Search returns relevant results with filters.
* Coverage/gaps and duplicates endpoints work.
* Agent tools callable (MCP or adapter) with examples.
* Tests green.

## Notes / guardrails

* Keep raw Anki data immutable; store derived fields separately.
* Prefer note-level embeddings; card-level only if needed.
* Make every step incremental/idempotent.
* Log enough to debug, but avoid dumping sensitive content by default.
