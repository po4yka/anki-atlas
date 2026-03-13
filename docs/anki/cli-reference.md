# CLI Reference

## Sync & Index

```bash
# Sync Anki collection to PostgreSQL + vector index
cargo run --bin anki-atlas -- sync --source ~/path/to/collection.anki2

# Sync without indexing
cargo run --bin anki-atlas -- sync --source ~/path/to/collection.anki2 --no-index

# Force re-embed all notes
cargo run --bin anki-atlas -- sync --source ~/path/to/collection.anki2 --force-reindex

# Run database migrations only
cargo run --bin anki-atlas -- migrate

# Index notes to vector database
cargo run --bin anki-atlas -- index

# Force re-index all
cargo run --bin anki-atlas -- index --force
```

## Search

```bash
# Hybrid search (semantic + full-text)
cargo run --bin anki-atlas -- search "query text"

# Filter by deck or tag
cargo run --bin anki-atlas -- search "query" --deck "Kotlin" --tag "kotlin::coroutines"

# Semantic-only or FTS-only
cargo run --bin anki-atlas -- search "query" --semantic
cargo run --bin anki-atlas -- search "query" --fts

# Verbose (show detailed scores)
cargo run --bin anki-atlas -- search "query" --verbose
```

## Topics & Coverage

```bash
# Show topic taxonomy
cargo run --bin anki-atlas -- topics

# Load taxonomy from file
cargo run --bin anki-atlas -- topics --file topics.yml

# Label notes with topics
cargo run --bin anki-atlas -- topics --label --min-confidence 0.3

# Show coverage for a topic
cargo run --bin anki-atlas -- coverage "programming/python"

# Detect coverage gaps
cargo run --bin anki-atlas -- gaps "programming" --min-coverage 1
```

## Duplicates

```bash
# Find near-duplicate notes
cargo run --bin anki-atlas -- duplicates --threshold 0.92

# Filter by deck
cargo run --bin anki-atlas -- duplicates --deck "Kotlin" --verbose
```

## Card Generation & Validation

```bash
# Parse Obsidian note and preview card generation
cargo run --bin anki-atlas -- generate path/to/note.md
cargo run --bin anki-atlas -- generate path/to/note.md --dry-run

# Validate card content
cargo run --bin anki-atlas -- validate card.md
cargo run --bin anki-atlas -- validate card.md --quality

# Scan Obsidian vault
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run

# Audit tags
cargo run --bin anki-atlas -- tag-audit tags.txt
cargo run --bin anki-atlas -- tag-audit tags.txt --fix
```

## MCP Tools

Available as `ankiatlas_*` when using the MCP server:

| Tool | Description |
|------|-------------|
| `ankiatlas_search` | Hybrid search across indexed notes |
| `ankiatlas_topic_coverage` | Coverage metrics for a topic |
| `ankiatlas_topic_gaps` | Find gaps in topic coverage |
| `ankiatlas_duplicates` | Find near-duplicate notes |
| `ankiatlas_sync` | Sync collection to index |
| `ankiatlas_generate` | Parse text and preview card generation |
| `ankiatlas_validate` | Validate card front/back/tags |
| `ankiatlas_obsidian_sync` | Discover and sync Obsidian vault notes |
| `ankiatlas_tag_audit` | Audit tags for convention violations |
