# CLI Reference

## Sync & Index

```bash
# Sync Anki collection to PostgreSQL + vector index
uv run anki-atlas sync --source ~/path/to/collection.anki2

# Sync without indexing
uv run anki-atlas sync --source ~/path/to/collection.anki2 --no-index

# Force re-embed all notes
uv run anki-atlas sync --source ~/path/to/collection.anki2 --force-reindex

# Run database migrations only
uv run anki-atlas migrate

# Index notes to vector database
uv run anki-atlas index

# Force re-index all
uv run anki-atlas index --force
```

## Search

```bash
# Hybrid search (semantic + full-text)
uv run anki-atlas search "query text"

# Filter by deck or tag
uv run anki-atlas search "query" --deck "Kotlin" --tag "kotlin::coroutines"

# Semantic-only or FTS-only
uv run anki-atlas search "query" --semantic
uv run anki-atlas search "query" --fts

# Verbose (show detailed scores)
uv run anki-atlas search "query" --verbose
```

## Topics & Coverage

```bash
# Show topic taxonomy
uv run anki-atlas topics

# Load taxonomy from file
uv run anki-atlas topics --file topics.yml

# Label notes with topics
uv run anki-atlas topics --label --min-confidence 0.3

# Show coverage for a topic
uv run anki-atlas coverage "programming/python"

# Detect coverage gaps
uv run anki-atlas gaps "programming" --min-coverage 1
```

## Duplicates

```bash
# Find near-duplicate notes
uv run anki-atlas duplicates --threshold 0.92

# Filter by deck
uv run anki-atlas duplicates --deck "Kotlin" --verbose
```

## Card Generation & Validation

```bash
# Parse Obsidian note and preview card generation
uv run anki-atlas generate path/to/note.md
uv run anki-atlas generate path/to/note.md --dry-run

# Validate card content
uv run anki-atlas validate card.md
uv run anki-atlas validate card.md --quality

# Scan Obsidian vault
uv run anki-atlas obsidian-sync /path/to/vault
uv run anki-atlas obsidian-sync /path/to/vault --dry-run

# Audit tags
uv run anki-atlas tag-audit tags.txt
uv run anki-atlas tag-audit tags.txt --fix
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
