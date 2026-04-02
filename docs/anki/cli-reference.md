# CLI Reference

## Sync & Index

```bash
# Sync Anki collection to PostgreSQL + vector index
cargo run --bin anki-atlas -- sync ~/path/to/collection.anki2

# Sync without indexing
cargo run --bin anki-atlas -- sync ~/path/to/collection.anki2 --no-index

# Sync without running migrations
cargo run --bin anki-atlas -- sync ~/path/to/collection.anki2 --no-migrate

# Force re-embed all notes
cargo run --bin anki-atlas -- sync ~/path/to/collection.anki2 --force-reindex

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

# Filter by deck or tag (repeatable)
cargo run --bin anki-atlas -- search "query" --deck "Kotlin" --tag "kotlin::coroutines"

# Semantic-only or FTS-only
cargo run --bin anki-atlas -- search "query" --semantic
cargo run --bin anki-atlas -- search "query" --fts

# Limit results (default 10)
cargo run --bin anki-atlas -- search "query" -n 5
cargo run --bin anki-atlas -- search "query" --limit 20

# Search RAG chunks instead of notes
cargo run --bin anki-atlas -- search "query" --chunks

# Verbose (show detailed scores)
cargo run --bin anki-atlas -- search "query" --verbose
```

## Topics

Topics is a subcommand group with three sub-commands:

```bash
# Show topic taxonomy tree
cargo run --bin anki-atlas -- topics tree
cargo run --bin anki-atlas -- topics tree --root-path "rust"

# Load taxonomy from file
cargo run --bin anki-atlas -- topics load --file topics.yml

# Label notes with topics
cargo run --bin anki-atlas -- topics label
cargo run --bin anki-atlas -- topics label --file topics.yml --min-confidence 0.6
```

## Coverage & Gaps

```bash
# Show coverage for a topic (includes subtree by default)
cargo run --bin anki-atlas -- coverage "programming/python"
cargo run --bin anki-atlas -- coverage "rust" --no-subtree

# Detect coverage gaps
cargo run --bin anki-atlas -- gaps "programming" --min-coverage 1
```

## Weak Notes

```bash
# Find low-retention notes for a topic
cargo run --bin anki-atlas -- weak-notes "kotlin" -n 20
cargo run --bin anki-atlas -- weak-notes "android" --limit 10
```

## Duplicates

```bash
# Find near-duplicate notes (default threshold 0.92, max 50)
cargo run --bin anki-atlas -- duplicates

# Custom threshold and limit
cargo run --bin anki-atlas -- duplicates --threshold 0.95 --max 100

# Filter by deck or tag
cargo run --bin anki-atlas -- duplicates --deck "Kotlin" --tag "kotlin::coroutines" --verbose
```

## Card Generation & Validation

```bash
# Parse Obsidian note and preview card generation
cargo run --bin anki-atlas -- generate path/to/note.md
cargo run --bin anki-atlas -- generate path/to/note.md --dry-run

# Validate card content
cargo run --bin anki-atlas -- validate card.md
cargo run --bin anki-atlas -- validate card.md --quality
```

## Obsidian Sync

```bash
# Scan and sync Obsidian vault
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run

# Limit to specific source directories (comma-delimited)
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --source-dirs "notes,references"
```

## Tag Audit

```bash
# Audit tags for convention violations
cargo run --bin anki-atlas -- tag-audit tags.txt

# Auto-fix violations
cargo run --bin anki-atlas -- tag-audit tags.txt --fix
```

## Card Quality Loop

The `cardloop` subcommand group provides a persistent queue-driven workflow for
systematic card improvement with score tracking and an audit trail.

```bash
# Scan cards and populate work queue (basic audit + generation issues)
cargo run --bin anki-atlas -- cardloop scan --registry path/to/registry.db

# Also pull FSRS retention signals from Anki collection
cargo run --bin anki-atlas -- cardloop scan --registry path/to/registry.db --anki-collection path/to/collection.anki2

# Also run LLM batch review
cargo run --bin anki-atlas -- cardloop scan --registry path/to/registry.db --llm-review

# Also detect semantic duplicates (requires Qdrant + PostgreSQL)
cargo run --bin anki-atlas -- cardloop scan --registry path/to/registry.db --detect-duplicates
cargo run --bin anki-atlas -- cardloop scan --registry path/to/registry.db --detect-duplicates --dup-threshold 0.85

# Show score dashboard (open_count, overall_score, strict_score)
cargo run --bin anki-atlas -- cardloop status
cargo run --bin anki-atlas -- cardloop status --json

# Get next work item(s)
cargo run --bin anki-atlas -- cardloop next
cargo run --bin anki-atlas -- cardloop next -n 5
cargo run --bin anki-atlas -- cardloop next --loop-kind audit
cargo run --bin anki-atlas -- cardloop next --loop-kind generation
cargo run --bin anki-atlas -- cardloop next --cluster <cluster-id>

# Resolve an item (--registry triggers verification re-scan)
cargo run --bin anki-atlas -- cardloop resolve ITEM_ID --status fixed --attest "Rewrote question to test reasoning"
cargo run --bin anki-atlas -- cardloop resolve ITEM_ID --status fixed --attest "Fixed tags" --registry path/to/registry.db
cargo run --bin anki-atlas -- cardloop resolve ITEM_ID --status skipped --attest "Not actionable"
cargo run --bin anki-atlas -- cardloop resolve ITEM_ID --status wontfix --attest "Intentional design choice"

# Show recent resolution history
cargo run --bin anki-atlas -- cardloop log
cargo run --bin anki-atlas -- cardloop log -n 20
```

## Other

```bash
# TUI operator console
cargo run --bin anki-atlas -- tui

# Show version
cargo run --bin anki-atlas -- version
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
