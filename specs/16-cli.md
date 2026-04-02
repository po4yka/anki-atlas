# Spec: crate `anki-atlas-cli`

## Source Reference

Current Rust source:

- [main.rs](bins/cli/src/main.rs)
- [args.rs](bins/cli/src/args.rs)
- [commands/mod.rs](bins/cli/src/commands/mod.rs)
- [output.rs](bins/cli/src/output.rs)

## Purpose

Provide the operator-facing command surface for Anki Atlas. The CLI is the only public surface that may execute sync and index directly. It also exposes search, analytics, taxonomy operations, and local preview workflows.

## Dependencies

```toml
[dependencies]
common = { path = "../../crates/common" }
database = { path = "../../crates/database" }
surface-contracts = { path = "../../crates/surface-contracts" }
surface-runtime = { path = "../../crates/surface-runtime" }

anyhow.workspace = true
clap.workspace = true
crossterm = "0.28"
ratatui = "0.29"
serde_json.workspace = true
tokio.workspace = true
```

## Public API

### Binary

- binary name: `anki-atlas`

### Command Set

```text
anki-atlas version
anki-atlas migrate
anki-atlas tui
anki-atlas sync <source> [--no-migrate] [--no-index] [--force-reindex]
anki-atlas index [--force]
anki-atlas search <query> [--deck <name>]... [--tag <tag>]... [-n <limit>] [--chunks] [--semantic] [--fts] [--verbose]
anki-atlas topics tree [--root-path <path>]
anki-atlas topics load --file <path>
anki-atlas topics label [--file <path>] [--min-confidence <float>]
anki-atlas coverage <topic> [--no-subtree]
anki-atlas gaps <topic> [--min-coverage <n>]
anki-atlas weak-notes <topic> [-n <limit>]
anki-atlas duplicates [--threshold <float>] [--max <n>] [--deck <name>]... [--tag <tag>]... [--verbose]
anki-atlas generate <file> [--dry-run]
anki-atlas validate <file> [--quality]
anki-atlas obsidian-sync <vault> [--source-dirs a,b,c] [--dry-run]
anki-atlas tag-audit <file> [--fix]
```

### Command Semantics

- `sync`
  - may run migrations first
  - runs direct sync execution
  - may run direct indexing afterward
- `tui`
  - boots the shared local runtime in a full-screen terminal UI
  - exposes search, analytics, taxonomy views, and workflow execution
  - renders runtime progress for sync, index, and obsidian preview workflows
- `index`
  - runs direct indexing over PostgreSQL notes
- `search`
  - default mode maps onto the shared contract `SearchRequest`
  - `--chunks` maps onto the shared contract `ChunkSearchRequest`
  - `--chunks` is semantic-only and rejects `--fts`
- `topics tree`
  - prints taxonomy tree data
- `topics load`
  - loads taxonomy YAML into PostgreSQL
- `topics label`
  - labels notes against the taxonomy
- `coverage`, `gaps`, `weak-notes`, `duplicates`
  - call shared analytics facades
- `generate`
  - previews parsed-note generation only
- `validate`
  - runs `ValidationPipeline` and optional quality scoring
- `obsidian-sync`
  - scans a vault and previews work
  - currently requires dry-run behavior
- `tag-audit`
  - validates tags, applies normalization when requested, and prints suggestions

## Runtime Wiring

The CLI uses [surface-runtime](crates/surface-runtime/src/services.rs) with direct execution enabled, and consumes shared DTOs from [surface-contracts](crates/surface-contracts/src/lib.rs).

That shared runtime provides:

- PostgreSQL pool
- embedding provider
- Qdrant-backed vector repository
- optional reranker
- search facade
- analytics facade
- direct sync executor
- direct index executor
- preview workflow wrappers

## Output Contract

The CLI is human-readable only. It does not promise a stable JSON schema.

Current output categories:

- tabular search and duplicate listings
- topic and coverage summaries
- sync and index summaries
- validation issue reports
- preview output for generation and Obsidian scans
- a full-screen TUI surface for local operator workflows

## Constraints

- `generate` does not persist cards
- `obsidian-sync` does not persist non-preview results yet
- CLI direct sync/index require PostgreSQL and Qdrant availability
- CLI direct sync/index may recreate an incompatible vector collection when the embedding model, dimension, or vector schema has changed
- the CLI should reuse shared runtime services rather than rewire dependencies ad hoc inside commands

## Module Layout

```text
bins/cli/src/
  main.rs
  args.rs
  output.rs
  runtime.rs
  tui/
    mod.rs
    app.rs
    bootstrap.rs
    input.rs
    tasks.rs
    widgets.rs
    screens/
      home.rs
      search.rs
      topics.rs
      workflows.rs
  commands/
    coverage.rs
    duplicates.rs
    gaps.rs
    generate.rs
    index.rs
    migrate.rs
    obsidian_sync.rs
    search.rs
    sync.rs
    tag_audit.rs
    topics.rs
    validate.rs
    version.rs
    weak_notes.rs
```

## Acceptance Criteria

- CLI command tree matches the clap definitions in `args.rs`
- search and analytics commands call shared facades
- sync and index execute directly only in CLI
- preview workflows fail explicitly for unsupported persistence behavior
- docs and examples do not mention removed or unwired CLI commands
