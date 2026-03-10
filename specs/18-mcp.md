# Spec: crate `anki-atlas-mcp`

## Source Reference

Current Rust source:

- [main.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/main.rs)
- [server.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/server.rs)
- [tools.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/tools.rs)
- [formatters.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/formatters.rs)
- [handlers.rs](/Users/po4yka/GitRep/anki-atlas/bins/mcp/src/handlers.rs)

## Purpose

Expose Anki Atlas capabilities to agent clients over MCP using real shared services and typed tool contracts.

The MCP server must:

- share runtime wiring with API and CLI
- keep sync and index async-only through jobs
- support `markdown` and `json` output modes
- use real local preview workflows for generation, validation, Obsidian scan, and tag audit

## Dependencies

```toml
[dependencies]
common = { path = "../../crates/common" }
jobs = { path = "../../crates/jobs" }
search = { path = "../../crates/search" }
surface-runtime = { path = "../../crates/surface-runtime" }

rmcp.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
tracing.workspace = true
anyhow.workspace = true
```

## Public API

### Transport

- MCP stdio server powered by `rmcp`
- server registers tools through `tool_router` and `tool_handler`

### Tool Set

Read tools:

- `ankiatlas_search`
- `ankiatlas_topics`
- `ankiatlas_topic_coverage`
- `ankiatlas_topic_gaps`
- `ankiatlas_topic_weak_notes`
- `ankiatlas_duplicates`

Async job tools:

- `ankiatlas_sync_job`
- `ankiatlas_index_job`
- `ankiatlas_job_status`
- `ankiatlas_job_cancel`

Local workflow tools:

- `ankiatlas_generate`
- `ankiatlas_validate`
- `ankiatlas_obsidian_sync`
- `ankiatlas_tag_audit`

### Input Contract

Every tool input includes:

- `output_mode`

`output_mode` values:

- `markdown`
- `json`

Tool-specific inputs map onto:

- search query and filters
- analytics query parameters
- job enqueue payloads
- file or vault paths for preview workflows

### Result Contract

Each tool has one canonical typed result struct. Markdown mode renders that result for humans. JSON mode returns the same data structurally.

Representative result families:

- `SearchToolResult`
- `TopicsToolResult`
- `TopicCoverageToolResult`
- `TopicGapsToolResult`
- `TopicWeakNotesToolResult`
- `DuplicatesToolResult`
- `JobAcceptedToolResult`
- `JobStatusToolResult`
- `WorkflowToolResult`

## Runtime Rules

- MCP uses [surface-runtime](/Users/po4yka/GitRep/anki-atlas/crates/surface-runtime/src/services.rs) with direct execution disabled
- read tools call the shared search and analytics facades
- sync/index tools enqueue jobs rather than executing directly
- preview workflow tools must fail explicitly for unsupported persistence paths
- logging must stay off stdout to avoid corrupting the MCP protocol stream

## Known Constraints

- `ankiatlas_generate` is preview-only
- `ankiatlas_obsidian_sync` requires dry-run behavior today
- MCP is not allowed to bypass job orchestration for sync/index

## Module Layout

```text
bins/mcp/src/
  main.rs
  lib.rs
  server.rs
  tools.rs
  formatters.rs
  handlers.rs
```

## Acceptance Criteria

- exact tool registration list matches this spec
- every tool supports both output modes
- JSON mode returns structured results, not markdown-in-a-string
- sync/index remain async-only in MCP
- docs do not describe removed or placeholder MCP tools
