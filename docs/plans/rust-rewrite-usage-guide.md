# Rust Rewrite: How To Use

## Prerequisites

1. Install Rust 1.85+ via rustup
2. Install ralph-orchestrator:
   ```bash
   npm install -g @ralph-orchestrator/ralph-cli
   # or: cargo install ralph-cli
   # or: brew install ralph-orchestrator
   ```
3. PostgreSQL, Qdrant, and Redis running (for integration tests in later crates)

## Quick Start

```bash
# Run all 19 crates sequentially via TDD ralph loops
./run-ralph.sh

# Resume from a specific crate (e.g., after a failure)
./run-ralph.sh 06-indexer
```

## What Happens

For each crate, ralph runs a TDD loop with 4 hats:

```
Test Architect (RED)
    -> writes failing tests
    -> commits: test(<crate>): red - <component>

Implementer (GREEN)
    -> writes minimum code to pass
    -> commits: feat(<crate>): green - <component>

Refactorer (REFACTOR)
    -> cleans up, runs clippy
    -> commits: refactor(<crate>): <what>
    -> loops back to Test Architect if more components remain

Verifier (DONE)
    -> checks all acceptance criteria
    -> outputs CRATE_COMPLETE to end the loop
```

## Execution Order

| Phase | Crates | Why this order |
|-------|--------|----------------|
| 1. Foundation | common, taxonomy | No dependencies |
| 2. Data | database, anki-reader | Depend on common |
| 3. Core | anki-sync, indexer, search | Depend on data layer |
| 4. Domain | analytics, card, validation, llm, obsidian, rag, generator, jobs | Various deps |
| 5. Binaries | cli, api, mcp, worker | Depend on all crates |

## Files Overview

```
PROMPT.md                       # Fed to claude each ralph iteration
ralph.yml                       # Ralph config (completion promise, limits)
presets/tdd-rewrite.yml         # Custom TDD hat collection (4 hats)
specs/CURRENT_SPEC.txt          # Points to current spec (updated by runner)
specs/01-common.md ... 19-worker.md  # One spec per crate
run-ralph.sh                    # Sequential runner script
Cargo.toml                      # Workspace root with shared deps
docs/plans/2026-03-08-rust-rewrite-design.md  # Approved design doc
```

## Manual Invocation

If you want to run a single crate manually:

```bash
# Set the spec
echo "01-common.md" > specs/CURRENT_SPEC.txt

# Run ralph with TDD preset
ralph run --config presets/tdd-rewrite.yml
```

## Monitoring Progress

Each crate produces commits during TDD phases. Check progress with:

```bash
git log --oneline | head -20
```

## After Completion

Once all 19 crates pass:

```bash
# Full workspace verification
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo build --release

# The Rust binaries will be in target/release/
# - anki-atlas-cli
# - anki-atlas-api
# - anki-atlas-mcp
# - anki-atlas-worker
```
