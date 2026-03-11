---
name: act-ci
description: "Use when running CI checks locally before pushing, testing specific CI jobs, or verifying code passes CI. Triggers on: act, CI, local CI, run CI, check CI, clippy, fmt, format check, cargo test, integration test, pre-push"
allowed-tools: Read, Glob, Grep, Bash
---

# Run CI Locally

Run GitHub Actions CI jobs locally using `act` or direct cargo commands.

## When to Use

- User wants to run CI before pushing
- User asks "will this pass CI?"
- User wants to test a specific CI job locally
- User mentions "act", "local CI", or "pre-push check"

## Prerequisites Check

Before running, verify:

```bash
# Check act is installed
command -v act >/dev/null 2>&1 || echo "act not installed: brew install act"

# Check Docker is running (only needed for act or integration tests)
docker info >/dev/null 2>&1 || echo "Docker not running"
```

## Decision Tree

```
What does the user want?
    |
    +-- "quick check" / "will this pass?"
    |   -> Run cargo directly: fmt + clippy + check + test
    |
    +-- Specific job (e.g., "run clippy")
    |   -> cargo directly (fastest) or act -j <job>
    |
    +-- "full CI" / "all jobs"
    |   -> Lightweight via cargo, then integration via docker-compose + cargo
    |
    +-- Integration tests specifically
        -> docker compose up -d, then cargo test with env vars
```

## Quick Commands (Cargo -- Preferred for Speed)

```bash
# Format check
cargo fmt --all -- --check

# Clippy
cargo clippy --workspace -- -D warnings

# Type check
cargo check --workspace

# Unit tests (excludes Docker-dependent crates)
cargo test --workspace --exclude anki-sync --exclude database

# All lightweight checks in sequence
cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace --exclude anki-sync --exclude database
```

## act Commands (Full CI Reproduction)

```bash
# Single lightweight job
act -j fmt --env-file .github/act/env.ci -e .act-event.json
act -j clippy --env-file .github/act/env.ci -e .act-event.json
act -j check --env-file .github/act/env.ci -e .act-event.json
act -j test --env-file .github/act/env.ci -e .act-event.json

# List all available jobs
act -l
```

## Integration Tests

act's `services:` blocks do not work reliably on macOS. Use docker-compose instead:

```bash
# Start services
docker compose up -d postgres qdrant redis

# Wait for services to be healthy
docker compose ps --format json | jq -r '.Health'

# Run crate integration tests
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas \
ANKIATLAS_QDRANT_URL=http://localhost:6333 \
ANKIATLAS_REDIS_URL=redis://localhost:6379/0 \
cargo test -p database --tests && cargo test -p anki-sync --tests

# Run surface integration tests
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas \
ANKIATLAS_QDRANT_URL=http://localhost:6333 \
ANKIATLAS_REDIS_URL=redis://localhost:6379/0 \
ANKIATLAS_EMBEDDING_PROVIDER=mock \
ANKIATLAS_EMBEDDING_MODEL=mock/test \
ANKIATLAS_EMBEDDING_DIMENSION=384 \
ANKIATLAS_RERANK_ENABLED=false \
cargo test -p anki-atlas-api --test project_surface_flows -- --test-threads=1 && \
cargo test -p anki-atlas-api --test project_cli_runtime_flows -- --test-threads=1

# Tear down when done
docker compose down
```

## act Quirks

| Issue | Workaround |
|-------|------------|
| `services:` unsupported on macOS | Use `docker compose up -d` + direct cargo |
| rust-cache misses | Builds are slower; mount host target dir with `--bind` |
| arm64 emulation cost | ~2-5x slower on Apple Silicon; prefer direct cargo |
| Container startup ~30s | Direct cargo avoids this overhead entirely |

## Output Parsing

After running, check for:

| Pattern | Meaning | Next Step |
|---------|---------|-----------|
| `Diff in` | Format failure | Run `cargo fmt --all` |
| `clippy::` | Clippy lint | Run `cargo clippy --fix --allow-dirty` |
| `error[E` | Compile error | Read error, fix source |
| `test result: FAILED` | Test failure | Check assertion, fix test or code |
| All green | CI passes | Safe to push |

## Related

- [ci-jobs.md](../_shared/ci-jobs.md) - Full CI job reference table
- `/ci/fix-ci` - Auto-fix common CI failures
