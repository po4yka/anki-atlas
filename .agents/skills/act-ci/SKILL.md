---
name: act-ci
description: "Use when running CI checks locally before pushing, testing specific CI jobs, or verifying code passes CI. Don't use when CI has already failed and needs fixing (use fix-ci). Triggers on: act, CI, local CI, run CI, check CI, clippy, fmt, format check, cargo test, integration test, pre-push"
globs: ["**/*.rs", ".github/workflows/*.yml", "Cargo.toml", "Cargo.lock"]
allowed-tools: ["Read", "Grep", "Glob", "Bash"]
---

# Run CI Locally

Run GitHub Actions CI jobs locally using `act` or direct cargo commands.

## CI Job Reference

| Job | Cargo Command | Services | Docker? |
|-----|--------------|----------|---------|
| check | `cargo check --workspace` | None | No |
| clippy | `cargo clippy --workspace -- -D warnings` | None | No |
| fmt | `cargo fmt --all -- --check` | None | No |
| test | `cargo test --workspace --exclude anki-sync --exclude database` | None | No |
| crate-integration | `cargo test -p database --tests && cargo test -p anki-sync --tests` | postgres, qdrant, redis | Yes |
| surface-integration | `cargo test -p anki-atlas-api --test project_surface_flows -- --test-threads=1 && cargo test -p anki-atlas-api --test project_cli_runtime_flows -- --test-threads=1` | postgres, qdrant, redis | Yes |

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

## Environment Variables

### Lightweight jobs (check, clippy, fmt, test)
```
CARGO_TERM_COLOR=always
RUSTFLAGS=-Dwarnings
```

### Integration jobs
```
ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas
ANKIATLAS_QDRANT_URL=http://localhost:6333
ANKIATLAS_REDIS_URL=redis://localhost:6379/0
ANKIATLAS_EMBEDDING_PROVIDER=mock
ANKIATLAS_EMBEDDING_MODEL=mock/test
ANKIATLAS_EMBEDDING_DIMENSION=384
ANKIATLAS_RERANK_ENABLED=false
```

## Service Images

| Service | Image | Ports |
|---------|-------|-------|
| postgres | `postgres:16-alpine` | 5432 |
| qdrant | `qdrant/qdrant:v1.16.3` | 6333, 6334 |
| redis | `redis:7-alpine` | 6379 |

## act Quirks

| Issue | Workaround |
|-------|------------|
| `services:` unsupported on macOS | Use `docker compose up -d` + direct cargo |
| rust-cache misses | Builds are slower; mount host target dir with `--bind` |
| arm64 emulation cost | ~2-5x slower on Apple Silicon; prefer direct cargo |
| Container startup ~30s | Direct cargo avoids this overhead entirely |

## Output Parsing

| Pattern | Meaning | Next Step |
|---------|---------|-----------|
| `Diff in` | Format failure | Run `cargo fmt --all` |
| `clippy::` | Clippy lint | Run `cargo clippy --fix --allow-dirty` |
| `error[E` | Compile error | Read error, fix source |
| `test result: FAILED` | Test failure | Check assertion, fix test or code |
| All green | CI passes | Safe to push |

## Related

- `patterns/common-failures.md` - Failure catalog with regex patterns
