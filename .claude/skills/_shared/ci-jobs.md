# CI Job Reference

Source of truth: `.github/workflows/ci.yml` and `.github/workflows/perf.yml`

## CI Jobs

| Job | Cargo Command | Services | Docker? |
|-----|--------------|----------|---------|
| check | `cargo check --workspace` | None | No |
| clippy | `cargo clippy --workspace -- -D warnings` | None | No |
| fmt | `cargo fmt --all -- --check` | None | No |
| test | `cargo test --workspace --exclude anki-sync --exclude database` | None | No |
| crate-integration | `cargo test -p database --tests && cargo test -p anki-sync --tests` | postgres, qdrant, redis | Yes |
| surface-integration | `cargo test -p anki-atlas-api --test project_surface_flows -- --test-threads=1 && cargo test -p anki-atlas-api --test project_cli_runtime_flows -- --test-threads=1` | postgres, qdrant, redis | Yes |

## Perf Jobs

| Job | Cargo Command | Services | Docker? |
|-----|--------------|----------|---------|
| perf-smoke | `cargo run -p perf-harness -- --profile pr --scenario full` | postgres, qdrant, redis | Yes |
| perf-nightly | `cargo run -p perf-harness -- --profile nightly --scenario full` + benchmarks | postgres, qdrant, redis | Yes |

## Environment Variables

### Lightweight jobs (check, clippy, fmt, test)
```
CARGO_TERM_COLOR=always
RUSTFLAGS=-Dwarnings
```

### Integration jobs (crate-integration, surface-integration, perf-*)
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

## act Commands

```bash
# Lightweight (no Docker services needed)
act -j check --env-file .github/act/env.ci -e .act-event.json
act -j clippy --env-file .github/act/env.ci -e .act-event.json
act -j fmt --env-file .github/act/env.ci -e .act-event.json
act -j test --env-file .github/act/env.ci -e .act-event.json

# Integration (requires docker-compose services running)
act -j crate-integration --env-file .github/act/env.integration -e .act-event.json
act -j surface-integration --env-file .github/act/env.integration -e .act-event.json
```

## Notes

- act does not reliably support `services:` blocks on macOS; use `docker compose up -d` instead
- rust-cache action may not work in act; builds will be slower
- arm64 hosts incur emulation cost with `--container-architecture linux/amd64`
- Direct cargo commands are 10-50x faster than act for lightweight jobs
