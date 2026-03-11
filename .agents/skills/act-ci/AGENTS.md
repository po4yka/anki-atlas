# act-ci - Quick Reference

## Job Summary

| Job | Command | Docker? |
|-----|---------|---------|
| fmt | `cargo fmt --all -- --check` | No |
| clippy | `cargo clippy --workspace -- -D warnings` | No |
| check | `cargo check --workspace` | No |
| test | `cargo test --workspace --exclude anki-sync --exclude database` | No |
| crate-integration | `cargo test -p database --tests && cargo test -p anki-sync --tests` | Yes |
| surface-integration | Surface flow tests (see SKILL.md) | Yes |

## Quick Decision Tree

```
Need to run CI?
    |
    +-- Quick validation? -> cargo fmt + clippy + test (no Docker)
    +-- Specific job? -> cargo <command> directly
    +-- Integration? -> docker compose up -d + cargo test with env vars
    +-- Full reproduction? -> act -j <job> --env-file .github/act/env.ci -e .act-event.json
```

## Common act Flags

```bash
act -l                          # List jobs
act -j <job>                    # Run specific job
act --env-file <file>           # Load env vars
act -e .act-event.json          # Fake push event
act --dryrun                    # Parse only, no execution
act --bind                      # Bind mount workdir (faster)
```

## Env Var Files

| File | Use |
|------|-----|
| `.github/act/env.ci` | Lightweight jobs (CARGO_TERM_COLOR, RUSTFLAGS) |
| `.github/act/env.integration` | Integration jobs (DB URLs, mock embedding config) |

## Troubleshooting

- [ ] `act` installed? `command -v act`
- [ ] Docker running? `docker info`
- [ ] Services healthy? `docker compose ps`
- [ ] Correct Rust toolchain? `rustc --version` (need 1.85+)
- [ ] On arm64? Use `--container-architecture linux/amd64` (in .actrc)
- [ ] act `services:` failing? Use `docker compose up -d` instead
