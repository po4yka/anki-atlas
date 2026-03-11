---
description: "Run CI checks locally using cargo or act"
argument-hint: "[job-name|all|quick] [--fix]"
allowed-tools: [Read, Grep, Glob, Bash]
---

# Run CI Locally

## Task

Run CI jobs locally and report pass/fail status.

## Arguments

- `job-name` (optional): One of `check`, `clippy`, `fmt`, `test`, `crate-integration`, `surface-integration`, `all`, `quick`
  - Default: `quick` (fmt + clippy + test)
- `--fix`: If any job fails, automatically chain to `/ci/fix-ci` for auto-fixable issues

## Process

### 1. Parse Arguments

Determine which jobs to run:

| Argument | Jobs |
|----------|------|
| `quick` (default) | fmt, clippy, test |
| `all` | fmt, clippy, check, test, crate-integration, surface-integration |
| `check` | cargo check --workspace |
| `clippy` | cargo clippy --workspace -- -D warnings |
| `fmt` | cargo fmt --all -- --check |
| `test` | cargo test --workspace --exclude anki-sync --exclude database |
| `crate-integration` | cargo test -p database --tests && cargo test -p anki-sync --tests |
| `surface-integration` | surface flow tests (see ci-jobs.md) |

### 2. Check Prerequisites

```bash
# For lightweight jobs: just need cargo
cargo --version

# For integration jobs: need Docker services
docker compose ps 2>/dev/null
```

If integration jobs requested but services not running:
```bash
docker compose up -d postgres qdrant redis
# Wait for healthy status
for i in {1..30}; do
  docker compose ps --format json | grep -q '"healthy"' && break
  sleep 2
done
```

### 3. Run Jobs

Run each job sequentially. For each job:

1. Print job name
2. Execute the cargo command
3. Capture exit code and output
4. Record pass/fail

For integration jobs, export env vars from `.github/act/env.integration`.

### 4. Report Results

```
CI Results
==========
fmt:     PASS
clippy:  PASS
test:    FAIL (3 tests failed)

Failed tests:
  - search::tests::test_hybrid_search
  - analytics::tests::test_coverage
  - card::tests::test_slug_generation

Overall: FAIL
```

### 5. Auto-fix (if --fix)

If `--fix` flag was passed and failures are auto-fixable:
- Format failures: run `cargo fmt --all`
- Clippy lints: run `cargo clippy --fix --allow-dirty`
- Then re-run the failing jobs to verify

For non-auto-fixable failures (compile errors, test failures), show context and suggest manual fixes.

## Error Handling

| Error | Action |
|-------|--------|
| cargo not found | Error: "Rust toolchain not installed" |
| Docker not running | Skip integration jobs, warn user |
| Services not healthy | Retry wait, then error with service logs |
| act not installed | Fall back to direct cargo commands |
