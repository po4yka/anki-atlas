---
description: "Auto-fix common CI failures (formatting, clippy lints)"
argument-hint: "[--job job-name] [--error 'paste error']"
allowed-tools: [Read, Grep, Glob, Bash, Edit, Write]
---

# Fix CI Failures

## Task

Diagnose and auto-fix CI failures. For auto-fixable issues (formatting, clippy), apply fixes directly. For manual issues (compile errors, test failures), provide structured diagnosis.

## Arguments

- `--job` (optional): Which job failed (fmt, clippy, check, test, crate-integration, surface-integration)
- `--error` (optional): Paste of the error output to diagnose

## Process

### 1. Gather Context

If no `--error` provided, run the failing job to capture output:
```bash
# If --job specified, run just that job
cargo fmt --all -- --check 2>&1
cargo clippy --workspace -- -D warnings 2>&1
cargo test --workspace --exclude anki-sync --exclude database 2>&1
```

If no `--job` specified, run quick CI to find failures:
```bash
cargo fmt --all -- --check 2>&1 || true
cargo clippy --workspace -- -D warnings 2>&1 || true
cargo test --workspace --exclude anki-sync --exclude database 2>&1 || true
```

### 2. Classify Failure

| Pattern | Category | Auto-fix? |
|---------|----------|-----------|
| `Diff in` / rustfmt output | Formatting | Yes |
| `warning:.*clippy::` | Clippy lint | Yes (most) |
| `error\[E\d+\]` | Compile error | No |
| `test result: FAILED` | Test failure | No |
| `services.*unhealthy` | Service issue | No (restart services) |

### 3. Apply Fixes

#### Formatting (auto-fix)

```bash
cargo fmt --all
```

Verify:
```bash
cargo fmt --all -- --check
```

#### Clippy (auto-fix)

```bash
cargo clippy --fix --allow-dirty --workspace
```

If `--fix` cannot resolve (e.g., structural changes needed):
1. Read the clippy warning
2. Identify the file and line
3. Apply the suggested fix using Edit tool

Verify:
```bash
cargo clippy --workspace -- -D warnings
```

#### Compile Error (manual)

1. Parse `error[E####]` code
2. Read the referenced file at the error line
3. Show surrounding context (5 lines before/after)
4. Suggest fix based on error type:
   - `E0308` type mismatch: show expected vs found types
   - `E0433` unresolved import: check Cargo.toml dependencies
   - `E0599` method not found: check trait imports
   - `E0277` trait bound not satisfied: suggest impl or constraint

#### Test Failure (manual)

1. Identify the failing test from output
2. Read the test source code
3. Parse assertion failure:
   - `assert_eq!`: show left vs right values
   - `assert!`: show the condition
4. Read the code under test
5. Suggest fix based on whether test or implementation is wrong

### 4. Re-run Verification

After applying fixes, re-run the failing job:

```bash
# Re-run only the job that failed
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace --exclude anki-sync --exclude database
```

### 5. Report

```
Fix Report
==========
Category: Formatting
Files fixed: 3
  - crates/search/src/lib.rs
  - crates/card/src/slug.rs
  - bins/api/src/routes.rs

Verification: PASS

Remaining issues: None
```

Or for manual fixes:

```
Fix Report
==========
Category: Compile Error (E0308)
Location: crates/search/src/query.rs:42
Auto-fix: No

Diagnosis:
  Expected `Vec<String>`, found `Vec<&str>`

Suggested fix:
  Change `.collect::<Vec<&str>>()` to `.map(|s| s.to_string()).collect::<Vec<String>>()`

Run `/ci/run-ci clippy` after applying the fix.
```
