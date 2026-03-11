---
name: fix-ci
description: "Use when CI checks fail and need fixing. Don't use when running CI proactively (use act-ci). Diagnoses and auto-fixes formatting, clippy lints; provides structured analysis for compile errors and test failures. Triggers on: fix CI, CI failing, clippy error, format error, cargo fmt, test failed, compile error"
globs: ["**/*.rs", "Cargo.toml"]
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]
---

# Fix CI Failures

Diagnose and fix CI failures. Auto-fix formatting and clippy; structured manual fix for compile and test errors.

## Failure Taxonomy

| Pattern | Category | Auto-fix? | Command |
|---------|----------|-----------|---------|
| `Diff in` | Formatting | Yes | `cargo fmt --all` |
| `clippy::` warning/error | Clippy lint | Yes (most) | `cargo clippy --fix --allow-dirty` |
| `error[E` | Compile error | No | Manual diagnosis |
| `test result: FAILED` | Test failure | No | Manual diagnosis |
| `connection refused` | Service down | No | `docker compose up -d` |

## Diagnostic Workflow

### Step 1: Reproduce

Run the failing job to capture current output:

```bash
# Try each in order, stop at first failure
cargo fmt --all -- --check 2>&1
cargo clippy --workspace -- -D warnings 2>&1
cargo test --workspace --exclude anki-sync --exclude database 2>&1
```

### Step 2: Isolate

From the output, identify:
1. Which job failed (fmt, clippy, check, test)
2. Which file(s) are affected
3. The specific error pattern

### Step 3: Parse and Classify

Match output against patterns in `../act-ci/patterns/common-failures.md`.

### Step 4: Fix

#### Auto-fix: Formatting

```bash
cargo fmt --all
# Verify
cargo fmt --all -- --check
```

#### Auto-fix: Clippy

```bash
cargo clippy --fix --allow-dirty --workspace
# Verify
cargo clippy --workspace -- -D warnings
```

If `--fix` fails:
1. Read the clippy warning carefully
2. Identify file and line from output
3. Apply the suggested change using Edit tool

#### Manual: Compile Error

1. Parse error code (`error[E####]`)
2. Read the file at the error location (5 lines context)
3. Analyze based on error type:

| Error Code | Type | Typical Fix |
|-----------|------|-------------|
| E0308 | Type mismatch | Convert types, change return type |
| E0433 | Unresolved import | Add `use` statement or Cargo.toml dep |
| E0599 | No method | Add trait import, check method name |
| E0277 | Trait not satisfied | Add bound, implement trait |
| E0382 | Use after move | Clone, use reference, restructure |
| E0502 | Borrow conflict | Split borrows, restructure code |

4. Apply fix using Edit tool
5. Verify with `cargo check --workspace`

#### Manual: Test Failure

1. Find failing test name from output
2. Read test source code
3. Read the code under test
4. Determine if test or implementation is wrong:
   - Test expectations outdated? Update test
   - Implementation bug? Fix implementation
5. Verify with `cargo test -p <crate> -- <test_name>`

## Verification Protocol

After every fix, re-run the specific failing command:

```bash
# After fmt fix
cargo fmt --all -- --check

# After clippy fix
cargo clippy --workspace -- -D warnings

# After compile fix
cargo check --workspace

# After test fix
cargo test --workspace --exclude anki-sync --exclude database
```

Never claim a fix is complete without passing verification.

## Related

- `../act-ci/SKILL.md` - Running CI locally
- `../act-ci/patterns/common-failures.md` - Failure pattern catalog
