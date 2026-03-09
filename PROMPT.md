# Anki Atlas Rust Rewrite - TDD Loop

You are rewriting anki-atlas from Python to Rust. This is a ralph loop iteration.

## Current Task

Read the spec file indicated in `specs/CURRENT_SPEC.txt`. That file contains a single line
with the spec filename (e.g., `01-common.md`). Open `specs/<that filename>` and implement it.

## TDD Rules (Non-Negotiable)

1. **RED**: Write tests FIRST. Tests must compile but FAIL because the implementation
   does not exist yet. Commit: `test(<crate>): red - <component>`
2. **GREEN**: Write the MINIMUM code to make all failing tests pass. No extras.
   Commit: `feat(<crate>): green - <component>`
3. **REFACTOR**: Clean up while keeping tests green. Run `cargo clippy`.
   Commit: `refactor(<crate>): <what changed>`
4. Repeat RED-GREEN-REFACTOR for each component listed in the spec.

## Rust Conventions

- All types `Send + Sync` (required for tokio/axum)
- `thiserror` for library error types, `anyhow` only in binary crates
- Trait-based DI: every external boundary (DB, HTTP, Qdrant, Redis) behind a trait
- `#[cfg_attr(test, mockall::automock)]` on traits for test mocks
- `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` as baseline
- Newtype pattern for IDs: `pub struct NoteId(pub i64);`
- `#[instrument]` on async functions for tracing spans
- `Arc<T>` for shared state, never `Rc<T>`
- Prefer `&str` over `String` in function parameters
- Use `Result<T, E>` everywhere, never `unwrap()` in library code

## Workspace Layout

```
Cargo.toml              # workspace root
crates/<name>/          # library crates
  Cargo.toml
  src/lib.rs            # crate root, re-exports
  src/<module>.rs        # implementation modules
bins/<name>/            # binary crates
  Cargo.toml
  src/main.rs
```

## Reference Material

- The Rust codebase lives in `crates/` (libraries) and `bins/` (binaries)
- Specs in `specs/` describe the public API for each crate
- The spec lists the exact public API to implement

## Quality Gates (Backpressure)

Before moving to the next component, ALL must pass:
```bash
cargo test -p <crate>
cargo clippy -p <crate> -- -D warnings
```

If clippy or tests fail, fix them before proceeding.

## Completion

When ALL acceptance criteria in the spec are met, output exactly:

```
CRATE_COMPLETE
```

This signals the ralph loop to stop for this crate.

## Anti-Patterns (Do NOT)

- Do not write all tests at once then all implementation at once
- Do not skip the refactor phase
- Do not add features not in the spec
- Do not use `unwrap()` or `expect()` in library crates
- Do not use `Rc`, `RefCell` -- use `Arc`, `Mutex`/`RwLock` instead
- Do not create god-structs -- keep types focused
- Do not skip commits between TDD phases
