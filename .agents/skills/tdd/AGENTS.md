# TDD - Quick Reference

Rust TDD workflow with Red-Green-Refactor cycle for anki-atlas.

## TDD Rules

| # | Rule |
|---|------|
| 1 | Write the failing test BEFORE any implementation code |
| 2 | One behavior per R-G-R cycle -- no batching |
| 3 | GREEN means simplest code that passes, not "finished" code |
| 4 | REFACTOR only when all tests are green |
| 5 | Never skip RED -- if the test passes immediately, rethink it |

## Cargo Test Cheat Sheet

```bash
cargo test -p <crate> -- <test_name>           # Single test
cargo test -p <crate> -- <test_name> --nocapture # With output
cargo test -p <crate>                            # All crate tests
cargo test --workspace                           # All tests (CI gate)
cargo test --workspace --exclude database --exclude anki-sync  # Skip Docker
cargo test -p <crate> --test <file>             # Specific integration test
```

## Mock Decision Tree

```
Need to mock a trait?
    |
    +-- Trait in SAME crate as test?
    |   +-- Yes: #[cfg_attr(test, mockall::automock)] on trait
    |   |        Use MockTraitName::new()
    |
    +-- Trait in DIFFERENT crate?
        +-- Yes: mock! { pub Name {} impl Trait for Name { ... } }
                 Use MockName::new()
```

## Where to Put Tests

```
Unit test (private logic)     -> #[cfg(test)] mod tests in source file
Integration test (public API) -> crates/<name>/tests/*.rs
Handler test (HTTP endpoints) -> bins/api/tests/
CLI test (subcommands)        -> bins/cli/tests/
```

## Sub-File Index

| File | Content |
|------|---------|
| `patterns/testing-patterns.md` | mockall, testcontainers, wiremock, tempfile examples |
| `patterns/red-green-refactor.md` | R-G-R cycle with cargo commands, worked example |

## Essential Checklist

- [ ] Each test failed first (RED was red)
- [ ] Implementation is minimal per GREEN step
- [ ] All tests pass: `cargo test -p <crate>`
- [ ] Clippy clean: `cargo clippy -p <crate> -- -D warnings`
- [ ] Format clean: `cargo fmt --all -- --check`
- [ ] No `unwrap()`/`expect()` in library production code
