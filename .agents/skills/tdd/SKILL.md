---
name: tdd
description: "Rust TDD workflow for anki-atlas. Use when writing new features, fixing bugs, or adding tests using Red-Green-Refactor with cargo test, mockall, and testcontainers. Don't use when running existing tests (use act-ci) or fixing CI failures (use fix-ci)."
globs: ["**/*.rs", "**/Cargo.toml"]
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]
---

# Rust TDD Workflow

## When to Use

- Writing a new feature or method
- Fixing a bug (write the failing test first)
- Adding test coverage to existing code
- Refactoring with safety net

## When NOT to Use

- Running existing tests -> use the act-ci skill
- Fixing CI failures -> use the fix-ci skill
- Docs-only or config-only changes

---

## The Cycle

One behavior per cycle. No skipping RED.

### RED: Write one failing test

```bash
cargo test -p <crate> -- <test_name>
```

The test MUST fail. If it passes, you're not testing new behavior.

### GREEN: Minimal implementation

Same command. Make it pass with the simplest code possible.
Then check for regressions:

```bash
cargo test -p <crate>
```

### REFACTOR: Clean up

Improve code structure while keeping all tests green:

```bash
cargo test -p <crate>
cargo clippy -p <crate> -- -D warnings
cargo fmt --all -- --check
```

See `patterns/red-green-refactor.md` for a worked example.

---

## Where to Put Tests

| Test Type | Location | When |
|-----------|----------|------|
| Unit test | `#[cfg(test)] mod tests` in source file | Testing private logic, pure functions |
| Integration test | `crates/<name>/tests/*.rs` | Testing public API, cross-module behavior |
| Handler test | `bins/api/tests/` | Testing HTTP endpoints with mock deps |
| CLI test | `bins/cli/tests/` | Testing CLI subcommands end-to-end |

**Decision tree:**
1. Does it test a single function's logic? -> Unit test in source file
2. Does it test the crate's public API? -> Integration test in `tests/`
3. Does it need Docker (Postgres, Redis)? -> Integration test, skip gracefully
4. Does it test an HTTP handler? -> Handler test with `mock!` deps

---

## Test Anatomy

```rust
#[tokio::test]  // or #[test] for sync
async fn estimate_tokens_empty_string_returns_zero() {
    // Arrange
    let provider = MockLlmProvider::new();

    // Act
    let result = provider.estimate_tokens("");

    // Assert
    assert_eq!(result, 0);
}
```

### Naming Convention

```
<behavior>_<condition>_<expected>
```

Examples: `slugify_simple_text`, `normalize_tag_unknown_kebab`, `version_sends_correct_payload`.

One assertion per test. Multiple related assertions are OK if testing one behavior.

---

## Mocking Quick Reference

### automock (same crate)

Add to trait definition:

```rust
#[cfg_attr(test, mockall::automock)]
pub trait MyTrait: Send + Sync {
    async fn do_thing(&self, input: &str) -> Result<Output, Error>;
}
```

Use in test:

```rust
let mut mock = MockMyTrait::new();
mock.expect_do_thing()
    .with(eq("input"))
    .returning(|_| Ok(Output::default()));
```

### mock! (cross-crate)

For traits defined in other crates (e.g., handler tests mocking job traits):

```rust
mock! {
    pub MyMock {}
    #[async_trait]
    impl ExternalTrait for MyMock {
        async fn method(&self, arg: Type) -> Result<Out, Err>;
    }
}
```

Use `MockMyMock::new()`.

See `patterns/testing-patterns.md` for full examples from this codebase.

---

## External Dependencies

### testcontainers (PostgreSQL)

```rust
let container = Postgres::default().start().await?;
let port = container.get_host_port_ipv4(5432).await?;
let url = format!("postgresql://postgres:postgres@localhost:{port}/postgres");
```

Requires Docker. Skip gracefully when unavailable:

```rust
let Some(ctx) = setup_pool().await else { return };
```

**Affected crates:** `database`, `anki-sync` -- exclude locally with:

```bash
cargo test --workspace --exclude database --exclude anki-sync
```

### wiremock (HTTP mocking)

```rust
let server = MockServer::start().await;
Mock::given(method("POST"))
    .and(body_json(json!({ "action": "version" })))
    .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": 6 })))
    .mount(&server)
    .await;
```

### tempfile

```rust
let file = NamedTempFile::new().expect("create temp file");
let dir = TempDir::new().expect("create temp dir");
```

Auto-cleaned on drop. Hold in scope for test duration.

See `patterns/testing-patterns.md` for full examples.

---

## Cargo Commands Cheat Sheet

```bash
# Single test by name (substring match)
cargo test -p <crate> -- <test_name>

# Single test with output
cargo test -p <crate> -- <test_name> --nocapture

# All tests in one crate
cargo test -p <crate>

# All workspace tests (CI gate)
cargo test --workspace

# Skip Docker-dependent crates
cargo test --workspace --exclude database --exclude anki-sync

# Specific integration test file
cargo test -p <crate> --test <test_file_name>
```

---

## CI Integration

Full CI check before committing:

```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

---

## Anti-Patterns

| Anti-Pattern | Do Instead |
|-------------|------------|
| Writing tests after implementation | Write the test FIRST (RED) |
| Testing private functions directly | Test through public API |
| Mocking everything | Mock at boundaries, test real logic |
| Horizontal slicing (all tests, then all code) | Vertical: one test + one impl per cycle |
| Giant test functions | One behavior per test, helper functions for setup |
| Skipping RED (test already passes) | If test passes immediately, it's not testing new behavior |

---

## Verification Checklist

Before completing a TDD session:

- [ ] Each new test failed first (RED was actually red)
- [ ] Implementation is minimal for each GREEN step
- [ ] All tests pass: `cargo test -p <crate>`
- [ ] Clippy clean: `cargo clippy -p <crate> -- -D warnings`
- [ ] Format clean: `cargo fmt --all -- --check`
- [ ] No `unwrap()` or `expect()` in library crate production code
- [ ] Test names follow `<behavior>_<condition>_<expected>` convention
