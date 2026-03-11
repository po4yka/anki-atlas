# Red-Green-Refactor Cycle for Rust

The R-G-R cycle adapted for Rust with cargo commands.
Philosophy lives in `superpowers:test-driven-development`. This file is the execution layer.

---

## RED: Write One Failing Test

Write the test FIRST. It must fail for the right reason (missing function, wrong return value).

```rust
#[test]
fn slugify_preserves_cjk_characters() {
    assert_eq!(SlugService::slugify("hello"), "hello");
}
```

Run it:

```bash
cargo test -p card -- slugify_preserves_cjk
```

Expected output: **FAILED** (compile error or assertion failure).

If it passes, the test is not testing new behavior -- rethink it.

### Checklist

- [ ] Test name follows `test_<behavior>_<condition>_<expected>` or `<behavior>_<condition>`
- [ ] One logical assertion per test
- [ ] Test is in the right location (see "Where to Put Tests" in SKILL.md)
- [ ] Test compiles (add minimal stubs if needed: `todo!()`, empty struct)

---

## GREEN: Minimal Implementation

Write the **simplest code** that makes the test pass. No more.

- Return a literal if that's all it takes
- Use `todo!()` for unrelated branches
- Don't optimize, don't refactor, don't add error handling beyond what the test requires

Run the same test:

```bash
cargo test -p card -- slugify_preserves_cjk
```

Expected output: **ok**

Then run the full crate to check for regressions:

```bash
cargo test -p card
```

### Checklist

- [ ] The previously red test is now green
- [ ] No other tests broke
- [ ] Implementation is minimal (resist the urge to "finish" the feature)

---

## REFACTOR: Clean Up

Now improve the code while keeping all tests green.

- Extract helpers, remove duplication
- Improve naming
- Simplify control flow

After refactoring, run the full validation:

```bash
cargo test -p <crate>
cargo clippy -p <crate> -- -D warnings
cargo fmt --all -- --check
```

### Checklist

- [ ] All tests still pass
- [ ] Clippy clean (no warnings)
- [ ] Format clean
- [ ] No dead code introduced
- [ ] Code is simpler than before (or unchanged)

---

## Worked Example: Adding a Method to an Existing Trait

Goal: Add `estimate_tokens(&self, text: &str) -> usize` to `LlmProvider`.

### RED

```rust
// In crates/llm/src/provider.rs tests
#[test]
fn estimate_tokens_empty_string_returns_zero() {
    let provider = MockLlmProvider::new();
    // This won't compile yet -- method doesn't exist
    assert_eq!(provider.estimate_tokens(""), 0);
}
```

```bash
cargo test -p llm -- estimate_tokens_empty
# FAILS: no method `estimate_tokens`
```

### GREEN

Add the method to the trait with a default implementation:

```rust
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait LlmProvider: Send + Sync {
    // ... existing methods ...

    /// Estimate token count for text. Default: split on whitespace.
    fn estimate_tokens(&self, text: &str) -> usize {
        if text.is_empty() { 0 } else { text.split_whitespace().count() }
    }
}
```

```bash
cargo test -p llm -- estimate_tokens_empty
# PASSES
```

### Add More Tests (repeat RED-GREEN)

```rust
#[test]
fn estimate_tokens_single_word() {
    let provider = create_test_provider();
    assert_eq!(provider.estimate_tokens("hello"), 1);
}

#[test]
fn estimate_tokens_multiple_words() {
    let provider = create_test_provider();
    assert_eq!(provider.estimate_tokens("hello world foo"), 3);
}
```

### REFACTOR

Extract test helper, verify all tests pass:

```bash
cargo test -p llm
cargo clippy -p llm -- -D warnings
```

---

## When to Skip TDD

These cases don't need the full R-G-R cycle:

- **Docs-only changes**: comments, README, rustdoc
- **Typo fixes**: obvious single-character fixes
- **Generated code**: protobuf, sqlx macros output
- **Config files**: Cargo.toml dependency bumps, CI yaml
- **Mechanical refactors**: renames via `ast-grep` or IDE, `cargo fmt`

When in doubt, write the test. The cost of an unnecessary test is near zero.
The cost of a missed regression is high.
