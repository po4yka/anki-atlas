# Error Handling - Quick Reference

Guides error handling strategy: thiserror vs anyhow, Result vs panic, error propagation.

## Decision Table

| Context | Strategy | Crate |
|---------|----------|-------|
| Library crate | Custom error enum | `thiserror` |
| Binary crate | Contextual errors | `anyhow` |
| Infallible invariant | `panic!` / `unreachable!` | std |
| Optional value | `Option<T>` | std |
| FFI boundary | Error codes | manual |
| Async context | `Result` + `?` | `thiserror` / `anyhow` |

## When to Panic vs Return Result

| Panic | Result |
|-------|--------|
| Programming bug (invariant violation) | Expected failure (I/O, network, parse) |
| Unrecoverable state | Caller can handle or propagate |
| Tests and examples | Library public API |

## Sub-File Index

| File | Content |
|------|---------|
| `examples/library-vs-app.md` | Side-by-side library vs application error handling |
| `patterns/error-patterns.md` | Error propagation, conversion, and context patterns |

## Essential Checklist

- [ ] Library crates: `thiserror` with enum variants
- [ ] Binary crates: `anyhow` with `.context()`
- [ ] No `unwrap()` / `expect()` in library code
- [ ] Add context at every `?` boundary crossing
- [ ] Map external errors to domain errors at crate boundary
