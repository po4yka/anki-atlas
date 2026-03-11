# Anti-Patterns - Quick Reference

Identifies common Rust anti-patterns and provides idiomatic alternatives.

## Top 5 Mistakes

| # | Anti-Pattern | Why Bad | Fix |
|---|--------------|---------|-----|
| 1 | `.clone()` everywhere | Hides ownership bugs, wastes memory | Proper `&T` / ownership design |
| 2 | `.unwrap()` in library code | Runtime panics in production | `?` operator, `thiserror` enums |
| 3 | `Rc` when single owner | Unnecessary overhead and complexity | Direct ownership or references |
| 4 | `unsafe` for convenience | UB risk, defeats Rust's guarantees | Find safe pattern or abstraction |
| 5 | OOP inheritance via `Deref` | Misleading API, fragile hierarchy | Composition + traits |

## Additional Anti-Patterns

| Anti-Pattern | Better |
|--------------|--------|
| `String` params everywhere | `&str` or `impl AsRef<str>` |
| Nested `match` on `Option/Result` | Combinator chains (`.map`, `.and_then`) |
| `Box<dyn Error>` in libraries | Concrete error enum with `thiserror` |
| Manual `impl Iterator` | `iter().map().filter()` chains |
| God struct with many fields | Smaller structs with clear ownership |

## Sub-File Index

| File | Content |
|------|---------|
| `patterns/common-mistakes.md` | Extended anti-pattern catalog with code examples |

## Essential Checklist

- [ ] Check for `.clone()` -- is ownership designed correctly?
- [ ] Check for `.unwrap()` -- should this return `Result`?
- [ ] Check for `Rc`/`Arc` -- is shared ownership truly needed?
- [ ] Check for `unsafe` -- is there a safe alternative?
- [ ] Check for OOP patterns -- prefer composition over inheritance
