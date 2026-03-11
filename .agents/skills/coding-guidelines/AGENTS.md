# Coding Guidelines - Quick Reference

Rust coding conventions: naming, formatting, data types, strings, and clippy rules.

## Rule Summary

| Category | Key Rules |
|----------|-----------|
| Naming | No `get_` prefix; `iter()`/`iter_mut()`/`into_iter()`; `as_`/`to_`/`into_` |
| Data Types | Newtypes for domain; slice patterns; `with_capacity()`; arrays for fixed |
| Strings | `&str` params; `format!` over concat; `write!` for building |
| Functions | Single responsibility; early return; builder pattern for 3+ params |
| Traits | Impl std traits; `From`/`Into` for conversions; `Display` for user output |
| Error Style | `thiserror` for libs; `anyhow` for bins; `?` over `unwrap` |

## Sub-File Index

| File | Content |
|------|---------|
| `index/rules-index.md` | Full 50-rule index with IDs and categories |

## Essential Checklist

- [ ] Follow Rust naming conventions (no `get_` prefix)
- [ ] Use newtypes for domain semantics
- [ ] Prefer `&str` over `String` in function parameters
- [ ] Run `cargo clippy` and `cargo fmt` before committing
