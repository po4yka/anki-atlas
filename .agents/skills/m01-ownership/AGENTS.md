# Ownership & Lifetimes - Quick Reference

Diagnoses and fixes ownership, borrowing, and lifetime errors in Rust.

## Error Code Quick Reference

| Error | Meaning | First Question |
|-------|---------|----------------|
| E0382 | Use of moved value | Who should own this data? |
| E0597 | Borrowed value doesn't live long enough | Is the scope boundary correct? |
| E0506 | Cannot assign; already borrowed | Should mutation happen elsewhere? |
| E0507 | Cannot move out of borrowed content | Why are we moving from a reference? |
| E0515 | Cannot return reference to local | Should caller own the data? |
| E0716 | Temporary value dropped while borrowed | Does the temporary need a binding? |
| E0106 | Missing lifetime specifier | What's the relationship between inputs and output? |

## Sub-File Index

| File | Content |
|------|---------|
| `comparison.md` | Ownership model comparison (Rust vs GC vs manual) |
| `examples/best-practices.md` | Idiomatic ownership patterns with examples |
| `patterns/common-errors.md` | Common ownership errors and fixes |
| `patterns/lifetime-patterns.md` | Lifetime annotation patterns and elision rules |

## Essential Checklist

- [ ] Identify the error code and map to design question
- [ ] Check if `&T`, `&mut T`, or owned `T` is appropriate
- [ ] Consider `Cow<T>` for conditional ownership
- [ ] Verify lifetime annotations match data flow
- [ ] Prefer references over `.clone()` unless ownership transfer needed
