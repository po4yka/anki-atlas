# fix-ci - Quick Reference

## Failure Pattern Table

| Pattern | Category | Auto-fix? |
|---------|----------|-----------|
| `Diff in` | Formatting | Yes |
| `clippy::` | Clippy lint | Yes (most) |
| `error[E` | Compile error | No |
| `test result: FAILED` | Test failure | No |
| `connection refused` | Service down | No |

## Auto-fix Cheat Sheet

```bash
# Fix formatting
cargo fmt --all

# Fix clippy lints
cargo clippy --fix --allow-dirty --workspace

# Verify formatting
cargo fmt --all -- --check

# Verify clippy
cargo clippy --workspace -- -D warnings

# Verify compilation
cargo check --workspace

# Verify tests
cargo test --workspace --exclude anki-sync --exclude database
```

## Escalation Criteria

Escalate to manual fix when:
- `cargo clippy --fix` reports "could not apply suggestion"
- Error involves lifetime or borrow checker issues
- Test failure requires understanding business logic
- Multiple interdependent compile errors
- Service configuration issues (Docker, ports)

## Common Error Code Quick Ref

| Code | Meaning | Quick Fix |
|------|---------|-----------|
| E0308 | Type mismatch | Convert types |
| E0433 | Missing import | Add `use` or dep |
| E0599 | No such method | Check trait import |
| E0277 | Trait not impl | Add bound/impl |
| E0382 | Use after move | Clone or borrow |
| E0502 | Borrow conflict | Restructure |
