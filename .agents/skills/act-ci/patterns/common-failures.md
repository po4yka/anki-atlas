# Common CI Failure Patterns

## Failure Catalog

### Formatting

| Regex Pattern | Example Output | Category | Auto-fix Command |
|---------------|---------------|----------|-----------------|
| `Diff in .*\.rs` | `Diff in /src/lib.rs` | fmt | `cargo fmt --all` |
| `left:.*\nright:.*\n.*rustfmt` | Diff output from rustfmt | fmt | `cargo fmt --all` |

### Clippy Lints

| Regex Pattern | Example Output | Category | Auto-fix Command |
|---------------|---------------|----------|-----------------|
| `warning:.*\[clippy::` | `warning: unused variable [clippy::unused_variables]` | clippy | `cargo clippy --fix --allow-dirty` |
| `error:.*\[clippy::` | `error: ... [clippy::unwrap_used]` | clippy-deny | `cargo clippy --fix --allow-dirty` |
| `could not compile.*due to.*warning` | Clippy with -D warnings | clippy-deny | Fix manually or `--fix` |

### Compile Errors

| Regex Pattern | Example Output | Category | Manual Fix |
|---------------|---------------|----------|------------|
| `error\[E0308\]` | Type mismatch | type-error | Check expected vs found types |
| `error\[E0433\]` | Unresolved import | import-error | Add missing `use` or dependency |
| `error\[E0599\]` | Method not found | method-error | Check trait imports, method signature |
| `error\[E0277\]` | Trait not satisfied | trait-error | Add trait bound or impl |
| `error\[E0382\]` | Use after move | ownership-error | Clone, borrow, or restructure |
| `error\[E0502\]` | Borrow conflict | borrow-error | Restructure to avoid aliasing |
| `error\[E0425\]` | Cannot find value | name-error | Check spelling, imports, scope |

### Test Failures

| Regex Pattern | Example Output | Category | Manual Fix |
|---------------|---------------|----------|------------|
| `test result: FAILED` | Test summary line | test-fail | Read test, check assertion |
| `thread '.*' panicked at` | Panic in test | test-panic | Check unwrap/expect calls |
| `assertion.*failed` | `assert_eq!` failure | assertion-fail | Compare left vs right values |
| `connection refused` | Service not running | service-error | Start docker-compose services |
| `timeout` | Service timeout | service-timeout | Increase timeout or restart |

### Service/Infrastructure

| Regex Pattern | Example Output | Category | Manual Fix |
|---------------|---------------|----------|------------|
| `error connecting to.*5432` | Postgres not available | postgres-down | `docker compose up -d postgres` |
| `error connecting to.*6333` | Qdrant not available | qdrant-down | `docker compose up -d qdrant` |
| `error connecting to.*6379` | Redis not available | redis-down | `docker compose up -d redis` |
| `Cannot connect to the Docker daemon` | Docker not running | docker-down | Start Docker Desktop |

## Auto-fix Decision Matrix

```
Is the failure auto-fixable?
    |
    +-- Formatting (Diff in) -> YES: cargo fmt --all
    +-- Clippy lint -> MAYBE: cargo clippy --fix --allow-dirty
    |   +-- Simple lint (unused, naming) -> Usually works
    |   +-- Structural lint (lifetime, borrow) -> Often needs manual fix
    +-- Compile error -> NO: manual diagnosis required
    +-- Test failure -> NO: manual diagnosis required
    +-- Service down -> NO: start services manually
```

## Escalation Criteria

Escalate to manual fix when:
- `cargo clippy --fix` reports "could not apply suggestion"
- Error involves lifetime or borrow checker issues
- Test failure requires understanding business logic
- Multiple interdependent compile errors
- Service configuration issues
