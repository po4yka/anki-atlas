# Contributing to Anki Atlas

Thank you for your interest in contributing to Anki Atlas.

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/anki-atlas.git
cd anki-atlas
```

### 2. Set Up Development Environment

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Start infrastructure dependencies
docker compose -f infra/docker-compose.yml up -d

# Verify build
cargo build
cargo test --workspace --exclude anki-sync --exclude database
```

### 3. Create a Branch

Use a descriptive branch name following this convention:

- `feat/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test additions or changes

```bash
git checkout -b feat/my-feature
```

### 4. Make Changes

- Follow existing code patterns and conventions
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes

```bash
# Run linting
cargo clippy --workspace -- -D warnings

# Check formatting
cargo fmt --all -- --check

# Run tests (excludes Docker-dependent crates)
cargo test --workspace --exclude anki-sync --exclude database

# Run single crate tests for faster iteration
cargo test -p <crate-name>
```

### 6. Commit Your Changes

Follow Conventional Commits format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
```bash
git commit -m "feat(search): add similarity threshold parameter"
git commit -m "fix(sync): handle empty collections gracefully"
git commit -m "test(indexer): add embedding provider mock tests"
```

### 7. Push and Create Pull Request

```bash
git push origin feat/my-feature
```

## Code Conventions

- Rust 1.85+ (edition 2024)
- All types must be `Send + Sync`
- `thiserror` for library error types, `anyhow` only in binary crates
- Trait-based DI at every external boundary (DB, HTTP, Qdrant, Redis)
- `#[cfg_attr(test, mockall::automock)]` on traits for test mocks
- `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` as baseline
- Newtype pattern for domain IDs: `pub struct NoteId(pub i64);`
- `#[instrument]` on public async functions for tracing
- `Arc<T>` for shared state, never `Rc<T>`
- No `unwrap()` or `expect()` in library crates

## Testing

- Unit tests in `#[cfg(test)] mod tests` within each source file
- Integration tests in `crates/<name>/tests/` and `bins/<name>/tests/`
- `#[tokio::test]` for async tests
- `mockall` for auto-generated mock implementations
- `tempfile::TempDir` for filesystem tests

## Reporting Issues

When reporting bugs, include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Rust version)
- Relevant logs or error messages
