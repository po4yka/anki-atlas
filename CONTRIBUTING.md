# Contributing to Anki Atlas

This workspace is still evolving quickly, but the contribution rule on `main` is stable: public surfaces must stay aligned with the real Rust services underneath them.

## Before You Start

- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- Prefer updating the shared domain crates over adding surface-specific behavior in `bins/*`.
- Keep API, CLI, and MCP documentation in sync with the code whenever you change a public contract.

## Local Setup

```bash
git clone https://github.com/your-username/anki-atlas.git
cd anki-atlas

docker compose -f infra/docker-compose.yml up -d

# Mock embeddings are fine for basic smoke tests
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384

cargo build
cargo run --bin anki-atlas -- migrate
```

If you need real embeddings, switch to `ANKIATLAS_EMBEDDING_PROVIDER=openai` or `google` and supply the matching API key.

## Branching and Commits

Use descriptive branch names and Conventional Commit messages.

Examples:

```bash
git checkout -b docs/update-runtime-docs
git checkout -b refactor/search-surface-alignment

git commit -m "docs: refresh runtime and public surface guides"
git commit -m "refactor(api): align search DTOs with service contract"
git commit -m "test(mcp): cover markdown and json tool output"
```

## Project Rules

- `bins/*` should translate transport concerns only.
- Domain rules belong in `crates/*`.
- Shared surface wiring belongs in `crates/surface-runtime`.
- Unsupported workflows must fail explicitly instead of returning placeholder success.
- Avoid reintroducing duplicate public entrypoints for the same behavior.

## Code Style

- Rust edition `2024`
- `thiserror` for library error types; `anyhow` only at binary/application edges
- Trait-based boundaries for infrastructure and external services
- Prefer typed DTOs over `serde_json::Value` at public interfaces unless the payload is intentionally schema-free
- No `unwrap()` / `expect()` in library code
- Keep comments sparse and useful

## Testing Expectations

Run the narrowest useful test set before sending a change:

```bash
# format and lint
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings

# surface-focused tests
cargo test -p anki-atlas-api
cargo test -p anki-atlas-cli
cargo test -p anki-atlas-mcp

# crate-focused tests
cargo test -p <crate-name>
```

For broader changes:

```bash
cargo test --workspace --exclude anki-sync --exclude database
```

## Documentation Expectations

Update docs when you change:

- API routes or schemas
- CLI commands or flags
- MCP tool names, parameters, or output modes
- environment variables
- deployment requirements
- intentional product limitations

The current source-of-truth docs are:

- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [specs/16-cli.md](specs/16-cli.md)
- [specs/17-api.md](specs/17-api.md)
- [specs/18-mcp.md](specs/18-mcp.md)

## Reporting Problems

When filing an issue or review note, include:

- the command or route you ran
- the environment variables or provider mode in use
- the exact error text
- whether the failure is in CLI, API, MCP, or worker
- reproduction steps
