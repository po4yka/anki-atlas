# Contributing to Anki Atlas

Thank you for your interest in contributing to Anki Atlas. This document
provides guidelines for contributing to the project.

## Development Workflow

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/anki-atlas.git
cd anki-atlas
```

### 2. Set Up Development Environment

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
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
uv run ruff check apps packages tests

# Run type checking
uv run mypy packages apps

# Run tests
uv run pytest tests/ -v

# Run single test file for faster iteration
uv run pytest tests/test_specific.py -v
```

### 6. Commit Your Changes

Follow Conventional Commits format:

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring (no feature change)
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes

Examples:
```bash
git commit -m "feat: add similarity threshold to duplicate detection"
git commit -m "fix: handle empty collections in sync"
git commit -m "docs: add troubleshooting guide"
```

Keep subject line under 72 characters. Use imperative mood ("add feature" not "added feature").

### 7. Push and Create Pull Request

```bash
git push origin feat/my-feature
```

Then create a PR on GitHub with:
- Clear title describing the change
- Description explaining what and why
- Reference to any related issues

## Code Style

### Python

- Follow PEP 8 with ruff as the linter/formatter
- Use type hints for all function signatures
- Prefer explicit types over inferred when it aids readability
- Maximum line length: 100 characters

### Naming Conventions

- Functions/methods: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`

### Documentation

- Docstrings for all public functions, classes, and modules
- Use Google-style docstrings
- Keep docstrings concise but complete

```python
def search(query: str, limit: int = 10) -> SearchResult:
    """Search for notes matching the query.

    Args:
        query: Search query text.
        limit: Maximum number of results.

    Returns:
        SearchResult with matching notes and statistics.

    Raises:
        DatabaseConnectionError: If database is unavailable.
    """
```

### Error Handling

- Use custom exceptions from `packages/common/exceptions.py`
- Always include context in log messages
- Never silently swallow exceptions without logging

## Testing

### Test Requirements

- All new features must have tests
- Bug fixes should include regression tests
- Maintain test coverage above 80%

### Test Organization

- Unit tests: `tests/test_<module>.py`
- Integration tests: `tests/test_integration.py`
- Use pytest fixtures for shared setup

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=packages --cov=apps

# Specific file
uv run pytest tests/test_search.py -v

# Specific test
uv run pytest tests/test_search.py::test_hybrid_search -v
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Get at least one review approval
4. Squash commits if requested
5. Maintainer will merge after approval

## Reporting Issues

When reporting bugs, include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version)
- Relevant logs or error messages

## Questions?

Open an issue with the `question` label or start a discussion.
