# Python Style Guide

## Immutability First

- Use `@dataclass(frozen=True, slots=True)` for value objects
- Use `Final` for constants
- Prefer tuples over lists for fixed collections
- Return new objects instead of mutating

## Type Hints

- Complete type hints on all public functions and methods
- Use `from __future__ import annotations` in every file
- mypy strict mode -- no `type: ignore` without explanation
- Use `TypeAlias` for complex types

## Code Organization

- Max 600 lines per file -- split into submodules if needed
- One class per file for large classes
- Group imports: stdlib, third-party, local (ruff handles this)
- Use `__all__` in `__init__.py` for public API

## Logging

- Use `structlog.get_logger()` -- never `print()` or `logging.getLogger()`
- Log at appropriate levels: debug for internals, info for operations, error for failures

## Error Handling

- Define domain exceptions in `packages/common/exceptions.py`
- Catch specific exceptions, not bare `except:`
- Use `raise ... from e` for exception chaining
