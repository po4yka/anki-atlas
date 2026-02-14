.PHONY: help install sync dev worker up down logs test lint format typecheck check clean lock

# Default target
help:
	@echo "Anki Atlas - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies with uv (creates venv)"
	@echo "  sync        Sync dependencies from lockfile"
	@echo "  lock        Update uv.lock"
	@echo ""
	@echo "Development:"
	@echo "  dev         Run API server locally"
	@echo "  worker      Run arq background worker"
	@echo "  up          Start postgres + qdrant + redis via docker compose"
	@echo "  down        Stop docker compose services"
	@echo "  logs        Tail docker compose logs"
	@echo ""
	@echo "Quality:"
	@echo "  test        Run tests"
	@echo "  lint        Run ruff linter"
	@echo "  format      Format code with ruff"
	@echo "  typecheck   Run mypy type checker"
	@echo "  check       Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean       Remove build artifacts and caches"

# Setup
install:
	uv sync --all-extras

sync:
	uv sync

lock:
	uv lock

# Development
dev:
	uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run arq apps.worker.WorkerSettings

up:
	docker compose up -d postgres qdrant redis

down:
	docker compose down

logs:
	docker compose logs -f

# Quality
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov --cov-report=term-missing

lint:
	uv run ruff check apps packages tests

format:
	uv run ruff format apps packages tests
	uv run ruff check --fix apps packages tests

typecheck:
	uv run mypy apps packages

check: lint typecheck test

# Cleanup
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
