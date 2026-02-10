.PHONY: help install dev up down logs test lint format typecheck check clean

# Default target
help:
	@echo "Anki Atlas - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install     Install dependencies with uv"
	@echo ""
	@echo "Development:"
	@echo "  dev         Run API server locally (requires deps installed)"
	@echo "  up          Start postgres + qdrant via docker compose"
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
	uv pip install -e ".[dev]"

# Development
dev:
	uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up -d postgres qdrant

down:
	docker compose down

logs:
	docker compose logs -f

# Quality
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov --cov-report=term-missing

lint:
	ruff check apps packages tests

format:
	ruff format apps packages tests
	ruff check --fix apps packages tests

typecheck:
	mypy apps packages

check: lint typecheck test

# Cleanup
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
