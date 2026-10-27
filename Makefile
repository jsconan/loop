.PHONY: run test test-architecture test-integration test-all

run:
	uv run loop

test: test-architecture
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/unit --cov=loop --cov-report=term-missing:skip-covered

test-architecture:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/architecture

test-integration:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration

test-all:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/architecture tests/unit tests/integration --cov=loop --cov-report=term-missing:skip-covered
