.PHONY: run test test-integration test-all

run:
	uv run loop

test:
	PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/unit --cov=loop --cov-report=term-missing:skip-covered

test-integration:
	PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/integration

test-all:
	PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/unit tests/integration --cov=loop --cov-report=term-missing:skip-covered
