.PHONY: run check format test test-architecture test-integration test-all

run:
	uv run loop

check:
	uv --no-cache run ruff check .
	uv --no-cache run ruff format --check .
	git --no-pager diff --check
	uv --no-cache run pylint src

format:
	uv --no-cache run ruff format .

test: test-architecture
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/unit --cov=loop --cov-report=term-missing:skip-covered

test-architecture:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/architecture

test-integration:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration

test-all:
	PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/architecture tests/unit tests/integration --cov=loop --cov-report=term-missing:skip-covered
