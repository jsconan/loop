PYTHON ?= .venv/bin/python

.PHONY: run test test-architecture test-integration test-all

run:
	uv run loop

test: test-architecture
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m pytest tests/unit --cov=loop --cov-report=term-missing:skip-covered

test-architecture:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m pytest tests/architecture

test-integration:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m pytest tests/integration

test-all:
	PYTHONDONTWRITEBYTECODE=1 "$(PYTHON)" -m pytest tests/architecture tests/unit tests/integration --cov=loop --cov-report=term-missing:skip-covered
