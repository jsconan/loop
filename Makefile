.PHONY: run test

run:
	uv run loop

test:
	PYTHONWARNINGS=error::ResourceWarning uv run pytest --cov --cov-report=term-missing:skip-covered
