.PHONY: test

test:
	PYTHONWARNINGS=error::ResourceWarning uv run pytest --cov --cov-report=term-missing:skip-covered
