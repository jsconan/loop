.PHONY: run test test-unit test-integration test-coverage test-failure-injection \
	test-concurrency test-native-macos release-provenance lint coverage-reset coverage-report

PYTEST_RESOURCE_SAFE = PYTHONWARNINGS=error::ResourceWarning uv run pytest -W error::pytest.PytestUnraisableExceptionWarning

run:
	uv run loop

test: test-coverage

test-unit:
	$(PYTEST_RESOURCE_SAFE) --ignore=tests/integration --cov=loop --cov-report= --cov-fail-under=0

test-integration:
	$(PYTEST_RESOURCE_SAFE) tests/integration --cov=loop --cov-append --cov-report= --cov-fail-under=0

test-failure-injection:
	$(PYTEST_RESOURCE_SAFE) tests/execution/sandbox/macos/test_execution.py tests/execution/sandbox/macos/test_jobs.py tests/execution/vfs/test_commit_broker.py -q

test-concurrency:
	$(PYTEST_RESOURCE_SAFE) tests/execution/vfs/test_agent_workspace.py tests/execution/sandbox/macos/test_workspace.py -q

test-native-macos:
	test -n "$(STATE_ROOT)"
	uv run python scripts/verify_macos_execution_native.py --state-root "$(STATE_ROOT)"

release-provenance:
	uv run python scripts/validate_runtime_provenance.py

lint:
	uv run ruff check .
	uv run ruff format --check .
	git diff --check
	git diff --cached --check

test-coverage:
	$(MAKE) coverage-reset
	$(MAKE) test-unit
	$(MAKE) test-integration
	$(MAKE) coverage-report

coverage-reset:
	uv run coverage erase

coverage-report:
	uv run coverage report --show-missing --skip-covered --fail-under=100
