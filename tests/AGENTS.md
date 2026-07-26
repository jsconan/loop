# Test Suite Guidelines

These rules apply to every file under `tests/`.

## Coverage

- Maintain 100% statement and branch coverage for active source code.
- Add comprehensive tests for every supported behavior, edge case, error path, and branch.
- Do not lower, bypass, exclude, or otherwise weaken coverage requirements to make a check pass.
- Do not add tests for passive barrel files such as `__init__.py` files that only re-export names.
  Test a barrel file only when it contains active behavior.

## Structure and ownership

- Mirror the package structure beneath `src/loop/` in `tests/`.
- Keep exactly one test suite for each active source module.
- Order test cases to follow the declaration order of the source module's public classes,
  functions, methods, and constants; keep multiple cases for the same declaration together.
- Place every test in the suite corresponding to the source module that owns the behavior.
- Do not spread one module's tests across multiple suites.
- Do not mix tests for unrelated source modules in one suite.
- Use empty `__init__.py` files only where needed to make mirrored test directories importable; do
  not place tests or active code in them.

Examples:

```text
src/loop/client.py          -> tests/test_client.py
src/loop/loop.py            -> tests/test_loop.py
src/loop/tools/tools.py     -> tests/tools/test_tools.py
src/loop/types/tooling.py   -> tests/types/test_tooling.py
src/loop/utils/tooling.py   -> tests/utils/test_tooling.py
```

## Test boundaries

- Test behavior through public interfaces.
- Do not import private helpers.
- Do not access, mutate, or assert directly against private members.
- Prefer forging realistic input payloads, response objects, events, and tool calls that trigger
  the behavior under test naturally.
- Assert observable results, public return values, emitted output, forwarded requests, filesystem
  effects, or interactions with injected dependencies.
- Test module-level helpers from their defining module in that module's corresponding suite. Do not
  duplicate their behavioral tests in consumer-module suites.

## Isolation and repeatability

- Every test must be standalone, deterministic, idempotent, and isolated from test execution order and external environment.
- Each test suite must pass when run independently.
- Do not share mutable state between tests.
- Use fresh registries, clients, mocks, payloads, and temporary paths in each test.
- Use `tmp_path` for filesystem tests and `monkeypatch` or scoped mocks for environment variables,
  user input, time-sensitive dependencies, SDK clients, and other external boundaries.
- Do not make real network calls or depend on locally running services.
- Restore patched state automatically by using pytest fixtures or scoped context managers.

## Verification

Run the complete suite with strict coverage enforcement:

```shell
.venv/bin/pytest --cov=loop --cov-report=term-missing --cov-fail-under=100
```

Run an affected suite directly to confirm that it is independently collectible and executable:

```shell
.venv/bin/pytest tests/path/to/test_module.py
```

Run quality checks after changing tests:

```shell
.venv/bin/ruff check tests
git diff --check
```
