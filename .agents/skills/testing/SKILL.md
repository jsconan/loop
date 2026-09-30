---
name: testing
description: Write, update, reorganize, diagnose, and verify isolated pytest unit and integration tests. Activate whenever a change adds, modifies, or removes tests or test fixtures, or when the task verifies behavior, mocks, fixtures, edge and error cases, or the repository's 100% statement-and-branch coverage requirement, including running the suite or coverage to validate any other change.
---

# Testing

Preserve 100% statement and branch coverage for active source code. Test supported behavior, edge
cases, error paths, and branches. Never lower, bypass, exclude, or weaken coverage requirements to
make checks pass. Do not test passive `__init__.py` barrel files unless they contain active
behavior.

## Place tests by ownership

Mirror the package structure beneath `src/loop/` in `tests/unit/` and keep exactly one unit suite
for each active source module:

```text
src/loop/client.py          -> tests/unit/test_client.py
src/loop/loop.py            -> tests/unit/test_loop.py
src/loop/tools/files.py     -> tests/unit/tools/test_files.py
src/loop/types/tooling.py   -> tests/unit/types/test_tooling.py
src/loop/utils/tooling.py   -> tests/unit/utils/test_tooling.py
```

Place unit behavior in the suite belonging to the module that owns it. Do not split one module's
unit tests across suites or mix unrelated source modules in one suite. Order cases by the source module's
declaration order and keep cases for the same declaration together. Use empty `__init__.py` files
only when mirrored test directories must be importable; do not place tests or active code in them.

Put native or dependency integration and end-to-end checks under `tests/integration/`, grouped
by feature, module or package. Mark platform-specific cases and skip unsupported platforms before
native fixture setup. Keep unit tests portable and free of platform skips.

Test filenames, names, docstrings, fixtures and documentation must describe durable behavior.
Do not reference transient plans, milestones, review history or the origin of a test. Preserve
out-of-scope unversioned scripts and their tests; do not move or collect them without authorization.

## Test observable behavior

- Exercise public interfaces; do not import private helpers or inspect private members.
- Build realistic payloads, responses, events, and tool calls that reach behavior naturally.
- Assert public results, emitted output, forwarded requests, filesystem effects, or interactions
  with injected dependencies.
- Mock external dependencies and collaborators at their public boundary. Test collaborator
  internals only in their owning suite.
- Test module-level helpers in their defining module's suite, not again through every consumer.

Fix unexpected warnings and errors at their cause. Capture intended warnings with `pytest.warns`,
logged errors with `caplog`, and exceptions with `pytest.raises`; assert the specific diagnostic.
Do not add broad warning/error filters, exclusions or permissive assertions to hide failures.

## Document test intent

- Add a concise module docstring describing the suite and a concise docstring to every test,
  fixture, and test helper.
- Describe the behavior or guarantee being exercised, not the test's implementation steps.
- When changing a test's scope or expected outcome, update its docstring so it still describes the
  complete behavior covered by the final test.

## Keep tests isolated

Make every test deterministic, idempotent, standalone, and independent of execution order or the
external environment. Ensure each test suite passes when run independently. Do not share mutable
state between tests. Create fresh registries, clients, mocks, payloads, and temporary paths for
each test. Use `tmp_path` for filesystem behavior and `monkeypatch` or scoped mocks for environment
variables, user input, time-sensitive dependencies, SDK clients, and other external boundaries.
Unit tests must not launch real processes or use real networks or services. Use controlled clocks,
mocks and stubs for timing; do not sleep or wait for elapsed time. Integration tests may exercise
real native boundaries only with owned disposable fixtures, including local endpoints, and must
close resources and remove fixture output after success or failure. Neither suite may affect
production resources or leave data behind, apart from requested coverage or monitoring output.
Restore patched state automatically with fixtures or scoped context managers.

## Verify in increasing scope

Run the affected suite independently first:

```shell
uv run pytest tests/path/to/test_module.py
```

Then run the complete suite with strict coverage:

```shell
uv run pytest --cov=loop --cov-report=term-missing:skip-covered --cov-fail-under=100
```

After changing tests, run:

```shell
uv run ruff format tests
uv run ruff check tests
git diff --check
```
