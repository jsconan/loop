---
name: testing
description: Write, update, reorganize, diagnose, and verify isolated pytest unit tests. Use whenever work affects tests, test placement, mocks or fixtures, behavioral coverage, edge and error cases, or this repository's 100% statement-and-branch coverage requirement.
---

# Testing

Preserve 100% statement and branch coverage for active source code. Test supported behavior, edge
cases, error paths, and branches. Never lower, bypass, exclude, or weaken coverage requirements to
make checks pass. Do not test passive `__init__.py` barrel files unless they contain active
behavior.

## Place tests by ownership

Mirror the package structure beneath `src/loop/` in `tests/` and keep exactly one suite for each
active source module:

```text
src/loop/client.py          -> tests/test_client.py
src/loop/loop.py            -> tests/test_loop.py
src/loop/tools/files.py     -> tests/tools/test_files.py
src/loop/types/tooling.py   -> tests/types/test_tooling.py
src/loop/utils/tooling.py   -> tests/utils/test_tooling.py
```

Place behavior in the suite belonging to the module that owns it. Do not split one module's tests
across suites or mix unrelated source modules in one suite. Order cases by the source module's
declaration order and keep cases for the same declaration together. Use empty `__init__.py` files
only when mirrored test directories must be importable; do not place tests or active code in them.

## Test observable behavior

- Exercise public interfaces; do not import private helpers or inspect private members.
- Build realistic payloads, responses, events, and tool calls that reach behavior naturally.
- Assert public results, emitted output, forwarded requests, filesystem effects, or interactions
  with injected dependencies.
- Mock external dependencies and collaborators at their public boundary. Test collaborator
  internals only in their owning suite.
- Test module-level helpers in their defining module's suite, not again through every consumer.

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
Never require real networks or locally running services. Restore patched state automatically with
fixtures or scoped context managers.

## Verify in increasing scope

Run the affected suite independently first:

```shell
.venv/bin/pytest tests/path/to/test_module.py
```

Then run the complete suite with strict coverage:

```shell
.venv/bin/pytest --cov=loop --cov-report=term-missing --cov-fail-under=100
```

After changing tests, run:

```shell
.venv/bin/ruff format tests
.venv/bin/ruff check tests
git diff --check
```
