# Test suites

The test suites separate portable module behavior from operating-system and dependency integration. Unit tests must provide **100% statement and branch coverage independently**.

## Organization

- `unit/` mirrors `src/loop/` by owning module. It is the default pytest target and runs on every platform without platform skips. Platform adapters use controlled platform, process, account, clock and diagnostic doubles.
- `integration/` exercises native boundaries in disposable workspaces. Shared process and search checks live under `execution/` and `utils/`; platform-specific checks live under `macos/` and `linux/`.
- The fresh-process platform import check verifies that loading the app does not import the other operating system's sandbox implementation.
- Public dispatch through permissions and execution is additionally marked `e2e`.

Platform markers skip unsupported operating systems before native setup. macOS enforcement checks require a parent process that can initialize Seatbelt. Linux native enforcement requires explicit host provisioning; public execution checks exercise unavailable behavior without a qualified installation. Native ripgrep checks skip when the optional executable is absent. See [Linux native integration prerequisites](../docs/development.md#linux-native-integration) before running those checks; skips do not establish enforcement.

The platform integration suites are organized by behavior:

| Suite | Guarantees |
| --- | --- |
| `macos/test_sandbox.py` | Native execution, exit status, authority binding, filesystem confinement, aliases, protected paths, grants, relocation, detached children, network/IPC and descriptors |
| `macos/test_system.py` | Public command dispatch, virtual paths, destination-aware redirection and Git authorization, host recovery, scratch authority, output bounds and registration aliases |
| `macos/test_search.py` | Public search, Unicode columns, regex, binary handling, result bounds, executable lookup and outside symlinks |
| `linux/test_host.py` | Trusted host and installed-tool discovery |
| `linux/test_seccomp.py` | Native ABI and seccomp behavior |
| `linux/test_namespace.py` | Namespace setup and selected system inputs |
| `linux/test_installed_execution.py` | Installed public execution, supervision and admission boundaries |

Portable Linux unit tests exercise their source owners with controlled collaborators; they do not establish kernel enforcement. Scripts and their adjacent tests are outside the configured suites.

## Running tests

Install development dependencies and run the desired suite:

```sh
uv sync
make test                 # portable unit tests with strict coverage
make test-integration     # native and dependency integration
make test-all             # both suites with strict coverage
```

Run an individual module or select end-to-end cases:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/unit/tools/test_system.py
PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest tests/integration -m e2e
```

To reject unexpected Python warnings during verification, add `-W error` to a pytest command. Tests that exercise expected warnings, logged errors or exceptions must capture and assert the specific diagnostic using `pytest.warns`, `caplog` or `pytest.raises`. Do not suppress diagnostics to make a failing test pass.

## Isolation

Each case receives a fresh workspace, working directory, home, temporary directory, application storage roots and content cache. Environment variables, cache metadata, permission-manager lifecycle state and telemetry are scoped and restored. Construct mutable agents and other collaborators inside test fixtures rather than at collection time.

Unit tests cannot launch real processes, signal host processes, open network connections or listeners, or perform DNS. Replace those boundaries with scoped mocks or stubs. Timing, timeout, cancellation and retained-pipe checks use controlled clocks and synchronous doubles; they must not depend on real sleeps or scheduling delays.

The native process integration suite also exercises short flushed output held by a blocking reader until timeout cleanup. The supervisor snapshots bounded output after owned termination and bounded pipe draining, including on cancellation; stalled draining reports unknown discard counts. Portable unit cases reproduce this ordering without launching or waiting for real children.

Filesystem and storage tests use synthetic files and databases in disposable scratch, preserving native inode, symlink, hardlink, transaction and permission semantics. Fixture data is removed after every case, including failed cases and sibling paths. Ordinary physical fixtures avoid `/tmp`, which the application treats as a virtual command path. Native tests that require the system temporary root or the listener filesystem use `native_directory` to register and own one fresh directory at a time, with cleanup before its output registration is removed.

A Python audit hook rejects filesystem and SQLite output outside the owned roots, including unregistered siblings of native fixture directories. A scoped `os.open` guard also checks descriptor-relative writes. `/dev/null` is allowed as an output sink. Coverage output remains allowed outside per-case isolation. Pytest cache writing is disabled, and the Make targets disable Python bytecode writing before interpreter startup. Cleanup completes before the next case starts.

These guards apply to trusted test code; they are not an operating-system sandbox. Native extensions and integration children can bypass Python auditing. Integration commands must use only explicitly constructed disposable fixtures, without production credentials, user data or external services. Loopback and Unix-domain endpoints used for native network/IPC checks are owned and closed by the test. Native enforcement remains real; diagnostic log polling is stubbed and its behavior is covered by the unit suite.

## Coverage and checks

Coverage measures both statements and branches with a 100% threshold. Keep security, trust-boundary and error-path assertions active; do not lower thresholds or add exclusions to hide missing coverage.

After changing tests, run the affected modules, the complete unit suite with coverage, and applicable integration suites. Check formatting and whitespace:

```sh
uv run ruff check tests
uv run ruff format --check tests
git diff --check
```

Sandbox setup tests live beside their owners in `unit/execution/sandbox/linux/test_setup.py`, `unit/execution/sandbox/test_commands.py` and `linux/resources/test_setup.py`. They cover fixed lifecycle composition, original refresh authority, terminal-only fresh consent, reviewed whole-input authentication before privileged imports, bounded regular input, private snapshot cleanup and sanitized failures. Mocked unit collaborators do not prove native enforcement.
