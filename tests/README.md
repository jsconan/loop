# Test suites

The test suites separate portable module behavior from operating-system and dependency integration. Unit tests must provide **100% statement and branch coverage independently**.

## Organization

- `architecture/` holds repository-wide static naming and import-boundary contracts that do not belong to an individual source module's unit suite. `make test` runs these checks with the unit suite.
- `unit/` mirrors `src/loop/` by owning module. It is the default pytest target and runs on every platform without platform skips. Platform adapters use controlled platform, process, account, clock and diagnostic doubles.
- `integration/` exercises native boundaries in disposable workspaces. Shared process and search checks live under `execution/` and `utils/`; `session/` verifies durable working files across process exits and resumes, cross-process ownership, crash lock release, independent-session concurrency and close-on-exec ownership descriptors; platform-specific checks live under `macos/` and `linux/`.
- Public dispatch through permissions and execution is additionally marked `e2e`.
- Agent runner cases simulate saves during model generation and exercise stale-edit recovery, partial reads, same-batch rereads, and isolated session observations. Tooling cases verify snapshot versions, visible byte coverage, trusted read receipts, immutable cached continuations, deletion and recreation, and interrupted batch recovery.
- Response collection cases validate terminal metadata and malformed streams independently of a UI. Presentation cases cover channel ordering and streamed-text deduplication. Session cases apply the same checks to synchronous and asynchronous streams, including cancellation and output finalization. Console cases cover synchronous and asynchronous choice validation and input. `integration/interaction/` exercises native prompt-toolkit input with owned pipes, captured output, and cancellation followed by prompt reuse.
- Conversation controller cases verify ordered mailbox admission, priority steering, scoped interruption, isolated approval responses, UI thread ownership, concurrent output backpressure, and shutdown cleanup. Runner cases verify stream closure, settled tool statistics, cancellation results, and guards after approval. Loop cases verify correction validation, session commands, persistence, and telemetry correlation through the controller. Session storage cases also cover database and workspace namespaces, concurrent first access, selection during active execution, stale saves, no-follow directory creation, durable generation recovery, adoption, retirement, ACL validation, and failed-constructor cleanup. Native session acceptance uses synthetic private paths with spaces, Unicode and shell metacharacters; verifies canonicalization and SQLite; and distinguishes confinement from detached-descendant termination on macOS. `integration/conversation/` exercises native queued input, steering and interruption, and preserves unsent drafts while approval is pending.

Platform markers skip unsupported operating systems before native setup. macOS enforcement checks require a parent process that can initialize Seatbelt. Linux native enforcement requires explicit host provisioning; public execution checks exercise unavailable behavior without a qualified installation. Native ripgrep checks skip when the optional executable is absent. See [Linux native integration prerequisites](../docs/development.md#linux-native-integration) before running those checks; skips do not establish enforcement.

The platform integration suites are organized by behavior:

| Suite | Guarantees |
| --- | --- |
| `macos/test_sandbox.py` | Native execution, exit status, authority binding, filesystem confinement, aliases, protected paths, grants, relocation, detached children, network/IPC and descriptors |
| `macos/test_writers.py` | Detached fork/spawn confinement, retained mmap and SQLite WAL resources, native ACL rejection and distinct-account access controls |
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

The Make targets disable the persistent `uv` cache and dependency synchronization so verification uses the installed environment and writable temporary storage.

Run an individual module or select end-to-end cases:

```sh
PYTHONDONTWRITEBYTECODE=1 uv --no-cache run --no-sync pytest tests/unit/tools/test_system.py
PYTHONDONTWRITEBYTECODE=1 uv --no-cache run --no-sync pytest tests/integration -m e2e
```

To reject unexpected Python warnings during verification, add `-W error` to a pytest command. Tests that exercise expected warnings, logged errors or exceptions must capture and assert the specific diagnostic using `pytest.warns`, `caplog` or `pytest.raises`. Do not suppress diagnostics to make a failing test pass.

## Native macOS writer prerequisites

The detached-writer checks compile a disposable native helper using `/usr/bin/clang` and the system SQLite library. They use bounded child alarms and kqueue process-exit events to settle owned writers, including after assertion failures. Command completion and ownership release do not establish descendant termination; retained mmap writes and SQLite WAL commits exercise that documented limitation while checking confinement.

The distinct-account ACL check requires existing noninteractive `sudo -n -u nobody` authority. It verifies that account's identity and a public read control before testing private access and ACL rejection. Without that authority the case skips; the suite does not change accounts, configure sudo, or request a password. A separate native ACL case runs without account switching and verifies that extended access lists are rejected even when mode bits remain `0700`.

## Native Linux test setup

Use a disposable test environment and a sandbox installation matching the package under test. Native checks must use fake data, without production credentials or external services. Missing prerequisites cause skips and do not establish enforcement.

| Variable | Required setup |
| --- | --- |
| `LOOP_NATIVE_BWRAP` | Explicitly selected, root-protected Bubblewrap helper for `linux/test_namespace.py`. |
| `LOOP_NATIVE_RUNTIME_FIXTURE` | Administrator-provisioned, root-owned, read-only fake-data directory under `/usr/local/share/`, named `loop-native-private-fixture-*`. |
| `LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY` | Directory owned by the test user, with mode `0700`, on the sandbox listener’s filesystem for socket-alias checks. |

The runtime fixture must contain the following synthetic data. Provision it with explicit administrator authentication, and remove only that owned fixture after testing. The tests neither create nor modify it and request no extra read grant, so the runtime sandbox must enforce the private-file restrictions.

| Entry | Contents |
| --- | --- |
| Marker file `marker` | `Loop private-runtime qualification fixture v1` followed by a newline. |
| `.ssh/key`, `nested/.AWS/key`, `.config/key`, `.loop/key`, `Library/key` | `private-runtime-canary` |
| Executable `.ssh/probe` | Script printing `private-runtime-canary`. |
| `neighbor` | `ordinary-runtime-canary` |

The capacity case in `linux/test_installed_execution.py` requires `LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host` and an installation with unused profile slots. It fills every available slot, checks refusal at the ceiling, and verifies reuse of existing profiles. Refresh the installation before further native testing.

Run `linux/test_management_lifecycle.py` separately with `pytest -s`, in a real terminal on a disposable Linux VM. Set `LOOP_NATIVE_MANAGEMENT_TEST=disposable-linux-host` and `LOOP_NATIVE_MANAGEMENT_WHEEL` to the wheel being tested. This test requires fresh consent and administrator authentication. It exercises installation, removal, live-child protection, concurrent admission, partial-state recovery, and uninstall after package removal. Keep the wheel and output directory outside the repository.

## Isolation

Each case receives a fresh workspace, working directory, home, temporary directory, application storage roots and content cache. Environment variables, cache metadata, permission-manager lifecycle state and telemetry are scoped and restored. Unit cases also own fresh fallback worker admission and executors, and close tool registries before removing their scratch. Construct mutable agents and other collaborators inside test fixtures rather than at collection time. Permission-catalog consumers receive fresh copies of validated package fixtures; the owning model suite exercises the actual catalog loader.

Unit tests cannot launch real processes, signal host processes, load native libraries, open network connections or listeners, or perform DNS. Scoped doubles and a Python audit guard reject these calls, including retained native function aliases. Default prompt sessions fail immediately on unconfigured input; account discovery and ACL inspection use synthetic collaborators. Replace these boundaries with explicit test responses. Timing, timeout, cancellation and retained-pipe checks use controlled clocks, futures and in-memory streams; they must not depend on real sleeps or scheduling delays. Thread synchronization uses observable handshakes with bounded failure waits and joins.

The native process integration suite also exercises short flushed output held by a blocking reader until timeout cleanup. The supervisor snapshots bounded output after owned termination and bounded pipe draining, including on cancellation; stalled draining reports unknown discard counts. Portable unit cases reproduce this ordering without launching or waiting for real children.

Filesystem and storage tests use synthetic files and databases in disposable scratch, preserving native inode, symlink, hardlink, transaction and permission semantics. Fixture data is removed after every case, including failed cases and sibling paths. Ordinary physical fixtures use owned scratch; `/tmp` remains an operating-system path and session file tools use `$TMPDIR`. Native tests that require the system temporary root or the listener filesystem use `native_directory` to register and own one fresh directory at a time, with cleanup before its output registration is removed.

A Python audit hook rejects filesystem and SQLite output outside the owned roots, including unregistered siblings of native fixture directories. A scoped `os.open` guard also checks descriptor-relative writes. `/dev/null` is allowed as an output sink. Coverage output remains allowed outside per-case isolation. Pytest cache writing is disabled, and the Make targets disable Python bytecode writing before interpreter startup. Cleanup completes before the next case starts.

Filesystem cases require scratch where synthetic `.loop` and `.git` entries can be created. An enclosing sandbox's name-based denials still apply within owned scratch.

These guards apply to trusted test code; they are not an operating-system sandbox. Native extensions and integration children can bypass Python auditing. Integration commands must use only explicitly constructed disposable fixtures, without production credentials, user data or external services. Loopback and Unix-domain endpoints used for native network/IPC checks are owned and closed by the test. Native enforcement remains real; diagnostic log polling is stubbed and its behavior is covered by the unit suite. Linux continuity children use the installed receiver's system temporary root for fresh probe scratch while keeping saved working files in synthetic private storage. Native virtual-environment checks inspect the active interpreter environment, including environments selected outside the default project directory.

## Coverage and checks

Coverage measures both statements and branches with a 100% threshold. Keep security, trust-boundary and error-path assertions active; do not lower thresholds or add exclusions to hide missing coverage.

After changing tests, run the affected modules, the complete unit suite with coverage, and applicable integration suites. Check formatting and whitespace:

```sh
uv run ruff check tests
uv run ruff format --check tests
git diff --check
```

Sandbox setup tests live beside their owners in `unit/execution/sandbox/linux/test_setup.py`, `unit/execution/sandbox/test_commands.py` and `linux/resources/test_setup.py`. They cover fixed lifecycle composition, original refresh authority, terminal-only fresh consent, reviewed whole-input authentication before privileged imports, bounded regular input, private snapshot cleanup and sanitized failures. Mocked unit collaborators do not prove native enforcement.
