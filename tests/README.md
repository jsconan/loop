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

Platform guards skip unsupported operating systems before native setup. Set `LOOP_NATIVE_MACOS_TEST=owned-volume` to run macOS enforcement checks; the ordinary integration run skips them without attaching a disk image. Native macOS qualification requires a parent process that can initialize Seatbelt and permission to attach, format, mount, and eject an owned 8 MiB RAM filesystem through `hdiutil` and `diskutil`; the integration fixture shares one disposable volume across cases and removes it at suite teardown. A surrounding tool sandbox that denies disk-image operations prevents native qualification, and a fixture failure after opt-in stops the suite. Set `LOOP_NATIVE_LINUX_TEST=installed` to run installed Linux enforcement checks; the ordinary integration run skips them without depending on a protected installation. Linux native qualification requires explicit host provisioning; public execution checks exercise unavailable behavior without a qualified installation. Native ripgrep checks skip when the optional executable is absent. See [Linux native integration prerequisites](../docs/development.md#linux-native-integration) before running those checks; skips do not establish enforcement.

The platform integration suites are organized by behavior:

| Suite | Guarantees |
| --- | --- |
| `macos/test_sandbox.py` | Native execution, exit status, authority binding, filesystem confinement, aliases, protected paths, grants, relocation, detached children, network/IPC and descriptors |
| `macos/test_writers.py` | Detached fork/spawn confinement, retained mmap and SQLite WAL resources, native ACL rejection and distinct-account access controls |
| `macos/test_system.py` | Public command dispatch, canonical paths across scripts/subprocesses/file tools, destination-aware redirection and Git authorization, host recovery, scratch authority, output bounds and registration aliases |
| `macos/test_command_jobs.py` | Public supervised continuation, live ordered output, session ownership, cancellation, interactive PTY controls, inner and foreign terminal isolation, descriptor receipt, and finite long-running commands |
| `macos/test_mediated_egress.py` | Public macOS Unix-SOCKS mediation, cross-attempt confinement, routed HTTP(S) and Git clients, and client compatibility |
| `macos/test_search.py` | Public search, Unicode columns, regex, binary handling, result bounds, executable lookup and outside symlinks |
| `linux/test_host.py` | Trusted host and installed-tool discovery |
| `linux/test_seccomp.py` | Native ABI and seccomp behavior |
| `linux/test_namespace.py` | Namespace setup and selected system inputs |
| `linux/test_apparmor_profile_lifetime.py` | Foreign attachment and live-profile removal and replacement behavior |
| `linux/test_installed_execution.py` | Installed public execution, supervised continuation, owned interactive terminals, cancellation, verified cleanup and admission boundaries |
| `linux/test_mediated_egress.py` | Public mediated HTTP(S) dispatch, isolated destinations, policy revisions, bounded authority loss, and representative clients against owned routed fixtures |
| `linux/test_spawn_disconnect.py` | Disconnect and watcher-failure behavior during a stalled spawn in an owned user-systemd service |

Portable Linux unit tests exercise their source owners with controlled collaborators; they do not establish kernel enforcement. Scripts and their adjacent tests are outside the configured suites.

## Running tests

Install development dependencies and run the desired suite:

```sh
uv sync
make test                 # architecture and portable unit tests with strict coverage
make test-integration     # portable integration; native qualification is opt-in
make test-all             # architecture, unit and integration tests with strict coverage
```

`make test` writes the `.coverage` data file in the checkout. A sandboxed command running it needs the default writable workspace scope (`read_only=false`).

The Make targets run pytest through `uv --no-cache run`, which may resolve dependencies and initialize a cache. They do not support a `PYTHON` override.

On a prepared native host, run `LOOP_NATIVE_MACOS_TEST=owned-volume make test-integration` or `LOOP_NATIVE_LINUX_TEST=installed make test-integration` to qualify its sandbox. These flags enable native checks but do not install anything or change host permissions.

Run an individual module or select end-to-end cases:

```sh
PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/unit/tools/test_system.py
PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration -m e2e
```

To reject unexpected Python warnings during verification, add `-W error` to a pytest command. Tests that exercise expected warnings, logged errors or exceptions must capture and assert the specific diagnostic using `pytest.warns`, `caplog` or `pytest.raises`. Do not suppress diagnostics to make a failing test pass.

## Native macOS writer prerequisites

The detached-writer checks compile a disposable native helper using `/usr/bin/clang` and the system SQLite library. They use bounded child alarms and kqueue process-exit events to settle owned writers, including after assertion failures. Command completion and ownership release do not establish descendant termination; retained mmap writes and SQLite WAL commits exercise that documented limitation while checking confinement.

The distinct-account ACL check requires existing noninteractive `sudo -n -u nobody` authority. It verifies that account's identity and a public read control before testing private access and ACL rejection. Without that authority the case skips; the suite does not change accounts, configure sudo, or request a password. A separate native ACL case runs without account switching and verifies that extended access lists are rejected even when mode bits remain `0700`.

The extended quiet and noisy command controls, plus the interactive terminal duration controls, in `macos/test_command_jobs.py` and `linux/test_installed_execution.py` require `LOOP_NATIVE_LONG_COMMAND_TEST=extended-native-check`. Run them explicitly on a host with its qualified native sandbox; each continuation or terminal handle retains its original command without replay.

The macOS public test-workflow case uses a small owned project during routine integration. Set `LOOP_NATIVE_FULL_PROJECT_TEST=extended-native-check` together with `LOOP_NATIVE_MACOS_TEST=owned-volume` to additionally run the copied real `make test` target through public sandbox dispatch; it checks the full project recipe without modifying the checkout.

## Native Linux test setup

Use a disposable test environment and a sandbox installation matching the package under test. Native checks must use fake data, without production credentials or external services. Missing prerequisites cause skips and do not establish enforcement.

| Variable | Required setup |
| --- | --- |
| `LOOP_NATIVE_BWRAP` | Explicitly selected, root-protected Bubblewrap helper for `linux/test_namespace.py`. |
| `LOOP_NATIVE_HOST_TEST` | Set to `disposable-linux-host` only on a disposable Linux machine to run the explicit unrestricted host-shell boundary case in `execution/test_coordinator.py`. It reads only synthetic test data. |
| `LOOP_NATIVE_INNER_ADVERSARIAL_TEST` | Set to `disposable-linux-host` only on a disposable Linux machine to probe inner namespace, mount, Bubblewrap, profile and cgroup attempts and detached-child cleanup. The ordinary inner-seccomp child and outer-boundary checks run without this flag. |
| `LOOP_NATIVE_RUNTIME_FIXTURE` | Administrator-provisioned, root-owned, read-only fake-data directory under `/usr/local/share/`, named `loop-native-private-fixture-*`. |
| `LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY` | Directory owned by the test user, with mode `0700`, on the sandbox listener’s filesystem for socket-alias checks. |
| `LOOP_NATIVE_ALTERNATIVE_CANARY` | Administrator-provisioned root-owned symlink under `/etc/alternatives/` named `loop-native-alternative-canary-*`, pointing to a present harmless system executable. |
| `LOOP_NATIVE_SYSTEM_ALIAS_FIXTURE` | Administrator-provisioned root-owned `bin` under `/usr/local/lib/loop-native-system-alias-*`, with a `loop-parent-tool` executable link routed through a separate `/etc/alternatives` directory link back into that `bin`. |
| `LOOP_NATIVE_MANAGED_POLICY_TEST` | Set to `disposable-linux-host` for the provisioned Linux `/etc/loop` fixture, `macos-root-fixture` for the macOS `/Library/Loop` fixture, or `owned-root-fixture` for the managed YOLO prohibition fixture described below. Remove the fixture after its isolated test. |
| `LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN`, `LOOP_NATIVE_EGRESS_HTTPS_FIXTURE_ORIGIN` | Owned HTTP and HTTPS origins routed to the synthetic egress fixture in a separate namespace. |
| `LOOP_NATIVE_EGRESS_MIXED_DNS_ORIGIN`, `LOOP_NATIVE_EGRESS_PRIVATE_DNS_ORIGIN` | Fixture names resolving to an allowed and a forbidden address, or only a forbidden address, respectively. |
| `LOOP_NATIVE_EGRESS_REBIND_DNS_ORIGIN`, `LOOP_NATIVE_EGRESS_REBIND_TEST` | The owned rebinding name and `disposable-linux-host` opt-in for changing only that marked host entry between two native connections. Set `LOOP_NATIVE_EGRESS_PRIVATE_IP` to the VM's private address. |
| `LOOP_NATIVE_EGRESS_SUPERVISOR_TEST` | Set to `disposable-linux-host` only on a disposable VM to enable the receiver-death control. |
| `LOOP_NATIVE_EGRESS_SUSPEND_TEST` | Set to `disposable-linux-host` only on a disposable VM to pause and resume the Loop policy-authority process while native tunnels are active. |

The protected administrator-policy integration tests use the actual fixed system anchors and must run separately from the general suites so their system-wide policy does not affect unrelated test managers. The expectation file has `{"version":1,"identity":"0123456789abcdef0123456789abcdef","minimum_revision":1}`. The policy file has `{"version":1,"identity":"0123456789abcdef0123456789abcdef","revision":1,"forbid_host":true,"deny_local_networks":true,"rules":[{"id":"native-deny-session","decision":"deny","action":"session.mutate"},{"id":"native-deny-egress","decision":"deny","action":"network.request"}]}`. On a disposable Linux VM with no pre-existing `/etc/loop` or `/etc/loop.managed.expect.json`, create a root-owned `0755` `/etc/loop` directory, a root-owned `0644` `/etc/loop.managed.expect.json` expectation, and a root-owned `0644` `/etc/loop/managed-policy.json` policy. Run `LOOP_NATIVE_MANAGED_POLICY_TEST=disposable-linux-host PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/permissions/test_managed_policy.py`, then remove only the fixture created for that run. On macOS, first confirm `/Library/Loop` and `/Library/loop.managed.expect.json` are absent and no other Loop process needs the optional policy. With administrator authentication, create root-owned `0755` `/Library/Loop`, root-owned `0644` `/Library/loop.managed.expect.json`, and root-owned `0644` `/Library/Loop/managed-policy.json` with those same contents. Run `LOOP_NATIVE_MANAGED_POLICY_TEST=macos-root-fixture PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/macos/test_system.py::test_root_managed_policy_controls_macos_yolo_host_and_native_network` outside an enclosing sandbox that prevents Seatbelt or the owned RAM volume. Remove only the exact two files and the directory created for the fixture, including after a failed test; recheck that the fixed source is absent. A missing fixture or skipped case is not active-policy qualification. For the managed YOLO prohibition, use the same expectation file and a separate root-owned `0644` policy file containing `{"version":1,"identity":"0123456789abcdef0123456789abcdef","revision":1,"forbid_yolo":true}`. With the same isolation and cleanup precautions, run `LOOP_NATIVE_MANAGED_POLICY_TEST=owned-root-fixture PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/permissions/test_managed_policy.py::test_native_root_policy_prohibits_yolo_activation`. The test reads the protected fixture without changing it; remove only the fixture that was provisioned for this run and verify both fixed anchors are absent afterward. Policy changes and trusted reload are covered by isolated portable tests.

For the mediated-egress native suite, run `LOOP_NATIVE_EGRESS_PRIVATE_IP=<disposable-VM-private-IPv4> sh tests/integration/linux/egress_fixture_setup.sh setup` on an otherwise unused disposable VM with noninteractive `sudo` for the owned `ip`, `systemd-run`, and `/etc/hosts` changes. The script creates a separate network namespace and synthetic HTTP, HTTPS, package-index, and Git endpoints at `11.0.0.2`, with mixed and private test names in `/etc/hosts`. Set `LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN=http://11.0.0.2:18080`, `LOOP_NATIVE_EGRESS_HTTPS_FIXTURE_ORIGIN=https://11.0.0.2:18443`, `LOOP_NATIVE_EGRESS_MIXED_DNS_ORIGIN=http://loop-egress-mixed.invalid:18080`, and `LOOP_NATIVE_EGRESS_PRIVATE_DNS_ORIGIN=http://loop-egress-private.invalid:18080`; then run `pytest tests/integration/linux/test_mediated_egress.py`. Enable the supervisor-death, process-suspension, and DNS-rebinding variables only on a disposable host; the last also needs `LOOP_NATIVE_EGRESS_REBIND_DNS_ORIGIN=http://loop-egress-rebind.invalid:18080` and the setup address left in the environment. Afterward run `sh tests/integration/linux/egress_fixture_setup.sh cleanup`, which stops the two owned services, removes its network namespace and marked host entries, and removes only its owned temporary fixture directory. A skipped origin or unavailable protected installation is a missing prerequisite, not a mediated-egress pass.

For positive macOS mediation checks, use an otherwise unused disposable Ubuntu VM on a host-only network. Run `sh tests/integration/macos/egress_fixture_setup.sh setup` on the VM; it creates synthetic HTTP, HTTPS, package-index and Git endpoints bound to `11.0.0.2` and changes only the VM's owned loopback alias and named services. On macOS, add a temporary host route with administrator authentication using `sudo /sbin/route -n add -host 11.0.0.2 <VM-private-IPv4>` and confirm `/sbin/route -n get -inet 11.0.0.2` uses the VM-facing, non-loopback interface. Run `LOOP_MACOS_EGRESS_FIXTURE_GATEWAY=<VM-private-IPv4> PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/macos/test_mediated_egress.py`. The fixture verifies the exact route gateway and the owned HTTP marker before any positive proxy test. Remove the exact host route with `sudo /sbin/route -n delete -host 11.0.0.2`, then run the fixture script with `cleanup` on the VM. A skipped route or unavailable owned endpoint is not a positive mediation pass; do not substitute a real public service or a host-local address for this fixture.

The macOS direct-resolver control additionally needs an owned VM DNS listener and a temporary administrator-installed resolver file. Run `LOOP_MACOS_EGRESS_PRIVATE_IP=<VM-private-IPv4> sh tests/integration/macos/egress_fixture_setup.sh dns-setup` on the VM. After confirming `/private/etc/resolver/loop-egress-owned.test` does not already exist and `/private/etc/resolver` is a real root-owned directory if present, create that one root-owned `0644` file on macOS containing exactly `nameserver <VM-private-IPv4>` followed by a newline. Set both `LOOP_MACOS_EGRESS_FIXTURE_GATEWAY` and `LOOP_MACOS_EGRESS_DNS_GATEWAY` to that VM address for the native suite. The test requires fresh host libc and DNSService queries to reach the owned listener, then checks that equivalent unique sandboxed queries add no listener records. Remove only that resolver file and the resolver directory if this fixture created it and it is empty; run `dns-cleanup` on the VM before the ordinary `cleanup`. A missing DNS fixture is an unqualified direct-resolver control, not a denial pass.

The separate `linux/egress_lifecycle_probe.py` exercises controls that cannot safely run inside the same pytest process. With the owned HTTPS fixture active, start `python tests/integration/linux/egress_lifecycle_probe.py control-death` in one external shell, then start the same script with `death` in another; the controller waits for an established tunnel, kills only that probe's ordinary Loop process, and requires both the host transport and receiver to stop within two seconds. For a whole-VM suspend/resume check, start the script with `suspend`, wait for `READY`, suspend and resume only that disposable VM using its hypervisor, then require `PASS resumed stream completed`. Keep the VM pause within the fixture's streaming interval. These controls create `/tmp/loop-egress-death-probe` and `/tmp/loop-egress-suspend-probe`; remove only those owned directories after inspection. Run them separately from the ordinary suite so no other Loop receiver or fixture traffic confuses the external observations.

The runtime fixture must contain the following synthetic data. Provision it with explicit administrator authentication, and remove only that owned fixture after testing. The tests neither create nor modify it and request no extra read grant, so the runtime sandbox must enforce the private-file restrictions.

| Entry | Contents |
| --- | --- |
| Marker file `marker` | `Loop private-runtime qualification fixture v1` followed by a newline. |
| `.ssh/key`, `nested/.AWS/key`, `.config/key`, `.loop/key`, `Library/key` | `private-runtime-canary` |
| Executable `.ssh/probe` | Script printing `private-runtime-canary`. |
| `neighbor` | `ordinary-runtime-canary` |

The profile lifecycle stress case in `linux/test_installed_execution.py` requires `LOOP_NATIVE_CAPACITY_TEST=disposable-linux-host`. It exercises more than three times the profile capacity with distinct completed requests and verifies that enforcing policy remains bounded without a refresh.

The external Python environment approval cases require an active test interpreter whose base runtime is outside the fixed system runtime mounts. A virtual environment based on `/usr` cannot exercise that approval boundary. The indirect build-tool cases require the system `make` executable; absent tools are reported as skipped prerequisites.

Run `linux/test_management_lifecycle.py` separately with `pytest -s`, in a real terminal on a disposable Linux VM from an isolated interpreter with the reviewed wheel, pytest and pip installed. Set `LOOP_NATIVE_MANAGEMENT_TEST=disposable-linux-host` and `LOOP_NATIVE_MANAGEMENT_WHEEL` to the wheel being tested. This test requires fresh consent and administrator authentication. It exercises installation, removal, idle tracking and boot recovery, live-child protection, concurrent admission, partial-state recovery, and uninstall after package removal. Keep the wheel and output directory outside the repository.

The AppArmor profile-lifetime check also requires `LOOP_NATIVE_MANAGEMENT_TEST=disposable-linux-host` and noninteractive administrator access. It creates isolated policy namespaces and test processes, verifies foreign attachment and live-profile removal and replacement behavior, and removes its fixtures.

The stalled-spawn disconnect check requires a disposable Linux host with a working per-user systemd manager. Set `LOOP_NATIVE_WATCHDOG_TEST=disposable-linux-host` to allow it to create, stop, and clear its own uniquely named transient user services. It does not exercise the installed receiver.

## Isolation

Filesystem identity cases require an owned, writable scratch directory with mode `0700` and no extended access list. The session fixture validates ownership, mode, directory identity and access lists before running cases; on Linux it also rejects inheritable default ACLs. Physical scratch paths may contain spaces, Unicode and private-name ancestors; synthetic policy identities retain their own validation. By default the suite creates and removes a private session directory under the inherited system temporary directory. To choose another location, create a private directory and set `LOOP_TEST_SCRATCH_ROOT` to its absolute path before starting pytest. The directory must already exist; the suite creates and removes fresh private children inside it and leaves the supplied root intact. Native Unix-domain socket cases may require a shorter path. These fixture rules do not change production path validation.

Each case receives a fresh workspace, working directory, home, temporary directory, application storage roots and content cache. Environment variables, cache metadata, permission-manager lifecycle state and telemetry are scoped and restored. Unit cases also own fresh fallback worker admission and executors, and close tool registries before removing their scratch. Construct mutable agents and other collaborators inside test fixtures rather than at collection time. Permission-catalog consumers receive fresh copies of validated package fixtures; the owning model suite exercises the actual catalog loader.

Tests that construct interpreter shebangs require an owned writable path without whitespace because the operating system separates the interpreter path at whitespace. When scratch contains whitespace, these fixtures use a disposable directory under the checkout or the conventional temporary directory and remove it after the case.

Unit tests cannot launch real processes, signal host processes, load native libraries, open network connections or listeners, or perform DNS. Scoped doubles and a Python audit guard reject these calls, including retained native function aliases. Default prompt sessions fail immediately on unconfigured input; account discovery and ACL inspection use synthetic collaborators. Replace these boundaries with explicit test responses. Timing, timeout, cancellation and retained-pipe checks use controlled clocks, futures and in-memory streams; they must not depend on real sleeps or scheduling delays. Application and case clocks are synthetic; real sleep calls fail immediately. Asynchronous delivery checkpoints use queued callbacks without timers. Telemetry unit cases drive queue responses and lifecycle completion directly; native writer intervals, executor ownership, cancellation settlement and holder thread stress are checked in integration.

The native process integration suite also exercises short flushed output held by a blocking reader until timeout cleanup. The supervisor snapshots bounded output after owned termination and bounded pipe draining, including on cancellation; stalled draining reports unknown discard counts. Portable unit cases reproduce this ordering without launching or waiting for real children.

Filesystem and storage fixtures use owned disposable scratch with real local files and SQLite databases. Fixture output, including sibling paths, is removed after each case. Native inode, descriptor, symlink, hardlink, journal and thread-contention guarantees belong in integration suites. Native cases requiring the system temporary root or listener filesystem use `native_directory` to register one fresh directory at a time, with cleanup before its output registration is removed.

A Python audit hook rejects filesystem and SQLite output outside the owned roots, including unregistered siblings of native fixture directories. A scoped `os.open` guard also checks descriptor-relative writes. `/dev/null` is allowed as an output sink. Coverage output remains allowed outside per-case isolation. Pytest cache writing is disabled, and the Make targets disable Python bytecode writing before interpreter startup. Cleanup completes before the next case starts.

Fixture environment and resource guards have independent patch scopes, so undoing a case's mocks cannot remove them. Filesystem fixtures need authority to create their synthetic names; an enclosing sandbox's private-name denials still apply inside disposable scratch.

These guards apply to trusted test code; they are not an operating-system sandbox. Native extensions and integration children can bypass Python auditing. Integration commands must use only explicitly constructed disposable fixtures, without production credentials, user data or external services. Loopback and Unix-domain endpoints used for native network/IPC checks are owned and closed by the test. Native enforcement remains real; diagnostic log polling is stubbed and its behavior is covered by the unit suite. Linux continuity children use the installed receiver's system temporary root for fresh probe scratch while keeping saved working files in synthetic private storage. Native virtual-environment checks inspect the active interpreter environment, including environments selected outside the default project directory.

## Coverage and checks

Coverage measures both statements and branches with a 100% threshold. Keep security, trust-boundary and error-path assertions active; do not lower thresholds or add exclusions to hide missing coverage.

After changing tests, run the affected modules, the complete unit suite with coverage, and applicable integration suites. Check formatting and whitespace:

```sh
uv --no-cache run ruff check tests
uv --no-cache run ruff format --check tests
git diff --check
```

Sandbox setup tests live beside their owners in `unit/execution/sandbox/linux/test_setup.py`, `unit/execution/sandbox/test_commands.py` and `linux/resources/test_setup.py`. They cover fixed lifecycle composition, original refresh authority, terminal-only fresh consent, reviewed whole-input authentication before privileged imports, bounded regular input, private snapshot cleanup and sanitized failures. Mocked unit collaborators do not prove native enforcement.
