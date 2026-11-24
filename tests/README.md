# Test suites

The test suites separate portable module behavior from operating-system and dependency integration. Unit tests must provide **100% statement and branch coverage independently**.

Changed behavior also requires an ad-hoc smoke test through a realistic application flow, using real configuration layout, interaction and relevant native boundaries. Compare observable results and effects with the requested behavior and applicable direct controls. Automated unit and regression checks support this evidence; they do not replace it. Report unavailable smoke prerequisites as unverified.

## Organization

- `architecture/` holds repository-wide static naming and import-boundary contracts that do not belong to an individual source module's unit suite. `make test` runs these checks with the unit suite.
- `unit/` mirrors `src/loop/` by owning module. It is the default pytest target and runs on every platform without platform skips. Platform adapters use controlled platform, process, account, clock and diagnostic doubles.
- `integration/` exercises native boundaries in disposable workspaces. Shared process and search checks live under `execution/` and `utils/`; `session/` verifies durable working files across process exits and resumes, cross-process ownership, crash lock release, independent-session concurrency and close-on-exec ownership descriptors; platform-specific checks live under `macos/` and `linux/`.
- Public dispatch through permissions and execution is additionally marked `e2e`.
- Agent runner cases simulate saves during model generation and exercise stale-edit recovery, partial reads, same-batch rereads, and isolated session observations. Tooling cases verify snapshot versions, visible byte coverage, trusted read receipts, immutable cached continuations, deletion and recreation, and interrupted batch recovery.
- Direct-file cases cover environment-value names, public template exceptions, trusted designations, policy changes during reads, private continuation provenance, and no-follow descriptor binding. `integration/permissions/test_direct_file_boundaries.py` exercises real filesystem aliases, inode reuse and separate-process path/parent swaps on macOS and Linux without requiring a native command backend.
- Response collection cases validate terminal metadata and malformed streams independently of a UI. Presentation cases cover channel ordering and streamed-text deduplication. Session cases apply the same checks to synchronous and asynchronous streams, including cancellation and output finalization. Console cases cover synchronous and asynchronous choice validation and input. `integration/interaction/` exercises native prompt-toolkit input with owned pipes, captured output, and cancellation followed by prompt reuse.
- Conversation controller cases verify ordered mailbox admission, priority steering, scoped interruption, isolated approval responses, UI thread ownership, concurrent output backpressure, and shutdown cleanup. Runner cases verify stream closure, settled tool statistics, cancellation results, and guards after approval. Loop cases verify correction validation, session commands, persistence, and telemetry correlation through the controller. Session storage cases also cover database and workspace namespaces, concurrent first access, selection during active execution, stale saves, no-follow directory creation, durable generation recovery, adoption, retirement, ACL validation, and failed-constructor cleanup. Native session acceptance uses synthetic private paths with spaces, Unicode and shell metacharacters; verifies canonicalization and SQLite; and distinguishes confinement from detached-descendant termination on macOS. `integration/conversation/` exercises native queued input, steering and interruption, and preserves unsent drafts while approval is pending.

Platform guards skip unsupported operating systems before native setup. Set `LOOP_NATIVE_MACOS_TEST=owned-volume` to run macOS enforcement checks; the ordinary integration run skips them without attaching a disk image. Native macOS qualification requires a parent process that can initialize Seatbelt and permission to attach, format, mount, and eject an owned 8 MiB RAM filesystem through `hdiutil` and `diskutil`; the integration fixture shares one disposable volume across cases and removes it at suite teardown. A surrounding tool sandbox that denies disk-image operations prevents native qualification, and a fixture failure after opt-in stops the suite. Set `LOOP_NATIVE_LINUX_TEST=installed` to run installed Linux enforcement checks; the ordinary integration run skips them without depending on a protected installation. Linux native qualification requires explicit host provisioning; public execution checks exercise unavailable behavior without a qualified installation. Native ripgrep checks skip when the optional executable is absent. See [Linux native integration prerequisites](../docs/development.md#linux-native-integration) before running those checks; skips do not establish enforcement.

The platform integration suites are organized by behavior:

| Suite | Guarantees |
| --- | --- |
| `macos/test_sandbox.py` | Native execution, exit status, authority binding, filesystem confinement, aliases, protected paths, grants, relocation, detached children, network/IPC, POSIX named semaphore denial and descriptors |
| `macos/test_writers.py` | Detached fork/spawn confinement, retained mmap and SQLite WAL resources, native ACL rejection and distinct-account access controls |
| `macos/test_system.py` | Public command dispatch, canonical paths across scripts/subprocesses/file tools, destination-aware redirection and Git authorization, sandbox-only refusal, scratch authority, small and large POSIX here-documents, output bounds and registration aliases |
| `macos/test_command_jobs.py` | Public supervised continuation, live ordered output, session ownership, cancellation, interactive PTY controls, inner and foreign terminal isolation, descriptor receipt, and finite long-running commands |
| `macos/test_mediated_egress.py` | Public macOS Unix-SOCKS mediation, cross-attempt confinement, routed HTTP(S) and Git clients, and client compatibility |
| `macos/test_search.py` | Public search, Unicode columns, regex, binary handling, result bounds, executable lookup and outside symlinks |
| `linux/test_host.py` | Trusted host and installed-tool discovery |
| `linux/test_seccomp.py` | Native ABI and seccomp behavior |
| `linux/test_namespace.py` | Namespace setup and selected system inputs |
| `linux/test_apparmor_profile_lifetime.py` | Foreign attachment and live-profile removal and replacement behavior |
| `linux/test_installed_execution.py` | Installed public execution, supervised continuation, owned interactive terminals, cancellation, verified cleanup, admission boundaries, read-only masks for linked Python environments, and independently prepared uv environments |
| `execution/test_application_storage.py` | Public native writes to bound session scratch, outside app data and the checkout, with control and exact hard-path negatives |
| `execution/test_optional_ipc.py` | Demand-driven macOS IPC, denied setup, overlapping leases, checked eject, counter capacity, cross-attempt endpoint denial, cancellation, and explicit spawned-client and private-account bindings |
| `execution/test_ingress.py` | Exact public loopback publication, foreground identity, active revocation/cancellation, endpoint swaps and payload outbound/descriptor denials |
| `linux/test_mediated_egress.py` | Public mediated HTTP(S) dispatch, isolated destinations, policy revisions, bounded authority loss, and representative clients against owned routed fixtures |
| `linux/test_spawn_disconnect.py` | Disconnect and watcher-failure behavior during a stalled spawn in an owned user-systemd service |

Portable Linux unit tests exercise their source owners with controlled collaborators; they do not establish kernel enforcement. Scripts and their adjacent tests are outside the configured suites.

`integration/execution/test_application_storage.py` exercises public commands with protected application-data storage, a durable session scratch generation, default process approval and remembered scopes. It checks scratch access alongside control/sibling data and explicit hard-denial negatives on either native backend.

The native application-storage suite uses an owned runtime prefix to exercise path policies through the public command tools. Its cases cover an indirect executable reached through a function and a symlink, complete same-root links, a deny beneath a read grant, built-in secret protection, lack of runtime writes, and removal of the grant. Explicit runtime-discovery fixtures use owned paths independently of host-installed presets. Smoke testing the default preset also requires a real installation and its runtime consumers.

`integration/execution/test_hard_policy.py` runs the real coordinator and native launcher with disposable canaries. It covers public template directories, direct/child/symlink/link/rename/future-name denials, complete and incomplete aliases, read-only source isolation, hard runtime-baseline refusal, inherited host descriptors, detached sessions, Unicode/newline private names, overlapping reader/writer attempts, cross-root copy fallback, and cancellation after a trusted policy reload. Run it with `LOOP_NATIVE_MACOS_TEST=owned-volume` or `LOOP_NATIVE_LINUX_TEST=installed`; unavailable backends and skipped cases do not prove enforcement. Offline macOS cases instantiate the production backend without initializing an unrelated socket volume.

`integration/permissions/test_direct_file_boundaries.py` exercises real-file direct reads, benign alias replacement/deletion, incomplete/protected aliases and separate-process path/parent swaps. These controls require no native sandbox opt-in.

## Running tests

Install development dependencies and run the desired suite:

```sh
uv sync
make test                 # architecture and portable unit tests with strict coverage
make test-integration     # portable integration; native qualification is opt-in
make test-all             # architecture, unit and integration tests with strict coverage
```

`make test` and `make test-all` write `.coverage` in the checkout by default. For a run with a read-only checkout, configure Coverage.py to write to an available output directory. In Loop, that can be a private output-path binding in `tools.sandbox_environment` or an explicit `COVERAGE_FILE` in the verification command. Use the prepared development environment and `UV_NO_SYNC=1` when the runtime must stay read-only. For ordinary checkout writes, use an owner-approved workspace write scope.

The Make targets run pytest through `uv --no-cache run`, which may resolve dependencies and initialize a cache. They do not support a `PYTHON` override.

On a prepared native host, run `LOOP_NATIVE_MACOS_TEST=owned-volume make test-integration` or `LOOP_NATIVE_LINUX_TEST=installed make test-integration` to qualify its sandbox. These flags enable native checks but do not install anything or change host permissions.

Set `LOOP_NATIVE_MACOS_BASH` to the absolute path of an installed stock Bash 5 executable to include macOS checks of workspace search-path selection, child lookup, read-only here-documents and stable installation upgrades. The executable and its loaded libraries must already be available on the test host.

Run native cleanup checks without other running sandbox jobs in the Linux test installation or other Loop RAM-attachment tests on macOS. The Linux assertions observe the installation generation's cgroups, and the macOS attachment assertions compare the host attachment set.

The Linux native application check uses owned POSIX semaphore names to verify Python spawn, isolation from host and previous attempts, and cleanup on namespace teardown. Refresh an installation built before private `/dev/shm` support when qualifying this check.

### Running external test wrappers on macOS

Start a fresh session with the project’s installed environment. Request the required runtime paths and, for trusted workloads, any needed account lookup or process-pool service before reviewing the command approval. Review the prepared scope before launch. Each service starts off and can be approved once or remembered for matching commands in this workspace. Keep the checkout read-only when verifying scratch-only outputs. The separately configured stock Bash fixture above tests interpreter selection independently. [Workspace permissions](../docs/permissions.md#optional-sandbox-capabilities).

Set generated output paths in the invocation and run the wrapper with its full test and coverage options. For example, a prepared offline uv environment can use this command.

```sh
UV_NO_SYNC=1 UV_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR="$TMPDIR/uv-cache" COVERAGE_FILE="$LOOP_OUTPUT_DIR/.coverage" \
    PYTEST_ADDOPTS="-o cache_dir=$LOOP_OUTPUT_DIR/pytest-cache" bash ./scripts/check_all_pytest.sh
```

For a lint wrapper that uses Ruff, set `RUFF_CACHE_DIR="$LOOP_OUTPUT_DIR/ruff-cache"` in the same way. Parallel coverage workers create suffixed databases alongside `COVERAGE_FILE`, so select a writable directory for the tool's base path. Loop supplies the private output directory. The project or command selects which artifacts to place there.

Keep test-created private names such as a synthetic `.env` in fresh Loop-owned session scratch. A separately approved external directory still applies the hard private-name denials, including when used as pytest's `--basetemp`. Quote scratch paths so the same command works when both private namespaces contain spaces.

### Clock timezone discovery

`integration/test_timezone.py` checks POSIX timezone discovery against the native offset and covers an unnamed local timezone. It runs on macOS and Linux without privileged setup.

### Selecting individual tests

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

Use a disposable test environment and a sandbox installation matching the package under test. When testing a source copy, verify that the test runner and its isolated native clients both load that copy. Native checks must use fake data, without production credentials or external services. Missing prerequisites cause skips and do not establish enforcement.

| Variable | Required setup |
| --- | --- |
| `LOOP_NATIVE_BWRAP` | Explicitly selected, root-protected Bubblewrap helper for `linux/test_namespace.py`. |
| `LOOP_NATIVE_INNER_ADVERSARIAL_TEST` | Set to `disposable-linux-host` only on a disposable Linux machine to probe inner namespace, mount, Bubblewrap, profile and cgroup attempts and detached-child cleanup. The ordinary inner-seccomp child and outer-boundary checks run without this flag. |
| `LOOP_NATIVE_RUNTIME_FIXTURE` | Administrator-provisioned, root-owned, read-only fake-data directory under `/usr/local/share/`, named `loop-native-private-fixture-*`. |
| `LOOP_NATIVE_SOCKET_ALIAS_DIRECTORY` | Directory owned by the test user, with mode `0700`, on the sandbox listener’s filesystem for socket-alias checks. |
| `LOOP_NATIVE_ALTERNATIVE_CANARY` | Administrator-provisioned root-owned symlink under `/etc/alternatives/` named `loop-native-alternative-canary-*`, pointing to a present harmless system executable. |
| `LOOP_NATIVE_SYSTEM_ALIAS_FIXTURE` | Administrator-provisioned root-owned tree under `/usr/local/lib/loop-native-system-alias-*`; set this variable to its `bin`. Provide `bin/real-tool` printing `parent-chain`, `bin/loop-parent-tool -> ../selected/real-tool`, `selected -> /etc/alternatives/<owned-name>`, and that protected alternative pointing back to `bin`. This exercises a selected ancestor-directory alias rather than an unsupported leaf link directly into an unmounted alternatives subtree. |
| `LOOP_NATIVE_MANAGED_POLICY_TEST` | Set to `disposable-linux-host` for the provisioned Linux `/etc/loop` fixture, `macos-root-fixture` for the macOS `/Library/Loop` fixture, or `owned-root-fixture` for the managed YOLO prohibition fixture described below. Remove the fixture after its isolated test. |
| `LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN`, `LOOP_NATIVE_EGRESS_HTTPS_FIXTURE_ORIGIN` | Owned HTTP and HTTPS origins routed to the synthetic egress fixture in a separate namespace. |
| `LOOP_NATIVE_EGRESS_MIXED_DNS_ORIGIN`, `LOOP_NATIVE_EGRESS_PRIVATE_DNS_ORIGIN` | Fixture names resolving to an allowed and a forbidden address, or only a forbidden address, respectively. |
| `LOOP_NATIVE_EGRESS_REBIND_DNS_ORIGIN`, `LOOP_NATIVE_EGRESS_REBIND_TEST` | The owned rebinding name and `disposable-linux-host` opt-in for changing only that marked host entry between two native connections. Set `LOOP_NATIVE_EGRESS_PRIVATE_IP` to the VM's private address. |
| `LOOP_NATIVE_EGRESS_SUPERVISOR_TEST` | Set to `disposable-linux-host` only on a disposable VM to enable the receiver- and client-death controls. |
| `LOOP_NATIVE_EGRESS_SUSPEND_TEST` | Set to `disposable-linux-host` only on a disposable VM to pause and resume the Loop policy-authority process while native tunnels are active. |

The standalone Bubblewrap namespace probes invoke the selected helper directly. The installed Loop receiver uses its own AppArmor setup profile. On Ubuntu 24.04 with unprivileged user-namespace restrictions enabled, the `apparmor-profiles` package supplies `/usr/share/apparmor/extra-profiles/bwrap-userns-restrict`. Load that enforcing profile for raw qualification and select the distro's `/usr/bin/bwrap`. Keep the kernel restriction enabled and verify the profile's restricted payload transition. [Ubuntu release notes](https://documentation.ubuntu.com/release-notes/24.04/#unprivileged-user-namespace-restrictions) · [Ubuntu package file list](https://packages.ubuntu.com/noble-updates/all/apparmor-profiles/filelist).

The protected administrator-policy integration tests use the actual fixed system anchors and must run separately from the general suites so their system-wide policy does not affect unrelated test managers. The expectation file has `{"version":1,"identity":"0123456789abcdef0123456789abcdef","minimum_revision":1}`. The policy file has `{"version":1,"identity":"0123456789abcdef0123456789abcdef","revision":1,"forbid_host":true,"deny_local_networks":true,"rules":[{"id":"native-deny-session","decision":"deny","action":"session.mutate"},{"id":"native-deny-egress","decision":"deny","action":"network.request"}]}`. On a disposable Linux VM with no pre-existing `/etc/loop` or `/etc/loop.managed.expect.json`, create a root-owned `0755` `/etc/loop` directory, a root-owned `0644` `/etc/loop.managed.expect.json` expectation, and a root-owned `0644` `/etc/loop/managed-policy.json` policy. Run `LOOP_NATIVE_MANAGED_POLICY_TEST=disposable-linux-host PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/permissions/test_managed_policy.py`, then remove only the fixture created for that run. On macOS, first confirm `/Library/Loop` and `/Library/loop.managed.expect.json` are absent and no other Loop process needs the optional policy. With administrator authentication, create root-owned `0755` `/Library/Loop`, root-owned `0644` `/Library/loop.managed.expect.json`, and root-owned `0644` `/Library/Loop/managed-policy.json` with those same contents. Run `LOOP_NATIVE_MANAGED_POLICY_TEST=macos-root-fixture PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/macos/test_system.py::test_root_managed_policy_controls_macos_yolo_host_and_native_network` outside an enclosing sandbox that prevents Seatbelt or the owned RAM volume. Remove only the exact two files and the directory created for the fixture, including after a failed test; recheck that the fixed source is absent. A missing fixture or skipped case is not active-policy qualification. For the managed YOLO prohibition, use the same expectation file and a separate root-owned `0644` policy file containing `{"version":1,"identity":"0123456789abcdef0123456789abcdef","revision":1,"forbid_yolo":true}`. With the same isolation and cleanup precautions, run `LOOP_NATIVE_MANAGED_POLICY_TEST=owned-root-fixture PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/permissions/test_managed_policy.py::test_native_root_policy_prohibits_yolo_activation`. The test reads the protected fixture without changing it; remove only the fixture that was provisioned for this run and verify both fixed anchors are absent afterward. Policy changes and trusted reload are covered by isolated portable tests.

For the mediated-egress native suite, run `LOOP_NATIVE_EGRESS_PRIVATE_IP=<disposable-VM-private-IPv4> sh tests/integration/linux/egress_fixture_setup.sh setup` on an otherwise unused disposable VM with noninteractive `sudo` for the owned `ip`, `systemd-run`, and `/etc/hosts` changes. The script creates a separate network namespace and synthetic HTTP, HTTPS, package-index, and Git endpoints at `11.0.0.2`, with mixed and private test names in `/etc/hosts`. Set `LOOP_NATIVE_EGRESS_FIXTURE_ORIGIN=http://11.0.0.2:18080`, `LOOP_NATIVE_EGRESS_HTTPS_FIXTURE_ORIGIN=https://11.0.0.2:18443`, `LOOP_NATIVE_EGRESS_MIXED_DNS_ORIGIN=http://loop-egress-mixed.invalid:18080`, and `LOOP_NATIVE_EGRESS_PRIVATE_DNS_ORIGIN=http://loop-egress-private.invalid:18080`; then run `pytest tests/integration/linux/test_mediated_egress.py`. Enable the supervisor-death, process-suspension, and DNS-rebinding variables only on a disposable host; the last also needs `LOOP_NATIVE_EGRESS_REBIND_DNS_ORIGIN=http://loop-egress-rebind.invalid:18080` and the setup address left in the environment. Afterward run `sh tests/integration/linux/egress_fixture_setup.sh cleanup`, which stops the two owned services, removes its network namespace and marked host entries, and removes only its owned temporary fixture directory. A skipped origin or unavailable protected installation is a missing prerequisite, not a mediated-egress pass.

Native service-publication tests live in `execution/test_ingress.py`. Run them with `LOOP_NATIVE_LINUX_TEST=installed` on the protected Linux installation or `LOOP_NATIVE_MACOS_TEST=owned-volume` outside an enclosing macOS sandbox. `LOOP_NATIVE_INGRESS_INTERFACE` may name an owned non-loopback interface for the public-bind negative. The fixtures own ephemeral host ports, synthetic HTTP services and temporary workspace/scratch data; macOS fixtures register exact future mount paths and require removal afterward. Cases check approval before bind, deny before launch, active cancellation and revision closure, payload-to-host refusal, Linux agent/child descriptor denial, and macOS foreground identity and endpoint swaps. A missing platform prerequisite is unverified.

For positive macOS mediation checks, connect the Mac and an otherwise unused disposable Ubuntu VM to the same host-only network. With two guests, use a host-only interface on each that permits guest-to-guest traffic. A shared NAT interface can let the host reach both guests while isolating them from each other. Use the Ubuntu fixture interface’s address as the gateway. Run `sh tests/integration/macos/egress_fixture_setup.sh setup` on the VM; it creates synthetic HTTP, HTTPS, package-index and Git endpoints bound to `11.0.0.2` and changes only the VM's owned loopback alias and named services. On macOS, add a temporary host route with administrator authentication using `sudo /sbin/route -n add -host 11.0.0.2 <VM-private-IPv4>` and confirm `/sbin/route -n get -inet 11.0.0.2` uses the VM-facing, non-loopback interface. From the Mac, verify the owned endpoint returns `egress-fixture-ok` with `curl --noproxy '*' --fail --max-time 3 http://11.0.0.2:18080/`. Run `LOOP_NATIVE_MACOS_TEST=owned-volume LOOP_MACOS_EGRESS_FIXTURE_GATEWAY=<VM-private-IPv4> PYTHONDONTWRITEBYTECODE=1 uv --no-cache run pytest tests/integration/macos/test_mediated_egress.py`. The fixture verifies the exact route gateway and the owned HTTP marker before any positive proxy test. Remove the exact host route with `sudo /sbin/route -n delete -host 11.0.0.2`, then run the fixture script with `cleanup` on the VM. A skipped route or unavailable owned endpoint is not a positive mediation pass; do not substitute a real public service or a host-local address for this fixture.

The macOS direct-resolver control additionally needs an owned VM DNS listener and a temporary administrator-installed resolver file. Run `LOOP_MACOS_EGRESS_PRIVATE_IP=<VM-private-IPv4> sh tests/integration/macos/egress_fixture_setup.sh dns-setup` on the VM. After confirming `/private/etc/resolver/loop-egress-owned.test` does not already exist and `/private/etc/resolver` is a real root-owned directory if present, create that one root-owned `0644` file on macOS containing exactly `nameserver <VM-private-IPv4>` followed by a newline. Set both `LOOP_MACOS_EGRESS_FIXTURE_GATEWAY` and `LOOP_MACOS_EGRESS_DNS_GATEWAY` to that VM address for the native suite. The test requires fresh host libc and DNSService queries to reach the owned listener, then checks that equivalent unique sandboxed queries add no listener records. Remove only that resolver file and the resolver directory if this fixture created it and it is empty; run `dns-cleanup` on the VM before the ordinary `cleanup`. A missing DNS fixture is an unqualified direct-resolver control, not a denial pass.

The independent macOS unicast DNS check uses a loopback listener and an owned split-DNS zone. On a disposable macOS test installation, choose an unused unprivileged UDP port on loopback. Create a root-owned `0644` file at `/private/etc/resolver/unicast.loop-egress-owned.test` containing `nameserver 127.0.0.1` and `port <port>` on separate lines. Use a real root-owned resolver directory without group or other write permission and create the file only if the name is unused. Run `test_public_general_network_resolves_owned_unicast_name` with `LOOP_NATIVE_MACOS_TEST=owned-volume`. The test binds the selected port, checks a host control, and records fresh queries for offline, mediated and general network decisions. Afterward remove the owned mapping and any empty directory created for it.

`linux/test_mediated_egress.py::test_loop_process_death_closes_established_tunnel` launches the production client through `linux/egress_lifecycle_probe.py` in owned disposable storage. An independent controller verifies an established synthetic HTTPS transport, kills only that client, and requires both the transport and receiver to close within two seconds. Enable `LOOP_NATIVE_EGRESS_SUPERVISOR_TEST=disposable-linux-host` with the owned HTTPS fixture. The helper also supports an explicitly selected whole-VM suspend/resume experiment: run `python tests/integration/linux/egress_lifecycle_probe.py suspend <new-owned-directory>`, wait for `READY`, suspend and resume only that disposable VM using its hypervisor, and require `PASS resumed stream completed`. Remove only the supplied fixture directory after inspection. Keep native lifecycle runs isolated from other Loop clients on that installation; a process-pause test does not qualify whole-VM resume.

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

Unit private-path classification uses the synthetic case root as its boundary: an enclosing host `Library` or `.ssh` directory does not make every synthetic public file private, while those names within the case remain protected. Native integration tests retain real path classification. Profile-size unit checks use explicit bounded logical paths so they measure the compiler contract independently of the runner's temporary-directory spelling. The subprocess scratch-admission checks exercise these rules under protected and unusual ancestor names.

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

Direct-host recovery qualification uses owned temporary canaries and the public `request_host_command` path. Run `tests/integration/execution/test_coordinator.py` with `LOOP_NATIVE_HOST_TEST=disposable-linux-host` on Linux or `LOOP_NATIVE_HOST_TEST=disposable-macos-host` on macOS to include persisted-grant and unrestricted-boundary controls. The default-off gate, enabled YOLO case, refusal effects and separate recovery tests also run in the portable integration suite. These tests never use real private data. Native sandbox opt-ins remain independently required to qualify confinement; a synthetic unavailable backend is only a host-recovery control.

`integration/execution/test_write_scopes.py` exercises the public tool with real native confinement. It covers scratch-only defaults, ordinary read-only developer tools, allow/ask/deny on output directories, creating and removing one exact output file, persistence of exact-file approvals, neighboring-file and hard-resource denial, runtime and scratch policy checks, and refusal of encompassing grants with withheld descendants. The compiler fixture uses the resolved installed executable. Runtime suites cover generic executable-alias discovery. Use the platform native-test opt-in. Skipped cases remain unverified.

`unit/application/test_verification.py` exercises private slash-command shortcuts, including plain shell input, approval refusal, replacement during approval and the saved task’s own write scope. `integration/execution/test_verification.py` exercises the saved-task command through the same native process approval and owned job path. It checks exact file creation and replacement, multiple parallel outputs in a bounded output directory, protected and undeclared output denial, and artifact retrieval using normal file tools. Use the same platform native-test opt-in.

Policy recovery checks cover legacy schema migration, missing-file defaults, explicit temporary recovery, archived replacement and concurrent-edit refusal. Resource-capability tests opt into automatic process approval where they need to isolate filesystem/native behavior; separate checks require the built-in process `ask` and verify that declining it launches no payload.
