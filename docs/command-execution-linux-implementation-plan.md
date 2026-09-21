# Production Linux command-execution architecture and implementation plan

**Status:** implementation-ready
**Platform:** Linux hosts on explicitly qualified distributions and CPU architectures
**Scope:** ordinary sandbox execution, transactional workspace effects, lifecycle, security, and release qualification

## 1. Outcome

Ordinary `run_command` executes only inside a Loop-private rootless containerd/runc sandbox. Loop
installs and manages its pinned runtime beneath application-owned data, uses immutable workspace
generations and private attempt overlays, observes effects before authorization, and publishes
approved changes transactionally. Host execution is a separate explicit request and authority.
Missing prerequisites or failed attestation produce typed unsupported or integrity results; they
never select direct host execution or weaker containment.

This file is self-contained. It owns its architecture, implementation order, exit gates, rollout
state, and definition of done. It is temporary planning material: source, tests, builds, packaging,
and releases must not read this file or any other file under `docs/`.

## 2. Product and maintenance boundary

The decision order is usefulness, safety, then operational refinement. A design is rejected when it
makes ordinary repository work harder without closing a concrete sandbox boundary. The Linux work
must reuse the existing platform-neutral contracts and add only the host-specific adapter needed to
make the public command work.

The following constraints are absolute:

- First use automatically installs the pinned private runtime, builds the command image locally,
  and then runs the original command. There is no separate setup workflow.
- There is exactly one complete sandbox image. There are no profiles, language-specific images,
  registries, image publication, registry authentication, or CI prerequisite.
- The image includes a POSIX shell, certificates, Git, SSH client, search/text/file/archive tools,
  Python, Node, C/C++, Rust, Go, their package managers, and representative build, test, lint,
  format, and generator tools.
- Provenance statements, SBOMs, SLSA/in-toto material, artifact signatures, published images, and
  external CI evidence are optional review inputs only. Their absence never blocks development,
  installation, qualification, or release.
- Security claims come from local verification of the running boundary and real public
  `run_command` behavior: isolation, containment, resource limits, failure handling, and cleanup.
- Ordinary commands have no host fallback. Host execution remains a separately constructed,
  warned, authorized, and audited capability.

Loop-owned implementation is Python plus declarative configuration and data. Loop does not build a
kernel, native launcher, container runtime, RootlessKit replacement, CNI plugin, proxy, firewall
component, daemon, or native extension. It invokes only fixed-shape operations from pinned upstream
artifacts or documented host facilities.

The supported runtime is an official `nerdctl-full` distribution installed under a private prefix.
Its containerd, runc, RootlessKit, CNI, slirp4netns, and transient BuildKit components use private
configuration, sockets, state, content, namespaces, and process ownership. Loop never connects to
Docker, Podman, system containerd, or user container configuration and never runs an upstream
system-wide installation script.

## 3. Security architecture

Each attempt uses:

- a rootless user namespace and private mount, PID, IPC, UTS, and network namespaces;
- all capabilities dropped, `no_new_privileges`, a read-only image root, minimal `/proc` and
  `/dev`, masked kernel paths, and the pinned runtime's qualified default seccomp policy;
- mandatory cgroup-v2 memory and pids limits;
- delegated cgroup CPU limits when available, otherwise a process-tree CPU-time budget plus wall
  deadline;
- filesystem, tmpfs, image, cache, and write-byte quotas instead of requiring cgroup I/O delegation;
- fd, file, core, stack, output, and duration limits;
- no ambient route, host/control socket, management credential, host path, or live workspace mount.

The live workspace is never an OverlayFS lower. Loop materializes an immutable generation using
`FICLONE` when safe and a verified userspace copy otherwise. Each agent has a private cumulative
branch; each command gets a fresh upper/work pair. Only a stopped attempt is inspected. Whiteouts,
opaque directories, metadata, and changed content become a canonical delta. The destination is
checked for representability and conflicts before authorization and journaled publication.

Persistent publication uses fsynced `PREPARED`, `HOST_APPLIED`, `BRANCH_APPLIED`, and
`COMMITTED` states with deterministic roll-forward. Agent forks preserve transaction identity but
never share mutable branch state.

## 4. Host support contract

Before any substantial runtime download, a read-only attestor must prove:

- a named supported distribution/version and minimum kernel;
- unprivileged user namespaces;
- working `newuidmap`/`newgidmap` and adequate subordinate IDs for the selected RootlessKit mode;
- cgroup v2 and a usable transient user-scope delegation with memory and pids controllers;
- the selected CPU enforcement mode;
- seccomp and required namespace/mount behavior;
- permission for the application-private RootlessKit binary under the active LSM policy;
- rootless OverlayFS and compatible immutable snapshot/upper storage;
- enforceable filesystem/image/tmpfs quotas and byte budgets;
- sufficient disk, process, and file-descriptor ceilings.

Loop never installs packages, edits `/etc/subuid`, `/etc/subgid`, sysctls, LSM policy, cgroup
delegation, system paths, login configuration, or services. A host requiring any such remediation
is unsupported. Diagnostics are typed, sanitized, and contain no credential or private host path.

## 5. Runtime and artifact contract

The trusted packaged manifest has one entry per supported architecture. Every entry pins the exact
version, immutable HTTPS source, expected size, SHA-256 digest, declared OS/architecture, install
layout, and component versions. Selection rejects missing, duplicate, mismatched, or cross-platform
entries before download.

Downloads enforce origin and redirect allowlists, byte ceilings, exact size/digest, safe archive
members, private temporary state, fsync, and atomic activation. Active attempts hold leases; one
known-good rollback set is retained; garbage collection removes only inactive owned data.
Installed executable identities and live runtime versions are attested independently of download
verification.

One checked-in Containerfile and locked inventory builds the complete command image locally on
first use. Exact package versions and signed repository metadata protect package installation; the
base image and every downloaded input use fixed origins, bounded sizes, pinned digests, and safe
extraction. The build context is trusted and fixed. Readiness proves image identity, source
version, non-root user, PATH, the complete ordinary-developer command inventory, and absence of
privileged defaults. Warm use performs no download, build, package installation, or runtime
restart. Readiness is deliberately local and small; it creates no provenance, SBOM, signature,
publication, registry, or CI dependency.

## 6. Execution, permissions, and lifecycle

The public command accepts an opaque POSIX shell script or explicit argv, virtual cwd/environment,
terminal request, resource limits, workspace identity, and requested capabilities. Model-visible
paths are stable virtual paths such as `/workspace`; host paths never enter requests, output,
diagnostics, logs, or results.

Predicted capabilities may constrain launch but never authorize persistent effects. Frozen observed
deltas are converted to typed permission operations and evaluated with deny precedence, subject,
boundary, effect, resource subset, constraints, workspace binding, scope, expiry, revocation, and
policy version. Only an authorized matching delta may publish.

The OCI supervisor owns create/start/attach/wait/signal/stop/inspect/remove. Pipe mode preserves
stdout/stderr with bounded backpressure. PTY mode owns resize, signal, hangup, and EOF semantics.
Completion requires task termination and zero remaining processes. Timeout/cancellation kills the
task and cleans the attempt. Existing durable-job behavior may be connected through the same
supervisor after ordinary foreground commands pass; this plan adds no new job protocol or daemon.

Network, listener, secret, IPC, device, package, system, and privilege effects are default-denied.
They are not prerequisites for the basic Linux sandbox. An already-supported effect may be reused
only through its existing enforcement and cleanup contract; this plan adds no proxy, broker,
firewall component, CNI plugin, or parallel permission path. Unsupported effects fail before
permission evaluation or runtime preparation.

## 7. Implementation packages

```text
L0 extension seams
  +-> L1 host attestation
  +-> L2 manifest and bootstrap
L1 + L2 -> L3 rootless runtime lifecycle
L0 + L3 -> L4 OCI transport and process lifecycle
L0 + L3 -> L5 workspace materializer
L4 + L5 -> L6 product composition
L6 -> L7 native security and recovery
L2 + L7 -> L8 release cutover
```

### L0 — extension seams

Inventory the existing platform selection, runtime, sandbox backend, OCI supervisor, VFS
materializer, permission, result, audit, and application-composition interfaces. Add architecture
checks that common packages import no platform implementation. Change a common protocol only when
it cannot express this document's semantics; record the rationale and cover every implementation
with contract tests.

Exit: exact extension points and owners are named, baseline tests pass, and no platform-specific
kernel, systemd, subid, cgroup, LSM, or OverlayFS mechanics appear in common modules.

### L1 — host attestation

Implement the complete section 4 preflight before runtime installation. Unit-test every absent,
malformed, partial, contradictory, and permission-denied condition. Validate one clean supported
host on the primary development architecture with only Loop installed.

Exit: the clean image passes with zero remediation; unsupported candidates fail early with precise
typed reasons. Local test output is sufficient; no external qualification evidence is required.

### L2 — manifest and bootstrap

Add verified artifacts for the architecture being implemented and reuse the common download,
extraction, activation, lease, rollback, and garbage-collection machinery. Attest every installed
executable and component. Add another architecture only when it is actually supported; do not add
an architecture matrix as speculative work.

Exit: selection and metadata tests cover every declared artifact; corrupt, redirected, oversized,
unsafe, truncated, and wrong-platform inputs fail atomically; no global state changes. No
provenance, SBOM, SLSA, artifact-signature, registry, publication, or CI check is required.

### L3 — private rootless runtime

Implement fixed-shape prepare, start, health, identity attestation, warm reuse, stop, crash recovery,
and ownership-checked cleanup for private RootlessKit/containerd state. Use a transient user scope
only on an already-qualified host. Serialize first use and retain ambiguous state rather than
deleting or killing an unproven owner.

Exit: native lifecycle and failure injection pass for every boundary; wrong PID, executable,
socket, epoch, namespace, cgroup, or ownership state fails closed.

### L4 — OCI transport and process lifecycle

Implement the local transport behind the OCI supervisor. Reuse the common specification, labels,
limits, results, events, pipe/PTY behavior, and durable-job contract. Attest namespaces, caps,
seccomp, mounts, root identity, resources, labels, and absence of control sockets after start.

Exit: native pipe, PTY, timeout, cancellation, output pressure, signal, descendant, job recovery,
spoofed inspection, stale handle, escape, and cleanup tests pass with no remaining processes.

### L5 — workspace materializer

Implement immutable generation creation, private cumulative branches, per-attempt overlays, stopped
upper inspection, canonical delta production, archive/content staging, and owned cleanup. Reuse the
representability, conflict, authorization, journal, recovery, lease, and transaction contracts.

Exit: mutation, denial, conflict, host race, crash boundary, fork, refresh, restart, quota, and
concurrent-agent suites pass; warm reads perform no recursive host scan, hash, or copy.

### L6 — product composition

Select the backend only after exact host and manifest attestation. Route ordinary `run_command`
through the existing sandbox service and real shell. Reuse permission, result, audit, publication,
and explicit-host boundaries. On first use, prepare the runtime, locally build the one complete
image, verify it, and execute the original request. Build and run the complete developer-tool
corpus covering shell, Git, search, archives, Python, Node, C/C++, Rust, Go, package managers, and
representative build/test/lint/format/generator commands.

Exit: the public shell, read/write permission, foreground lifecycle, offline, recovery, cleanup,
failure-injection, and developer-tool corpora pass with zero ordinary host starts. A second command
reuses the warm runtime and image without download, build, package installation, or setup.

### L7 — local native security, recovery, and performance

On the named clean image, prove namespace/mount/capability/seccomp/cgroup/non-root/device/proc/sys/
socket isolation, immutable bases, no ambient route, path-canary absence, resource containment,
crash recovery, cleanup, bounded caches, concurrency, cold installation, and warm reuse. At least
20 post-warm-up offline no-op samples must satisfy the declared release latency budget and perform
no recursive workspace work, runtime restart, image pull, or broker start.

Exit: every local native negative, recovery, concurrency, and performance gate passes without
skips or host remediation; repeat execution is deterministic. Real public `run_command` tests—not
provenance or external evidence—are the release proof.

### L8 — release cutover

Embed the verified manifest in packaged resources and verify clean install, offline reuse, update,
rollback, uninstall ownership, notices, and diagnostics locally on a supported host. Remove any
feature switch that permits host execution or weaker containment. CI may repeat these checks but is
never a prerequisite for development, installation, qualification, or release.

Exit: all unit, integration, public end-to-end, and native suites pass locally. Unsupported hosts
fail closed before installation. Every advertised architecture has an exact verified manifest
artifact. No registry, published image, provenance, SBOM, SLSA/in-toto statement, artifact
signature, or external CI evidence is required.

## 8. Rollout state

`docs/command-execution-linux-work-packages.json` is the sole current-state ledger for this plan.
This file owns scope, dependencies, and exit gates; the ledger owns package state, exact verification
commands, blockers, and the first resume action. Neither file is a runtime, build, packaging, or
test input.

Allowed states are `pending`, `in_progress`, `blocked`, and `complete`. Before work, reload
both files and select the earliest dependency-eligible package, preferring `in_progress` or
`blocked` over `pending`. Mark it `complete` only after its exit gate passes and record the
exact verification commands. For a genuine external block, record the exact condition and first
resume action. Reopen completed work only when owned code changes or a gate regresses.

## 9. Definition of done

- Ordinary execution is sandbox-only and every failure remains typed and fail-closed.
- At least one clean named host needs no administrator action, package install, persistent service,
  or global configuration change.
- Runtime artifacts and image inputs are pinned, verified, atomically installed, leased,
  rollbackable, and garbage-collected.
- First use installs the runtime, builds exactly one complete image locally, and runs the original
  command without a separate setup action.
- The image supports ordinary repository work across shell, Git, search, archives, Python, Node,
  C/C++, Rust, Go, package management, build, test, lint, format, and generation.
- Host paths and control endpoints never reach untrusted code or model-visible output.
- Immutable workspace generations and transactional publication pass every race and recovery gate.
- Permissions, lifecycle, durable jobs, enabled effects, cleanup, audit, and explicit host execution
  satisfy their public and native negative tests.
- Unsupported prerequisites never weaken containment or trigger host execution.
- Local release checks, packaged manifests/notices, operator diagnostics, and the rollout ledger
  agree with the implemented product; no external evidence system is part of the trust boundary.
