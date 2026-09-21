# Command-execution operations and release blockers

This document records operational behavior and unresolved release blockers without changing the
normative architecture or work-package ledger.

## Supported product boundary

Ordinary `run_command` requests use the managed macOS sandbox. The application composition keeps
network connections, listeners, and all raw or header secret exposure disabled. A request for any
of those effects returns `UnsupportedCapability` before permission evaluation, runtime preparation,
Envoy startup, secret resolution, or command-container construction.

The pinned Envoy/runtime stack has no primitive that atomically enforces one cumulative byte budget
for both directions across every connection belonging to a lease or attempt, including concurrent
connections, redirects, retries, and reused connections. Envoy buffer limits are instantaneous
memory limits; bandwidth filters enforce a rate; and counters observed after forwarding cannot
prevent the transfer that crosses the ceiling. None is acceptable cumulative enforcement.
Networking remains unsupported until an upstream, attestable primitive supplies that shared
pre-forward accounting without a custom proxy, TLS interception, ambient route, dynamic forward
proxy, CNI plugin, firewall daemon, or native helper.

Raw environment and file secrets are also unsupported at the public boundary. Arbitrary code that
receives raw bytes can print, copy, encode, or persist them, so raw exposure cannot simultaneously
guarantee non-disclosure from results, logs, diagnostics, process listings, durable state, and
workspace deltas. Enabling it requires an explicit product decision to narrow the non-disclosure
contract. Header injection additionally remains blocked on cumulative network enforcement and the
complete broker control set.

## Resource-limit attribution

The runtime enforces its configured OCI memory, PID, CPU, file-descriptor, output, duration,
read-only-root, scratch, temporary, cache, and persistent-write controls. Typed attribution is
returned only when the pinned Lima/nerdctl/containerd/runc interfaces provide unambiguous evidence.
An exit code, signal, or stderr phrase alone is not evidence. The pinned CLI does not expose a
single authenticated terminal record that reliably distinguishes every cgroup, RLIMIT, tmpfs,
overlay, and workspace quota exhaustion from program-selected failures. Those controls remain
enforced, but release qualification must not claim `ResourceLimitExceeded` for an ambiguous case.

## Recovery, reset, and uninstall

Runtime collection removes only inactive, unleased content under the exact application-owned
runtime root. Active and rollback artifact sets and valid leases remain protected. Reset removes
only inactive digest-addressed artifacts and owned temporary downloads. The journaled Lima
lifecycle stops and deletes only the exact instance whose manifest, artifact set, configuration,
name, epoch, and ownership record match; malformed, substituted, symlinked, or ambiguous state
fails closed and is retained for operator review.

`/sandbox` and `/sandbox status` report the active workspace's attested sandbox state without
starting an absent or stopped VM. `/sandbox list` displays the same workspace-scoped owned
inventory, which contains at most one VM in the current architecture. `/sandbox stop` stops the
active workspace's journal-owned VM while retaining its private disk and
locally built image for reuse. `/sandbox delete` stops and deletes that VM and private disk, which
removes its containers, guest overlays, and locally built images. Both operations reject cleanup
while a live durable job owns the sandbox, validate the exact ownership journal, and serialize with
preparation. A later command prepares or starts the sandbox again.

Verified runtime downloads are shared, lease-protected application assets rather than workspace
sandbox content, so `/sandbox delete` deliberately leaves them to runtime garbage collection. There
is not yet one application-facing inventory/dry-run/uninstall transaction spanning those shared
runtime artifacts, snapshots and generations, attempt archives, publication journals, and broker
state. Until that ownership graph is composed, operators must not recursively delete an application
root. Ambiguous or pre-existing roots are retained. Failed native verification likewise retains
state whenever removal would lose the ownership evidence needed to stop a live VM safely.

## Runtime integrity

Release qualification validates every downloaded input against its pinned HTTPS origin, redirect
allowlist, byte ceiling, expected size, and SHA-256 or OCI digest. Installed executable identities,
runtime component versions, OCI policy, and the live isolation boundary are independently attested.
Optional upstream SBOMs, signatures, and provenance statements may support review when they name
the exact selected digest; their absence is not a release blocker and Loop does not synthesize
them.

## Release verification

Release qualification runs the documented native verifier locally on a controlled physical Apple
Silicon host. That command proves the running Lima/VZ/rootless-OCI boundary and the public
`run_command` behavior; its passing local result is sufficient for release qualification.

The repository may repeat the same verifier in the `macOS Apple Silicon native and security` job
on a `self-hosted`, `macOS`, `ARM64`, `loop-native-security` runner. That automation is a
convenience and regression signal, not an external evidence authority or release prerequisite.
Missing runner capacity, GitHub rulesets, retained CI artifacts, provenance, SBOMs, SLSA/in-toto
statements, or artifact signatures do not block development, installation, qualification, or
release.
