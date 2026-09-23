# Staged-Change Review — Managed Command Execution (macOS/OCI sandbox, effect broker, typed grants)

**Review type:** Senior technical lead review for bugs, correctness, and state-of-the-art fit.
**Scope:** all staged changes (`git diff --cached`): 168 files, +33,285 / −3,408; ~100 files under `src/`.
**Method:** static review of the staged content (working tree matches the index).

---

## Verdict

The architecture is genuinely modern and, in most places, correctly implemented: closed Pydantic contracts with
capability↔lease cross-validation, a single typed grant evaluator with deny-dominance, data-only CNI/Envoy broker
plans, start-gated `--add-host` hardening, content-addressed staging, per-attempt dedicated bridges with no default
route, digest-pinned everything, and a fail-closed `ExecutionService`. The code I read closely is above average.

It is **not clean**. There is one capability-containment defect, one disclosure defect that undermines the host
approval boundary, several documented-but-unenforced limits, and a set of contract/path/typing bugs. Fix P0 and P1
before merge; P2 is a judgment call I would still take.

Recommendation: **request changes** — P0 items 1–4 are blocking; P1 items 5–10 should land in the same change.

### Verification status and limitations

- Static review only. The managed sandbox is failing in this session (every non-`git` `run_command` returns
  `InfrastructureFailure` with empty output), so `ruff check`, `ruff format --check`, `git diff --check`, the pytest
  suite, and the 100 % coverage gate **were not executed**. No claim below depends on a test run.
- Read closely (~15 files, the security- and lifecycle-critical paths): `execution/{broker,command,contracts,results,
  service}.py`, `execution/host/{models,broker,supervisor}.py`, `execution/infrastructure/process.py`,
  `execution/oci/{spec,process,control,adapter}.py`, `execution/runtime/{image,provenance}.py`,
  `permissions/{execution,execution_adapter,host_adapter}.py`, `permissions/manager.py` (`authorize_execution`),
  `application/{execution,secrets,runtime}.py`, `execution/vfs/{staging,models}.py`,
  `execution/sandbox/macos/{composition,backend:excerpt,execution:excerpt}.py`, `tools/{system,files:excerpt}.py`,
  `utils/search.py`, `Containerfile`, both new workflows, and the `Makefile`.
- Not audited line-by-line (~9k lines): `macos/{control_plane,workspace,lima,jobs,images,candidate,requirements}.py`,
  `oci/{artifacts,attestation}.py`, `vfs/{commit_broker,agent_workspace,journal,snapshot,coordinator,delta,
  materializer}.py`, `scripts/verify_macos_execution_native.py`. The seams where these meet leases, network effects,
  and delta publication were checked and are internally consistent.
- The `execution/policy/*` and `utils/{process,path}.py` deletions leave **no** dangling references (the only
  remaining `..policy` imports are the unrelated `loop.telemetry.policy`).

---

## P0 — blocking

### 1. The untrusted container runs as root, discarding the identity the image layer just attested

- `execution/sandbox/oci/control.py:505-508` compiles `--user 0:0` for the **command** container.
- `execution/runtime/image.py:71-76` requires `(uid, gid) == (1000, 1000)` — "Sandbox image requires the reviewed
  non-root environment"; `image.py:202` attests image config `User in {"agent:agent", "1000:1000"}`;
  `runtime/Containerfile:33` ends with `USER agent:agent`. The property is verified and then thrown away at spawn.

Consequences:

1. Under rootless OCI, container uid 0 maps to the *guest owner* uid — the same identity that owns the secret-bearing
   Envoy bootstrap and every staged broker file. Isolation then rests entirely on mount selection, not identity.
2. `--tmpfs /home/agent:…,uid=65534,gid=65534,mode=700` (`control.py:539`) is **not writable by container root** with
   `--cap-drop ALL` (no `CAP_FOWNER`, no DAC override). `git`, `pip --user`, `npm`, `go build` fail on `$HOME`.
3. The `/cache` tmpfs is created but nothing points at it (no `XDG_CACHE_HOME`), so a reviewed mount is dead weight.

**Fix.** Carry the reviewed identity through the contract instead of hard-coding it twice:

- add `run_as_user: tuple[int, int]` to `OciExecutionSpec`, sourced from `SandboxImageDefinition.environment` so a
  lease-bound image cannot silently change it, emitted as `--user {uid}:{gid}`;
- mount `/home/agent` with `uid=1000,gid=1000,mode=700`;
- set `XDG_CACHE_HOME=/cache` in `spec._environment` (`oci/spec.py:_environment`);
- keep `--user 0:0` only for the broker container, whose justification comment already exists (`control.py:611-614`);
- add a native-gate assertion that `$HOME` and `$XDG_CACHE_HOME` are writable inside an attempt.

### 2. The host-boundary prompt never shows the real working directory

`execution/host/models.py:130-150` renders the **model-supplied** `display_cwd` (tool default `"host filesystem"`,
`tools/system.py` `run_host_command`), never `request.cwd` — which `execution/host/broker.py:150-166` resolves, maps
from `/workspace`, and binds into the lease fingerprint. `permissions/host_adapter.py:104` passes `prompt.render()`
verbatim. `run_host_command`'s `cwd` defaults to `"/"`, so a user can be shown "Working directory: host filesystem"
while the process runs at `/` or any host path. That is prompt laundering on the one boundary whose own text says
"sandbox protections do not apply to this operation."

**Fix.** Render the broker-resolved host path (plus the workspace-relative form when it maps from `/workspace`). Keep
`display_cwd` only if it is validated against the resolved path; otherwise remove the parameter from the tool and from
`HostExecutionRequest`. Consider also disclosing `NetworkEndpoint.addresses` in `manager._execution_prompt`, which
today shows only `protocol://host:port` while the grant authority is the pinned address set.

### 3. Documented lease limits that are not enforced

- `NetworkConnectionLease.max_bytes` is documented as "Maximum brokered bytes for the attempt"
  (`execution/contracts.py:101,111`, plus the model-facing tool schema) but is used **only** as Envoy's
  `per_connection_buffer_limit_bytes` (`execution/broker.py:335,391`) — a flow-control watermark, not a quota. Repo-wide
  grep confirms no byte accounting anywhere.
- `ExecutionLease.expires_at_ns` is never checked on the foreground path: it appears only in
  `execution/command.py:174,361`, `execution/host/*`, and `macos/jobs.py:624` (durable operations). The effective
  authority lifetime for ordinary commands is `request.deadline_seconds`.

**Fix.** Implement cumulative byte accounting in the guest supervisor (or from Envoy access logs / a rate-limit
descriptor), **or** remove the field and every claim to it from contract, tool schema, and docs. For the lease, either
enforce expiry once at the adapter entry point (defense in depth against a stale or mis-built request) or delete the
field and document the deadline as the authority lifetime. Shipping a silent overclaim is the worst of the three
options.

### 4. Eight `assert` statements in production paths — banned by this repo's own rule and disabled under `-O`

`execution/sandbox/macos/broker.py:144,246,248,250` and `execution/sandbox/macos/execution.py:379,380,555,556`.
Four sit in the secret/network staging path, where a skipped assert passes `None` into CNI or Envoy configuration.
No `assert` exists in `src/loop/agent`, `backend`, `tooling`, `utils`, `instructions`, or `workspace`, so this is a
regression against house style, not consistency.

**Fix.** Raise the accurate exception (`BrokerPlanError`, `OciSpecificationError`) — or, better, make the `None` case
unrepresentable by splitting `BrokerPlan` into an offline plan and a networked plan with non-optional fields. That
removes both the asserts and the branching.

---

## P1 — fix in this change

### 5. `VirtualPathTree` root `/` matches nothing

`permissions/execution.py:309-312`: with `root == "/"` the prefix test degenerates to
`path.startswith("//")`, so an admin or product grant on `/**` never covers `/workspace/…` → silent re-prompt or
unexplained denial.

```python
prefix = "" if root == "/" else root
return (path == root or path.startswith(f"{prefix}/")) and requested.effects <= granted.effects
```

Add a unit test for the root case; nothing in `src/` currently uses a `/` tree, which is why it is latent.

### 6. `manage_sandbox()` breaks the constructor invariant it documents

`execution/command.py:216-218` clears `runtime_digest` and `runtime_ready`. In the documented "already-attested digest"
mode (`runtime_resolver is None`, the basis for the `# pragma: no cover - constructor invariant` at
`command.py:353`), every later `execute()` returns `InfrastructureFailure` forever. Either do not clear the digest when
there is no resolver, or make the resolver mandatory.

### 7. Unconstrained identifiers reach paths and container names

- `BaseExecutionRequest.request_id` has only `min_length=1` (`contracts.py`) yet becomes
  `container_name = f"loop-{request_id}"` (`oci/spec.py:214`) and is then checked by `_VALUE`
  (`oci/control.py:896`), which permits `/` and `:`. Sibling identities **are** constrained
  (`JobHandle.job_id`, `SecretExposure.secret_id`).
- `BaseOpaqueIdentifier.value` (`execution/vfs/models.py:20`) is unconstrained and is interpolated into a filesystem
  path in `macos/composition.py:351-352` (`store / snapshot_id.value / "manifest.json"`).
- `_attempt_arguments` passes `spec.cwd` and `spec.argv` through **without** `_value()`/`_absolute_path` and without a
  `--` separator, unlike every other slot (`oci/control.py:560-576`), so a `DirectExecutionRequest` whose `argv[0]`
  starts with `-` relies on CLI parsing behaviour rather than contract enforcement.

**Fix.** One shared identity pattern declared in `contracts.py`; validate `cwd` with `_absolute_path`; emit `--` before
argv.

### 8. PTY input can be silently truncated; `wait()` has an undocumented coupling

`oci/process.py:141` ignores the return value of `os.write`, while `vfs/staging.py:80-86` implements the correct
`memoryview` loop for the same problem. Tool input is allowed up to 64 KB, which exceeds a tty buffer; a partial write
is silent data loss. Reuse the staging loop — a good candidate for a shared helper in `loop/utils/` per the coding
skill's guidance.

Related: `oci/process.py:186-200` (`wait()`) does not drain the frame queue, so any caller that waits without a
concurrent drain thread trips `_backpressure_exceeded` → `OciSessionOutputFailure`. `macos/execution.py` gets this
right; document the precondition on `wait()` and on `queue_limit`.

### 9. Mixed clocks in durable-job authorization

`permissions/execution_adapter.py:299` combines the injected wall clock (`now_ns`, line 320) with a hardcoded
`time.monotonic_ns()`, so an injected clock cannot control that path even though the class advertises `clock_ns`.
Derive "remaining lease" through one named monotonic helper and keep the injected clock for grant matching only. The
`macos/jobs.py:624` ↔ `command.py:361` monotonic pairing is otherwise correct.

### 10. CI runs pull-request code on a self-hosted macOS ARM runner

`.github/workflows/release-promotion.yml:4-8` triggers on `pull_request` with no `environment:` gate, so untrusted PR
code executes on the security-sensitive self-hosted runner (the canonical self-hosted-runner abuse vector). Both
workflows also omit `permissions:` (default write-all), pin actions to mutable major tags, and have no `concurrency`
group. In `quality.yml`, `git diff --check HEAD^` fails on single-commit or shallow checkouts, and the new
`make lint` target (which also checks `git diff --cached --check`) is not what CI runs.

**Fix.** Drop `pull_request` from the promotion workflow (or gate it behind an `environment:` with required reviewers),
add `permissions: contents: read`, add `concurrency`, pin actions to SHAs, and have CI call `make lint`.

---

## P2 — recommendations

- **Ship the capability or stop advertising it.** `application/execution.py:120-130` never passes
  `supports_network_effects` (defaults `False`) and sets `supports_secret_exposures=False`, so every `run_command` with
  effect leases returns `UnsupportedCapability` (`execution/command.py:146-149`) — yet the tool schema still exposes
  `network_connections` / `network_listeners` / `secret_exposures` to the model (visible in this very session). Build
  the schema from the composed backend's declared support, or do not register those parameters. Roughly 8k lines of
  security-critical broker code are then exercised only by `scripts/verify_macos_execution_native.py`, never by the
  product.
- **Subnet allocation.** `execution/broker.py:112-116` derives `10.240.{16 + h % 224}.0/24` from a random lease id: a
  224-slot birthday space, no availability probe, no guaranteed release on partial failure, inside a range real
  corporate/VPN networks occupy. Use a per-guest allocator that probes `ip route` / bridge state and retries.
- **Layering.** `tools/system.py:14` imports `OciSignal` from `execution.sandbox.oci.control` into the public tool
  layer, so the model-facing tool hard-codes one backend's enum. Move the signal vocabulary to `contracts.py` (like
  `JobOperation`) and map per backend. `JobOperation.SUSPEND` / `RESUME` have no caller and no tool — dead enum members
  that the coverage gate must paper over.
- **String-coupled control flow** in three places: `permissions/host_adapter.py:107-110`
  (`reason == "Rejected by the user."`, `"no interactive user" in reason`) and `tools/files.py:770`
  (`"regex parse error" in detail`). Return a typed `denial_reason` from `ExecutionAuthorizationResult`; raise a typed
  `InvalidSearchPattern`.
- **Two parallel authorization systems.** File tools and presets still use `Action` + `PermissionManager.authorize`;
  execution uses typed `PolicyGrant` + `authorize_execution`, and `_command_plan` (`tools/system.py:30-36`) now returns
  no actions at all, so `run_command` bypasses the Action pipeline entirely. Acceptable as a migration state; document
  the enforcement point and the deprecation path. In `manager._evaluate_execution:505-511`, prefer enum `is` / `in`
  comparisons over `request.effect.value == "fs.read"` and `boundary.value == "sandbox"`.
- **`search_text_paths` lost its wall-clock guard** by moving in-process (`utils/search.py`): a model-supplied
  catastrophic pattern (`(a+)+$`) now blocks the agent, and one unreadable file aborts the entire search — only the
  caller's broad `except` keeps it a `Problem`. Bound per-line work, use the `re` timeout parameter (3.13+), and
  isolate failures per file.
- **Provenance overclaims.** `execution/runtime/provenance.py` verifies that SBOM/SLSA *statements* name the pinned
  digest but never verifies a signature, while the module docstring says "independently verifiable". Verify provenance
  signatures or narrow the claim to "relationship validation anchored on digest pinning and TLS". Also
  `subject_digest = f"sha256:{…}"` (lines 68-70) silently assumes hex SHA-256 — require the algorithm; and artifacts
  outside `LIMA`/`NERDCTL` with non-OCI acquisition (e.g. the lima disk) need no evidence at all — confirm that is
  intended and record the ceiling.
- **Static analysis gaps.** No type checker runs in CI despite pervasive annotations;
  `host/supervisor.py:175` (`_host_environment(entries, executable_dir)`), `macos/execution.py:471`
  (`_await_running(control_plane, …)`), and `macos/composition.py:351` (`_load_manifest(self, snapshot_id)`) are
  unannotated. `pylint` is configured in `pyproject.toml` but never invoked; `[tool.black]` and `[tool.isort]` are
  stale now that `ruff format` is authoritative.
- **Docstring conventions.** `application/secrets.py:74` (`__repr__`) and the `__post_init__` methods in
  `oci/spec.py:63` and `oci/control.py:63` carry docstrings, which the project rule forbids for dunders; three
  `__init__` docstrings duplicate the class `Args:` block that the rule says should own them.
- **Local state paths.** `application/execution.py:80-82` and the promotion workflow use predictable
  `/private/tmp/loop-{uid}-{hash}` roots. Ownership is correctly re-checked (`macos/backend.py:185-192` checks
  `st_uid`, symlink, and mode), so the residual risk is local DoS by pre-creating the path — worth a random suffix or
  `mkdtemp`, plus a note on why `/private/tmp` is required (Lima path length) if that is the reason.

---

## What is good, and should be preserved

- Contract-level capability↔lease cross-validation (`contracts.py:validate_virtual_cwd`), uniqueness checks, and the
  `is_global` / no-IP-literal authority rules in `NetworkConnectionLease.validate_destination` are exactly right.
- Fail-closed posture is consistent: `ExecutionService.execute` wraps the adapter, `HostExecutionBroker._record`
  refuses to start host work when the audit sink throws, and unknown capabilities return `UnsupportedCapability` rather
  than degrading.
- The `create` → harden `/etc/hosts` → `start-gate` → `run` sequence (`macos/execution.py:333-357`,
  `oci/control.py:566-576`) is a correct answer to the DNS/hosts TOCTOU problem that most sandboxes get wrong.
- Secret hygiene: `repr=False` on `BrokerPlan` binary fields and `BrokerSecretFile`, `JobHandle.token` excluded from
  repr, ASCII/CRLF/NUL validation on header and env material, and no ambient environment fallback in
  `ApplicationSecretAuthority`.
- `StagedContentStore` (O_EXCL|O_NOFOLLOW, fsynced links, dev/inode root identity, reference-pattern validation) is a
  model of what the rest of the boundary should look like.
- `InfrastructureProcessCommand`'s descriptor-derived executable and directory identities, revalidated immediately
  before spawn, are the right primitive; the `ns:`-scoped Envoy sidecar with no dynamic discovery and no default
  filter chain is a sound, deny-by-default design.

---

## Suggested minimal change set

1. `oci/spec.py` + `oci/control.py` + `runtime/image.py`: non-root `run_as_user` in the spec, `/home/agent` ownership,
   `XDG_CACHE_HOME=/cache`, `--` before argv, `_absolute_path` on `cwd` — plus a native-gate writability assertion.
2. `host/models.py` + `host/broker.py`: disclose the resolved host cwd; drop or validate `display_cwd`.
3. `contracts.py` + `broker.py` + tool schema: make `max_bytes` real or remove the claim; enforce or remove
   `ExecutionLease.expires_at_ns`.
4. Replace all eight `assert`s with typed failures or non-optional plan shapes.
5. `permissions/execution.py:309` root-`/` fix + test; `command.py:216` runtime-digest fix; shared identity pattern for
   `request_id` / `BaseOpaqueIdentifier`; PTY write loop; clock helper; workflow triggers, `permissions:`, and
   `concurrency`.

Each item is small and local; none requires rethinking the architecture.

---

## Assumptions

- The working tree equals the index for every file reviewed (git status showed only staged entries, no second-column
  modifications), so reading files from disk reproduces the staged content.
- The `run_command` failures observed during this session are environmental; they are **not** presented as evidence of
  a defect in the staged code, though the failure shape (`InfrastructureFailure` with empty output and no
  `diagnostic_id` propagation to the tool detail) is worth reproducing once the sandbox is healthy.
- Severity assumes the current product posture: macOS/arm64 only, network and secret effects disabled at
  `application/execution.py`, host execution gated by `allow_host_processes`.
