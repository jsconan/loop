# Review: staged command-execution changeset

Scope: all staged changes (168 files, +33,285 / −3,408 lines).
Method: direct source inspection of the security-critical paths (contracts, broker,
process infrastructure, permissions, host boundary, vfs staging, tools, workflows).
Limitation: the managed sandbox runtime was down during review, so the test suite,
ruff, and `git diff --check` were not executed. Re-run the workflow checks before merging.

## Overall verdict

State-of-the-art design, not the common "shell + proxy env vars" pattern. The changeset
implements the stronger model used by production agent sandboxes:

- **Explicit effect leases** (`NetworkConnectionLease` / `NetworkListenerLease` /
  `SecretExposure`) validated by Pydantic. No capability inference from script parsing;
  fail-closed on anything unrepresentable. Non-public and localhost upstream addresses are
  rejected at the contract layer.
- **Broker-enforced network**: dedicated no-default-route CNI bridge + static Envoy
  bootstrap with SNI/authority-exact routing, circuit breakers, per-connection byte
  buffers, and a start-gate file preventing staging TOCTOU. Correct approach for
  per-attempt egress control inside a container.
- **Secret hygiene**: audience-bound resolution, delimiter validation, `repr=False`
  everywhere, in-memory overwrite on close, `/run/secrets/`-only raw files, and a
  cleanup script for the obsolete session-snapshot field.
- **Host boundary** as a separately authorized path: descriptor-walk (`O_NOFOLLOW`)
  identity verification, request-fingerprint lease, sealed scrubbed environment
  (`LD_*`/`DYLD_*`/`IFS`/`PATH` etc. rejected), process-group kill, sanitized audit.
- **Permissions** as typed grants with deny-dominance, scope lifetimes, revocation, and
  audit; durable jobs authenticated by 256-bit bearer tokens.
- Removal of the old `execution/policy` package and `utils/process` leaves no dangling
  imports (verified by search).

The architecture is correct and modern. The defects follow, in priority order.

## Bugs to fix before merging

### 1. Envoy vhost domain `{hostname}:*` is not a valid pattern — plain HTTP on non-standard ports will 404

`src/loop/execution/broker.py:359`

```python
"domains": [lease.hostname, f"{lease.hostname}:*"],
```

Envoy vhost domain matching supports exact strings and leading `*.`-label wildcards only.
`host:*` is treated as a *literal exact* domain that no real `:authority` can equal, so
the second entry is dead weight — and worse, clients connecting to a non-default port
send `Host: api.example.com:8080` (the test at `tests/execution/test_broker.py:196` uses
port 8080), which matches neither entry, so Envoy 404s every request. Only HTTP leases
on port 80 work; the SNI path (HTTPS/TLS) is unaffected.

**Proposed fix:** give the listener's virtual host `domains: ["*"]` and enforce
authority at the route level, where Envoy does support it — a `:authority` route match
(exact `hostname` plus a `safe_regex` such as `^hostname:port$`, or two exact-match
routes). Keep one vhost per lease so the per-credential `request_headers_to_add` stays
authority-scoped. Add a test that asserts the *match semantics* (current tests only
assert JSON shape, which is why this slipped through), ideally one integration test that
boots the pinned Envoy image and asserts a `Host: host:8080` request is routed.

### 2. Tree grant on `/` matches nothing

`src/loop/permissions/execution.py`, `resource_matches`

```python
root = granted.root.removesuffix("/**").rstrip("/") or "/"
return (path == root or path.startswith(f"{root}/")) and requested.effects <= granted.effects
```

When the granted root is `/`, `rstrip("/")` yields `""` → `"/"`, and `startswith("//")`
is never true, so a deny (or allow) on the root tree covers *no* subpath. A product or
ADMIN deny on `/` — the canonical "lock everything" grant — is silently inert.

**Fix:** `path == root or path.startswith(root + "/") or (root == "/" and path.startswith("/"))`.
Note the current form is safe in the prefix direction (`/workspace` does not match
`/workspace-secret`); preserve that property in the fix.

### 3. Job-operation effects are semantically inverted

`src/loop/permissions/execution_adapter.py`, `authorize_job_operation`

```python
effect = (
    GrantEffect.PROCESS_SPAWN if operation is JobOperation.STATUS else GrantEffect.PROCESS_SIGNAL
)
```

`ATTACH` / `STDIN` / `RESIZE` (reads, input, terminal geometry) all map to
`PROCESS_SIGNAL`, and a *status query* maps to *spawn*. Functionally the flow stays
closed (each `durable-job:<id>:<op>` resource still prompts once), but remembered grants
and audit records will state that a user approved "process signal" for reading a job —
confusing for the permission UI and for anyone later auditing the grant store.

**Fix:** map by true semantics (status → a read/query effect; attach/stdin/resize →
process I/O; cancel/signal → `PROCESS_SIGNAL`), adding effects to `GrantEffect` if needed.

## Smaller issues

4. **String-coupled denial reasons** (`src/loop/permissions/host_adapter.py`): denial
   classification matches exact reason strings (`"Rejected by the user."`,
   `"no interactive user" in result.reason`). One doc-string edit in the manager silently
   reclassifies a user denial as a generic policy denial. Give
   `ExecutionAuthorizationResult` a typed denial-reason enum and match on that.

5. **Temp-file leak in `StagedContentStore.stage`** (`src/loop/execution/vfs/staging.py`):
   cleanup of `.incoming-*` happens only in the write-loop `except`. If `os.link` raises
   anything other than `FileExistsError`, the temp object leaks in the 0700 bucket until
   manual cleanup. Wrap the link in the same cleanup path (or `unlink` on `OSError`).

6. **Stale verification command** in `docs/command-execution-work-packages.json` (WP0.5):
   `pytest tests/execution/policy` references the deleted policy package — re-running that
   "complete" verification will fail. Update or annotate the package removal.
   (`docs/command-execution-linux-work-packages.json` is a separate L0–L8 plan file, not a
   duplicate.)

7. **Obscure expression** in `broker.py` `plan()`:
   `b"\n".join(environment) + (...) or None` works (empty bytes → `None`) but invites a
   precedence foot-gun; an explicit `if environment else None` is clearer.

8. `report-1.md` at the repo root is **untracked** — not part of the staged change, but
   delete it or keep it out of a later `git add -A`.

## Verification status

Could not run the test suite, ruff, or `git diff --check`: the managed sandbox runtime
failed mid-review (plain `echo` returned `process.infrastructure_failure`); all findings
were verified by direct source inspection. The one git-derived check needed (no dangling
references to the deleted `execution/policy`, `utils/process`, and `runtime/commands`
modules) passed via code search.

**Before merging:** re-run `make test-unit`, `make coverage-report`, and ruff. Items 1 and
2 are the only correctness defects that change behavior for a legitimate, authorized
request; the architecture itself needs no rework.
