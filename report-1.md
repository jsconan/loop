# Code Review: Sandboxed Command Execution & macOS Subsystem

## Summary

This is a foundational architectural change introducing a complete sandbox execution subsystem with macOS (Lima VM) and OCI backends, typed permission grants, durable job lifecycle, host execution broker, and staged VFS. The approach is state-of-the-art for a security-isolated agent framework.

## Positive Aspects (State-of-the-Art Approvals)

1. **Descriptor-based file identity verification** in `infrastructure/process.py` — Using `O_NOFOLLOW`, device+inode tracking, and `hashlib.file_digest` for executable provenance is excellent practice for mitigating TOCTOU race conditions.

2. **Broker plan pattern** in `broker.py` — Decoupling the declarative CNI/Envoy/secret plan from lifecycle management allows testing without external dependencies.

3. **Typed permission grants** replacing the old `ProcessTarget` model — The new `PolicyRequest`/`PolicyGrant` model with explicit effect types (`PROCES_SPAWN`, `PROCESS_SIGNAL`, etc.) is a significant improvement over implicit command-line parsing.

4. **Secret authority isolation** via `ApplicationSecretAuthority` — The in-memory, scope-bound, zero-fallback design with buffer zeroing on close is correct for secret lifecycle management.

5. **Durable job handles with opaque tokens** — Using `secrets.token_hex(32)` per job is correct for unguessable bearer authentication.

6. **Staged content store** — Using `O_EXCL | O_NOFOLLOW`, hard links for atomicity, and sha256-based referencing is the right pattern for immutable content stores.

7. **CNI/Envoy plan generation** — The static, no-dynamic-discovery approach with DNS-only isolation is correct for a security-constrained sandbox.

## Critical Bugs & Issues

### 1. **`_network_connection` validates `hostname` by stripping trailing dots, but raises on mismatch** (contracts.py, line ~210)

```python
hostname = self.hostname.rstrip(".").lower()
if (
    hostname != self.hostname  # BUG: original hostname isn't stripped
    ...
):
    raise ValueError("Network destination authority is invalid.")
```

This makes any hostname with a trailing dot fail validation, even though DNS allows trailing dots. The check should compare `hostname` (stripped) against `self.hostname` (stripped), not the original. **Fix**: `if hostname != self.hostname.rstrip("."): `

### 2. **`NetworkConnectionLease` rejects IP address hostnames but allows bare IPs in validation** (contracts.py)

The `validate_destination` validator checks `ip_address(hostname)` for IP addresses and rejects them ("require a DNS authority"), but the regex `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?` also rejects pure numeric labels. This is correct in intent but the error message says "Hostname-scoped network leases require a DNS authority" which is misleading since the hostname itself might be numeric but valid in DNS. The validation itself is correct — this is a documentation issue.

### 3. **`BrokerPlan` `_cni` function uses `name[-10:]` for bridge naming** (broker.py, line ~262)

```python
f"br{name[-10:]}"  # Linux bridge names have 15-char limit
```

This is technically correct but brittle — the bridge name could collide if two network names share the same suffix. For a security product, consider a hash-based bridge name for uniqueness.

### 4. **`_envoy` generates DNS listener with `gateway` but `_socket` forces TCP protocol** (broker.py, line ~318)

```python
def _udp_listener(gateway: str, port: int, cluster: str) -> dict[str, object]:
    address = _socket(gateway, port)  # Sets protocol="TCP" even for UDP
```

The `_socket` helper hardcodes `"protocol": "TCP"` which is incorrect for the UDP listener. This needs `_socket_udp` or a conditional.

### 5. **`InfrastructureProcessRunner` ignores `_BoundedDrain.error` in `run()`** (infrastructure/process.py)

If a reader thread hits an OSError (pipe closed, etc.), the error is stored in `_BoundedDrain.error` but never checked in the `run()` method before returning. The caller sees truncated output but no error — the error should either be surfaced or documented as expected behavior.

### 6. **`ApplicationSecretAuthority.close()` is not thread-safe** (application/secrets.py)

The `close()` method iterates `_material.values()` and clears the dict without a lock. If a concurrent `resolve()` call starts iteration after `clear()`, it raises a `RuntimeError: dictionary changed size during iteration`. In the current architecture this shouldn't happen (the authority is used only during a single request), but this assumption is implicit. **Fix**: Add a lock or document the single-threaded contract explicitly.

### 7. **`_open_absolute` uses `O_NOFOLLOW` on non-final components but not on the final one for directories** (infrastructure/process.py)

The `directory=True` path omits `O_NOFOLLOW` on the final component, allowing symlinks for the target directory itself. For a security product, the target should also be checked with `O_NOFOLLOW`. The comment "followed by checking it's a directory" in the calling code only checks the stat result, not the open path.

### 8. **`LimaInstanceContext.create` checks `root in forbidden` but not `root in forbidden.parents`** (macos/lima.py)

The check `root in forbidden.parents` exists but `forbidden in root.parents` should also be checked (i.e., the state root should not be *inside* the working directory or home directory). Currently if state_root equals `/workspace/.local/loop`, the check passes even though it's inside the workspace.

### 9. **`JobHandle` regex allows `job_id` up to 128 chars but `create()` doesn't validate** (contracts.py)

`JobHandle.create(job_id)` doesn't validate the `job_id` before constructing the handle. If the caller passes an invalid ID, Pydantic's `Field(pattern=...)` is applied at `model_validate`/`model_validate_python` time, not at construction. If `JobHandle` is used via `model_validate()`, it's fine, but the `create()` method bypasses validation.

### 10. **`_validate_target` in `SecretExposure` uses `re.fullmatch` but `self.target` is already validated by `Field(min_length=1, max_length=512)`** — Minor: redundant but correct.

## Design Concerns

### 11. **`SandboxCommandExecutor` swallows all exceptions from `_resolve_runtime()`**

```python
except Exception as error:  # Preparation failures become closed public results.
```

This catches `ValueError` (invalid digest), `RuntimeError` (no resolver), and any unexpected exception, converting them all to `InfrastructureFailure`. This is intentionally broad (fail-closed), but the diagnostic_id generation could be deterministic (e.g., hashed from the request) for better observability.

### 12. **`_BoundedDrain` runs as a daemon thread**

Daemon threads won't prevent process exit, which is correct for bounded reads, but means errors in the drain might be silently lost on early process termination.

### 13. **No validation that `OciAttemptBindings` values are unique**

The `OciAttemptBindings` class combines `network_name`, `command_address`, `host_aliases`, etc. without uniqueness validation. If two network leases resolve to the same `command_address`, this could cause Envoy misconfiguration.

### 14. **`StagedContentStore` uses `shutil.rmtree` for discard without revalidation**

After `shutil.rmtree(bucket)`, the directory itself is not re-validated (symlink check, permissions, identity). A TOCTOU window exists between the discard call and the next `_validate_root` call. For a security product, consider `os.open` + `fstat` to verify the bucket is truly gone.

### 15. **`HostExecutionBroker` allows `cwd` to escape workspace without `..`**

The `_host_cwd` method only rejects `..` in relative parts. A `cwd` of `/workspace/../../etc` (via `/workspace` followed by `../../etc` with `..` in different components) could still escape via `Path.relative_to().parts` that doesn't catch nested escaping. Actually, checking `..` in `relative.parts` catches this — it's fine.

## Minor Issues

16. **`_stream_output` loses the `handle`/`cursor` pagination from the original** — The old `run_command` tool returned a `handle` and `next_cursor` for large output. The new version only returns `truncated`/`capture_complete`. This is a behavior regression; consumers of large output lose the ability to fetch more. The `read_cached_content` tool still exists, but the integration between the two is broken.

17. **`ApplicationRuntime` creates a new `threading.Event()` for cancellation in `initialize` but also accepts one via `__init__`** — There's a subtle race: if `cancellation` is passed via `__init__` and then `initialize` creates a new one, the `__init__` version is lost. The code correctly does `cancellation = threading.Event()` in `initialize`, shadowing the parameter.

18. **`_command_plan` and `_host_command_plan` create `OperationPlan(arguments=dict(arguments))`** which loses any nested structure if `arguments` contains non-serializable types. This is fine for Pydantic but should be documented.

19. **`validate_infrastructure_command` calls `validate_verified_executable` which opens the file again** — Double-opening adds a small window for the file to change between the two validations. Acceptable for a security product but worth documenting.

20. **No `__all__` declarations in any of the new modules** — The coding skill requires module-level `__all__` immediately after the docstring in `__init__.py` files.

## Recommendations

1. **Fix bug #4** (UDP listener TCP protocol) — This is a functional bug that will break DNS sandboxing.
2. **Fix bug #1** (trailing dot handling) — This silently rejects valid DNS hostnames.
3. **Restore output pagination** (issue #16) — The integration between `run_command` output and `read_cached_content` is broken.
4. **Add `__all__`** to all new module `__init__.py` files.
5. **Add the `_validate_root` check after `shutil.rmtree`** in `StagedContentStore.discard()` for TOCTOU safety.
6. **Document the single-threaded contract** for `ApplicationSecretAuthority`.
7. **Consider deterministic diagnostic_id** (e.g., `hashlib.sha256(request_id.encode()).hexdigest()`) for observability.

## Verdict

**This is a strong architectural change** that correctly isolates execution, uses descriptor-based verification, implements typed permissions, and follows fail-closed semantics. The core patterns (broker plan, no-follow descriptor walks, immutable content stores, typed grants) are state-of-the-art.

The critical bugs (#4 — UDP protocol, #1 — hostname trailing dots, and #16 — lost output pagination) should be fixed before merge. The design concerns (#11, #13, #14) are edge cases that should be addressed in the next iteration.

**Estimated effort to fix critical issues: ~2 hours. Total recommended pre-merge fixes: ~4 hours.**
