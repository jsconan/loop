"""Compile bound payload authority without kernel loading or unbounded parser work.

Use only in trusted host preparation. This owner does not authorize requests, create the
administrator-owned compiler lock, admit generations or establish backend readiness.
"""

from __future__ import annotations

import fcntl
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from .host import (
    COMPILER_SECONDS,
    MAX_SOURCE_BYTES,
    HostFile,
    compile_policy_source,
    runtime_read_roots,
)
from .policy import payload_policy

if TYPE_CHECKING:
    from ..contracts import SandboxRequest


def compile_payload(
    request: SandboxRequest,
    scratch: Path,
    cache_lock: Path,
    profile: str,
    compiler_lock: HostFile,
) -> bytes:
    """Serialize and strictly compile a current request within fixed operational limits.

    The parent must already have authorized the request and authenticated installation-wide
    authority. The separate read-only compiler lock must be shared across all retained
    generations. Compilation uses protected installed parser/prlimit inputs, an empty parser
    configuration, one worker, no cache and no kernel loading. Source is a private anonymous
    regular input, never a command argument. All diagnostics cause refusal and stay internal.
    Limits are candidate operational bounds pending native compatibility qualification.

    Args:
        request (SandboxRequest): Current authorized request with an absolute deadline.
        scratch (Path): Private scratch bound as the request's TMPDIR.
        cache_lock (Path): Authenticated payload cache lock denied by the filename policy.
        profile (str): Simple payload identity accepted by the renderer.
        compiler_lock (HostFile): Host-authenticated installation-wide read-only lock inode.

    Returns:
        bytes: Nonempty compiled single-profile policy, at most one MiB.

    Raises:
        ValueError: Request/source/lock authority is invalid, oversized or changes.
        TimeoutError: The request budget expires before or during compilation.
        OSError: Lock contention, missing/changed native inputs, parser diagnostics, failed
            compilation, cancellation or output overflow prevents preparation.
    """
    source = payload_policy(
        request, scratch, cache_lock, profile, runtime_roots=runtime_read_roots()
    ).encode("utf-8")

    if len(source) > MAX_SOURCE_BYTES:
        raise ValueError("Payload policy source exceeds the preparation bound.")

    if compiler_lock.identity[2] & 0o222:
        raise ValueError("Compiler lock must be read-only.")

    deadline = min(request.deadline, time.monotonic() + COMPILER_SECONDS)
    if time.monotonic() >= deadline:
        raise TimeoutError("Payload compilation deadline expired.")

    with ExitStack() as inputs:
        lock_descriptor = inputs.enter_context(compiler_lock.open_verified(deadline=deadline))
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compiled = compile_policy_source(source, deadline)
        inputs.enter_context(compiler_lock.open_verified(deadline=deadline))
        if not request.paths_are_current():
            raise ValueError("Policy request authority changed during compilation.")
        return compiled
