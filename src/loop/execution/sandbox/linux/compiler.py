"""Compile bound payload authority without kernel loading or unbounded parser work.

Use only in trusted host preparation. This owner does not authorize requests, create the
administrator-owned compiler lock, admit generations or establish backend readiness.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from .... import constants
from .host import (
    COMPILER_SECONDS,
    MAX_SOURCE_BYTES,
    HostFile,
    compile_policy_source,
)
from .policy import payload_policy

if TYPE_CHECKING:
    from ..contracts import SandboxRequest

_LOCK_RETRY_SECONDS = 0.01


class CompilationCancelled(Exception):
    """Signal cancellation before a request enters native execution."""


def compile_payload(
    request: SandboxRequest,
    scratch: Path,
    cache_lock: Path,
    profile: str,
    runtime_roots: tuple[Path, ...],
    compiler_lock: HostFile,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> bytes:
    """Serialize and strictly compile a current request within fixed operational limits.

    The parent must already have authorized the request and authenticated installation-wide
    authority. The separate read-only compiler lock must be shared across all retained
    generations. Compilation uses protected installed parser/prlimit inputs, an empty parser
    configuration, one worker, no cache and no kernel loading. Source is a private anonymous
    regular input, never a command argument. All diagnostics cause refusal and stay internal.
    Concurrent requests wait for the compiler lock within their original request deadline;
    parser work remains separately bounded once the lock has been acquired.

    Args:
        request (SandboxRequest): Current authorized request with an absolute deadline.
        scratch (Path): Private scratch bound as the request's TMPDIR.
        cache_lock (Path): Authenticated payload cache lock denied by the filename policy.
        profile (str): Simple payload identity accepted by the renderer.
        runtime_roots (tuple[Path, ...]): Authenticated protected runtime directories included in
            the canonical policy used to derive ``profile``.
        compiler_lock (HostFile): Host-authenticated installation-wide read-only lock inode.
        cancelled (Callable[[], bool] | None): Optional trusted cancellation check while waiting
            for the compiler lock; None does not check cancellation during the wait.

    Returns:
        bytes: Nonempty compiled single-profile policy, at most one MiB.

    Raises:
        ValueError: Request/source/lock authority is invalid, oversized or changes.
        TimeoutError: The request budget expires before or during compilation.
        CompilationCancelled: The trusted cancellation check signals a cancelled request.
        OSError: Missing/changed native inputs, parser diagnostics, failed
            compilation, cancellation or output overflow prevents preparation.
    """
    source = payload_policy(
        request, scratch, cache_lock, profile, runtime_roots=runtime_roots
    ).encode("utf-8")

    if len(source) > MAX_SOURCE_BYTES:
        raise ValueError("Payload policy source exceeds the preparation bound.")

    if compiler_lock.identity[2] & constants.WRITE_MODE_MASK:
        raise ValueError("Compiler lock must be read-only.")

    if time.monotonic() >= request.deadline:
        raise TimeoutError("Payload compilation deadline expired.")

    with ExitStack() as inputs:
        lock_descriptor = inputs.enter_context(
            compiler_lock.open_verified(deadline=request.deadline)
        )
        while True:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if cancelled is not None and cancelled():
                    raise CompilationCancelled("Payload compilation cancelled.") from error
                remaining = request.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Payload compilation deadline expired.") from error
                time.sleep(min(_LOCK_RETRY_SECONDS, remaining))
        deadline = min(request.deadline, time.monotonic() + COMPILER_SECONDS)
        compiled = compile_policy_source(source, deadline)
        inputs.enter_context(compiler_lock.open_verified(deadline=deadline))
        if not request.paths_are_current():
            raise ValueError("Policy request authority changed during compilation.")
        return compiled
