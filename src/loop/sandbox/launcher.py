"""Apply command limits from inside an already-created sandbox."""

from __future__ import annotations

import os
import platform
import resource
import sys

_LIMITS = (
    (resource.RLIMIT_CPU, 30),
    (resource.RLIMIT_AS, 1_073_741_824),
    (resource.RLIMIT_NPROC, 64),
    (resource.RLIMIT_NOFILE, 256),
    (resource.RLIMIT_FSIZE, 16_777_216),
    (resource.RLIMIT_CORE, 0),
)


def _apply_resource_limits() -> None:
    """Lower soft limits before their hard ceilings, then make them irrevocable."""
    for kind, requested in _LIMITS:
        soft, hard = resource.getrlimit(kind)
        effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        try:
            if soft > effective:
                resource.setrlimit(kind, (effective, hard))
            if hard > effective:
                resource.setrlimit(kind, (effective, effective))
        except ValueError:
            if platform.system() != "Darwin":
                raise


def main() -> None:
    """Apply limits and replace this trusted launcher with the approved command.

    Raises:
        ValueError: If no command follows the launcher separator.
        OSError: If limits cannot be installed or the approved executable cannot start.
    """
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise ValueError("Sandbox launcher requires a command separator.") from exc
    command = sys.argv[separator + 1 :]
    if not command:
        raise ValueError("Sandbox launcher requires an executable.")
    _apply_resource_limits()
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":  # pragma: no cover - exercised by the packaged launcher entrypoint.
    main()
