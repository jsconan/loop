"""Restore an installed Linux sandbox's volatile authority at system startup."""

import os
import stat
import sys
import time
from pathlib import Path

_INITIAL_USER_NAMESPACE_UID_MAP = ("0", "0", "4294967295")
_UNSAFE_HOST_PATH_MODE_BITS = 0o6022
_BOOT_RECOVERY_SECONDS = 60.0


def main() -> None:
    """Authenticate the fixed root-owned entry and restore its installed generation.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: The fixed entry or its protected ancestry is invalid.
        TimeoutError: Boot recovery exceeds its bounded startup time.
        OSError: Installed authority or native boot recovery is unavailable.
    """
    if os.getuid() != 0 or tuple(
        Path("/proc/self/uid_map").read_text(encoding="ascii").split()
    ) != (_INITIAL_USER_NAMESPACE_UID_MAP):
        raise PermissionError("Boot recovery requires initial-host administrator authority.")
    entry = Path(__file__).absolute()
    if entry.name != "loop_boot.py" or entry.resolve(strict=True) != entry:
        raise ValueError("Boot entry must use its protected canonical installed path.")
    for path in (entry, *entry.parents):
        details = path.lstat()
        if (
            details.st_uid != 0
            or details.st_mode & _UNSAFE_HOST_PATH_MODE_BITS
            or stat.S_ISLNK(details.st_mode)
        ):
            raise ValueError("Boot entry ancestry is not administrator protected.")
    sys.path.insert(0, str(entry.parent))
    from loop.execution.sandbox.linux.management import recover_boot_generation

    recover_boot_generation(
        entry.parent / "management.json", time.monotonic() + _BOOT_RECOVERY_SECONDS
    )


if __name__ == "__main__":
    main()
