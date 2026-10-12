"""Run the retained exact installer-owned uninstaller after private system authentication.

Copy as loop_uninstall.py alongside the immutable isolated management modules. This entry point
works without the application distribution. It never elevates or reads credentials and requires
the record SHA256 shown in the freshly reviewed administrator action.
"""

import hashlib
import os
import re
import stat
import sys
import time
from pathlib import Path

_MAX_MANAGEMENT_RECORD_BYTES = 65_536
_UNSAFE_HOST_PATH_MODE_BITS = 0o6022
_INITIAL_USER_NAMESPACE_UID_MAP = ("0", "0", "4294967295")


def _reviewed_record_digest() -> str:
    """Require initial-host administrator authority and one reviewed record digest."""
    if os.getuid() != 0 or (
        tuple(Path("/proc/self/uid_map").read_text(encoding="ascii").split())
        != _INITIAL_USER_NAMESPACE_UID_MAP
    ):
        raise PermissionError("Uninstall requires initial-host administrator authentication.")

    if len(sys.argv) != 2 or re.fullmatch("[a-f0-9]{64}", sys.argv[1]) is None:
        raise ValueError("Supply the freshly reviewed management record SHA256.")
    return sys.argv[1]


def _validate_installed_entry() -> Path:
    """Require the fixed uninstaller path and administrator-protected ancestry."""
    entry = Path(__file__).absolute()
    if entry.name != "loop_uninstall.py" or entry.resolve(strict=True) != entry:
        raise ValueError("Uninstaller must use its protected canonical installed path.")

    for path in (entry, *entry.parents):
        details = path.lstat()
        if (
            details.st_uid != 0
            or details.st_mode & _UNSAFE_HOST_PATH_MODE_BITS
            or stat.S_ISLNK(details.st_mode)
        ):
            raise ValueError("Uninstaller ancestry is not administrator protected.")
    return entry


def _validate_management_record(entry: Path, expected_digest: str) -> Path:
    """Authenticate the protected management record against the reviewed digest."""
    record = entry.parent / "management.json"
    details = record.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != 0
        or details.st_nlink != 1
        or details.st_mode & _UNSAFE_HOST_PATH_MODE_BITS
        or details.st_size > _MAX_MANAGEMENT_RECORD_BYTES
        or hashlib.sha256(record.read_bytes()).hexdigest() != expected_digest
    ):
        raise ValueError("Reviewed uninstall authority changed.")
    return record


def _retire_generation(entry: Path, record: Path, expected_digest: str) -> None:
    """Load the retained verified owner and retire its authenticated idle generation."""
    sys.path.insert(0, str(entry.parent))

    # Only now load the retained immutable distribution, independently of the application.
    from loop.constants import NATIVE_MANAGEMENT_TIMEOUT_SECONDS
    from loop.execution.sandbox.linux.management import uninstall_authority, uninstall_generation

    authority, installed_entry = uninstall_authority(record)
    if authority.digest != expected_digest or installed_entry.path != entry:
        raise ValueError("Reviewed uninstall entry authority changed.")

    with installed_entry.open_verified():
        uninstall_generation(
            record,
            time.monotonic() + NATIVE_MANAGEMENT_TIMEOUT_SECONDS,
        )
    print("Authenticated idle generation uninstalled; capacity released.", flush=True)


def main() -> None:
    """Authenticate retained code ancestry and exact record bytes before checked idle retirement.

    Raises:
        PermissionError: Initial-host administrator authority is absent.
        ValueError: Arguments, retained code ancestry or reviewed record bytes differ.
        OSError: Protected state inspection or checked retirement fails.
    """
    expected_digest = _reviewed_record_digest()
    entry = _validate_installed_entry()
    record = _validate_management_record(entry, expected_digest)
    _retire_generation(entry, record, expected_digest)


if __name__ == "__main__":
    main()
