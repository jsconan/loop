"""Read the current process mount inventory and descriptor mount identities."""

import sys
from pathlib import Path

MOUNTINFO_PATH = Path("/proc/self/mountinfo")
FDINFO_DIRECTORY = Path("/proc/self/fdinfo")
MINIMUM_MOUNTINFO_FIELDS = 10


def read_mountinfo() -> list[list[str]]:
    """Return the current process mount table as whitespace-separated fields.

    Returns:
        list[list[str]]: Mountinfo rows in kernel order, with each row split into fields.

    Raises:
        OSError: The current process mount inventory cannot be read.
    """
    return [
        line.split()
        for line in MOUNTINFO_PATH.read_text(
            encoding=sys.getfilesystemencoding(), errors="surrogateescape"
        ).splitlines()
    ]


def read_descriptor_mount_ids(descriptor: int) -> list[list[str]]:
    """Return mount IDs listed in fdinfo for one open descriptor.

    Args:
        descriptor (int): Open descriptor whose visible mount identity is required.

    Returns:
        list[list[str]]: Values following each ``mnt_id:`` field, including malformed empty
            values for the caller to reject as ambiguous.

    Raises:
        OSError: The descriptor's fdinfo cannot be read.
    """
    path = FDINFO_DIRECTORY / str(descriptor)
    return [
        line.split()[1:]
        for line in path.read_text(encoding="ascii").splitlines()
        if line.startswith("mnt_id:")
    ]
