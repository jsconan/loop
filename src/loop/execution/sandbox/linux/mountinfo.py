"""Read the current process mount inventory and descriptor mount identities."""

import sys
from pathlib import Path

MOUNTINFO_PATH = Path("/proc/self/mountinfo")
FDINFO_DIRECTORY = Path("/proc/self/fdinfo")
MINIMUM_MOUNTINFO_FIELDS = 10
_PATH_ESCAPES = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


def _decode_path(value: str) -> str:
    """Decode kernel mount path escapes once, rejecting ambiguous sequences."""
    parts = []
    while "\\" in value:
        prefix, _, value = value.partition("\\")
        parts.append(prefix)
        escape = value[:3]
        if escape not in _PATH_ESCAPES:
            raise ValueError("Invalid mountinfo path escape.")
        parts.append(_PATH_ESCAPES[escape])
        value = value[3:]
    parts.append(value)
    return "".join(parts)


def read_mountinfo() -> list[list[str]]:
    """Return the current process mount table as space-separated fields.

    Returns:
        list[list[str]]: Mountinfo rows in kernel order; root and mountpoint fields are decoded.

    Raises:
        OSError: The current process mount inventory cannot be read.
        ValueError: A root or mountpoint contains an invalid kernel path escape.
    """
    contents = MOUNTINFO_PATH.read_bytes().decode(sys.getfilesystemencoding(), "surrogateescape")
    rows = []
    for line in contents.split("\n"):
        if line:
            fields = line.split(" ")
            if len(fields) >= 5:
                fields[3] = _decode_path(fields[3])
                fields[4] = _decode_path(fields[4])
            rows.append(fields)
    return rows


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
