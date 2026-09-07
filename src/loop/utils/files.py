"""Provide general filesystem mutation utilities."""

import stat
import tempfile
from pathlib import Path

from .hashing import sha256_digest


def is_binary_file(path: Path, probe_bytes: int = 8192) -> bool:
    """Return whether a file begins with a binary NUL-byte marker.

    Args:
        path (Path): File to inspect.
        probe_bytes (int): Maximum leading bytes inspected. Defaults to 8192.

    Returns:
        bool: Whether the inspected prefix contains a NUL byte.
    """
    with path.open("rb") as stream:
        return b"\0" in stream.read(probe_bytes)


def write_text_atomically(
    path: Path,
    content: str,
    *,
    expected_digest: str | None,
    expected_device: int | None = None,
    expected_inode: int | None = None,
    expected_mode: int | None = None,
    expected_parent_device: int | None = None,
    expected_parent_inode: int | None = None,
) -> None:
    """Atomically write UTF-8 text when the destination has the expected state.

    Args:
        path (Path): Destination file to create or replace.
        content (str): UTF-8 text to write.
        expected_digest (str | None): Required SHA-256 digest for an existing destination, or
            ``None`` to require that the destination does not exist.
        expected_device (int | None): Expected device identity for replacement, when captured.
        expected_inode (int | None): Expected inode identity for replacement, when captured.
        expected_mode (int | None): Expected permission and type bits for replacement.
        expected_parent_device (int | None): Expected parent-directory device identity.
        expected_parent_inode (int | None): Expected parent-directory inode identity.

    Raises:
        RuntimeError: If the destination does not have the expected kind, identity, or content.
        OSError: If staging or committing the content fails.

    Existing regular-file permission bits are copied to the staged file before replacement.
    Ownership, timestamps, extended attributes, ACLs, and platform-specific metadata are not
    preserved. The digest and identity check immediately precedes atomic name replacement, but is
    not an atomic compare-and-swap against unrelated writers on platforms without such a primitive.
    """
    temporary_path = None
    try:
        try:
            parent = path.parent.lstat()
        except OSError as exc:
            raise RuntimeError(
                "The target parent changed after approval; mutation was cancelled."
            ) from exc
        if (
            not stat.S_ISDIR(parent.st_mode)
            or expected_parent_device is not None
            and parent.st_dev != expected_parent_device
            or expected_parent_inode is not None
            and parent.st_ino != expected_parent_inode
        ):
            raise RuntimeError("The target parent changed after approval; mutation was cancelled.")
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(content)
            temporary_path = Path(file.name)

        if expected_digest is not None:
            try:
                status = path.lstat()
            except OSError as exc:
                raise RuntimeError(
                    "The target changed after approval; replacement was cancelled."
                ) from exc
            identity_matches = (
                stat.S_ISREG(status.st_mode)
                and path.is_file()
                and (expected_device is None or status.st_dev == expected_device)
                and (expected_inode is None or status.st_ino == expected_inode)
                and (expected_mode is None or status.st_mode == expected_mode)
            )
            if not identity_matches or sha256_digest(path.read_bytes()) != expected_digest:
                raise RuntimeError("The target changed after approval; replacement was cancelled.")
            temporary_path.chmod(stat.S_IMODE(status.st_mode))
            temporary_path.replace(path)
        else:
            if path.exists() or path.is_symlink():
                raise RuntimeError("The target changed after approval; creation was cancelled.")
            path.hardlink_to(temporary_path)
            temporary_path.unlink()
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
