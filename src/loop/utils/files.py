"""Provide general filesystem mutation utilities."""

import os
import stat
from pathlib import Path
from uuid import uuid4

from .. import constants
from .hashing import sha256_digest
from .path import open_selected_parent


def is_binary_file(path: Path, probe_bytes: int = constants.DEFAULT_STREAM_CHUNK_SIZE) -> bool:
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
    expected_ctime_ns: int | None = None,
    expected_link_count: int | None = None,
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
        expected_ctime_ns (int | None): Change timestamp from authorized alias accounting.
            Without a bound timestamp, multiply linked destinations remain unavailable.
        expected_link_count (int | None): Link count from the same alias snapshot; required
            with its timestamp to replace a multiply linked destination.
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
    parent_identity = (
        (expected_parent_device, expected_parent_inode)
        if expected_parent_device is not None and expected_parent_inode is not None
        else None
    )
    try:
        with open_selected_parent(path, parent_identity) as parent:
            parent_status = os.fstat(parent)
            if (
                expected_parent_device is not None
                and parent_status.st_dev != expected_parent_device
                or expected_parent_inode is not None
                and parent_status.st_ino != expected_parent_inode
            ):
                raise RuntimeError(
                    "The target parent changed after approval; mutation was cancelled."
                )
            temporary = f".{path.name}.{uuid4().hex}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(content)
                if expected_digest is not None:
                    try:
                        selected = os.open(
                            path.name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                            dir_fd=parent,
                        )
                    except OSError as error:
                        raise RuntimeError(
                            "The target changed after approval; replacement was cancelled."
                        ) from error
                    with os.fdopen(selected, "rb") as stream:
                        status = os.fstat(stream.fileno())
                        if (
                            not stat.S_ISREG(status.st_mode)
                            or status.st_nlink != 1
                            and (expected_ctime_ns is None or expected_link_count is None)
                            or expected_device is not None
                            and status.st_dev != expected_device
                            or expected_inode is not None
                            and status.st_ino != expected_inode
                            or expected_ctime_ns is not None
                            and status.st_ctime_ns != expected_ctime_ns
                            or expected_link_count is not None
                            and status.st_nlink != expected_link_count
                            or expected_mode is not None
                            and status.st_mode != expected_mode
                            or sha256_digest(stream.read()) != expected_digest
                        ):
                            raise RuntimeError(
                                "The target changed after approval; replacement was cancelled."
                            )
                    os.chmod(temporary, stat.S_IMODE(status.st_mode), dir_fd=parent)
                    os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
                else:
                    try:
                        os.link(
                            temporary,
                            path.name,
                            src_dir_fd=parent,
                            dst_dir_fd=parent,
                            follow_symlinks=False,
                        )
                    except FileExistsError as error:
                        raise RuntimeError(
                            "The target changed after approval; creation was cancelled."
                        ) from error
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
    except OSError as error:
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise RuntimeError(
                "The target parent changed after approval; mutation was cancelled."
            ) from error
        raise
