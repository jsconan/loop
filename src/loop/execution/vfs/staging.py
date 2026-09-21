"""Persist immutable transaction content outside authenticated workspaces."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from ...utils import sha256_digest
from .models import ContentReference

_REFERENCE = re.compile(r"^staged:([0-9a-f]{64}):([0-9a-f]{64})$")


class StagedContentStore:
    """Own immutable content needed for journaled publication and recovery.

    Args:
        directory (Path): Loop-private root outside every workspace.
    """

    directory: Path
    _identity: tuple[int, int]

    def __init__(self, directory: Path) -> None:
        """Create and authenticate the private content root."""
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Staged content root must be a private directory.")
        directory.chmod(0o700)
        self.directory = directory.resolve(strict=True)
        metadata = self.directory.stat()
        self._identity = (metadata.st_dev, metadata.st_ino)

    def stage(
        self,
        staging_id: str,
        source: BinaryIO,
        *,
        maximum_bytes: int,
    ) -> ContentReference:
        """Durably stage one bounded byte stream and return its opaque identity.

        Args:
            staging_id (str): Attempt or transaction identity owning the staged bytes.
            source (BinaryIO): Readable binary stream positioned at its payload.
            maximum_bytes (int): Maximum accepted payload size.

        Returns:
            ContentReference: Immutable digest, size, and opaque store reference.

        Raises:
            ValueError: The identity, limit, stream, or private store is invalid.
        """
        if not staging_id or maximum_bytes < 0:
            raise ValueError("Staged content bounds are invalid.")
        self._validate_root()
        bucket_name = sha256_digest(staging_id)
        bucket = self.directory / bucket_name
        bucket.mkdir(mode=0o700, exist_ok=True)
        if bucket.is_symlink() or not bucket.is_dir():
            raise ValueError("Staged content bucket is invalid.")
        temporary = bucket / f".incoming-{uuid4().hex}"
        digest = hashlib.sha256()
        size = 0
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            while chunk := source.read(1024 * 1024):
                if not isinstance(chunk, bytes):
                    raise TypeError("Staged content source returned non-bytes data.")
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("Staged content exceeds its byte limit.")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Staged content write made no progress.")
                    view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        os.close(descriptor)
        digest_value = digest.hexdigest()
        target = bucket / digest_value
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            pass
        temporary.unlink()
        self._fsync(bucket)
        return ContentReference(
            digest=f"sha256:{digest_value}",
            size=size,
            reference=f"staged:{bucket_name}:{digest_value}",
        )

    def open(self, content: ContentReference) -> BinaryIO:
        """Open one authenticated immutable content record without following links.

        Args:
            content (ContentReference): Expected opaque staged-content identity.

        Returns:
            BinaryIO: Readable descriptor-backed stream owned by the caller.

        Raises:
            ValueError: The reference or stored object is invalid.
            FileNotFoundError: The staged object has already been reclaimed.
        """
        self._validate_root()
        match = _REFERENCE.fullmatch(content.reference)
        if match is None or content.digest != f"sha256:{match.group(2)}":
            raise ValueError("Staged content reference is invalid.")
        bucket_name, digest = match.groups()
        bucket = self.directory / bucket_name
        bucket_metadata = bucket.lstat()
        if stat.S_ISLNK(bucket_metadata.st_mode) or not stat.S_ISDIR(bucket_metadata.st_mode):
            raise ValueError("Staged content bucket is invalid.")
        descriptor = os.open(bucket / digest, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != content.size:
            os.close(descriptor)
            raise ValueError("Staged content object is invalid.")
        return os.fdopen(descriptor, "rb")

    def discard(self, staging_id: str) -> None:
        """Reclaim one staging bucket after denial or durable commit.

        Args:
            staging_id (str): Attempt or transaction identity owning the staged bytes.

        Raises:
            ValueError: The identity or private store is invalid.
        """
        if not staging_id:
            raise ValueError("Staged content identity is invalid.")
        self._validate_root()
        bucket = self.directory / sha256_digest(staging_id)
        try:
            metadata = bucket.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Staged content bucket is invalid.")
        shutil.rmtree(bucket)
        self._fsync(self.directory)

    def _validate_root(self) -> None:
        """Reject root replacement, aliases, and permission widening."""
        metadata = self.directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self._identity
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ValueError("Staged content root changed.")

    @staticmethod
    def _fsync(directory: Path) -> None:
        """Synchronize one private directory entry set."""
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
