"""Install verified runtime artifacts into private content-addressed directories."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import uuid
from pathlib import Path
from typing import Protocol

from filelock import FileLock

from ... import constants
from .models import Artifact


class InstallError(RuntimeError):
    """Report an unsafe or failed artifact installation."""


class UnsupportedArtifactProvider(InstallError):
    """Report an acquisition kind absent from the composition root."""


class UnsafeArchive(InstallError):
    """Report archive content that cannot be extracted safely."""


class ArtifactInstaller(Protocol):
    """Install a verified artifact for one acquisition kind."""

    def install(self, artifact: Artifact, source: Path | None, destination: Path) -> Path:
        """Install an artifact and return its immutable content directory.

        Args:
            artifact (Artifact): Manifest-selected artifact to install.
            source (Path | None): Verified source file, when required.
            destination (Path): Staging directory for installed content.

        Returns:
            Path: Installed content directory.
        """


class FileInstaller:
    """Install a verified ordinary file beneath its declared layout."""

    def install(self, artifact: Artifact, source: Path | None, destination: Path) -> Path:
        """Install the verified source without executing it.

        Args:
            artifact (Artifact): Manifest-selected file artifact.
            source (Path | None): Verified downloaded source file.
            destination (Path): Staging directory for installed content.

        Returns:
            Path: Staging directory containing the installed file.

        Raises:
            InstallError: If the artifact layout is not a single file.
        """
        if source is None or artifact.layout is None or len(artifact.layout.files) != 1:
            raise InstallError(
                "File artifact layout requires exactly one verified source destination."
            )
        destination.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        target = destination / artifact.layout.files[0]
        target.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True)
        shutil.copyfile(source, target, follow_symlinks=False)
        os.chmod(
            target,
            constants.PRIVATE_DIRECTORY_MODE
            if target.relative_to(destination).as_posix() in artifact.layout.executables
            else constants.PRIVATE_FILE_MODE,
        )
        return destination


class ArchiveInstaller:
    """Extract a verified tar archive through a restrictive entry policy."""

    _maximum_files: int
    _maximum_bytes: int

    def __init__(
        self,
        maximum_files: int = constants.RUNTIME_DEFAULT_ARCHIVE_FILES,
        maximum_bytes: int = constants.RUNTIME_DEFAULT_ARCHIVE_BYTES,
    ) -> None:
        self._maximum_files = maximum_files
        self._maximum_bytes = maximum_bytes

    def install(self, artifact: Artifact, source: Path | None, destination: Path) -> Path:
        """Extract safe regular files and directories only.

        Args:
            artifact (Artifact): Manifest-selected archive artifact.
            source (Path | None): Verified downloaded archive.
            destination (Path): Staging directory for extracted content.

        Returns:
            Path: Staging directory containing extracted content.

        Raises:
            InstallError: If archive content is unsafe or does not match layout.
        """
        if source is None or artifact.layout is None:
            raise InstallError("Archive artifact requires verified source and layout.")
        try:
            with tarfile.open(source, "r:*") as archive:
                destination.mkdir(
                    mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True
                )
                members = [
                    member
                    for member in archive.getmembers()
                    if not (member.isdir() and member.name in {".", "./"})
                ]
                if len(members) > self._maximum_files:
                    raise UnsafeArchive("Archive exceeds the file-count limit.")
                names: set[str] = set()
                portable_names: set[str] = set()
                total = 0
                for member in members:
                    name = _safe_member_name(member.name)
                    portable_name = name.casefold()
                    declared_link = artifact.layout.symlinks.get(name)
                    if (
                        name in names
                        or portable_name in portable_names
                        or member.islnk()
                        or (member.issym() and member.linkname != declared_link)
                        or not (member.isdir() or member.isfile() or member.issym())
                        or member.pax_headers
                    ):
                        raise UnsafeArchive("Archive contains duplicate, link, or special entry.")
                    if member.mode & (stat.S_ISUID | stat.S_ISGID):
                        raise UnsafeArchive("Archive contains privileged mode bits.")
                    names.add(name)
                    portable_names.add(portable_name)
                    total += member.size
                    if total > self._maximum_bytes:
                        raise UnsafeArchive("Archive exceeds unpacked-byte limit.")
                for member in members:
                    name = _safe_member_name(member.name)
                    target = destination / name
                    parents = target.parent.relative_to(destination).parents
                    if destination.is_symlink() or any(
                        (destination / parent).is_symlink() for parent in parents
                    ):
                        raise UnsafeArchive("Archive destination contains a symbolic link.")
                    if member.isdir():
                        target.mkdir(
                            mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=False
                        )
                    elif member.issym():
                        target.parent.mkdir(
                            mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True
                        )
                        target.symlink_to(member.linkname)
                    else:
                        target.parent.mkdir(
                            mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True
                        )
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise InstallError("Archive regular file has no content.")
                        written = 0
                        with extracted, target.open("xb") as output:
                            while chunk := extracted.read(
                                constants.RUNTIME_ARCHIVE_READ_CHUNK_BYTES
                            ):
                                written += len(chunk)
                                if written > member.size:
                                    raise UnsafeArchive("Archive member exceeds declared size.")
                                output.write(chunk)
                        if written != member.size:
                            raise UnsafeArchive("Archive member was truncated.")
                        os.chmod(
                            target,
                            constants.PRIVATE_DIRECTORY_MODE
                            if name in artifact.layout.executables
                            else constants.PRIVATE_FILE_MODE,
                        )
        except (tarfile.TarError, OSError) as error:
            raise InstallError("Archive extraction failed.") from error
        return destination


def install_content(
    artifact: Artifact,
    source: Path | None,
    root: Path,
    installer: ArtifactInstaller,
) -> Path:
    """Atomically install one artifact under its content digest directory.

    Args:
        artifact (Artifact): Manifest-selected artifact to install.
        source (Path | None): Verified source file, when required.
        root (Path): Private artifact storage root.
        installer (ArtifactInstaller): Installer for the artifact acquisition kind.

    Returns:
        Path: Immutable content directory for the artifact.

    Raises:
        InstallError: If installation is unsafe or does not match its declared layout.
    """
    digest = artifact.digest.removeprefix(constants.SHA256_PREFIX)
    artifact_root = root / artifact.artifact_id
    artifact_root.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise InstallError("Artifact root is not a private directory.")
    final = artifact_root / digest
    lock = FileLock(str(artifact_root / ".install.lock"))
    with lock:
        if final.is_dir() and not final.is_symlink() and _verify_layout(final, artifact):
            return final
        if final.exists():
            raise InstallError("Existing content digest directory failed verification.")
        staging = artifact_root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True)
        try:
            installer.install(artifact, source, staging)
            if not _verify_layout(staging, artifact, require_immutable=False):
                raise InstallError("Installed artifact does not match declared layout.")
            _fsync_tree(staging)
            _freeze_tree(staging, artifact)
            os.replace(staging, final)
            _fsync_directory(artifact_root)
            return final
        except BaseException:
            _remove_private_tree(staging, artifact_root)
            raise


def _safe_member_name(name: str) -> str:
    """Return a normalized archive member name or reject it."""
    if not name or name.startswith("/") or "\\" in name:
        raise UnsafeArchive("Archive contains an absolute or nonportable path.")
    parts = Path(name).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArchive("Archive contains an escaping path.")
    return "/".join(parts)


def _verify_layout(directory: Path, artifact: Artifact, *, require_immutable: bool = True) -> bool:
    """Verify all required declared files and executable modes."""
    if artifact.layout is None:
        return True
    expected = set(artifact.layout.files)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if (not artifact.layout.allow_unlisted_files and actual != expected) or not expected.issubset(
        actual
    ):
        return False
    actual_links = {
        path.relative_to(directory).as_posix(): os.readlink(path)
        for path in directory.rglob("*")
        if path.is_symlink()
    }
    if actual_links != artifact.layout.symlinks:
        return False
    for relative in artifact.layout.files:
        path = directory / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(constants.RUNTIME_ARCHIVE_READ_CHUNK_BYTES):
                digest.update(chunk)
        if digest.hexdigest() != artifact.layout.identities[relative]:
            return False
        mode = stat.S_IMODE(path.stat().st_mode)
        if require_immutable:
            expected_mode = 0o500 if relative in artifact.layout.executables else 0o400
            if mode != expected_mode:
                return False
        elif relative in artifact.layout.executables and not mode & stat.S_IXUSR:
            return False
    return True


def verify_content(directory: Path, artifact: Artifact) -> bool:
    """Return whether one existing content directory is safe and complete.

    Args:
        directory (Path): Candidate immutable content directory.
        artifact (Artifact): Artifact layout the directory must satisfy.

    Returns:
        bool: Whether the directory is a non-link directory matching the declared layout.
    """
    return directory.is_dir() and not directory.is_symlink() and _verify_layout(directory, artifact)


def _fsync_tree(directory: Path) -> None:
    """Fsync regular files and directories before publication."""
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    """Fsync one private directory."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_tree(directory: Path, artifact: Artifact) -> None:
    """Make verified installed content immutable to ordinary application writes."""
    if artifact.layout is not None:
        for relative in artifact.layout.files:
            os.chmod(
                directory / relative,
                0o500 if relative in artifact.layout.executables else 0o400,
            )
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_dir() and not item.is_symlink()),
        reverse=True,
    ):
        os.chmod(path, 0o500)
    os.chmod(directory, 0o500)


def _remove_private_tree(path: Path, root: Path) -> None:
    """Remove a resolved staging tree only when it remains under ``root``."""
    if (
        path.exists()
        and path.is_dir()
        and not path.is_symlink()
        and path.parent.resolve() == root.resolve()
        and path.name.startswith(".staging-")
    ):
        for item in path.rglob("*"):
            if not item.is_symlink():
                os.chmod(
                    item,
                    constants.PRIVATE_DIRECTORY_MODE
                    if item.is_dir()
                    else constants.PRIVATE_FILE_MODE,
                )
        os.chmod(path, constants.PRIVATE_DIRECTORY_MODE)
        shutil.rmtree(path)
