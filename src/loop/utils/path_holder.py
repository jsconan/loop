"""Provide mutable path references and path discovery helpers."""

import os
from collections.abc import Iterator, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from loop.utils.value_holder import ValueHolder

PathInput = str | PathLike[str]


class PathHolder(ValueHolder[Path], os.PathLike[str]):
    """Hold a mutable reference to a filesystem path.

    Like every value holder, this class provides :attr:`value`, :meth:`get`, and :meth:`set` as
    one stable mutable indirection. Changing the value retargets future operations through this
    object. Paths derived from the holder are ordinary, independent snapshots::

        holder = PathHolder("/old")
        child = holder / "child"
        holder.set("/new")
        assert child == Path("/old/child")

    Equality compares the current path with another holder or an ordinary ``Path``. Because that
    value can change, holders are deliberately unhashable.

    Args:
        path (str | os.PathLike[str]): Initial filesystem path.
    """

    def __init__(self, path: PathInput) -> None:
        super().__init__(path)

    def _coerce(self, value: PathInput) -> Path:
        """Normalize a path-like input into the sole contained Path value."""
        return Path(value)

    @property
    def anchor(self) -> str:
        """Return the path anchor.

        Returns:
            str: Drive and root combination.
        """
        return self.value.anchor

    @property
    def drive(self) -> str:
        """Return the drive prefix.

        Returns:
            str: Drive prefix, if any.
        """
        return self.value.drive

    @property
    def name(self) -> str:
        """Return the final path component.

        Returns:
            str: Final path component.
        """
        return self.value.name

    @property
    def parent(self) -> Path:
        """Return a snapshot of the logical parent.

        Returns:
            Path: Parent of the current path.
        """
        return self.value.parent

    @property
    def parents(self) -> Sequence[Path]:
        """Return the current path's immutable ancestor sequence.

        Returns:
            Sequence[Path]: Logical ancestors of the current path.
        """
        return self.value.parents

    @property
    def parts(self) -> tuple[str, ...]:
        """Return the individual path components.

        Returns:
            tuple[str, ...]: Current path components.
        """
        return self.value.parts

    @property
    def root(self) -> str:
        """Return the local or global root marker.

        Returns:
            str: Root marker, if any.
        """
        return self.value.root

    @property
    def stem(self) -> str:
        """Return the final component without its suffix.

        Returns:
            str: Current path stem.
        """
        return self.value.stem

    @property
    def suffix(self) -> str:
        """Return the final component's last suffix.

        Returns:
            str: Last suffix, including its leading dot.
        """
        return self.value.suffix

    @property
    def suffixes(self) -> list[str]:
        """Return the final component's suffixes.

        Returns:
            list[str]: Suffixes, each including its leading dot.
        """
        return self.value.suffixes

    def __fspath__(self) -> str:
        return os.fspath(self.value)

    def __truediv__(self, key: PathInput) -> Path:
        return self.value / key

    def __rtruediv__(self, key: PathInput) -> Path:
        return Path(key) / self.value

    def absolute(self) -> Path:
        """Return an absolute snapshot without resolving symbolic links.

        Returns:
            Path: Absolute form of the current path.
        """
        return self.value.absolute()

    def as_posix(self) -> str:
        """Return the current path with forward slashes.

        Returns:
            str: POSIX-formatted path.
        """
        return self.value.as_posix()

    def as_uri(self) -> str:
        """Return the current absolute path as a file URI.

        Returns:
            str: File URI for the current path.

        Raises:
            ValueError: If the current path is relative.
        """
        return self.value.as_uri()

    def chmod(self, mode: int, *, follow_symlinks: bool = True) -> None:
        """Change the current path's mode.

        Args:
            mode (int): New permission bits.
            follow_symlinks (bool): Whether to modify a symbolic link's target.
        """
        self.value.chmod(mode, follow_symlinks=follow_symlinks)

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        """Return whether the current path exists.

        Args:
            follow_symlinks (bool): Whether to follow symbolic links.

        Returns:
            bool: Whether the target exists.
        """
        return self.value.exists(follow_symlinks=follow_symlinks)

    def expanduser(self) -> Path:
        """Return a snapshot with a leading tilde expanded.

        Returns:
            Path: Expanded path.
        """
        return self.value.expanduser()

    def glob(self, pattern: str, *, case_sensitive: bool | None = None) -> Iterator[Path]:
        """Yield snapshot paths matching a relative pattern.

        Args:
            pattern (str): Relative glob pattern.
            case_sensitive (bool | None): Explicit case behavior, or platform default.

        Yields:
            Path: Each matching path.
        """
        yield from self.value.glob(pattern, case_sensitive=case_sensitive)

    def group(self) -> str:
        """Return the group name owning the current path.

        Returns:
            str: Owning group name.
        """
        return self.value.group()

    def hardlink_to(self, target: PathInput) -> None:
        """Make the current path a hard link to a target.

        Args:
            target (str | os.PathLike[str]): Existing target path.
        """
        self.value.hardlink_to(target)

    def is_block_device(self) -> bool:
        """Return whether the current path is a block device.

        Returns:
            bool: Whether the path is a block device.
        """
        return self.value.is_block_device()

    def is_absolute(self) -> bool:
        """Return whether the current path is absolute.

        Returns:
            bool: Whether the path is absolute.
        """
        return self.value.is_absolute()

    def is_char_device(self) -> bool:
        """Return whether the current path is a character device.

        Returns:
            bool: Whether the path is a character device.
        """
        return self.value.is_char_device()

    def is_dir(self) -> bool:
        """Return whether the current path is a directory.

        Returns:
            bool: Whether the path is a directory.
        """
        return self.value.is_dir()

    def is_fifo(self) -> bool:
        """Return whether the current path is a FIFO.

        Returns:
            bool: Whether the path is a FIFO.
        """
        return self.value.is_fifo()

    def is_file(self) -> bool:
        """Return whether the current path is a regular file.

        Returns:
            bool: Whether the path is a regular file.
        """
        return self.value.is_file()

    def is_junction(self) -> bool:
        """Return whether the current path is a junction.

        Returns:
            bool: Whether the path is a junction.
        """
        return self.value.is_junction()

    def is_mount(self) -> bool:
        """Return whether the current path is a mount point.

        Returns:
            bool: Whether the path is a mount point.
        """
        return self.value.is_mount()

    def is_relative_to(self, other: PathInput) -> bool:
        """Return whether the current path is relative to another path.

        Args:
            other (str | os.PathLike[str]): Candidate ancestor.
        Returns:
            bool: Whether the current path is below the candidate.
        """
        return self.value.is_relative_to(other)

    def is_reserved(self) -> bool:
        """Return whether the current path is reserved by the platform.

        Returns:
            bool: Whether the path is reserved.
        """
        return self.value.is_reserved()  # pylint: disable=deprecated-method

    def is_socket(self) -> bool:
        """Return whether the current path is a socket.

        Returns:
            bool: Whether the path is a socket.
        """
        return self.value.is_socket()

    def is_symlink(self) -> bool:
        """Return whether the current path is a symbolic link.

        Returns:
            bool: Whether the path is a symbolic link.
        """
        return self.value.is_symlink()

    def iterdir(self) -> Iterator[Path]:
        """Yield snapshots of children in the current directory.

        Yields:
            Path: Each direct child path.
        """
        yield from self.value.iterdir()

    def joinpath(self, *pathsegments: PathInput) -> Path:
        """Return a snapshot combining the current path and child components.

        Args:
            *pathsegments (str | os.PathLike[str]): Components to append.

        Returns:
            Path: Combined path.
        """
        return self.value.joinpath(*pathsegments)

    def lchmod(self, mode: int) -> None:
        """Change the mode of the current symbolic link itself.

        Args:
            mode (int): New permission bits.
        """
        self.value.lchmod(mode)

    def lstat(self) -> os.stat_result:
        """Return metadata for the current path without following symbolic links.

        Returns:
            os.stat_result: Filesystem metadata.
        """
        return self.value.lstat()

    def match(self, path_pattern: str, *, case_sensitive: bool | None = None) -> bool:
        """Return whether the current path matches a pattern.

        Args:
            path_pattern (str): Pattern to match.
            case_sensitive (bool | None): Explicit case behavior, or platform default.

        Returns:
            bool: Whether the path matches.
        """
        return self.value.match(path_pattern, case_sensitive=case_sensitive)

    def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        """Create the current path as a directory.

        Args:
            mode (int): Requested permission bits.
            parents (bool): Whether to create missing ancestors.
            exist_ok (bool): Whether an existing directory is accepted.
        """
        self.value.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> Any:
        """Open the current path.

        Args:
            mode (str): File access mode.
            buffering (int): Buffering policy.
            encoding (str | None): Text encoding.
            errors (str | None): Text decoding error policy.
            newline (str | None): Universal-newline policy.

        Returns:
            Any: Open binary or text stream selected by ``mode``.
        """
        return self.value.open(mode, buffering, encoding, errors, newline)

    def owner(self) -> str:
        """Return the user name owning the current path.

        Returns:
            str: Owning user name.
        """
        return self.value.owner()

    def read_bytes(self) -> bytes:
        """Read the current file as bytes.

        Returns:
            bytes: File contents.
        """
        return self.value.read_bytes()

    def read_text(self, encoding: str | None = None, errors: str | None = None) -> str:
        """Read the current file as text.

        Args:
            encoding (str | None): Text encoding, or the platform default.
            errors (str | None): Decoding error policy.

        Returns:
            str: File contents.
        """
        return self.value.read_text(encoding=encoding, errors=errors)

    def readlink(self) -> Path:
        """Return a snapshot of the current symbolic link's target.

        Returns:
            Path: Link target.
        """
        return self.value.readlink()

    def relative_to(self, other: PathInput, *, walk_up: bool = False) -> Path:
        """Return a snapshot relative to another path.

        Args:
            other (str | os.PathLike[str]): Base path.
            walk_up (bool): Whether parent components may be added to reach the base.

        Returns:
            Path: Relative path snapshot.
        """
        return self.value.relative_to(other, walk_up=walk_up)

    def rename(self, target: PathInput) -> Path:
        """Rename the current path and return the destination snapshot.

        Args:
            target (str | os.PathLike[str]): Destination path.

        Returns:
            Path: Destination path.
        """
        return self.value.rename(target)

    def replace(self, target: PathInput) -> Path:
        """Replace a destination with the current path and return its snapshot.

        Args:
            target (str | os.PathLike[str]): Destination path.

        Returns:
            Path: Destination path.
        """
        return self.value.replace(target)

    def resolve(self, strict: bool = False) -> Path:
        """Return a resolved absolute snapshot.

        Args:
            strict (bool): Whether a missing target raises ``FileNotFoundError``.

        Returns:
            Path: Resolved path.
        """
        return self.value.resolve(strict=strict)

    def rglob(self, pattern: str, *, case_sensitive: bool | None = None) -> Iterator[Path]:
        """Yield snapshot paths recursively matching a relative pattern.

        Args:
            pattern (str): Relative glob pattern.
            case_sensitive (bool | None): Explicit case behavior, or platform default.

        Yields:
            Path: Each matching path.
        """
        yield from self.value.rglob(pattern, case_sensitive=case_sensitive)

    def rmdir(self) -> None:
        """Remove the current empty directory."""
        self.value.rmdir()

    def samefile(self, other_path: PathInput | int) -> bool:
        """Return whether another path or descriptor identifies the same file.

        Args:
            other_path (str | os.PathLike[str] | int): Path or open file descriptor to compare.

        Returns:
            bool: Whether both references identify the same file.
        """
        return self.value.samefile(other_path)

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        """Return metadata for the current path.

        Args:
            follow_symlinks (bool): Whether to follow symbolic links.

        Returns:
            os.stat_result: Filesystem metadata.
        """
        return self.value.stat(follow_symlinks=follow_symlinks)

    def symlink_to(self, target: PathInput, target_is_directory: bool = False) -> None:
        """Make the current path a symbolic link to a target.

        Args:
            target (str | os.PathLike[str]): Link target.
            target_is_directory (bool): Windows hint that the target is a directory.
        """
        self.value.symlink_to(target, target_is_directory=target_is_directory)

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """Create the current file or update its modification time.

        Args:
            mode (int): Requested permission bits for a new file.
            exist_ok (bool): Whether an existing file is accepted.
        """
        self.value.touch(mode=mode, exist_ok=exist_ok)

    def unlink(self, missing_ok: bool = False) -> None:
        """Remove the current file or symbolic link.

        Args:
            missing_ok (bool): Whether a missing target is accepted.
        """
        self.value.unlink(missing_ok=missing_ok)

    def walk(
        self, top_down: bool = True, on_error: Any = None, follow_symlinks: bool = False
    ) -> Iterator[tuple[Path, list[str], list[str]]]:
        """Walk the current directory tree using snapshot root paths.

        Args:
            top_down (bool): Whether to yield parents before children.
            on_error (Any): Optional callback receiving traversal errors.
            follow_symlinks (bool): Whether to descend through symbolic links.

        Yields:
            tuple[Path, list[str], list[str]]: Root snapshot, directory names, and file names.
        """
        yield from self.value.walk(
            top_down=top_down, on_error=on_error, follow_symlinks=follow_symlinks
        )

    def with_name(self, name: str) -> Path:
        """Return a snapshot with a different final component.

        Args:
            name (str): Replacement filename.

        Returns:
            Path: Modified path.
        """
        return self.value.with_name(name)

    def with_stem(self, stem: str) -> Path:
        """Return a snapshot with a different stem.

        Args:
            stem (str): Replacement stem.

        Returns:
            Path: Modified path.
        """
        return self.value.with_stem(stem)

    def with_suffix(self, suffix: str) -> Path:
        """Return a snapshot with a different suffix.

        Args:
            suffix (str): Replacement suffix, including a leading dot when non-empty.

        Returns:
            Path: Modified path.
        """
        return self.value.with_suffix(suffix)

    def write_bytes(self, data: bytes) -> int:
        """Write bytes to the current file.

        Args:
            data (bytes): Content to write.

        Returns:
            int: Number of bytes written.
        """
        return self.value.write_bytes(data)

    def write_text(
        self,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        """Write text to the current file.

        Args:
            data (str): Content to write.
            encoding (str | None): Text encoding, or the platform default.
            errors (str | None): Encoding error policy.
            newline (str | None): Newline translation policy.

        Returns:
            int: Number of characters written.
        """
        return self.value.write_text(data, encoding=encoding, errors=errors, newline=newline)
