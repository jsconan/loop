"""Provide general filesystem mutation utilities."""

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
) -> None:
    """Atomically write UTF-8 text when the destination has the expected state.

    Args:
        path (Path): Destination file to create or replace.
        content (str): UTF-8 text to write.
        expected_digest (str | None): Required SHA-256 digest for an existing destination, or
            ``None`` to require that the destination does not exist.

    Raises:
        RuntimeError: If the destination does not have the expected existence or content.
        OSError: If staging or committing the content fails.
    """
    temporary_path = None
    try:
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
            if not path.is_file() or sha256_digest(path.read_bytes()) != expected_digest:
                raise RuntimeError("The target changed after approval; replacement was cancelled.")
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
