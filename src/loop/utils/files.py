"""Provide general filesystem mutation utilities."""

import tempfile
from pathlib import Path

from .hashing import sha256_digest


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
