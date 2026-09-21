"""Atomically activate already verified immutable runtime artifact sets."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from filelock import FileLock

from ... import constants


class ActivationError(RuntimeError):
    """Report an unsafe active-version transition."""


def active_set(root: Path) -> str | None:
    """Return the active artifact-set digest when its pointer is valid.

    Args:
        root (Path): Private runtime root containing the active pointer.

    Returns:
        str | None: Active artifact-set digest, or None when no pointer exists.

    Raises:
        ActivationError: If the active pointer is malformed.
    """
    pointer = root / constants.RUNTIME_ACTIVE_FILENAME
    try:
        value = pointer.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    if len(value) != constants.SHA256_HEX_LENGTH or any(
        character not in constants.HEX_DIGITS for character in value
    ):
        raise ActivationError("Runtime active pointer is invalid.")
    return value


def rollback_set(root: Path) -> str | None:
    """Return the retained known-good artifact-set digest.

    Args:
        root (Path): Private runtime root containing the rollback pointer.

    Returns:
        str | None: Retained digest, or ``None`` when none is available.

    Raises:
        ActivationError: If the pointer is malformed.
    """
    pointer = root / constants.RUNTIME_ROLLBACK_FILENAME
    try:
        value = pointer.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    if len(value) != constants.SHA256_HEX_LENGTH or any(
        character not in constants.HEX_DIGITS for character in value
    ):
        raise ActivationError("Runtime rollback pointer is invalid.")
    return value


def activate(root: Path, artifact_set_digest: str, expected: str | None = None) -> str | None:
    """Compare-and-swap the active pointer to an already verified set.

    Args:
        root (Path): Private runtime root containing the active pointer.
        artifact_set_digest (str): Digest of the verified set to activate.
        expected (str | None): Digest expected to be active before the swap.

    Returns:
        str | None: Previous active digest.

    Raises:
        ActivationError: If a digest is malformed or the pointer changed concurrently.
    """
    if len(artifact_set_digest) != constants.SHA256_HEX_LENGTH or any(
        char not in constants.HEX_DIGITS for char in artifact_set_digest
    ):
        raise ActivationError("Artifact-set digest is invalid.")
    root.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    with FileLock(str(root / ".activation.lock")):
        current = active_set(root)
        if current != expected:
            raise ActivationError("Runtime active pointer changed concurrently.")
        temporary = root / f".active-{uuid.uuid4().hex}.new"
        rollback_temporary: Path | None = None
        try:
            with temporary.open("x", encoding="ascii") as output:
                output.write(f"{artifact_set_digest}\n")
                output.flush()
                os.fsync(output.fileno())
            if current is not None:
                rollback_temporary = root / f".rollback-{uuid.uuid4().hex}.new"
                with rollback_temporary.open("x", encoding="ascii") as output:
                    output.write(f"{current}\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(rollback_temporary, root / constants.RUNTIME_ROLLBACK_FILENAME)
            os.replace(temporary, root / constants.RUNTIME_ACTIVE_FILENAME)
            descriptor = os.open(root, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return current
        finally:
            temporary.unlink(missing_ok=True)
            if rollback_temporary is not None:
                rollback_temporary.unlink(missing_ok=True)


def rollback(root: Path) -> str:
    """Activate the retained verified set without downloading content.

    Args:
        root (Path): Private runtime root containing activation state.

    Returns:
        str: Newly active artifact-set digest.

    Raises:
        ActivationError: If no retained set exists or its record is unavailable.
    """
    target = rollback_set(root)
    current = active_set(root)
    if target is None or not (root / constants.RUNTIME_SETS_DIRECTORY / f"{target}.json").is_file():
        raise ActivationError("No verified runtime rollback set is available.")
    activate(root, target, current)
    return target
