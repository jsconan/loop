"""Define the session persistence adapter contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..models import SessionInfo
from ..session import Session


@dataclass(frozen=True)
class WorkingDirectoryBinding:
    """Retain an opaque relative generation and its original filesystem identity.

    Args:
        generation (str): Store-relative working root, never an absolute pathname.
        device (int): Original filesystem device.
        inode (int): Original directory inode.
        retired (bool): Whether this generation has been permanently retired.
    """

    generation: str
    device: int
    inode: int
    retired: bool = False


class SessionStore(Protocol):
    """Persist and retrieve sessions by their identifier."""

    def working_binding(self, session_id: str) -> WorkingDirectoryBinding | None:
        """Read durable working storage identity independently of conversation snapshots.

        Args:
            session_id (str): Opaque session identifier.

        Returns:
            WorkingDirectoryBinding | None: Existing binding, including retirement, if present.
        """

    def bind_working_directory(self, session_id: str, binding: WorkingDirectoryBinding) -> None:
        """Commit a working generation before admitting filesystem operations.

        Args:
            session_id (str): Opaque session identifier.
            binding (WorkingDirectoryBinding): Validated generation or retirement tombstone.
        """

    @property
    def temporary_root(self) -> Path:
        """Return the store-owned root for isolated session working directories.

        Returns:
            Path: Canonical root whose lifetime matches this store's persistence.
        """

    def save(self, session: Session) -> str:
        """Persist a session under its stable identifier.

        Args:
            session (Session): Session to persist.

        Returns:
            str: The session's stable identifier.

        Raises:
            SessionRevisionConflictError: If the snapshot is based on a stale revision.
            ValueError: If an artifact is invalid or the session references missing content.
        """

    def load(self, session_id: str) -> Session:
        """Load a persisted session.

        Args:
            session_id (str): Identifier of the session to load.

        Returns:
            Session: Reconstructed session state.

        Raises:
            SessionNotFoundError: If the requested session does not exist.
            UnsupportedConversationItemError: If a serialized conversation item type is not
                supported.
            ValueError: If its persisted format is invalid or unsupported.
        """

    def list(self) -> list[SessionInfo]:
        """List persisted sessions from most to least recently updated.

        Returns:
            list[SessionInfo]: Lightweight persisted-session descriptions.
        """

    def load_reference(self, digest: str) -> bytes | None:
        """Load immutable referenced content.

        Args:
            digest (str): Algorithm-qualified full content digest.

        Returns:
            bytes | None: Complete immutable content, or ``None`` when unavailable.
        """

    def store_reference(self, content: bytes, *, handle: str, digest: str) -> None:
        """Persist one verified immutable reference without changing a session snapshot.

        Args:
            content (bytes): Complete immutable reference content.
            handle (str): Opaque content-read capability associated with the reference.
            digest (str): Algorithm-qualified digest of ``content``.

        Raises:
            ValueError: If the supplied identity does not match the content or conflicts with an
                existing artifact.
        """

    def collect_unreferenced(self) -> int:
        """Delete artifacts not owned by any persisted session.

        Returns:
            int: Number of artifacts deleted.
        """
