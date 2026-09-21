"""Persist checksummed write-ahead records outside authenticated workspaces."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ...utils import json_encode, matches_digest, sha256_digest
from .models import AffectedPathIdentity, BaseSnapshotId, CanonicalDelta, HostObjectIdentity


class JournalPhase(StrEnum):
    """Identify the durable host-publication phase of a transaction."""

    PREPARED = "prepared"
    HOST_APPLIED = "host_applied"
    BRANCH_APPLIED = "branch_applied"
    COMMITTED = "committed"


class HostPrecondition(BaseModel):
    """Record one identity that must still hold before host publication.

    Args:
        path (str): Workspace-relative object path.
        identity (HostObjectIdentity | None): Expected object identity, or ``None`` for absence.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    identity: HostObjectIdentity | None = None


class RollbackMaterial(BaseModel):
    """Name destination-local material retained until a prepared transaction is final.

    Args:
        path (str): Workspace-relative original destination path.
        backup_name (str): Opaque sibling name that receives the old destination object.
        original_mode (int | None): Original portable mode for metadata-only effects.
        source_backup_name (str | None): Source-local staging name for a rename.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    backup_name: str
    original_mode: int | None = Field(default=None, ge=0, le=0o7777)
    source_backup_name: str | None = None


class JournalTransaction(BaseModel):
    """Carry one versioned, checksummed host-publication write-ahead transaction.

    Args:
        transaction_id (str): Idempotency identifier supplied by the caller.
        workspace_id (str): Authenticated external workspace identity.
        agent_run_id (str): Agent identity that owns the publication.
        base_snapshot_id (BaseSnapshotId): Immutable base used by the attempt.
        generation_id (str): Agent generation that produced the delta.
        delta (CanonicalDelta): Fully normalized portable effects.
        preconditions (tuple[HostPrecondition, ...]): Identity/absence facts checked before apply.
        rollback_material (tuple[RollbackMaterial, ...]): Destination-local old-object evidence.
        phase (JournalPhase): Durable host-publication phase.
        completed_operations (int): Number of durably completed operations.
        resulting_identities (tuple[AffectedPathIdentity, ...]): Current host identities after the
            host phase completes.
    """

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1)
    transaction_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    agent_run_id: str = Field(min_length=1)
    base_snapshot_id: BaseSnapshotId
    generation_id: str = Field(min_length=1)
    delta: CanonicalDelta
    preconditions: tuple[HostPrecondition, ...]
    rollback_material: tuple[RollbackMaterial, ...] = ()
    phase: JournalPhase = JournalPhase.PREPARED
    completed_operations: int = Field(default=0, ge=0)
    resulting_identities: tuple[AffectedPathIdentity, ...] = ()


class JournalCorruptionError(RuntimeError):
    """Report a missing, malformed, or checksum-invalid transaction record."""


class TransactionJournal:
    """Store atomic checksummed transaction records in Loop-private application data.

    Args:
        directory (Path): Loop-private directory outside every workspace root.
    """

    directory: Path

    def __init__(self, directory: Path) -> None:
        """Initialize the external journal directory."""
        self.directory = directory
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def load(self, transaction_id: str) -> JournalTransaction | None:
        """Load and authenticate one transaction record when it exists.

        Args:
            transaction_id (str): Transaction id to retrieve.

        Returns:
            JournalTransaction | None: Authenticated record, if it was persisted.

        Raises:
            JournalCorruptionError: The external record fails integrity validation.
        """
        path = self._path(transaction_id)
        try:
            return self._load_path(path)
        except FileNotFoundError:
            return None

    def persist(self, transaction: JournalTransaction) -> None:
        """Atomically write and fsync one checksummed transaction record.

        Args:
            transaction (JournalTransaction): New durable record state.
        """
        payload = transaction.model_dump(mode="json")
        encoded = json_encode(payload).encode()
        envelope = json_encode({"checksum": sha256_digest(encoded), "payload": payload}).encode()
        target = self._path(transaction.transaction_id)
        temporary = target.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, envelope)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        directory_descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    def incomplete(self) -> tuple[JournalTransaction, ...]:
        """Enumerate authenticated nonterminal transactions for startup recovery.

        Returns:
            tuple[JournalTransaction, ...]: Incomplete records in deterministic identity order.

        Raises:
            JournalCorruptionError: A transaction record is malformed, misplaced, or invalid.
        """
        return tuple(
            transaction
            for transaction in self.transactions()
            if transaction.phase is not JournalPhase.COMMITTED
        )

    def transactions(self) -> tuple[JournalTransaction, ...]:
        """Enumerate every authenticated transaction in deterministic identity order.

        Returns:
            tuple[JournalTransaction, ...]: Authenticated durable transaction records.

        Raises:
            JournalCorruptionError: A transaction record is malformed, misplaced, or invalid.
        """
        transactions = []
        for path in self.directory.iterdir():
            if not path.is_file() or path.suffix in {".lock", ".tmp"}:
                continue
            transaction = self._load_path(path)
            if path != self._path(transaction.transaction_id):
                raise JournalCorruptionError("Transaction journal identity is invalid.")
            transactions.append(transaction)
        return tuple(
            sorted(
                transactions,
                key=lambda transaction: (
                    transaction.workspace_id,
                    transaction.agent_run_id,
                    transaction.generation_id,
                    transaction.transaction_id,
                ),
            )
        )

    def _load_path(self, path: Path) -> JournalTransaction:
        """Load and authenticate one known transaction record path."""
        try:
            raw = path.read_bytes()
            envelope = json.loads(raw)
            payload = envelope["payload"]
            checksum = envelope["checksum"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise JournalCorruptionError("Transaction journal is malformed.") from error
        encoded = json_encode(payload).encode()
        if not isinstance(checksum, str) or not matches_digest(encoded, checksum):
            raise JournalCorruptionError("Transaction journal checksum is invalid.")
        try:
            return JournalTransaction.model_validate(payload)
        except ValueError as error:
            raise JournalCorruptionError("Transaction journal payload is invalid.") from error

    def _path(self, transaction_id: str) -> Path:
        """Return a confined journal path for an opaque transaction identifier."""
        return self.directory / sha256_digest(transaction_id)
