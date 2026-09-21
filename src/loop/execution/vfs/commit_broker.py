"""Publish validated canonical deltas to an authenticated host workspace."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from typing import BinaryIO, Protocol, cast

from filelock import FileLock

from ...utils import sha256_digest
from ..contracts import DeltaEffect
from .journal import (
    HostPrecondition,
    JournalPhase,
    JournalTransaction,
    RollbackMaterial,
    TransactionJournal,
)
from .models import (
    AffectedPathIdentity,
    BaseSnapshotId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    ContentReference,
    HostObjectIdentity,
    ObjectKind,
)
from .path_broker import AuthenticatedWorkspaceRoot, WorkspaceRootChangedError


class CommitConflictError(RuntimeError):
    """Report a host state that no longer satisfies an attempted delta's preconditions."""


class ContentSource(Protocol):
    """Open trusted immutable staged content addressed by an opaque reference."""

    def open(self, content: ContentReference) -> BinaryIO:
        """Return a readable stream for exactly the requested opaque content reference."""


class CommitBroker:
    """Apply journaled host effects under a short per-workspace commit lock.

    Args:
        root (AuthenticatedWorkspaceRoot): Retained authenticated live workspace root.
        journal (TransactionJournal): Loop-private external write-ahead journal.
        content_source (ContentSource): Trusted resolver for opaque staged file content.
    """

    root: AuthenticatedWorkspaceRoot
    journal: TransactionJournal
    content_source: ContentSource
    _lock: FileLock

    def __init__(
        self,
        root: AuthenticatedWorkspaceRoot,
        journal: TransactionJournal,
        content_source: ContentSource,
    ) -> None:
        """Bind publication to one authenticated root and external journal."""
        self.root = root
        self.journal = journal
        self.content_source = content_source
        identity = root.identity
        self._lock = FileLock(
            str(journal.directory / f"workspace-{identity.device}-{identity.inode}.lock")
        )

    def commit(
        self,
        *,
        transaction_id: str,
        workspace_id: str,
        agent_run_id: str,
        base_snapshot_id: BaseSnapshotId,
        generation_id: str,
        delta: CanonicalDelta,
    ) -> JournalTransaction:
        """Publish one delta exactly once or fail before changing a conflicting workspace.

        Args:
            transaction_id (str): Stable idempotency key for this publication.
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent identity.
            base_snapshot_id (BaseSnapshotId): Base used to produce the delta.
            generation_id (str): Agent generation used to produce the delta.
            delta (CanonicalDelta): Already representability-validated portable effects.

        Returns:
            JournalTransaction: Durable ``HOST_APPLIED`` transaction evidence.

        Raises:
            CommitConflictError: Live objects differ from the captured preconditions.
            JournalCorruptionError: Existing journal evidence is invalid or conflicts with the call.
        """
        with self._locked():
            existing = self.journal.load(transaction_id)
            if existing is not None:
                if existing.delta != delta or existing.workspace_id != workspace_id:
                    raise CommitConflictError(
                        "Transaction id was reused for different publication."
                    )
                if existing.phase in {
                    JournalPhase.HOST_APPLIED,
                    JournalPhase.BRANCH_APPLIED,
                    JournalPhase.COMMITTED,
                }:
                    return existing
                return self._recover_prepared(existing)
            transaction = JournalTransaction(
                transaction_id=transaction_id,
                workspace_id=workspace_id,
                agent_run_id=agent_run_id,
                base_snapshot_id=base_snapshot_id,
                generation_id=generation_id,
                delta=delta,
                preconditions=_preconditions(delta),
                rollback_material=_rollback_material(delta, transaction_id),
            )
            _require_precondition_evidence(delta)
            self._revalidate(transaction.preconditions)
            self.journal.persist(transaction)
            return self._apply(transaction)

    def recover(self, transaction_id: str) -> JournalTransaction | None:
        """Recover one prepared transaction while holding its workspace commit lock.

        Args:
            transaction_id (str): Persisted transaction id to recover.

        Returns:
            JournalTransaction | None: Final transaction evidence, if the record exists.
        """
        with self._locked():
            transaction = self.journal.load(transaction_id)
            if transaction is None or transaction.phase is not JournalPhase.PREPARED:
                return transaction
            return self._recover_prepared(transaction)

    @contextmanager
    def _locked(self):
        """Hold the narrow per-workspace publication lock and reauthenticate its root."""
        with self._lock:
            try:
                self.root.verify_root()
                yield
            except WorkspaceRootChangedError as error:
                raise CommitConflictError("Workspace root changed during publication.") from error

    def _recover_prepared(self, transaction: JournalTransaction) -> JournalTransaction:
        """Roll forward a prepared transaction from durable names, never path inference."""
        return self._apply(transaction)

    def _apply(self, transaction: JournalTransaction) -> JournalTransaction:
        """Apply remaining operations and persist progress after each durable boundary."""
        self._stage_rename_sources(transaction)
        for index in range(transaction.completed_operations, len(transaction.delta.entries)):
            entry = transaction.delta.entries[index]
            self._apply_entry(
                entry, transaction.rollback_material[index], transaction.transaction_id, index
            )
            transaction = transaction.model_copy(update={"completed_operations": index + 1})
            self.journal.persist(transaction)
        root_descriptor = self.root.duplicate_descriptor()
        try:
            _fsync_directory(root_descriptor)
        finally:
            os.close(root_descriptor)
        transaction = transaction.model_copy(
            update={
                "phase": JournalPhase.HOST_APPLIED,
                "resulting_identities": tuple(
                    AffectedPathIdentity(
                        path=condition.path, identity=_identity_at(self.root, condition.path)
                    )
                    for condition in transaction.preconditions
                ),
            }
        )
        self.journal.persist(transaction)
        return transaction

    def _stage_rename_sources(self, transaction: JournalTransaction) -> None:
        """Move every rename source aside before any destination can overwrite a chain."""
        for entry, recovery in zip(
            transaction.delta.entries, transaction.rollback_material, strict=True
        ):
            if entry.effect is not DeltaEffect.RENAME:
                continue
            source_parent, source_name = _open_parent(self.root, entry.source_path or "")
            try:
                staged = recovery.source_backup_name
                if staged is None:
                    raise CommitConflictError("Rename recovery material is incomplete.")
                if _exists(staged, source_parent):
                    continue
                if not _exists(source_name, source_parent):
                    destination_parent, destination_name = _open_parent(
                        self.root, entry.destination_path
                    )
                    try:
                        if _exists(destination_name, destination_parent):
                            continue
                    finally:
                        os.close(destination_parent)
                    raise CommitConflictError("Rename source disappeared before publication.")
                try:
                    os.rename(
                        source_name,
                        staged,
                        src_dir_fd=source_parent,
                        dst_dir_fd=source_parent,
                    )
                except OSError as error:
                    raise CommitConflictError(
                        f"Unable to apply workspace effect at {entry.source_path!r}."
                    ) from error
                _fsync_directory(source_parent)
            finally:
                os.close(source_parent)

    def finalize(self, transaction: JournalTransaction) -> None:
        """Reclaim host-side recovery material only after durable transaction completion.

        Args:
            transaction (JournalTransaction): Durable committed transaction to reclaim.

        Raises:
            ValueError: The transaction is not durably committed.
        """
        if transaction.phase is not JournalPhase.COMMITTED:
            raise ValueError("Only committed transaction material can be reclaimed.")
        with self._locked():
            self._discard_rollback_material(transaction.rollback_material)

    def _revalidate(self, preconditions: tuple[HostPrecondition, ...]) -> None:
        """Check all captured object identities before the first host mutation."""
        for condition in preconditions:
            actual = _identity_at(self.root, condition.path)
            if actual != condition.identity:
                raise CommitConflictError(f"Workspace conflict at {condition.path!r}.")

    def _apply_entry(
        self,
        entry: CanonicalDeltaEntry,
        rollback: RollbackMaterial,
        transaction_id: str,
        index: int,
    ) -> None:
        """Apply one descriptor-confined effect after all transaction preconditions passed."""
        parent, name = _open_parent(
            self.root, entry.destination_path, create=entry.effect is DeltaEffect.CREATE
        )
        try:
            if entry.effect is DeltaEffect.DELETE:
                _delete_entry(name, parent)
            elif entry.effect is DeltaEffect.RENAME:
                source_parent, _ = _open_parent(self.root, entry.source_path or "")
                try:
                    _backup_destination(entry, name, parent, rollback.backup_name)
                    staged = cast(str, rollback.source_backup_name)
                    try:
                        os.rename(staged, name, src_dir_fd=source_parent, dst_dir_fd=parent)
                    except FileNotFoundError:
                        if not _exists(name, parent):
                            raise
                finally:
                    os.close(source_parent)
            elif entry.object_kind is ObjectKind.DIRECTORY:
                _backup_destination(entry, name, parent, rollback.backup_name)
                try:
                    os.mkdir(name, entry.mode or 0o755, dir_fd=parent)
                except FileExistsError:
                    pass
            elif entry.object_kind is ObjectKind.SYMLINK:
                _backup_destination(entry, name, parent, rollback.backup_name)
                temporary = _temporary_name(transaction_id, index)
                _create_symlink(temporary, parent, entry.symlink_target or "")
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                _chmod(name, parent, entry.mode)
            elif entry.effect is DeltaEffect.METADATA:
                _chmod(name, parent, entry.mode or entry.metadata.basic_mode)
            else:
                _stage_file(
                    self.content_source,
                    entry.content,
                    parent,
                    name,
                    transaction_id,
                    index,
                    entry.mode,
                )
        except OSError as error:
            raise CommitConflictError(
                f"Unable to apply workspace effect at {entry.destination_path!r}."
            ) from error
        finally:
            os.close(parent)

    def _discard_rollback_material(self, material: tuple[RollbackMaterial, ...]) -> None:
        """Remove transaction-owned backup names only after durable host completion."""
        for rollback in material:
            parent, _ = _open_parent(self.root, rollback.path)
            try:
                try:
                    _remove(rollback.backup_name, parent)
                except FileNotFoundError:
                    continue
                _fsync_directory(parent)
            finally:
                os.close(parent)


def _preconditions(delta: CanonicalDelta) -> tuple[HostPrecondition, ...]:
    """Derive unique live-object preconditions from canonical inspector identity evidence."""
    conditions: dict[str, HostObjectIdentity | None] = {}
    for entry in delta.entries:
        if entry.effect is DeltaEffect.RENAME:
            conditions[entry.source_path or ""] = entry.source_identity
        conditions[entry.destination_path] = entry.destination_identity
    return tuple(
        HostPrecondition(path=path, identity=identity)
        for path, identity in sorted(conditions.items())
    )


def _require_precondition_evidence(delta: CanonicalDelta) -> None:
    """Reject mutations whose inspector omitted identity facts needed for safe recovery."""
    for entry in delta.entries:
        if entry.effect in {DeltaEffect.DELETE, DeltaEffect.REPLACE, DeltaEffect.METADATA} and (
            entry.destination_identity is None
        ):
            raise CommitConflictError(
                f"Missing destination identity for {entry.destination_path!r}."
            )
        if entry.effect is DeltaEffect.RENAME and entry.source_identity is None:
            raise CommitConflictError(f"Missing rename source identity for {entry.source_path!r}.")


def _rollback_material(delta: CanonicalDelta, transaction_id: str) -> tuple[RollbackMaterial, ...]:
    """Assign deterministic private sibling names before any host mutation occurs."""
    prefix = sha256_digest(transaction_id)[:16]
    return tuple(
        RollbackMaterial(
            path=entry.destination_path,
            backup_name=f".loop-backup-{prefix}-{index}",
            source_backup_name=(
                f".loop-rename-{prefix}-{index}" if entry.effect is DeltaEffect.RENAME else None
            ),
        )
        for index, entry in enumerate(delta.entries)
    )


def _identity_at(root: AuthenticatedWorkspaceRoot, path: str) -> HostObjectIdentity | None:
    """Read one no-follow object identity or a trustworthy absence result."""
    try:
        parent, name = _open_parent(root, path)
    except CommitConflictError:
        return None
    try:
        try:
            return _identity(os.stat(name, dir_fd=parent, follow_symlinks=False))
        except FileNotFoundError:
            return None
    finally:
        os.close(parent)


def _identity(metadata: os.stat_result) -> HostObjectIdentity:
    """Avoid importing host concepts into canonical records at call sites."""
    return HostObjectIdentity.from_metadata(metadata)


def _open_parent(
    root: AuthenticatedWorkspaceRoot, path: str, *, create: bool = False
) -> tuple[int, str]:
    """Open a no-follow parent descriptor, optionally making absent parent directories."""
    parts = path.split("/")
    descriptor = root.duplicate_descriptor()
    try:
        for part in parts[:-1]:
            try:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise CommitConflictError(f"Missing parent directory for {path!r}.") from None
                os.mkdir(part, 0o755, dir_fd=descriptor)
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _remove(name: str, parent: int) -> None:
    """Remove one object without recursively following or erasing concurrent descendants."""
    metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISDIR(metadata.st_mode):
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def _move_to_backup(name: str, parent: int, backup_name: str) -> None:
    """Move an original destination to its persisted private rollback sibling once."""
    if _exists(backup_name, parent):
        return
    try:
        os.rename(name, backup_name, src_dir_fd=parent, dst_dir_fd=parent)
    except FileNotFoundError:
        return


def _backup_destination(
    entry: CanonicalDeltaEntry, name: str, parent: int, backup_name: str
) -> None:
    """Back up only a destination that the inspector proved existed before the attempt."""
    if entry.destination_identity is not None:
        _move_to_backup(name, parent, backup_name)


def _delete_entry(name: str, parent: int) -> None:
    """Idempotently delete one file, link, or already-empty directory."""
    try:
        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def _exists(name: str, parent: int) -> bool:
    """Check one direct child without following a symbolic link."""
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _temporary_name(transaction_id: str, index: int) -> str:
    """Return the destination-local deterministic staging name for one operation."""
    return f".loop-stage-{sha256_digest(transaction_id)[:16]}-{index}"


def _create_symlink(name: str, parent: int, target: str) -> None:
    """Create a staging symlink once so recovery can repeat the final rename."""
    try:
        os.symlink(target, name, dir_fd=parent)
    except FileExistsError:
        pass


def _chmod(name: str, parent: int, mode: int | None) -> None:
    """Apply a portable basic mode only when one was explicitly captured."""
    if mode is not None:
        os.chmod(name, mode, dir_fd=parent, follow_symlinks=False)


def _stage_file(
    source: ContentSource,
    content: ContentReference | None,
    parent: int,
    name: str,
    transaction_id: str,
    index: int,
    mode: int | None,
) -> None:
    """Stage verified bytes beside their destination and atomically publish the final name."""
    if content is None:
        raise CommitConflictError("Regular-file effect has no trusted staged content.")
    temporary = _temporary_name(transaction_id, index)
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode or 0o600, dir_fd=parent
        )
    except FileExistsError:
        # A process may have stopped after creating this deterministic name but before
        # writing or synchronizing all bytes. Never infer completeness from existence.
        _remove(temporary, parent)
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode or 0o600, dir_fd=parent
        )
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open(content) as stream:
            while chunk := stream.read(1024 * 1024):
                if not isinstance(chunk, bytes):
                    raise CommitConflictError("Trusted content source returned non-bytes data.")
                digest.update(chunk)
                size += len(chunk)
                _write_all(descriptor, chunk)
        if size != content.size or f"sha256:{digest.hexdigest()}" != content.digest:
            raise CommitConflictError(
                "Trusted staged content does not match its canonical reference."
            )
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=parent)
        raise
    os.close(descriptor)
    os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
    _fsync_directory(parent)


def _write_all(descriptor: int, value: bytes) -> None:
    """Write every byte even when the operating system accepts a short write."""
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("Workspace staging made no write progress.")
        remaining = remaining[written:]


def _fsync_directory(descriptor: int) -> None:
    """Synchronize a changed directory without taking ownership of its descriptor."""
    os.fsync(descriptor)
