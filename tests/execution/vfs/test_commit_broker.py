"""Test journaled, descriptor-confined host workspace publication."""

from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from loop.execution.contracts import DeltaEffect
from loop.execution.vfs import (
    AuthenticatedWorkspaceRoot,
    BaseSnapshotId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    CommitBroker,
    CommitConflictError,
    ContentReference,
    HostObjectIdentity,
    JournalPhase,
    JournalTransaction,
    ObjectKind,
    TransactionJournal,
)
from loop.execution.vfs.commit_broker import _delete_entry, _remove
from loop.execution.vfs.journal import JournalCorruptionError, RollbackMaterial


class _ContentSource:
    """Serve fixed trusted staged bytes by their opaque canonical reference."""

    values: dict[str, bytes]

    def __init__(self, values: dict[str, bytes]) -> None:
        """Remember immutable test content by opaque reference."""
        self.values = values

    def open(self, content: ContentReference) -> io.BytesIO:
        """Open the bytes selected by a trusted opaque reference."""
        return io.BytesIO(self.values[content.reference])


def _content(value: bytes = b"new") -> ContentReference:
    """Return canonical content evidence for one byte payload."""
    return ContentReference(
        digest=f"sha256:{hashlib.sha256(value).hexdigest()}",
        size=len(value),
        reference="content:value",
    )


def _identity(path: Path) -> HostObjectIdentity:
    """Capture native no-follow identity evidence for a test workspace object."""
    return HostObjectIdentity.from_metadata(os.lstat(path))


def _broker(tmp_path: Path) -> tuple[Path, CommitBroker]:
    """Create one authenticated workspace and a broker with staged content."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    root = AuthenticatedWorkspaceRoot(workspace)
    broker = CommitBroker(
        root,
        TransactionJournal(tmp_path / "journal"),
        _ContentSource({"content:value": b"new"}),
    )
    return workspace, broker


def _commit(broker: CommitBroker, delta: CanonicalDelta, transaction_id: str = "tx"):
    """Publish a delta using stable test execution identity evidence."""
    return broker.commit(
        transaction_id=transaction_id,
        workspace_id="workspace",
        agent_run_id="agent",
        base_snapshot_id=BaseSnapshotId(value="base"),
        generation_id="1",
        delta=delta,
    )


def test_commit_stages_verified_content_and_is_idempotent(tmp_path: Path):
    """A file create is atomically published once and replays from durable evidence."""
    workspace, broker = _broker(tmp_path)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="nested/file",
                object_kind=ObjectKind.FILE,
                content=_content(),
                mode=0o640,
            ),
        )
    )

    first = _commit(broker, delta)
    second = _commit(broker, delta)

    assert first.phase is JournalPhase.HOST_APPLIED
    assert second == first
    assert (workspace / "nested" / "file").read_bytes() == b"new"
    assert (workspace / "nested" / "file").stat().st_mode & 0o777 == 0o640
    assert not list(workspace.rglob(".loop-stage-*"))


def test_commit_rejects_conflicts_and_nonempty_directory_deletes(tmp_path: Path):
    """Concurrent destination changes and directory children fail without destructive recursion."""
    workspace, broker = _broker(tmp_path)
    target = workspace / "target"
    target.write_text("old", encoding="utf-8")
    conflict = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.REPLACE,
                destination_path="target",
                destination_identity=_identity(target),
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(CommitConflictError, match="conflict"):
        _commit(broker, conflict)

    directory = workspace / "directory"
    directory.mkdir()
    (directory / "child").write_text("keep", encoding="utf-8")
    deletion = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.DELETE,
                destination_path="directory",
                destination_identity=_identity(directory),
            ),
        )
    )
    with pytest.raises(CommitConflictError, match="Unable to apply"):
        _commit(broker, deletion, "delete")
    assert (directory / "child").read_text(encoding="utf-8") == "keep"


def test_commit_preserves_rename_preconditions_and_rejects_root_replacement(tmp_path: Path):
    """Rename source identity and the retained root identity are both mandatory gates."""
    workspace, broker = _broker(tmp_path)
    source = workspace / "source"
    source.write_text("old", encoding="utf-8")
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="destination",
                source_path="source",
                source_identity=_identity(source),
                object_kind=ObjectKind.FILE,
            ),
        )
    )
    _commit(broker, delta, "rename")
    assert not source.exists()
    assert (workspace / "destination").read_text(encoding="utf-8") == "old"

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    workspace.rename(tmp_path / "old-workspace")
    replacement.rename(workspace)
    with pytest.raises(CommitConflictError, match="root changed"):
        _commit(broker, CanonicalDelta(), "root-replaced")


def test_journal_rejects_tampering_and_transaction_id_reuse(tmp_path: Path):
    """External journal integrity and idempotency identities fail closed."""
    _, broker = _broker(tmp_path)
    delta = CanonicalDelta()
    _commit(broker, delta)
    record = next(path for path in (tmp_path / "journal").glob("*") if path.suffix != ".lock")
    record.write_text('{"checksum":"bad","payload":{}}', encoding="utf-8")
    with pytest.raises(JournalCorruptionError, match="checksum"):
        _commit(broker, delta)

    _, other = _broker(tmp_path / "other")
    _commit(other, CanonicalDelta(), "same")
    with pytest.raises(CommitConflictError, match="reused"):
        _commit(
            other,
            CanonicalDelta(
                entries=(
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.CREATE,
                        destination_path="another",
                        object_kind=ObjectKind.FILE,
                        content=_content(),
                    ),
                )
            ),
            "same",
        )


def test_commit_applies_all_portable_effect_kinds_and_preserves_disjoint_changes(tmp_path: Path):
    """Independent creates, replacements, deletes, renames, links, modes, and directories commit."""
    workspace, broker = _broker(tmp_path)
    replaced = workspace / "replaced"
    deleted = workspace / "deleted"
    renamed = workspace / "renamed"
    metadata = workspace / "metadata"
    for path in (replaced, deleted, renamed, metadata):
        path.write_text(path.name, encoding="utf-8")
    original_replaced = _identity(replaced)
    original_deleted = _identity(deleted)
    original_renamed = _identity(renamed)
    original_metadata = _identity(metadata)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.DELETE,
                destination_path="deleted",
                destination_identity=original_deleted,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="directory",
                object_kind=ObjectKind.DIRECTORY,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.METADATA,
                destination_path="metadata",
                destination_identity=original_metadata,
                object_kind=ObjectKind.FILE,
                mode=0o600,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="moved",
                source_path="renamed",
                source_identity=original_renamed,
                object_kind=ObjectKind.FILE,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.REPLACE,
                destination_path="replaced",
                destination_identity=original_replaced,
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="symlink",
                object_kind=ObjectKind.SYMLINK,
                symlink_target="moved",
            ),
        )
    )

    _commit(broker, delta, "effects")

    assert (workspace / "directory").is_dir()
    assert not deleted.exists()
    assert metadata.stat().st_mode & 0o777 == 0o600
    assert not renamed.exists()
    assert (workspace / "moved").read_text(encoding="utf-8") == "renamed"
    assert replaced.read_bytes() == b"new"
    assert os.readlink(workspace / "symlink") == "moved"


def test_concurrent_disjoint_agent_publications_share_only_the_host_phase_lock(
    tmp_path: Path,
) -> None:
    """Concurrent agents serialize host mutation without conflicting on disjoint paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journal = TransactionJournal(tmp_path / "journal")
    content = _ContentSource({"content:value": b"new"})
    brokers = (
        CommitBroker(AuthenticatedWorkspaceRoot(workspace), journal, content),
        CommitBroker(AuthenticatedWorkspaceRoot(workspace), journal, content),
    )

    def publish(index: int) -> JournalTransaction:
        """Publish one agent's disjoint file through an independent broker instance."""
        delta = CanonicalDelta(
            entries=(
                CanonicalDeltaEntry(
                    effect=DeltaEffect.CREATE,
                    destination_path=f"agent-{index}",
                    object_kind=ObjectKind.FILE,
                    content=_content(),
                ),
            )
        )
        return brokers[index].commit(
            transaction_id=f"concurrent-{index}",
            workspace_id="workspace",
            agent_run_id=f"agent-{index}",
            base_snapshot_id=BaseSnapshotId(value="base"),
            generation_id="0",
            delta=delta,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        transactions = tuple(executor.map(publish, range(2)))

    assert all(transaction.phase is JournalPhase.HOST_APPLIED for transaction in transactions)
    assert (workspace / "agent-0").read_bytes() == b"new"
    assert (workspace / "agent-1").read_bytes() == b"new"


def test_prepared_transaction_recovers_by_replaying_durable_operation_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A progress-record failure after a destination rename resumes to journal-described output."""
    workspace, broker = _broker(tmp_path)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )
    original_persist = broker.journal.persist
    calls = 0

    def fail_after_prepare(transaction: object) -> None:
        """Simulate a crash boundary immediately after the first host operation."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected journal fsync failure")
        original_persist(transaction)  # type: ignore[arg-type]

    monkeypatch.setattr(broker.journal, "persist", fail_after_prepare)
    with pytest.raises(OSError, match="injected"):
        _commit(broker, delta, "recover")
    monkeypatch.setattr(broker.journal, "persist", original_persist)

    recovered = broker.recover("recover")

    assert recovered is not None
    assert recovered.phase is JournalPhase.HOST_APPLIED
    assert (workspace / "file").read_bytes() == b"new"


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            CanonicalDeltaEntry(effect=DeltaEffect.DELETE, destination_path="file"),
            "Missing destination identity",
        ),
        (
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="new",
                source_path="old",
                object_kind=ObjectKind.FILE,
            ),
            "Missing rename source identity",
        ),
    ),
)
def test_commit_fails_closed_when_inspector_omits_mutation_identity(
    tmp_path: Path, entry: CanonicalDeltaEntry, message: str
):
    """A delete, replacement, metadata change, or rename cannot commit without identity evidence."""
    _, broker = _broker(tmp_path)
    with pytest.raises(CommitConflictError, match=message):
        _commit(broker, CanonicalDelta(entries=(entry,)))


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="missing-content",
                object_kind=ObjectKind.FILE,
            ),
            "no trusted staged content",
        ),
        (
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="wrong-content",
                object_kind=ObjectKind.FILE,
                content=_content(b"other"),
            ),
            "does not match",
        ),
    ),
)
def test_commit_rejects_missing_or_mismatched_staged_content(
    tmp_path: Path, entry: CanonicalDeltaEntry, message: str
):
    """Opaque content must be present and exactly match its digest and byte count before rename."""
    _, broker = _broker(tmp_path)
    with pytest.raises(CommitConflictError, match=message):
        _commit(broker, CanonicalDelta(entries=(entry,)), entry.destination_path)


def test_recovery_returns_absent_and_completed_transactions_without_mutation(tmp_path: Path):
    """Recovery is a harmless lookup for unknown and already-final transaction identifiers."""
    _, broker = _broker(tmp_path)
    assert broker.recover("missing") is None
    completed = _commit(broker, CanonicalDelta(), "complete")
    assert broker.recover("complete") == completed


def test_recovery_rejects_incomplete_rename_material_and_handles_durable_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Rename recovery fails closed on missing evidence and resumes durable source stages."""
    workspace, broker = _broker(tmp_path)
    rename = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="destination",
                source_path="source",
                source_identity=HostObjectIdentity(device=1, inode=1, ctime_ns=1),
                object_kind=ObjectKind.FILE,
            ),
        )
    )

    def persist(transaction_id: str, source_backup_name: str | None) -> None:
        """Persist one deliberately interrupted rename transaction."""
        broker.journal.persist(
            JournalTransaction(
                transaction_id=transaction_id,
                workspace_id="workspace",
                agent_run_id="agent",
                base_snapshot_id=BaseSnapshotId(value="base"),
                generation_id="0",
                delta=rename,
                preconditions=(),
                rollback_material=(
                    RollbackMaterial(
                        path="destination",
                        backup_name=f".{transaction_id}-destination",
                        source_backup_name=source_backup_name,
                    ),
                ),
            )
        )

    persist("missing-material", None)
    with pytest.raises(CommitConflictError, match="incomplete"):
        broker.recover("missing-material")

    persist("missing-both", ".missing-source")
    with pytest.raises(CommitConflictError, match="disappeared"):
        broker.recover("missing-both")

    (workspace / ".durable-source").write_text("staged", encoding="utf-8")
    persist("durable", ".durable-source")
    recovered = broker.recover("durable")
    assert recovered is not None and recovered.phase is JournalPhase.HOST_APPLIED
    assert (workspace / "destination").read_text(encoding="utf-8") == "staged"

    (workspace / "destination").write_text("already", encoding="utf-8")
    persist("already-applied", ".already-missing")
    recovered = broker.recover("already-applied")
    assert recovered is not None and recovered.phase is JournalPhase.HOST_APPLIED
    assert (workspace / "destination").read_text(encoding="utf-8") == "already"

    with pytest.raises(ValueError, match="Only committed"):
        broker.finalize(recovered)

    persist("apply-missing", ".apply-missing")
    monkeypatch.setattr(broker, "_stage_rename_sources", lambda transaction: None)
    (workspace / "destination").unlink()
    with pytest.raises(CommitConflictError, match="Unable to apply"):
        broker.recover("apply-missing")


def test_native_removal_helpers_delete_files_and_tolerate_absence(tmp_path: Path) -> None:
    """Recovery cleanup removes files and treats an already-absent delta target idempotently."""
    target = tmp_path / "target"
    target.write_text("value", encoding="utf-8")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _remove("target", descriptor)
        _delete_entry("already-absent", descriptor)
    finally:
        os.close(descriptor)
    assert not target.exists()


@pytest.mark.parametrize("payload", ({}, {"phase": "invalid"}))
def test_journal_rejects_malformed_and_schema_invalid_authenticated_records(
    tmp_path: Path, payload: dict[str, str]
):
    """Both malformed envelopes and authenticated invalid transaction schemas fail closed."""
    journal = TransactionJournal(tmp_path / "journal")
    transaction_id = "record"
    path = journal.directory / hashlib.sha256(transaction_id.encode()).hexdigest()
    if not payload:
        path.write_text("{}", encoding="utf-8")
    else:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        envelope = {"checksum": hashlib.sha256(encoded).hexdigest(), "payload": payload}
        path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(JournalCorruptionError):
        journal.load(transaction_id)


def test_journal_enumeration_rejects_a_record_under_the_wrong_identity(tmp_path: Path):
    """Startup enumeration authenticates the filename against its transaction identity."""
    journal = TransactionJournal(tmp_path / "journal")
    transaction = JournalTransaction(
        transaction_id="correct",
        workspace_id="workspace",
        agent_run_id="agent",
        base_snapshot_id=BaseSnapshotId(value="base"),
        generation_id="0",
        delta=CanonicalDelta(),
        preconditions=(),
    )
    journal.persist(transaction)
    record = next(path for path in journal.directory.iterdir() if path.suffix != ".lock")
    record.rename(journal.directory / "wrong")

    with pytest.raises(JournalCorruptionError, match="identity"):
        journal.incomplete()


def test_replay_handles_completed_rename_and_directory_operations(tmp_path: Path):
    """A prepared replay accepts an already-moved source and an already-created directory."""
    workspace, broker = _broker(tmp_path)
    source = workspace / "source"
    source.write_text("source", encoding="utf-8")
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="directory",
                object_kind=ObjectKind.DIRECTORY,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="moved",
                source_path="source",
                source_identity=_identity(source),
                object_kind=ObjectKind.FILE,
            ),
        )
    )
    original_persist = broker.journal.persist
    calls = 0

    def fail_before_progress(transaction: object) -> None:
        """Leave both native effects applied while their journal stays PREPARED."""
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("interrupt")
        original_persist(transaction)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(broker.journal, "persist", fail_before_progress)
        with pytest.raises(OSError, match="interrupt"):
            _commit(broker, delta, "replay")
    _commit(broker, delta, "replay")
    assert (workspace / "directory").is_dir()
    assert (workspace / "moved").read_text(encoding="utf-8") == "source"


def test_commit_preserves_rename_chains_and_recursively_deletes_explicit_trees(
    tmp_path: Path,
) -> None:
    """Source staging preserves every rename value and deepest-first deletes remove full trees."""
    workspace, broker = _broker(tmp_path)
    first = workspace / "first"
    second = workspace / "second"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    rename = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="second",
                source_path="first",
                source_identity=_identity(first),
                destination_identity=_identity(second),
                object_kind=ObjectKind.FILE,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="third",
                source_path="second",
                source_identity=_identity(second),
                object_kind=ObjectKind.FILE,
            ),
        )
    )

    _commit(broker, rename, "rename-chain")

    assert not first.exists()
    assert second.read_text(encoding="utf-8") == "one"
    assert (workspace / "third").read_text(encoding="utf-8") == "two"

    directory = workspace / "directory"
    child = directory / "child"
    directory.mkdir()
    child.mkdir()
    file = child / "file"
    file.write_text("value", encoding="utf-8")
    deletion = CanonicalDelta(
        entries=tuple(
            sorted(
                (
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.DELETE,
                        destination_path="directory",
                        destination_identity=_identity(directory),
                    ),
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.DELETE,
                        destination_path="directory/child",
                        destination_identity=_identity(child),
                    ),
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.DELETE,
                        destination_path="directory/child/file",
                        destination_identity=_identity(file),
                    ),
                ),
                key=lambda entry: entry.ordering_key,
            )
        )
    )
    _commit(broker, deletion, "delete-tree")
    assert not directory.exists()


def test_replay_uses_existing_destination_local_staging_after_rename_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A failed final staging rename is retried from its same-filesystem private sibling."""
    workspace, broker = _broker(tmp_path)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )
    original_replace = os.replace

    def fail_stage(source: object, destination: object, **kwargs: object) -> None:
        """Interrupt only the destination-local final stage rename once."""
        if isinstance(source, str) and source.startswith(".loop-stage-"):
            raise OSError("rename interrupted")
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", fail_stage)
    with pytest.raises(CommitConflictError, match="Unable to apply"):
        _commit(broker, delta, "staged")
    monkeypatch.setattr(os, "replace", original_replace)
    _commit(broker, delta, "staged")
    assert (workspace / "file").read_bytes() == b"new"


def test_replay_rebuilds_an_incomplete_destination_local_staging_file(tmp_path: Path) -> None:
    """Recovery never promotes bytes left by a process that stopped during staging."""
    workspace, broker = _broker(tmp_path)
    transaction_id = "partial-stage"
    staging_name = f".loop-stage-{hashlib.sha256(transaction_id.encode()).hexdigest()[:16]}-0"
    (workspace / staging_name).write_bytes(b"par")
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )

    _commit(broker, delta, transaction_id)

    assert (workspace / "file").read_bytes() == b"new"
    assert not (workspace / staging_name).exists()


def test_commit_retries_short_content_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication writes every content byte when the host accepts only a prefix."""
    workspace, broker = _broker(tmp_path)
    original_write = os.write

    def short_write(descriptor: int, value: bytes | memoryview) -> int:
        """Accept at most one byte to exercise the public commit path's write loop."""
        return original_write(descriptor, value[:1])

    monkeypatch.setattr(os, "write", short_write)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )

    _commit(broker, delta, "short-write")

    assert (workspace / "file").read_bytes() == b"new"


def test_commit_rejects_content_writes_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication removes staging state when a host write makes no progress."""
    workspace, broker = _broker(tmp_path)
    monkeypatch.setattr(os, "write", lambda _descriptor, _value: 0)
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="file",
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )

    with pytest.raises(CommitConflictError, match="Unable to apply workspace effect") as failure:
        _commit(broker, delta, "stalled-write")

    assert isinstance(failure.value.__cause__, OSError)
    assert "no write progress" in str(failure.value.__cause__)
    assert not (workspace / "file").exists()
    assert not list(workspace.glob(".loop-stage-*"))


def test_replay_retains_recovery_material_until_committed_finalization(tmp_path: Path):
    """Replacement keeps private backups until branch durability permits final reclamation."""
    workspace, broker = _broker(tmp_path)
    file = workspace / "file"
    directory = workspace / "directory"
    file.write_text("old", encoding="utf-8")
    directory.mkdir()
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.REPLACE,
                destination_path="directory",
                destination_identity=_identity(directory),
                object_kind=ObjectKind.DIRECTORY,
            ),
            CanonicalDeltaEntry(
                effect=DeltaEffect.REPLACE,
                destination_path="file",
                destination_identity=_identity(file),
                object_kind=ObjectKind.FILE,
                content=_content(),
            ),
        )
    )
    original_persist = broker.journal.persist
    calls = 0

    def interrupt(transaction: object) -> None:
        """Leave a backup in place after the first replacement's host mutation."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupt")
        original_persist(transaction)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(broker.journal, "persist", interrupt)
        with pytest.raises(OSError, match="interrupt"):
            _commit(broker, delta, "backups")
    transaction = _commit(broker, delta, "backups")
    assert directory.is_dir()
    assert file.read_bytes() == b"new"
    assert list(workspace.glob(".loop-backup-*"))
    broker.finalize(transaction.model_copy(update={"phase": JournalPhase.COMMITTED}))
    assert not list(workspace.glob(".loop-backup-*"))


def test_symlink_stage_recovery_and_non_bytes_content_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Interrupted symlink staging replays, while a malformed trusted content stream is rejected."""
    workspace, broker = _broker(tmp_path)
    symlink = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="link",
                object_kind=ObjectKind.SYMLINK,
                symlink_target="target",
            ),
        )
    )
    original_replace = os.replace
    failed = False

    def fail_once(source: object, destination: object, **kwargs: object) -> None:
        """Leave exactly one private symlink stage for recovery."""
        nonlocal failed
        if not failed and isinstance(source, str) and source.startswith(".loop-stage-"):
            failed = True
            raise OSError("interrupt")
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(CommitConflictError):
        _commit(broker, symlink, "link")
    monkeypatch.setattr(os, "replace", original_replace)
    _commit(broker, symlink, "link")
    assert os.readlink(workspace / "link") == "target"

    class BadContent:
        """Expose an invalid stream type from the trusted content boundary."""

        def open(self, content: ContentReference) -> io.StringIO:
            """Return text to verify the broker rejects non-byte staged data."""
            del content
            return io.StringIO("not bytes")

    bad = CommitBroker(
        AuthenticatedWorkspaceRoot(workspace),
        TransactionJournal(tmp_path / "bad-journal"),
        BadContent(),
    )
    with pytest.raises(CommitConflictError, match="non-bytes"):
        _commit(
            bad,
            CanonicalDelta(
                entries=(
                    CanonicalDeltaEntry(
                        effect=DeltaEffect.CREATE,
                        destination_path="bad",
                        object_kind=ObjectKind.FILE,
                        content=_content(),
                    ),
                )
            ),
            "bad-content",
        )


def test_rename_fails_closed_when_source_disappears_after_precondition_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A source removed in the commit race window cannot be mistaken for a completed rename."""
    workspace, broker = _broker(tmp_path)
    source = workspace / "source"
    source.write_text("source", encoding="utf-8")
    delta = CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.RENAME,
                destination_path="destination",
                source_path="source",
                source_identity=_identity(source),
                object_kind=ObjectKind.FILE,
            ),
        )
    )
    original_rename = os.rename

    def lose_source(source_name: object, destination_name: object, **kwargs: object) -> None:
        """Make the source move report the native missing-source race."""
        if source_name == "source":
            raise FileNotFoundError
        original_rename(source_name, destination_name, **kwargs)

    monkeypatch.setattr(os, "rename", lose_source)
    with pytest.raises(CommitConflictError, match="Unable to apply"):
        _commit(broker, delta, "lost-source")
