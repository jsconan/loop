"""Test isolated agent workspace lineage and ordered publication."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from loop.execution.contracts import DeltaEffect
from loop.execution.vfs import (
    AgentWorkspaceContext,
    AgentWorkspaceCorruptionError,
    AgentWorkspaceManager,
    AuthenticatedWorkspaceRoot,
    BaseSnapshotId,
    BranchId,
    CanonicalDelta,
    CanonicalDeltaEntry,
    CommitBroker,
    ContentReference,
    GenerationId,
    GenerationView,
    JournalPhase,
    JournalTransaction,
    LineageEpoch,
    ObjectKind,
    PublicationCoordinator,
    SanitizedWorkspaceContext,
    TransactionJournal,
    sanitize_workspace_context,
)


class _Materializer:
    """Record opaque branch operations without supplying a production backend."""

    calls: list[tuple[str, str]]
    applied: set[tuple[str, str]]

    def __init__(self) -> None:
        """Initialize deterministic opaque materializer evidence."""
        self.calls = []
        self.applied = set()

    def create_branch(self, base: BaseSnapshotId) -> BranchId:
        """Create a distinct branch for each request."""
        branch = BranchId(value=f"branch:{len(self.calls)}:{base.value}")
        self.calls.append(("create", branch.value))
        return branch

    def recover_branch(
        self,
        base: BaseSnapshotId,
        branch: BranchId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> None:
        """Record restoration of one persisted branch."""
        del committed_transactions
        self.calls.append(("recover", f"{base.value}:{branch.value}"))

    def generation_view(self, branch: BranchId, generation: GenerationId) -> GenerationView:
        """Record generation lookup without revealing a path."""
        self.calls.append(("view", f"{branch.value}:{generation.value}"))
        return GenerationView()

    def apply_delta(self, branch: BranchId, transaction_id: str, delta: CanonicalDelta) -> None:
        """Record idempotent delta application."""
        del delta
        self.applied.add((branch.value, transaction_id))
        self.calls.append(("apply", f"{branch.value}:{transaction_id}"))

    def fork_branch(
        self,
        base: BaseSnapshotId,
        committed_transactions: tuple[tuple[str, CanonicalDelta], ...],
    ) -> BranchId:
        """Record replay of only identified committed effects into a separate branch."""
        identities = ",".join(identity for identity, _ in committed_transactions)
        branch = BranchId(value=f"fork:{base.value}:{len(committed_transactions)}")
        self.calls.append(("fork", f"{branch.value}:{identities}"))
        return branch

    def dispose_branch(self, branch: BranchId) -> None:
        """Record explicit branch disposal."""
        self.calls.append(("dispose", branch.value))


class _ContentSource:
    """Open the one immutable content payload used by publication tests."""

    def open(self, content: ContentReference) -> io.BytesIO:
        """Return bytes selected by opaque content evidence."""
        del content
        return io.BytesIO(b"new")


def _delta() -> CanonicalDelta:
    """Return one representability-ready create effect."""
    value = b"new"
    return CanonicalDelta(
        entries=(
            CanonicalDeltaEntry(
                effect=DeltaEffect.CREATE,
                destination_path="result",
                object_kind=ObjectKind.FILE,
                content=ContentReference(
                    digest=f"sha256:{hashlib.sha256(value).hexdigest()}",
                    size=len(value),
                    reference="result",
                ),
            ),
        )
    )


def _coordinator(
    tmp_path: Path,
) -> tuple[AgentWorkspaceManager, _Materializer, PublicationCoordinator]:
    """Build one fully authenticated C5 publication assembly."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    journal = TransactionJournal(tmp_path / "journal")
    materializer = _Materializer()
    manager = AgentWorkspaceManager(
        materializer,
        lambda context: True,
        tmp_path / "contexts",
    )
    broker = CommitBroker(AuthenticatedWorkspaceRoot(workspace), journal, _ContentSource())
    return manager, materializer, PublicationCoordinator(manager, broker, journal)


def test_agents_have_independent_branches_and_child_replays_only_committed_deltas(tmp_path: Path):
    """Agents and child forks never share writable opaque branch identity."""
    manager, materializer, _ = _coordinator(tmp_path)
    parent = manager.create("workspace", "parent", BaseSnapshotId(value="base"))
    manager.advance("workspace", "parent", "one", _delta())
    sibling = manager.create("workspace", "sibling", BaseSnapshotId(value="base"))
    child = manager.fork("workspace", "parent", "child")

    assert len({parent.branch_id, sibling.branch_id, child.branch_id}) == 3
    assert child.committed_deltas == (_delta(),)
    assert child.committed_transaction_ids == ("one",)
    assert ("fork", f"{child.branch_id.value}:one") in materializer.calls


@pytest.mark.parametrize(
    "transaction_ids, deltas",
    (
        (("one",), ()),
        (("one", "one"), (_delta(), _delta())),
    ),
)
def test_context_requires_one_unique_identity_per_committed_delta(
    transaction_ids: tuple[str, ...],
    deltas: tuple[CanonicalDelta, ...],
):
    """Persisted lineage cannot separate or duplicate transaction idempotency identities."""
    with pytest.raises(ValidationError, match="transaction"):
        AgentWorkspaceContext(
            workspace_id="workspace",
            agent_run_id="agent",
            base_snapshot_id=BaseSnapshotId(value="base"),
            branch_id=BranchId(value="branch"),
            generation_id=GenerationId(value="1"),
            lineage_epoch=LineageEpoch(value="0"),
            committed_deltas=deltas,
            committed_transaction_ids=transaction_ids,
        )


def test_generation_leases_block_mutation_refresh_and_disposal(tmp_path: Path):
    """A leased generation cannot be advanced, refreshed, or disposed."""
    manager, _, _ = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))

    with manager.lease_generation("workspace", "agent") as (leased, view):
        assert leased == context
        assert isinstance(view, GenerationView)
        with pytest.raises(RuntimeError, match="active generation leases"):
            manager.advance("workspace", "agent", "one", _delta())
        with pytest.raises(RuntimeError, match="active generation leases"):
            manager.refresh("workspace", "agent", BaseSnapshotId(value="next"))
        with pytest.raises(RuntimeError, match="active generation leases"):
            manager.dispose("workspace", "agent")

    refreshed = manager.refresh("workspace", "agent", BaseSnapshotId(value="next"))
    assert refreshed.base_snapshot_id == BaseSnapshotId(value="next")
    assert refreshed.committed_deltas == ()
    with pytest.raises(RuntimeError, match="stale workspace generation"):
        manager.require_publishable(context)


def test_refresh_fails_closed_when_publication_cannot_be_verified(tmp_path: Path):
    """An agent branch is retained unless trusted publication verification succeeds."""
    materializer = _Materializer()
    manager = AgentWorkspaceManager(
        materializer,
        lambda context: False,
        tmp_path / "contexts",
    )
    manager.create("workspace", "agent", BaseSnapshotId(value="base"))

    with pytest.raises(RuntimeError, match="published commits are verified"):
        manager.refresh("workspace", "agent", BaseSnapshotId(value="next"))


def test_publication_orders_host_branch_and_durable_completion_idempotently(tmp_path: Path):
    """Host publication precedes one delta-only branch advance and completion marker."""
    manager, materializer, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))

    updated = coordinator.publish("tx", context, _delta())
    replayed = coordinator.publish("tx", context, _delta())
    transaction = coordinator.journal.load("tx")

    assert updated.generation_id == GenerationId(value="1")
    assert replayed == updated
    assert transaction is not None and transaction.phase is JournalPhase.COMMITTED
    assert [call for call in materializer.calls if call[0] == "apply"] == [
        ("apply", f"{updated.branch_id.value}:tx")
    ]


def test_recovery_advances_host_applied_transaction_once(tmp_path: Path):
    """Recovery resumes the branch half after a durable host-only transaction."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    transaction = coordinator.broker.commit(
        transaction_id="tx",
        workspace_id=context.workspace_id,
        agent_run_id=context.agent_run_id,
        base_snapshot_id=context.base_snapshot_id,
        generation_id=context.generation_id.value,
        delta=_delta(),
    )

    recovered = coordinator.recover("tx")

    assert transaction.phase is JournalPhase.HOST_APPLIED
    assert recovered is not None and recovered.generation_id == GenerationId(value="1")
    assert coordinator.recover("tx") is None


def test_manager_rejects_duplicate_contexts_and_unknown_contexts(tmp_path: Path):
    """Context identities are exact rather than process-global fallbacks."""
    manager, _, _ = _coordinator(tmp_path)
    manager.create("workspace", "agent", BaseSnapshotId(value="base"))

    with pytest.raises(ValueError, match="already exists"):
        manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    with pytest.raises(KeyError):
        manager.get("workspace", "missing")


def test_fork_duplicate_advance_and_disposal_are_idempotent_at_the_manager_boundary(
    tmp_path: Path,
):
    """A branch records transaction replay once and can be explicitly released when quiescent."""
    manager, materializer, _ = _coordinator(tmp_path)
    manager.create("workspace", "parent", BaseSnapshotId(value="base"))
    manager.fork("workspace", "parent", "child")
    with pytest.raises(ValueError, match="already exists"):
        manager.fork("workspace", "parent", "child")
    first = manager.advance("workspace", "parent", "one", _delta())
    assert manager.advance("workspace", "parent", "one", _delta()) == first
    manager.dispose("workspace", "parent")
    assert ("dispose", first.branch_id.value) in materializer.calls


def test_recovery_returns_nothing_for_missing_and_completed_transactions(tmp_path: Path):
    """Recovery has no implicit fallback when durable evidence is absent or complete."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    assert coordinator.recover("missing") is None
    coordinator.publish("tx", context, _delta())
    assert coordinator.recover("tx") is None


def test_recovery_uses_the_broker_for_a_prepared_host_phase(tmp_path: Path, monkeypatch):
    """A prepared journal state is recovered through the confined host broker before branching."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    prepared = JournalTransaction(
        transaction_id="prepared",
        workspace_id="workspace",
        agent_run_id="agent",
        base_snapshot_id=context.base_snapshot_id,
        generation_id="0",
        delta=_delta(),
        preconditions=(),
    )
    coordinator.journal.persist(prepared)
    monkeypatch.setattr(
        coordinator.broker,
        "recover",
        lambda transaction_id: prepared.model_copy(update={"phase": JournalPhase.HOST_APPLIED}),
    )

    recovered = coordinator.recover("prepared")

    assert recovered is not None and recovered.generation_id == GenerationId(value="1")


def test_recovery_tolerates_a_prepared_record_removed_by_another_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prepared record that disappears during broker recovery is already resolved."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    coordinator.journal.persist(
        JournalTransaction(
            transaction_id="prepared",
            workspace_id="workspace",
            agent_run_id="agent",
            base_snapshot_id=context.base_snapshot_id,
            generation_id="0",
            delta=CanonicalDelta(),
            preconditions=(),
        )
    )
    monkeypatch.setattr(coordinator.broker, "recover", lambda transaction_id: None)

    assert coordinator.recover("prepared") is None


def test_restart_restores_lineage_and_recovers_every_incomplete_phase(tmp_path: Path):
    """Startup reconstructs lineages and deterministically finishes all journal phases."""
    manager, materializer, coordinator = _coordinator(tmp_path)
    phases = (
        JournalPhase.PREPARED,
        JournalPhase.HOST_APPLIED,
        JournalPhase.BRANCH_APPLIED,
    )
    for phase in phases:
        agent_run_id = phase.value
        context = manager.create("workspace", agent_run_id, BaseSnapshotId(value="base"))
        delta = CanonicalDelta()
        transaction = JournalTransaction(
            transaction_id=f"tx-{phase.value}",
            workspace_id="workspace",
            agent_run_id=agent_run_id,
            base_snapshot_id=context.base_snapshot_id,
            generation_id="0",
            delta=delta,
            preconditions=(),
            phase=phase,
        )
        if phase is JournalPhase.BRANCH_APPLIED:
            manager.advance("workspace", agent_run_id, transaction.transaction_id, delta)
        coordinator.journal.persist(transaction)

    (tmp_path / "contexts" / "ignored.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "contexts" / "ignored-directory").mkdir()
    restarted_manager = AgentWorkspaceManager(
        materializer,
        lambda context: True,
        tmp_path / "contexts",
    )
    restarted = PublicationCoordinator(
        restarted_manager,
        coordinator.broker,
        coordinator.journal,
    )
    (coordinator.journal.directory / "ignored.tmp").write_text("partial", encoding="utf-8")
    (coordinator.journal.directory / "ignored-directory").mkdir()

    recovered = restarted.recover_incomplete()

    assert len(recovered) == len(phases)
    for phase in phases:
        transaction_id = f"tx-{phase.value}"
        context = restarted_manager.get("workspace", phase.value)
        assert context.committed_transaction_ids == (transaction_id,)
        transaction = restarted.journal.load(transaction_id)
        assert transaction is not None and transaction.phase is JournalPhase.COMMITTED
    assert restarted.journal.incomplete() == ()


def test_startup_recovery_rejects_a_transaction_for_another_generation(tmp_path: Path):
    """Recovery fails closed when journal identity does not match durable lineage."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    coordinator.journal.persist(
        JournalTransaction(
            transaction_id="wrong-generation",
            workspace_id="workspace",
            agent_run_id="agent",
            base_snapshot_id=context.base_snapshot_id,
            generation_id="9",
            delta=CanonicalDelta(),
            preconditions=(),
            phase=JournalPhase.HOST_APPLIED,
        )
    )

    with pytest.raises(RuntimeError, match="durable agent generation"):
        coordinator.recover_incomplete()


def test_startup_enumeration_tolerates_a_transaction_completed_during_recovery(
    tmp_path: Path,
    monkeypatch,
):
    """Startup ignores a record another recovery owner completed after enumeration."""
    manager, _, coordinator = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    coordinator.journal.persist(
        JournalTransaction(
            transaction_id="raced",
            workspace_id="workspace",
            agent_run_id="agent",
            base_snapshot_id=context.base_snapshot_id,
            generation_id="0",
            delta=CanonicalDelta(),
            preconditions=(),
            phase=JournalPhase.HOST_APPLIED,
        )
    )
    monkeypatch.setattr(coordinator, "recover", lambda transaction_id: None)

    assert coordinator.recover_incomplete() == ()


@pytest.mark.parametrize(
    "contents, message",
    (
        ("not-json", "malformed"),
        ('{"checksum":"bad","payload":{}}', "checksum"),
    ),
)
def test_restart_rejects_corrupt_lineage_state(
    tmp_path: Path,
    contents: str,
    message: str,
):
    """Durable lineage corruption fails closed instead of creating empty state."""
    state_directory = tmp_path / "contexts"
    state_directory.mkdir()
    (state_directory / "record").write_text(contents, encoding="utf-8")

    with pytest.raises(AgentWorkspaceCorruptionError, match=message):
        AgentWorkspaceManager(_Materializer(), lambda context: True, state_directory)


def test_restart_rejects_invalid_and_misplaced_lineage_payloads(tmp_path: Path):
    """Authenticated state fails closed on schema and mismatched logical identity paths."""
    state_directory = tmp_path / "contexts"
    manager = AgentWorkspaceManager(_Materializer(), lambda context: True, state_directory)
    manager.create("workspace", "agent", BaseSnapshotId(value="base"))
    state_path = next(path for path in state_directory.iterdir() if path.suffix != ".tmp")
    envelope = json.loads(state_path.read_bytes())
    payload = envelope["payload"]
    misplaced = state_directory / "misplaced"
    state_path.rename(misplaced)
    with pytest.raises(AgentWorkspaceCorruptionError, match="identity"):
        AgentWorkspaceManager(_Materializer(), lambda context: True, state_directory)

    payload["format_version"] = 2
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    misplaced.write_text(
        json.dumps(
            {"checksum": hashlib.sha256(encoded).hexdigest(), "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(AgentWorkspaceCorruptionError, match="invalid"):
        AgentWorkspaceManager(_Materializer(), lambda context: True, state_directory)


def test_context_sanitizer_never_exposes_private_branch_or_delta_state(tmp_path: Path):
    """Model-safe lineage summaries retain only opaque public identifiers."""
    manager, _, _ = _coordinator(tmp_path)
    context = manager.create("workspace", "agent", BaseSnapshotId(value="private-base"))

    sanitized = sanitize_workspace_context(context)

    assert sanitized == SanitizedWorkspaceContext(
        workspace_id="workspace", agent_run_id="agent", generation_id="0", lineage_epoch="0"
    )
    assert "branch" not in sanitized.model_dump_json()
    assert "private-base" not in sanitized.model_dump_json()
