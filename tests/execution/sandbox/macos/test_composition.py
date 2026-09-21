"""Tests for lazy production macOS sandbox composition."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import loop.execution.sandbox.macos.composition as composition_module
from loop.execution.contracts import (
    ExecutionLease,
    ExecutionMode,
    JobHandle,
    ShellExecutionRequest,
    TerminalMode,
)
from loop.execution.results import Completed
from loop.execution.sandbox.macos import LimaLifecycleState, load_macos_runtime_release
from loop.execution.sandbox.macos.composition import MacosProductAdapter
from loop.execution.sandbox.oci.spec import OciResourceLimits
from loop.execution.vfs import BaseSnapshotId, SnapshotManifest


def _request(agent="agent"):
    """Return one authorized request for composition tests."""
    return ShellExecutionRequest(
        request_id="request",
        lease=ExecutionLease(
            lease_id="lease",
            workspace_id="workspace",
            agent_run_id=agent,
            policy_version="2",
            runtime_digest="sha256:" + "b" * 64,
            expires_at_ns=2**63,
            capabilities=frozenset(),
        ),
        script="printf ok",
    )


def _adapter(tmp_path, monkeypatch):
    """Compose an adapter with every external boundary replaced by a deterministic double."""
    snapshot_store = tmp_path / "snapshots"
    snapshot_store.mkdir()
    prepared = MagicMock()
    prepared.instance.context.state_root = tmp_path / "control"
    backend = MagicMock(snapshot_store=snapshot_store)
    backend.prepare.return_value = prepared
    root = MagicMock()
    manager = MagicMock()
    manager.get.side_effect = [KeyError("missing"), object(), object()]
    publication = MagicMock()
    delegate = MagicMock()
    result = Completed(request_id="request", exit_code=0)
    delegate.execute.return_value = result
    snapshot = SnapshotManifest(snapshot_id=BaseSnapshotId(value="snapshot-one"), entries=())
    builder = MagicMock()
    builder.build.return_value = SimpleNamespace(manifest=snapshot)
    coordinator_arguments = {}

    def coordinator(*args):
        coordinator_arguments["manifest_loader"] = args[4]
        return MagicMock()

    monkeypatch.setattr(composition_module, "AuthenticatedWorkspaceRoot", lambda _: root)
    monkeypatch.setattr(composition_module, "StagedContentStore", MagicMock())
    monkeypatch.setattr(composition_module, "MacosWorkspaceMaterializer", MagicMock())
    journal = MagicMock()
    journal.load.return_value = SimpleNamespace(phase=composition_module.JournalPhase.COMMITTED)
    monkeypatch.setattr(composition_module, "TransactionJournal", MagicMock(return_value=journal))
    monkeypatch.setattr(
        composition_module, "AgentWorkspaceManager", MagicMock(return_value=manager)
    )
    monkeypatch.setattr(composition_module, "CommitBroker", MagicMock())
    monkeypatch.setattr(
        composition_module, "PublicationCoordinator", MagicMock(return_value=publication)
    )
    monkeypatch.setattr(composition_module, "MacosWorkspaceCoordinator", coordinator)
    monkeypatch.setattr(composition_module, "MacosDeltaArchiveInspector", MagicMock())
    capabilities = MagicMock()
    capabilities.discover.return_value = MagicMock()
    monkeypatch.setattr(
        composition_module, "FilesystemCapabilities", MagicMock(return_value=capabilities)
    )
    monkeypatch.setattr(composition_module, "SnapshotBuilder", MagicMock(return_value=builder))
    monkeypatch.setattr(
        composition_module, "MacosExecutionAdapter", MagicMock(return_value=delegate)
    )
    jobs = MagicMock()
    jobs.start.return_value = JobHandle(job_id="job", token="a" * 64)
    jobs.has_live_jobs.return_value = False
    monkeypatch.setattr(composition_module, "DurableJobStore", MagicMock())
    monkeypatch.setattr(composition_module, "MacosDurableJobManager", MagicMock(return_value=jobs))
    manifest = load_macos_runtime_release()
    adapter = MacosProductAdapter(
        backend,
        manifest,
        tmp_path,
        tmp_path / "state",
        MagicMock(),
        OciResourceLimits(1024, 2, 1000),
    )
    return adapter, backend, prepared, root, manager, publication, delegate, coordinator_arguments


def test_product_adapter_initializes_recovers_creates_agents_and_closes(tmp_path, monkeypatch):
    """First use prepares once, recovers journals, creates one lineage, and reuses it."""
    (
        adapter,
        backend,
        prepared,
        root,
        manager,
        publication,
        delegate,
        coordinator_arguments,
    ) = _adapter(tmp_path, monkeypatch)

    first = adapter.execute(_request(), MagicMock(), lambda: False)
    second = adapter.execute(_request(), MagicMock(), lambda: False)
    durable = _request().model_copy(
        update={
            "mode": ExecutionMode.DURABLE_JOB,
            "terminal": TerminalMode.PTY,
            "terminal_columns": 80,
            "terminal_rows": 24,
        }
    )
    handle = adapter.start_job(durable)

    assert first.exit_code == second.exit_code == 0
    backend.prepare.assert_called_once_with()
    publication.recover_incomplete.assert_called_once_with()
    manager.create.assert_called_once_with(
        "workspace", "agent", BaseSnapshotId(value="snapshot-one")
    )
    assert delegate.execute.call_count == 2
    jobs = composition_module.MacosDurableJobManager.return_value
    jobs.recover.assert_called_once_with()
    jobs.start.assert_called_once()
    assert handle.job_id == "job"
    assert composition_module.MacosWorkspaceMaterializer.call_args.args[2] == (
        tmp_path / "control" / "archives"
    )
    assert adapter.job_manager() is jobs
    verifier = composition_module.AgentWorkspaceManager.call_args.args[1]
    context = SimpleNamespace(committed_transaction_ids=())
    assert verifier(context) is True
    context.committed_transaction_ids = ("transaction",)
    assert verifier(context) is True
    journal = composition_module.TransactionJournal.return_value
    journal.load.return_value = None
    assert verifier(context) is False
    journal.load.return_value = SimpleNamespace(phase=composition_module.JournalPhase.PREPARED)
    assert verifier(context) is False

    snapshot_directory = backend.snapshot_store / "snapshot-one"
    snapshot_directory.mkdir()
    manifest = SnapshotManifest(snapshot_id=BaseSnapshotId(value="snapshot-one"), entries=())
    (snapshot_directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    assert coordinator_arguments["manifest_loader"](manifest.snapshot_id) == manifest

    adapter.close()
    adapter.close()
    delegate.close.assert_called_once_with()
    jobs.close.assert_called_once_with()
    prepared.close.assert_not_called()
    root.close.assert_called_once_with()


def test_durable_entrypoints_can_initialize_the_product_lazily(tmp_path, monkeypatch) -> None:
    """Allow either durable start or recovery access to own first-use composition."""
    (tmp_path / "start").mkdir()
    adapter, *_ = _adapter(tmp_path / "start", monkeypatch)
    durable = _request().model_copy(
        update={
            "mode": ExecutionMode.DURABLE_JOB,
            "terminal": TerminalMode.PTY,
            "terminal_columns": 80,
            "terminal_rows": 24,
        }
    )
    assert adapter.start_job(durable).job_id == "job"
    adapter.close()

    (tmp_path / "manager").mkdir()
    adapter, *_ = _adapter(tmp_path / "manager", monkeypatch)
    assert adapter.job_manager() is composition_module.MacosDurableJobManager.return_value
    adapter.close()


def test_ensure_sandbox_uses_the_shared_warm_runtime(tmp_path, monkeypatch) -> None:
    """Lazy image preparation returns the one descriptor from a shared composition."""
    adapter, backend, prepared, *_ = _adapter(tmp_path, monkeypatch)
    digest = "sha256:" + "f" * 64
    prepared.sandbox_image = SimpleNamespace(
        image=SimpleNamespace(identity=SimpleNamespace(manifest_digest=digest))
    )

    assert adapter.ensure_sandbox() == digest
    assert adapter.ensure_sandbox() == digest
    backend.prepare.assert_called_once_with()
    adapter.close()


@pytest.mark.parametrize("fail_after_root", [False, True])
def test_product_adapter_closes_partial_initialization(tmp_path, monkeypatch, fail_after_root):
    """Every initialization failure releases its runtime lease and any opened root."""
    manifest = load_macos_runtime_release()
    prepared = MagicMock()
    backend = MagicMock(snapshot_store=tmp_path / "snapshots")
    backend.prepare.return_value = prepared
    root = MagicMock()
    if fail_after_root:
        monkeypatch.setattr(composition_module, "AuthenticatedWorkspaceRoot", lambda _: root)
        monkeypatch.setattr(
            composition_module,
            "StagedContentStore",
            MagicMock(side_effect=RuntimeError("failed")),
        )
    else:
        monkeypatch.setattr(
            composition_module,
            "AuthenticatedWorkspaceRoot",
            MagicMock(side_effect=RuntimeError("failed")),
        )
    adapter = MacosProductAdapter(
        backend,
        manifest,
        tmp_path,
        tmp_path / "state",
        MagicMock(),
        OciResourceLimits(1024, 2, 1000),
    )

    with pytest.raises(RuntimeError, match="failed"):
        adapter.execute(_request(), MagicMock(), lambda: False)

    prepared.close.assert_called_once_with()
    if fail_after_root:
        root.close.assert_called_once_with()


def test_product_adapter_does_not_close_borrowed_runtime_on_initialization_failure(
    tmp_path, monkeypatch
) -> None:
    """A borrowed prepared runtime remains owned by its outer composition on failure."""
    prepared = MagicMock()
    backend = MagicMock(snapshot_store=tmp_path / "snapshots")
    monkeypatch.setattr(
        composition_module,
        "AuthenticatedWorkspaceRoot",
        MagicMock(side_effect=RuntimeError("failed")),
    )
    adapter = MacosProductAdapter(
        backend,
        load_macos_runtime_release(),
        tmp_path,
        tmp_path / "state",
        MagicMock(),
        OciResourceLimits(1024, 2, 1000),
        prepared_runtime=prepared,
    )

    with pytest.raises(RuntimeError, match="failed"):
        adapter.execute(_request(), MagicMock(), lambda: False)

    backend.prepare.assert_not_called()
    prepared.close.assert_not_called()


def test_product_adapter_stops_an_idle_owned_vm_on_close(tmp_path, monkeypatch) -> None:
    """Graceful shutdown stops an idle owned VM instead of leaving a Lima process running."""
    adapter, _, prepared, *_ = _adapter(tmp_path, monkeypatch)
    adapter.execute(_request(), MagicMock(), lambda: False)

    adapter.close()

    prepared.stop.assert_called_once_with()


def test_product_adapter_reacquires_ownership_when_close_lease_is_stale(
    tmp_path, monkeypatch
) -> None:
    """Graceful shutdown still stops an owned VM after its original lease becomes stale."""
    adapter, backend, prepared, *_ = _adapter(tmp_path, monkeypatch)
    adapter.execute(_request(), MagicMock(), lambda: False)
    prepared.stop.side_effect = composition_module.LimaConfigurationError("stale lease")
    backend.manage.return_value = True

    adapter.close()

    backend.manage.assert_called_once_with(delete=False)


def test_product_adapter_keeps_vm_for_a_live_durable_job(tmp_path, monkeypatch) -> None:
    """Recoverable durable jobs retain their VM across application shutdown."""
    adapter, _, prepared, *_ = _adapter(tmp_path, monkeypatch)
    adapter.execute(_request(), MagicMock(), lambda: False)
    adapter._jobs.has_live_jobs.return_value = True

    adapter.close()

    prepared.stop.assert_not_called()


@pytest.mark.parametrize("delete", [False, True])
def test_product_adapter_manages_initialized_owned_sandbox(tmp_path, monkeypatch, delete) -> None:
    """Explicit management cleans initialized owned state in lifecycle order."""
    adapter, backend, prepared, root, _, _, delegate, _ = _adapter(tmp_path, monkeypatch)
    adapter.execute(_request(), MagicMock(), lambda: False)

    assert adapter.manage_sandbox(delete=delete) is True

    if delete:
        prepared.delete.assert_called_once_with()
        prepared.stop.assert_not_called()
    else:
        prepared.stop.assert_called_once_with()
        prepared.delete.assert_not_called()
    delegate.close.assert_called_once_with()
    root.close.assert_called_once_with()
    backend.manage.assert_not_called()


def test_product_adapter_manages_durable_uninitialized_sandbox(tmp_path, monkeypatch) -> None:
    """Explicit management reaches durable state without starting a new sandbox."""
    adapter, backend, *_ = _adapter(tmp_path, monkeypatch)
    backend.manage.return_value = True

    assert adapter.manage_sandbox(delete=True) is True

    backend.prepare.assert_not_called()
    backend.manage.assert_called_once_with(delete=True)


def test_product_adapter_rejects_cleanup_with_live_jobs(tmp_path, monkeypatch) -> None:
    """Explicit cleanup cannot destroy the runtime of a recoverable live job."""
    adapter, _, prepared, *_ = _adapter(tmp_path, monkeypatch)
    adapter.execute(_request(), MagicMock(), lambda: False)
    adapter._jobs.has_live_jobs.return_value = True

    with pytest.raises(RuntimeError, match="Live durable jobs"):
        adapter.manage_sandbox(delete=True)

    prepared.delete.assert_not_called()


def test_product_adapter_rejects_management_of_borrowed_runtime(tmp_path, monkeypatch) -> None:
    """An inner adapter cannot stop or delete a runtime owned by its caller."""
    prepared = MagicMock()
    backend = MagicMock(snapshot_store=tmp_path / "snapshots")
    backend.snapshot_store.mkdir()
    adapter = MacosProductAdapter(
        backend,
        load_macos_runtime_release(),
        tmp_path,
        tmp_path / "state",
        MagicMock(),
        OciResourceLimits(1024, 2, 1000),
        prepared_runtime=prepared,
    )
    adapter._prepared = prepared

    with pytest.raises(RuntimeError, match="Borrowed sandbox"):
        adapter.manage_sandbox(delete=True)

    prepared.delete.assert_not_called()


def test_product_adapter_exposes_workspace_sandbox_status_and_inventory(
    tmp_path, monkeypatch
) -> None:
    """Sandbox inspection normalizes backend lifecycle states for application use."""
    adapter, backend, *_ = _adapter(tmp_path, monkeypatch)
    backend.status.return_value = ("loop-one", LimaLifecycleState.STOPPED)
    backend.list_sandboxes.return_value = (("loop-one", LimaLifecycleState.STOPPED),)

    assert adapter.sandbox_status() == ("loop-one", "STOPPED")
    assert adapter.list_sandboxes() == (("loop-one", "STOPPED"),)
    backend.status.return_value = None
    assert adapter.sandbox_status() is None
