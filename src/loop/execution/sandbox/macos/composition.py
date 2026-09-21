"""Compose the production macOS sandbox lazily at the platform boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import RLock

from ....permissions.execution_adapter import ExecutionPermissionAdapter
from ....utils import is_workspace_path_ignored
from ...broker import SecretAuthority
from ...contracts import JobHandle, SandboxExecutionRequest
from ...results import ExecutionResult
from ...runtime.manifest import RuntimeManifest
from ...service import AttemptObserver
from ...state_machine import AttemptState, AttemptStateMachine
from ...vfs import (
    AgentWorkspaceContext,
    AgentWorkspaceManager,
    AuthenticatedWorkspaceRoot,
    CommitBroker,
    FilesystemCapabilities,
    JournalPhase,
    PublicationCoordinator,
    SnapshotBuilder,
    SnapshotManifest,
    StagedContentStore,
    TransactionJournal,
)
from ..oci.spec import OciResourceLimits
from .backend import MacosSandboxBackend, PreparedMacosRuntime
from .execution import MacosExecutionAdapter
from .jobs import DurableJobStore, MacosDurableJobManager
from .lima import LimaConfigurationError
from .workspace import (
    MacosDeltaArchiveInspector,
    MacosWorkspaceCoordinator,
    MacosWorkspaceMaterializer,
)


class MacosProductAdapter:
    """Lazily compose one workspace-bound production sandbox.

    Args:
        backend (MacosSandboxBackend): Lazy managed runtime substrate.
        manifest (RuntimeManifest): Embedded qualified production manifest.
        workspace_root (Path): Authenticated host publication root.
        state_root (Path): Loop-private durable execution state.
        permissions (ExecutionPermissionAdapter): Sole publication authorization adapter.
        limits (OciResourceLimits): Mandatory limits for every ordinary command.
        secret_authority (SecretAuthority | None): Application-owned exact secret resolver.
        prepared_runtime (PreparedMacosRuntime | None): Borrowed pre-attested runtime for an
            already composed product boundary.
    """

    backend: MacosSandboxBackend
    manifest: RuntimeManifest
    workspace_root: Path
    state_root: Path
    permissions: ExecutionPermissionAdapter
    limits: OciResourceLimits
    secret_authority: SecretAuthority | None
    prepared_runtime: PreparedMacosRuntime | None
    _lock: RLock
    _prepared: PreparedMacosRuntime | None
    _root: AuthenticatedWorkspaceRoot | None
    _manager: AgentWorkspaceManager | None
    _publication: PublicationCoordinator | None
    _delegate: MacosExecutionAdapter | None
    _jobs: MacosDurableJobManager | None

    def __init__(
        self,
        backend: MacosSandboxBackend,
        manifest: RuntimeManifest,
        workspace_root: Path,
        state_root: Path,
        permissions: ExecutionPermissionAdapter,
        limits: OciResourceLimits,
        *,
        secret_authority: SecretAuthority | None = None,
        prepared_runtime: PreparedMacosRuntime | None = None,
    ) -> None:
        self.backend = backend
        self.manifest = manifest
        self.workspace_root = workspace_root.resolve()
        self.state_root = state_root.resolve()
        self.permissions = permissions
        self.limits = limits
        self.secret_authority = secret_authority
        self.prepared_runtime = prepared_runtime
        self._lock = RLock()
        self._prepared = None
        self._root = None
        self._manager = None
        self._publication = None
        self._delegate = None
        self._jobs = None

    def execute(
        self,
        request: SandboxExecutionRequest,
        observer: AttemptObserver,
        cancellation: Callable[[], bool],
    ) -> ExecutionResult:
        """Execute through the lazily prepared sandbox and durable agent lineage.

        Args:
            request (SandboxExecutionRequest): Authorized virtual request.
            observer (AttemptObserver): Service-owned lifecycle observer.
            cancellation (Callable[[], bool]): Cancellation predicate.

        Returns:
            ExecutionResult: Closed program outcome or fail-closed infrastructure result.
        """
        with self._lock:
            if self._delegate is None:
                self._initialize()
            self._ensure_agent(request.lease.workspace_id, request.lease.agent_run_id)
            delegate = self._delegate
        if delegate is None:  # pragma: no cover - guarded by fail-closed initialization.
            raise RuntimeError("Managed macOS execution did not initialize.")
        return delegate.execute(request, observer, cancellation)

    def close(self) -> None:
        """Stop an idle owned VM and release authenticated workspace resources."""
        with self._lock:
            prepared = self._prepared
            keep_running = self._jobs is not None and self._jobs.has_live_jobs()
            if prepared is not None and self.prepared_runtime is None and not keep_running:
                try:
                    prepared.stop()
                except LimaConfigurationError:
                    # A long-lived application may outlast its original runtime lease.
                    # Reacquire and validate exact journal ownership before stopping so
                    # shutdown does not strand the Lima hostagent.
                    self.backend.manage(delete=False)
            if self._jobs is not None:
                self._jobs.close()
                self._jobs = None
            if self._delegate is not None:
                self._delegate.close()
                self._delegate = None
                self._prepared = None
            if self._root is not None:
                self._root.close()
                self._root = None

    def ensure_sandbox(self) -> str:
        """Prepare the single local image and return its attested manifest digest.

        Returns:
            str: Exact selected platform-manifest digest used by authorization and execution.

        Raises:
            RuntimeError: If managed preparation or attestation fails closed.
        """
        with self._lock:
            if self._delegate is None:
                self._initialize()
            prepared = self._prepared
            if prepared is None:  # pragma: no cover - initialized atomically.
                raise RuntimeError("Managed macOS execution did not initialize.")
            return prepared.sandbox_image.image.identity.manifest_digest

    def manage_sandbox(self, *, delete: bool) -> bool:
        """Stop or delete the owned sandbox while rejecting live durable jobs.

        Args:
            delete (bool): Delete the VM and its private disk when ``True``; otherwise stop it.

        Returns:
            bool: Whether an initialized or durable sandbox existed.

        Raises:
            RuntimeError: If a live durable job requires the VM to remain running.
        """
        with self._lock:
            if self._jobs is not None and self._jobs.has_live_jobs():
                raise RuntimeError("Live durable jobs prevent sandbox cleanup.")
            prepared = self._prepared
            if prepared is not None and self.prepared_runtime is not None:
                raise RuntimeError("Borrowed sandbox runtime cannot be managed by this adapter.")
            if prepared is not None:
                if delete:
                    prepared.delete()
                else:
                    prepared.stop()
            if self._jobs is not None:
                self._jobs.close()
                self._jobs = None
            if self._delegate is not None:
                self._delegate.close()
                self._delegate = None
            if self._root is not None:
                self._root.close()
                self._root = None
            self._prepared = None
            self._manager = None
            self._publication = None
            if prepared is None:
                return self.backend.manage(delete=delete)
            return True

    def sandbox_status(self) -> tuple[str, str] | None:
        """Return the attested current-workspace sandbox state without preparing it.

        Returns:
            tuple[str, str] | None: Instance name and lifecycle state, or ``None`` when absent.
        """
        with self._lock:
            status = self.backend.status()
        return None if status is None else (status[0], status[1].value)

    def list_sandboxes(self) -> tuple[tuple[str, str], ...]:
        """List attested sandboxes owned by the current workspace.

        Returns:
            tuple[tuple[str, str], ...]: Owned instance names and lifecycle states.
        """
        with self._lock:
            records = self.backend.list_sandboxes()
        return tuple((name, state.value) for name, state in records)

    def _initialize(self) -> None:
        """Prepare upstream runtime state and portable workspace authorities once."""
        prepared = self.prepared_runtime or self.backend.prepare()
        owns_prepared = self.prepared_runtime is None
        root: AuthenticatedWorkspaceRoot | None = None
        try:
            self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root = AuthenticatedWorkspaceRoot(self.workspace_root)
            content_store = StagedContentStore(self.state_root / "content")
            materializer = MacosWorkspaceMaterializer(
                prepared,
                content_store,
                prepared.instance.context.state_root / "archives",
                self.limits.persistent_write_bytes,
            )
            journal = TransactionJournal(self.state_root / "journal")

            def committed(context: AgentWorkspaceContext) -> bool:
                """Return whether every lineage transaction is durably terminal."""
                return all(
                    (record := journal.load(transaction_id)) is not None
                    and record.phase is JournalPhase.COMMITTED
                    for transaction_id in context.committed_transaction_ids
                )

            manager = AgentWorkspaceManager(
                materializer,
                committed,
                self.state_root / "lineages",
            )
            broker = CommitBroker(root, journal, content_store)
            publication = PublicationCoordinator(manager, broker, journal)
            coordinator = MacosWorkspaceCoordinator(
                manager,
                materializer,
                MacosDeltaArchiveInspector(
                    content_store,
                    maximum_content_bytes=self.limits.persistent_write_bytes,
                    maximum_archive_bytes=self.limits.disk_bytes,
                    ignore_path=lambda relative: is_workspace_path_ignored(
                        root.path / relative, root.path
                    ),
                ),
                publication,
                self._load_manifest,
                FilesystemCapabilities().discover(root),
            )
            delegate = MacosExecutionAdapter(
                self.backend,
                self.manifest,
                coordinator,
                self.limits,
                self.permissions.authorize_publication,
                prepared_runtime=prepared,
                owns_prepared_runtime=owns_prepared,
                secret_authority=self.secret_authority,
            )
            jobs = MacosDurableJobManager(
                self.backend,
                self.manifest,
                coordinator,
                self.limits,
                self.permissions.authorize_job_operation,
                self.permissions.authorize_publication,
                DurableJobStore(self.state_root / "jobs"),
                prepared_runtime=prepared,
            )
            jobs.recover()
            publication.recover_incomplete()
        except BaseException:
            if root is not None:
                root.close()
            if owns_prepared:
                prepared.close()
            raise
        self._prepared = prepared
        self._root = root
        self._manager = manager
        self._publication = publication
        self._delegate = delegate
        self._jobs = jobs

    def start_job(self, request: SandboxExecutionRequest) -> JobHandle:
        """Start one authorized durable request through the shared workspace lifecycle.

        Args:
            request (SandboxExecutionRequest): Explicit durable sandbox request.

        Returns:
            JobHandle: Authenticated durable handle.
        """
        attempt = AttemptStateMachine()
        attempt.transition(AttemptState.AUTHORIZED)
        observer = AttemptObserver(attempt)
        with self._lock:
            if self._delegate is None:
                self._initialize()
            self._ensure_agent(request.lease.workspace_id, request.lease.agent_run_id)
            jobs = self._jobs
        if jobs is None:  # pragma: no cover - defensive atomicity invariant.
            raise RuntimeError("Managed durable jobs are unavailable.")  # pragma: no cover
        return jobs.start(request, observer)

    def job_manager(self) -> MacosDurableJobManager:
        """Return the initialized durable lifecycle owner for authenticated operations.

        Returns:
            MacosDurableJobManager: Shared non-parallel durable lifecycle owner.
        """
        with self._lock:
            if self._delegate is None:
                self._initialize()
            jobs = self._jobs
        if jobs is None:  # pragma: no cover - defensive atomicity invariant.
            raise RuntimeError("Managed durable jobs are unavailable.")  # pragma: no cover
        return jobs

    def _ensure_agent(self, workspace_id: str, agent_run_id: str) -> None:
        """Create one immutable starting snapshot for a new logical agent."""
        manager = self._manager
        root = self._root
        if manager is None or root is None:  # pragma: no cover - initialized atomically.
            raise RuntimeError("Managed macOS workspace is unavailable.")
        try:
            manager.get(workspace_id, agent_run_id)
        except KeyError:
            snapshot = SnapshotBuilder(self.backend.snapshot_store).build(root)
            manager.create(workspace_id, agent_run_id, snapshot.manifest.snapshot_id)

    def _load_manifest(self, snapshot_id) -> SnapshotManifest:
        """Load one immutable snapshot manifest by its validated opaque identity."""
        path = self.backend.snapshot_store / snapshot_id.value / "manifest.json"
        return SnapshotManifest.model_validate_json(path.read_bytes())
