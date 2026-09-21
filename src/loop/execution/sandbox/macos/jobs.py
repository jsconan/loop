"""Own authenticated durable jobs through the managed macOS OCI boundary."""

from __future__ import annotations

import hmac
import json
import os
import time
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ....utils import json_encode, matches_digest, sha256_digest
from ...contracts import (
    Capability,
    ExecutionLease,
    ExecutionMode,
    JobHandle,
    JobOperation,
    SandboxExecutionRequest,
    TerminalMode,
)
from ...runtime.manifest import RuntimeManifest
from ...service import AttemptObserver
from ...vfs import CanonicalDelta
from ..oci import OciSandboxAdapter, OciSpecCompiler
from ..oci.control import OciControlSession, OciSignal
from ..oci.process import OciStreamFrame
from ..oci.spec import OciResourceLimits
from .backend import MacosSandboxBackend, PreparedMacosRuntime
from .execution import WorkspacePublicationAuthorizer
from .workspace import (
    MacosAttemptLayer,
    MacosDurableWorkspaceLease,
    MacosWorkspaceCoordinator,
)


class DurableJobState(StrEnum):
    """Identify the durable lifecycle state visible to an authorized caller."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DENIED = "denied"
    LOST = "lost"
    FAILED = "failed"


class DurableJobStatus(BaseModel):
    """Report sanitized durable-job state without exposing platform identities.

    Args:
        job_id (str): Stable public job identity.
        state (DurableJobState): Current durable lifecycle state.
        exit_code (int | None): Terminal program status when safely attested.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    state: DurableJobState
    exit_code: int | None = None


class DurableJobAuthorizer(Protocol):
    """Authorize one operation using a freshly issued sandbox lease."""

    def __call__(
        self,
        lease: ExecutionLease,
        operation: JobOperation,
        job_id: str,
    ) -> bool:
        """Return whether the exact operation is currently authorized."""


class StoredLayer(BaseModel):
    """Persist the platform-private overlay identity required for restart recovery."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str
    branch_reference: str
    mount_source: str
    guest_directory: str
    guest_archive: str
    host_archive: str

    @classmethod
    def from_layer(cls, layer: MacosAttemptLayer) -> StoredLayer:
        """Capture one private layer without weakening its validated identities."""
        return cls(
            attempt_id=layer.attempt_id,
            branch_reference=layer.branch_reference,
            mount_source=layer.mount_source,
            guest_directory=layer.guest_directory,
            guest_archive=layer.guest_archive,
            host_archive=str(layer.host_archive),
        )

    def restore(self) -> MacosAttemptLayer:
        """Restore the exact private layer record for trusted management only."""
        return MacosAttemptLayer(
            self.attempt_id,
            self.branch_reference,
            self.mount_source,
            self.guest_directory,
            self.guest_archive,
            Path(self.host_archive),
        )


class JobRecord(BaseModel):
    """Persist the minimum authority and attestation identities for one job."""

    model_config = ConfigDict(frozen=True)

    format_version: int = Field(default=1, ge=1, le=1)
    job_id: str
    token_digest: str
    lease_epoch: str
    runtime_epoch: str
    request: SandboxExecutionRequest
    container_name: str
    generation_reference: str
    layer: StoredLayer
    state: DurableJobState
    exit_code: int | None = None


class DurableJobStore:
    """Persist checksum-protected durable-job records in private application state.

    Args:
        directory (Path): Loop-private job-state directory.
    """

    directory: Path

    def __init__(self, directory: Path) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Durable job state must be a private directory.")
        self.directory = directory.resolve()

    def load(self, job_id: str) -> JobRecord:
        """Load and authenticate one durable job record.

        Args:
            job_id (str): Stable public job identity.

        Returns:
            JobRecord: Validated private job state.

        Raises:
            KeyError: If the job is unknown.
            RuntimeError: If durable state is malformed or corrupted.
        """
        path = self._path(job_id)
        try:
            envelope = json.loads(path.read_bytes())
            payload = envelope["payload"]
            encoded = json_encode(payload).encode()
            if not matches_digest(encoded, envelope["checksum"]):
                raise ValueError
            record = JobRecord.model_validate(payload)
        except FileNotFoundError as error:
            raise KeyError(job_id) from error
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("Durable job state is invalid.") from error
        if record.job_id != job_id:
            raise RuntimeError("Durable job identity is invalid.")
        return record

    def list(self) -> tuple[JobRecord, ...]:
        """Return all validated durable job records in stable identity order.

        Returns:
            tuple[JobRecord, ...]: Persisted jobs ordered by public identity.
        """
        records = [self.load(path.stem) for path in self.directory.glob("*.json")]
        return tuple(sorted(records, key=lambda record: record.job_id))

    def save(self, record: JobRecord) -> None:
        """Atomically persist and fsync one complete durable job record.

        Args:
            record (JobRecord): Complete private job state.
        """
        payload = record.model_dump(mode="json")
        encoded = json_encode(payload).encode()
        envelope = json_encode({"checksum": sha256_digest(encoded), "payload": payload}).encode()
        target = self._path(record.job_id)
        temporary = target.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, envelope)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def delete(self, job_id: str) -> None:
        """Durably delete state for a start that never returned a handle.

        Args:
            job_id (str): Stable public job identity.
        """
        self._path(job_id).unlink(missing_ok=True)
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _path(self, job_id: str) -> Path:
        """Map one untrusted public identity to a confined state filename."""
        if (
            not job_id
            or len(job_id) > 128
            or not job_id[0].isalnum()
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for character in job_id
            )
        ):
            raise ValueError("Durable job identity is invalid.")
        return self.directory / f"{job_id}.json"


class MacosDurableJobManager:
    """Own durable container, attachment, recovery, and terminal publication lifecycles.

    Args:
        backend (MacosSandboxBackend): Managed workspace-bound runtime substrate.
        candidate (RuntimeManifest): Trusted manifest containing the pinned OCI profile.
        workspace (MacosWorkspaceCoordinator): Durable workspace and publication owner.
        limits (OciResourceLimits): Mandatory resource controls for every job.
        authorize_operation (DurableJobAuthorizer): Fresh authorization callback per operation.
        authorize_publication (WorkspacePublicationAuthorizer): Observed-delta authorizer.
        store (DurableJobStore): Private restart-recovery record store.
        prepared_runtime (PreparedMacosRuntime | None): Borrowed existing warm runtime lease.
        clock_ns (Callable[[], int]): Monotonic clock used to reject expired operation leases.
    """

    _backend: MacosSandboxBackend
    _candidate: RuntimeManifest
    _workspace: MacosWorkspaceCoordinator
    _limits: OciResourceLimits
    _authorize_operation: DurableJobAuthorizer
    _authorize_publication: WorkspacePublicationAuthorizer
    _store: DurableJobStore
    _prepared: PreparedMacosRuntime | None
    _owns_prepared: bool
    _clock_ns: Callable[[], int]
    _workspaces: dict[str, MacosDurableWorkspaceLease]
    _sessions: dict[str, OciControlSession]

    def __init__(
        self,
        backend: MacosSandboxBackend,
        candidate: RuntimeManifest,
        workspace: MacosWorkspaceCoordinator,
        limits: OciResourceLimits,
        authorize_operation: DurableJobAuthorizer,
        authorize_publication: WorkspacePublicationAuthorizer,
        store: DurableJobStore,
        *,
        prepared_runtime: PreparedMacosRuntime | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._backend = backend
        self._candidate = candidate
        self._workspace = workspace
        self._limits = limits
        self._authorize_operation = authorize_operation
        self._authorize_publication = authorize_publication
        self._store = store
        self._prepared = prepared_runtime
        self._owns_prepared = prepared_runtime is None
        self._clock_ns = clock_ns
        self._workspaces = {}
        self._sessions = {}

    def start(self, request: SandboxExecutionRequest, observer: AttemptObserver) -> JobHandle:
        """Start one explicitly durable sandbox job and return its authenticated handle.

        Args:
            request (SandboxExecutionRequest): Authorized durable sandbox request.
            observer (AttemptObserver): Service-owned attempt lifecycle observer.

        Returns:
            JobHandle: Authenticated handle required for every later operation.

        Raises:
            ValueError: If the request is not durable or uses an unsupported broker capability.
            RuntimeError: If creation, attestation, or detached start fails.
        """
        if request.mode is not ExecutionMode.DURABLE_JOB:
            raise ValueError("Durable job start requires explicit durable mode.")
        if request.terminal is not TerminalMode.PTY:
            raise ValueError("Durable jobs require PTY mode for reliable upstream reattachment.")
        if request.network_connections or request.network_listeners or request.secret_exposures:
            raise ValueError("Durable broker leases are not supported by this lifecycle.")
        handle = JobHandle.create(request.request_id)
        try:
            self._store.load(handle.job_id)
        except KeyError:
            pass
        else:
            raise ValueError("Durable job identity already exists.")
        workspace = self._workspace.lease_job(
            request.lease.workspace_id,
            request.lease.agent_run_id,
            request.request_id,
        )
        prepared = self._runtime()
        prepared.renew(request.deadline_seconds + 60.0)
        oci = self._oci(prepared, request.lease.runtime_digest)
        compiled = oci.prepare(request, workspace.leased.generation_reference)
        lease_epoch = uuid4().hex
        runtime_epoch = _runtime_epoch(prepared)
        labels = dict(compiled.spec.labels)
        labels.update(
            {
                "io.loop.epoch": lease_epoch,
                "io.loop.generation": workspace.leased.generation_reference,
                "io.loop.job": handle.job_id,
                "io.loop.runtime": request.lease.runtime_digest,
                "io.loop.runtime-epoch": runtime_epoch,
            }
        )
        spec = replace(compiled.spec, labels=tuple(sorted(labels.items())))
        record = JobRecord(
            job_id=handle.job_id,
            token_digest=_token_digest(handle.token),
            lease_epoch=lease_epoch,
            runtime_epoch=runtime_epoch,
            request=request,
            container_name=spec.container_name,
            generation_reference=workspace.leased.generation_reference,
            layer=StoredLayer.from_layer(workspace.leased.layer),
            state=DurableJobState.RUNNING,
        )
        self._store.save(record)
        self._workspaces[record.job_id] = workspace
        try:
            starter = prepared.control_plane.run_job(
                spec,
                workspace.leased.layer.mount_source,
                deadline_seconds=min(30.0, request.deadline_seconds),
            )
            try:
                self._require_container(record, prepared, expected_running=True)
                observer.backend_attested()
            finally:
                starter.detach()
            observer.process_started()
            return handle
        except BaseException:
            self._abandon(record, prepared)
            raise

    def recover(self) -> tuple[DurableJobStatus, ...]:
        """Reattest persisted jobs and fail closed on every ambiguous recovery.

        Returns:
            tuple[DurableJobStatus, ...]: Sanitized state for every persisted job.
        """
        statuses = []
        for record in self._store.list():
            if record.state in _TERMINAL_STATES:
                statuses.append(_status(record))
                continue
            try:
                self._activate(record)
                statuses.append(self._refresh(record))
            except Exception:  # noqa: BLE001 - recovery ambiguity must fail closed.
                lost = self._lose(record)
                statuses.append(_status(lost))
        return tuple(statuses)

    def status(self, handle: JobHandle, lease: ExecutionLease) -> DurableJobStatus:
        """Reauthorize, attest, and return current durable job state.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh operation authorization.

        Returns:
            DurableJobStatus: Current sanitized state.
        """
        record = self._authorize(handle, lease, JobOperation.STATUS)
        return self._refresh(record)

    def attach(
        self,
        handle: JobHandle,
        lease: ExecutionLease,
        *,
        deadline_seconds: float = 86400,
    ) -> None:
        """Open the sole authorized attachment for one running job.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh attach authorization.
            deadline_seconds (float): Positive bound for the management attachment.

        Raises:
            RuntimeError: If another attachment exists or the job is not running.
        """
        record = self._authorize(handle, lease, JobOperation.ATTACH)
        if handle.job_id in self._sessions:
            raise RuntimeError("Durable job already has an active attachment.")
        if self._refresh(record).state not in {DurableJobState.RUNNING, DurableJobState.PAUSED}:
            raise RuntimeError("Durable job is not attachable.")
        self._sessions[handle.job_id] = self._runtime().control_plane.attach(
            record.container_name,
            pty=record.request.terminal is TerminalMode.PTY,
            deadline_seconds=deadline_seconds,
        )

    def write(self, handle: JobHandle, lease: ExecutionLease, data: bytes) -> None:
        """Reauthorize and write one bounded input frame to an active attachment.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh stdin authorization.
            data (bytes): Opaque input frame.
        """
        self._authorize(handle, lease, JobOperation.STDIN)
        self._session(handle.job_id).write(data)

    def close_stdin(self, handle: JobHandle, lease: ExecutionLease) -> None:
        """Reauthorize and deliver end-of-input to an active attachment.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh stdin authorization.
        """
        self._authorize(handle, lease, JobOperation.STDIN)
        self._session(handle.job_id).close_stdin()

    def read(
        self,
        handle: JobHandle,
        lease: ExecutionLease,
        timeout: float | None = None,
    ) -> OciStreamFrame | None:
        """Reauthorize and read one ordered frame from an active attachment.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh attach authorization.
            timeout (float | None): Optional nonnegative queue wait.

        Returns:
            OciStreamFrame | None: Next frame, or ``None`` when none is available.
        """
        self._authorize(handle, lease, JobOperation.ATTACH)
        return self._session(handle.job_id).read(timeout)

    def resize(
        self,
        handle: JobHandle,
        lease: ExecutionLease,
        columns: int,
        rows: int,
    ) -> None:
        """Reauthorize and resize one active durable PTY attachment.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh resize authorization.
            columns (int): Positive terminal width.
            rows (int): Positive terminal height.
        """
        self._authorize(handle, lease, JobOperation.RESIZE)
        self._session(handle.job_id).resize(columns, rows)

    def detach(self, handle: JobHandle, lease: ExecutionLease) -> None:
        """Reauthorize and close only the management attachment, not the job.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh attach authorization.
        """
        self._authorize(handle, lease, JobOperation.ATTACH)
        session = self._sessions.pop(handle.job_id, None)
        if session is not None:
            session.detach()

    def signal(
        self,
        handle: JobHandle,
        lease: ExecutionLease,
        signal: OciSignal,
    ) -> None:
        """Reauthorize and deliver one closed-set container signal.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh signal authorization.
            signal (OciSignal): Reviewed signal to deliver.
        """
        record = self._authorize(handle, lease, JobOperation.SIGNAL)
        _require_success(
            self._runtime().control_plane.signal_container(record.container_name, signal),
            "Durable job signal failed.",
        )

    def suspend(self, handle: JobHandle, lease: ExecutionLease) -> DurableJobStatus:
        """Reauthorize and pause all processes in one running job.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh suspend authorization.

        Returns:
            DurableJobStatus: Attested paused state.
        """
        record = self._authorize(handle, lease, JobOperation.SUSPEND)
        _require_success(
            self._runtime().control_plane.pause_container(record.container_name),
            "Durable job suspend failed.",
        )
        return self._refresh(record)

    def resume(self, handle: JobHandle, lease: ExecutionLease) -> DurableJobStatus:
        """Reauthorize and resume all processes in one paused job.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh resume authorization.

        Returns:
            DurableJobStatus: Attested running state.
        """
        record = self._authorize(handle, lease, JobOperation.RESUME)
        _require_success(
            self._runtime().control_plane.unpause_container(record.container_name),
            "Durable job resume failed.",
        )
        return self._refresh(record)

    def cancel(self, handle: JobHandle, lease: ExecutionLease) -> DurableJobStatus:
        """Reauthorize, stop the complete job boundary, and finalize its workspace.

        Args:
            handle (JobHandle): Authenticated durable job handle.
            lease (ExecutionLease): Fresh cancellation authorization.

        Returns:
            DurableJobStatus: Terminal cancelled state after cleanup.
        """
        record = self._authorize(handle, lease, JobOperation.CANCEL)
        control = self._runtime().control_plane
        stopped = control.stop_container(record.container_name)
        if _failed(stopped):
            _require_success(
                control.kill_container(record.container_name),
                "Durable job cancellation failed.",
            )
        _require_success(control.wait_container(record.container_name), "Durable job reap failed.")
        return self._finalize(record, DurableJobState.CANCELLED)

    def close(self) -> None:
        """Close attachments and release local leases without stopping durable jobs."""
        for session in tuple(self._sessions.values()):
            session.close()
        self._sessions.clear()
        for workspace in tuple(self._workspaces.values()):
            workspace.release()
        self._workspaces.clear()
        if self._prepared is not None and self._owns_prepared:
            self._prepared.close()
        self._prepared = None

    def has_live_jobs(self) -> bool:
        """Return whether a persisted job requires the managed VM to remain running.

        Returns:
            bool: ``True`` when a running or paused durable job remains recoverable.
        """
        return any(
            record.state in {DurableJobState.RUNNING, DurableJobState.PAUSED}
            for record in self._store.list()
        )

    def _authorize(
        self,
        handle: JobHandle,
        lease: ExecutionLease,
        operation: JobOperation,
    ) -> JobRecord:
        """Authenticate the handle and validate one fresh least-authority lease."""
        record = self._store.load(handle.job_id)
        if not hmac.compare_digest(record.token_digest, _token_digest(handle.token)):
            raise PermissionError("Durable job handle is invalid.")
        required = (
            Capability.PROCESS_SPAWN
            if operation is JobOperation.STATUS
            else Capability.PROCESS_SIGNAL
        )
        if (
            lease.workspace_id != record.request.lease.workspace_id
            or lease.agent_run_id != record.request.lease.agent_run_id
            or lease.runtime_digest != record.request.lease.runtime_digest
            or lease.expires_at_ns <= self._clock_ns()
            or required not in lease.capabilities
            or not self._authorize_operation(lease, operation, record.job_id)
        ):
            raise PermissionError("Durable job operation is not authorized.")
        if record.state in _TERMINAL_STATES:
            return record
        self._activate(record)
        return record

    def _activate(self, record: JobRecord) -> MacosDurableWorkspaceLease:
        """Reacquire and attest one nonterminal persisted job after restart."""
        if _runtime_epoch(self._runtime()) != record.runtime_epoch:
            raise RuntimeError("Durable job runtime epoch changed.")
        if record.job_id not in self._workspaces:
            self._workspaces[record.job_id] = self._workspace.resume_job(
                record.request.lease.workspace_id,
                record.request.lease.agent_run_id,
                record.generation_reference,
                record.layer.restore(),
            )
        self._require_container(record, self._runtime(), expected_running=None)
        return self._workspaces[record.job_id]

    def _refresh(self, record: JobRecord) -> DurableJobStatus:
        """Inspect current task state and finalize only an unambiguously exited job."""
        if record.state in _TERMINAL_STATES:
            return _status(record)
        value = self._inspect(record)
        state = value.get("State")
        if not isinstance(state, dict):
            raise TypeError("Durable job state evidence is invalid.")
        if state.get("Paused") is True:
            return DurableJobStatus(job_id=record.job_id, state=DurableJobState.PAUSED)
        if state.get("Running") is True:
            return DurableJobStatus(job_id=record.job_id, state=DurableJobState.RUNNING)
        if state.get("Status") != "exited" or not isinstance(state.get("ExitCode"), int):
            raise RuntimeError("Durable job terminal state is ambiguous.")
        return self._finalize(record, DurableJobState.COMPLETED, state["ExitCode"])

    def _finalize(
        self,
        record: JobRecord,
        terminal: DurableJobState,
        exit_code: int | None = None,
    ) -> DurableJobStatus:
        """Inspect and optionally publish only after the complete task is terminal."""
        workspace = self._workspaces[record.job_id]
        final = record.model_copy(update={"state": DurableJobState.FAILED})
        try:
            delta = CanonicalDelta(entries=())
            if Capability.WORKSPACE_WRITE in record.request.lease.capabilities:
                delta = self._workspace.observe(workspace.leased)
            workspace.release()
            if delta.entries:
                if self._authorize_publication(record.request, delta):
                    self._workspace.publish(workspace.leased, record.job_id, delta)
                else:
                    self._workspace.deny(workspace.leased)
                    terminal = DurableJobState.DENIED
            elif Capability.WORKSPACE_WRITE in record.request.lease.capabilities:
                self._workspace.deny(workspace.leased)
            final = record.model_copy(update={"state": terminal, "exit_code": exit_code})
        except Exception:  # noqa: BLE001 - terminal publication failures stay closed.
            try:
                self._workspace.deny(workspace.leased)
            except (OSError, RuntimeError, ValueError):
                ...
            final = record.model_copy(update={"state": DurableJobState.FAILED})
        finally:
            session = self._sessions.pop(record.job_id, None)
            if session is not None:
                session.close()
            workspace.discard()
            self._workspaces.pop(record.job_id, None)
            cleanup = self._runtime().control_plane.remove_container(record.container_name)
            if _failed(cleanup):
                final = record.model_copy(update={"state": DurableJobState.FAILED})
            self._store.save(final)
        return _status(final)

    def _lose(self, record: JobRecord) -> JobRecord:
        """Kill ambiguous runtime state, discard its delta, and retain a lost tombstone."""
        prepared = self._runtime()
        try:
            prepared.control_plane.kill_container(record.container_name)
            prepared.control_plane.wait_container(record.container_name)
            prepared.control_plane.remove_container(record.container_name)
        finally:
            workspace = self._workspaces.pop(record.job_id, None)
            if workspace is not None:
                self._workspace.deny(workspace.leased)
                workspace.discard()
            else:
                try:
                    self._workspace.materializer.discard_attempt(record.layer.restore())
                except (OSError, RuntimeError, ValueError):
                    # Ambiguous recovery is already terminal; cleanup remains best effort.
                    ...
        lost = record.model_copy(update={"state": DurableJobState.LOST})
        self._store.save(lost)
        return lost

    def _abandon(self, record: JobRecord, prepared: PreparedMacosRuntime) -> None:
        """Reclaim every partial start boundary before surfacing its failure."""
        try:
            prepared.control_plane.remove_container(record.container_name)
        finally:
            self._workspaces.pop(record.job_id).discard()
            self._store.delete(record.job_id)

    def _inspect(self, record: JobRecord) -> dict[str, object]:
        """Return structured container evidence after fixed identity checks."""
        result = self._runtime().control_plane.inspect_container(record.container_name)
        _require_success(result, "Durable job inspection failed.")
        try:
            value = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Durable job inspection evidence is invalid.") from error
        if not isinstance(value, dict):
            raise TypeError("Durable job inspection evidence is invalid.")
        labels = value.get("Config", {}).get("Labels")
        expected = {
            "io.loop.agent": record.request.lease.agent_run_id,
            "io.loop.epoch": record.lease_epoch,
            "io.loop.generation": record.generation_reference,
            "io.loop.job": record.job_id,
            "io.loop.lease": record.request.lease.lease_id,
            "io.loop.request": record.request.request_id,
            "io.loop.runtime": record.request.lease.runtime_digest,
            "io.loop.runtime-epoch": record.runtime_epoch,
            "io.loop.workspace": record.request.lease.workspace_id,
        }
        if (
            not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in expected.items())
            or any(key.startswith("io.loop.") and key not in expected for key in labels)
        ):
            raise RuntimeError("Durable job ownership evidence is invalid.")
        return value

    def _require_container(
        self,
        record: JobRecord,
        prepared: PreparedMacosRuntime,
        expected_running: bool | None,
    ) -> None:
        """Require exact labels and optional running state from the attested runtime."""
        del prepared
        value = self._inspect(record)
        state = value.get("State")
        if not isinstance(state, dict) or (
            expected_running is not None and state.get("Running") is not expected_running
        ):
            raise RuntimeError("Durable job runtime state is invalid.")

    def _oci(self, prepared: PreparedMacosRuntime, runtime_digest: str) -> OciSandboxAdapter:
        """Build the closed OCI compiler against the attested warm runtime."""
        image = prepared.image_for_digest(runtime_digest)
        return OciSandboxAdapter(
            OciSpecCompiler(image, self._limits),
            prepared.endpoint.namespace,
            prepared.endpoint.platform.os,
            prepared.endpoint.platform.architecture,
        )

    def _runtime(self) -> PreparedMacosRuntime:
        """Prepare once and retain the runtime lease while jobs are locally active."""
        if self._prepared is None:
            self._prepared = self._backend.prepare()
            self._owns_prepared = True
        return self._prepared

    def _session(self, job_id: str) -> OciControlSession:
        """Return the sole active attachment or fail without creating one implicitly."""
        try:
            return self._sessions[job_id]
        except KeyError as error:
            raise RuntimeError("Durable job has no active attachment.") from error


_TERMINAL_STATES = frozenset(
    {
        DurableJobState.COMPLETED,
        DurableJobState.CANCELLED,
        DurableJobState.DENIED,
        DurableJobState.LOST,
        DurableJobState.FAILED,
    }
)


def _token_digest(token: str) -> str:
    """Hash one handle authenticator before durable storage."""
    return sha256_digest(token.encode("ascii"))


def _runtime_epoch(prepared: PreparedMacosRuntime) -> str:
    """Derive an opaque boot identity from the attested rootless runtime endpoint."""
    return sha256_digest(
        f"{prepared.endpoint.socket_path}:{prepared.endpoint.network_namespace_identity}".encode()
    )


def _status(record: JobRecord) -> DurableJobStatus:
    """Project private durable state into its sanitized public result."""
    return DurableJobStatus(
        job_id=record.job_id,
        state=record.state,
        exit_code=record.exit_code,
    )


def _failed(result) -> bool:
    """Return whether fixed-shape infrastructure evidence is unsuccessful."""
    return bool(result.exit_code or result.stdout_truncated or result.stderr_truncated)


def _require_success(result, message: str) -> None:
    """Require one bounded management operation to succeed without truncation."""
    if _failed(result):
        raise RuntimeError(message)
