"""Execute foreground commands through the managed macOS sandbox runtime."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from ...broker import EffectBrokerPlanner, SecretAuthority
from ...contracts import Capability, ExecutionMode, SandboxExecutionRequest, TerminalMode
from ...results import (
    Cancelled,
    CapabilityDenied,
    CommitConflict,
    Completed,
    ExecutionResult,
    InfrastructureFailure,
    TimedOut,
    UnrepresentableDelta,
    UnsupportedCapability,
)
from ...runtime.manifest import RuntimeManifest
from ...service import AttemptObserver
from ...vfs import CanonicalDelta, CommitConflictError, UnrepresentableDeltaError
from ..oci import OciArtifactInstaller, OciSandboxAdapter, OciSpecCompiler
from ..oci.control import OciControlSession
from ..oci.process import OciSessionCancelled, OciSessionTimedOut, OciStreamFrame
from ..oci.spec import OciResourceLimits
from .backend import MacosSandboxBackend, PreparedMacosRuntime
from .broker import MacosBrokerLease, MacosEffectBroker, guest_owner_home
from .candidate import macos_artifact
from .workspace import MacosLeasedWorkspace

_LOGGER = logging.getLogger(__name__)


class UnavailableSecretAuthority:
    """Deny secret resolution when product composition supplied no authority."""

    def resolve(self, secret_id: str, audience: str) -> bytes:
        """Reject every lookup without inspecting or persisting its identifiers."""
        raise KeyError((secret_id, audience))


class MacosWorkspaceProvider(Protocol):
    """Own macOS attempt overlays and post-execution workspace transactions."""

    def lease(
        self, workspace_id: str, agent_run_id: str, attempt_id: str
    ) -> AbstractContextManager[MacosLeasedWorkspace]:
        """Return a context manager yielding one fresh leased attempt.

        Args:
            workspace_id (str): Authenticated workspace identity.
            agent_run_id (str): Owning agent execution identity.
            attempt_id (str): Unique attempt identity.

        Returns:
            AbstractContextManager[MacosLeasedWorkspace]: Fresh attempt over a leased generation.
        """

    def observe(self, leased: MacosLeasedWorkspace) -> CanonicalDelta:
        """Return the complete representable delta of one stopped attempt."""

    def publish(
        self,
        leased: MacosLeasedWorkspace,
        transaction_id: str,
        delta: CanonicalDelta,
    ) -> object:
        """Publish one approved delta after its attempt lease ends."""

    def deny(self, leased: MacosLeasedWorkspace) -> None:
        """Reclaim staged effects after denial or failed execution."""


class WorkspacePublicationAuthorizer(Protocol):
    """Authorize or deny one complete observed representable workspace delta."""

    def __call__(self, request: SandboxExecutionRequest, delta: CanonicalDelta) -> bool:
        """Return whether the complete observed delta may be published."""


class BoundedOutput:
    """Retain bounded bytes while continuing to drain both attached streams."""

    _limit: int
    _stdout: bytearray
    _stderr: bytearray
    _stdout_truncated: bool
    _stderr_truncated: bool
    _lock: threading.Lock

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._stdout_truncated = False
        self._stderr_truncated = False
        self._lock = threading.Lock()

    def add(self, frame: OciStreamFrame) -> None:
        """Retain one frame up to its stream's byte bound."""
        with self._lock:
            target = self._stdout if frame.stream in {"stdout", "pty"} else self._stderr
            available = max(0, self._limit - len(target))
            target.extend(frame.data[:available])
            if len(frame.data) > available:
                if frame.stream in {"stdout", "pty"}:
                    self._stdout_truncated = True
                else:
                    self._stderr_truncated = True

    @property
    def result_fields(self) -> dict[str, object]:
        """Return immutable result fields for the retained streams.

        Returns:
            dict[str, object]: Bounded stream bytes and truncation flags.
        """
        with self._lock:
            return {
                "stdout": bytes(self._stdout),
                "stderr": bytes(self._stderr),
                "stdout_truncated": self._stdout_truncated,
                "stderr_truncated": self._stderr_truncated,
            }


def _infrastructure_failure(
    request_id: str,
    output: BoundedOutput,
    error: BaseException | None = None,
) -> InfrastructureFailure:
    """Return one opaque correlated failure with only bounded untrusted output."""
    diagnostic_id = f"diag_{uuid4().hex}"
    if error is not None:
        _LOGGER.error(
            "Managed macOS execution failed (%s).",
            diagnostic_id,
            exc_info=error,
        )
    return InfrastructureFailure(
        request_id=request_id,
        diagnostic_id=diagnostic_id,
        **output.result_fields,
    )


class MacosExecutionAdapter:
    """Run one foreground attempt through managed macOS workspace and OCI boundaries.

    Args:
        backend (MacosSandboxBackend): Managed workspace-bound runtime substrate.
        candidate (RuntimeManifest): Trusted manifest containing the pinned core OCI profile.
        workspace (MacosWorkspaceProvider): Attempt overlay and transaction authority.
        limits (OciResourceLimits): Mandatory resource controls for each attempt.
        authorize_publication (WorkspacePublicationAuthorizer): Observed-delta authorization
            callback. It is never called for read-only requests or empty deltas.
        prepared_runtime (PreparedMacosRuntime | None): Existing owned warm runtime lease.
        owns_prepared_runtime (bool): Whether ``close`` releases an injected prepared lease.
        secret_authority (SecretAuthority | None): Host-owned exact secret resolver.
    """

    _prepared: PreparedMacosRuntime | None
    _owns_prepared: bool
    _runtime_lock: threading.Lock
    _envoy_prepared: bool
    _backend: MacosSandboxBackend
    _candidate: RuntimeManifest
    _workspace: MacosWorkspaceProvider
    _limits: OciResourceLimits
    _authorize_publication: WorkspacePublicationAuthorizer
    _secrets: SecretAuthority

    def __init__(
        self,
        backend: MacosSandboxBackend,
        candidate: RuntimeManifest,
        workspace: MacosWorkspaceProvider,
        limits: OciResourceLimits,
        authorize_publication: WorkspacePublicationAuthorizer,
        *,
        prepared_runtime: PreparedMacosRuntime | None = None,
        owns_prepared_runtime: bool = True,
        secret_authority: SecretAuthority | None = None,
    ) -> None:
        self._backend = backend
        self._candidate = candidate
        self._workspace = workspace
        self._limits = limits
        self._authorize_publication = authorize_publication
        self._prepared = prepared_runtime
        self._owns_prepared = owns_prepared_runtime
        self._secrets = secret_authority or UnavailableSecretAuthority()
        self._runtime_lock = threading.Lock()
        self._envoy_prepared = False

    def close(self) -> None:
        """Release the cached runtime lease while leaving the warm VM available."""
        with self._runtime_lock:
            if self._prepared is not None and self._owns_prepared:
                self._prepared.close()
            self._prepared = None
            self._envoy_prepared = False

    def execute(
        self,
        request: SandboxExecutionRequest,
        observer: AttemptObserver,
        cancellation: Callable[[], bool],
    ) -> ExecutionResult:
        """Execute one foreground request and return a closed result.

        Args:
            request (SandboxExecutionRequest): Authorized virtual sandbox request.
            observer (AttemptObserver): Service-owned lifecycle observer.
            cancellation (Callable[[], bool]): Predicate requesting cancellation.

        Returns:
            ExecutionResult: Completed program or typed closed failure.
        """
        if request.mode is not ExecutionMode.FOREGROUND:
            return UnsupportedCapability(request_id=request.request_id, capability="durable_job")
        output = BoundedOutput(request.output_limit_bytes)
        leased: MacosLeasedWorkspace | None = None
        delta: CanonicalDelta | None = None
        result: ExecutionResult = _infrastructure_failure(request.request_id, output)
        try:
            with self._workspace.lease(
                request.lease.workspace_id,
                request.lease.agent_run_id,
                request.request_id,
            ) as current:
                leased = current
                result = self._execute_leased(request, observer, cancellation, current, output)
                if Capability.WORKSPACE_WRITE in request.lease.capabilities and isinstance(
                    result, Completed
                ):
                    delta = self._workspace.observe(current)
        except UnrepresentableDeltaError:
            result = UnrepresentableDelta(request_id=request.request_id, **output.result_fields)
        except CommitConflictError:
            result = CommitConflict(request_id=request.request_id, **output.result_fields)
        except Exception as error:  # noqa: BLE001 - platform failures become closed results.
            result = _infrastructure_failure(request.request_id, output, error)
        if leased is None or Capability.WORKSPACE_WRITE not in request.lease.capabilities:
            return result
        if delta is None or not delta.entries:
            self._workspace.deny(leased)
            return result
        try:
            if not self._authorize_publication(request, delta):
                self._workspace.deny(leased)
                return CapabilityDenied(
                    request_id=request.request_id,
                    capability=Capability.WORKSPACE_WRITE.value,
                    **output.result_fields,
                )
            self._workspace.publish(leased, request.request_id, delta)
        except CommitConflictError:
            return CommitConflict(request_id=request.request_id, **output.result_fields)
        except Exception as error:  # noqa: BLE001 - publication failures remain sandbox failures.
            return _infrastructure_failure(request.request_id, output, error)
        return result

    def _execute_leased(
        self,
        request: SandboxExecutionRequest,
        observer: AttemptObserver,
        cancellation: Callable[[], bool],
        leased: MacosLeasedWorkspace,
        output: BoundedOutput,
    ) -> ExecutionResult:
        """Own runtime preparation through container cleanup under one generation lease."""
        prepared = self._runtime()
        prepared.renew(request.deadline_seconds + 60.0)
        session: OciControlSession | None = None
        broker_lease: MacosBrokerLease | None = None
        container_name: str | None = None
        result: ExecutionResult = _infrastructure_failure(request.request_id, output)
        cleanup_failed = False
        hosts_hardened = False
        try:
            image = prepared.image_for_digest(request.lease.runtime_digest)
            compiler = OciSpecCompiler(image, self._limits)
            oci = OciSandboxAdapter(
                compiler,
                prepared.endpoint.namespace,
                prepared.endpoint.platform.os,
                prepared.endpoint.platform.architecture,
            )
            attempt = oci.prepare(request, leased.generation_reference)
            container_name = attempt.spec.container_name
            observer.backend_attested()
            bindings = None
            if request.network_connections or request.network_listeners or request.secret_exposures:
                envoy = macos_artifact(self._candidate, "envoy")
                if request.network_connections or request.network_listeners:
                    self._prepare_envoy(prepared, envoy)
                plan = EffectBrokerPlanner().plan(request, envoy, self._secrets)
                broker = MacosEffectBroker(
                    prepared.control_plane,
                    prepared.endpoint.owner_uid,
                    guest_owner_home(prepared.endpoint.state_path),
                )
                broker_lease = broker.prepare(
                    plan,
                    frozenset(lease.protocol for lease in request.network_connections),
                )
                bindings = broker_lease.bindings
                if plan.network_name is not None:
                    broker.activate(
                        plan,
                        broker_lease,
                        envoy.source,
                        request.network_listeners,
                    )
            if bindings is None:
                session = prepared.control_plane.run_attempt(
                    attempt.spec,
                    leased.layer.mount_source,
                    deadline_seconds=request.deadline_seconds,
                )
            else:
                created = prepared.control_plane.create_attempt(
                    attempt.spec,
                    leased.layer.mount_source,
                    bindings=bindings,
                    deadline_seconds=request.deadline_seconds,
                )
                if created.exit_code or created.stdout_truncated or created.stderr_truncated:
                    _LOGGER.error(
                        "Managed command container creation failed: "
                        "exit=%s stdout=%r stderr=%r truncated=(%s,%s).",
                        created.exit_code,
                        created.stdout,
                        created.stderr,
                        created.stdout_truncated,
                        created.stderr_truncated,
                    )
                    raise RuntimeError("Managed command container creation failed.")
                hosts_hardened = bool(bindings.host_aliases)
                session = prepared.control_plane.start_attempt(
                    attempt.spec,
                    deadline_seconds=request.deadline_seconds,
                )
                if bindings.host_aliases:
                    if not self._await_running(prepared.control_plane, container_name):
                        raise RuntimeError("Managed command start gate was not reached.")
                    hardened_hosts = prepared.control_plane.harden_container_hosts(container_name)
                    if (
                        hardened_hosts.exit_code
                        or hardened_hosts.stdout_truncated
                        or hardened_hosts.stderr_truncated
                        or hardened_hosts.stdout != b"444\n"
                    ):
                        raise RuntimeError("Managed command hosts mapping is not read-only.")
                    if bindings.start_gate is None:
                        raise RuntimeError("Managed command start gate is unavailable.")
                    opened_gate = prepared.control_plane.open_container_start_gate(
                        bindings.start_gate
                    )
                    if (
                        opened_gate.exit_code
                        or opened_gate.stdout_truncated
                        or opened_gate.stderr_truncated
                    ):
                        raise RuntimeError("Managed command start gate could not be released.")
            observer.process_started()
            if request.terminal is TerminalMode.PTY:
                assert request.terminal_columns is not None
                assert request.terminal_rows is not None
                session.resize(request.terminal_columns, request.terminal_rows)
            _transport_exit_code, terminal = self._capture(
                session,
                cancellation,
                output,
                request.request_id,
                request.terminal,
                request.terminal_columns,
                request.terminal_rows,
                request.stdin,
            )
            if terminal is not None:
                if self._terminate_attempt(prepared.control_plane, container_name):
                    result = terminal
                else:
                    result = _infrastructure_failure(request.request_id, output)
            elif (
                attested_exit := self._attest_stopped(
                    prepared.control_plane,
                    container_name,
                    _transport_exit_code,
                )
            ) is None:
                result = _infrastructure_failure(request.request_id, output)
            else:
                result = Completed(
                    request_id=request.request_id,
                    exit_code=attested_exit,
                    **output.result_fields,
                )
        except Exception as error:  # noqa: BLE001 - platform failures become closed results.
            result = _infrastructure_failure(request.request_id, output, error)
        finally:
            if session is not None:
                session.close()
            if container_name is not None:
                if hosts_hardened:
                    try:
                        released_hosts = prepared.control_plane.release_container_hosts(
                            container_name
                        )
                        if (
                            released_hosts.exit_code
                            or released_hosts.stdout_truncated
                            or released_hosts.stderr_truncated
                            or released_hosts.stdout != b"600\n"
                        ):
                            _LOGGER.error(
                                "Managed hosts cleanup returned invalid evidence: "
                                "exit=%s stdout=%r stderr=%r truncated=(%s,%s).",
                                released_hosts.exit_code,
                                released_hosts.stdout,
                                released_hosts.stderr,
                                released_hosts.stdout_truncated,
                                released_hosts.stderr_truncated,
                            )
                            cleanup_failed = True
                    except Exception as error:
                        _LOGGER.error("Managed hosts cleanup failed.", exc_info=error)
                        cleanup_failed = True
                try:
                    cleanup = prepared.control_plane.remove_container(container_name)
                    remove_failed = bool(
                        cleanup.exit_code or cleanup.stdout_truncated or cleanup.stderr_truncated
                    )
                    cleanup_failed |= remove_failed
                    if remove_failed:
                        _LOGGER.error(
                            "Managed container cleanup returned invalid evidence: "
                            "exit=%s stdout=%r stderr=%r truncated=(%s,%s).",
                            cleanup.exit_code,
                            cleanup.stdout,
                            cleanup.stderr,
                            cleanup.stdout_truncated,
                            cleanup.stderr_truncated,
                        )
                except Exception as error:
                    _LOGGER.error("Managed container cleanup failed.", exc_info=error)
                    cleanup_failed = True
            if broker_lease is not None:
                try:
                    broker_lease.close()
                except Exception as error:
                    _LOGGER.error("Managed effect-broker cleanup failed.", exc_info=error)
                    cleanup_failed = True
        if cleanup_failed:
            return _infrastructure_failure(request.request_id, output)
        return result

    @staticmethod
    def _await_running(control_plane, container_name: str) -> bool:
        """Wait briefly for the fixed start-gate wrapper to enter running state."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state_result = control_plane.inspect_container_state(container_name)
            if not (
                state_result.exit_code
                or state_result.stdout_truncated
                or state_result.stderr_truncated
            ):
                try:
                    state = json.loads(state_result.stdout)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    state = None
                if isinstance(state, dict) and state.get("Running") is True:
                    return True
            time.sleep(0.01)
        return False

    def _prepare_envoy(self, prepared: PreparedMacosRuntime, envoy) -> None:
        """Lazily pull and attest the pinned broker image once per runtime lease."""
        with self._runtime_lock:
            if self._envoy_prepared:
                return
            with tempfile.TemporaryDirectory(
                prefix="envoy-attestation-",
                dir=prepared.snapshot_store.parent,
            ) as directory:
                OciArtifactInstaller(prepared.control_plane, prepared.endpoint).install(
                    envoy,
                    None,
                    Path(directory),
                )
            self._envoy_prepared = True

    @staticmethod
    def _terminate_attempt(control_plane, container_name: str) -> bool:
        """Kill and reap one timed-out or cancelled container task before removal."""
        killed = control_plane.kill_container(container_name)
        if not (killed.exit_code or killed.stdout_truncated or killed.stderr_truncated):
            waited = control_plane.wait_container(container_name)
            if not (waited.exit_code or waited.stdout_truncated or waited.stderr_truncated):
                return True
        # The attached client may have proxied its forced termination before this
        # idempotent control request reached containerd. Accept only fresh attested
        # evidence that the exact attempt is already terminal.
        return (
            MacosExecutionAdapter._attest_stopped(
                control_plane, container_name, expected_exit_code=None
            )
            is not None
        )

    def _runtime(self) -> PreparedMacosRuntime:
        """Prepare once and retain the attested warm runtime lease."""
        with self._runtime_lock:
            if self._prepared is None:
                self._prepared = self._backend.prepare()
                self._owns_prepared = True
            return self._prepared

    @staticmethod
    def _capture(
        session: OciControlSession,
        cancellation: Callable[[], bool],
        output: BoundedOutput,
        request_id: str,
        terminal_mode: TerminalMode,
        terminal_columns: int | None,
        terminal_rows: int | None,
        stdin: bytes = b"",
    ) -> tuple[int, ExecutionResult | None]:
        """Drain output and deliver EOF after bounded PTY attachment readiness."""
        done = threading.Event()
        first_output = threading.Event()
        failures: list[Exception] = []

        def drain() -> None:
            """Drain until terminal wait completes and the queue becomes empty."""
            try:
                while True:
                    frame = session.read(timeout=0.05)
                    if frame is not None:
                        if terminal_mode is TerminalMode.PTY and not first_output.is_set():
                            assert terminal_columns is not None
                            assert terminal_rows is not None
                            session.resize(terminal_columns, terminal_rows)
                        output.add(frame)
                        first_output.set()
                    elif done.is_set():
                        return
            except Exception as error:  # noqa: BLE001 - sanitize reader failures.
                if not done.is_set():
                    failures.append(error)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()

        def close_input() -> None:
            """Close pipes immediately and give nested PTYs a bounded attach grace."""
            if terminal_mode is TerminalMode.PTY:
                first_output.wait(0.25)
            try:
                if stdin:
                    session.write(stdin)
                session.close_stdin()
            except RuntimeError:
                if not done.is_set():
                    failures.append(RuntimeError("Attached input could not close."))

        closer = threading.Thread(target=close_input, daemon=True)
        closer.start()
        terminal: ExecutionResult | None = None
        exit_code = 0
        try:
            exit_code = session.wait(cancellation)
        except OciSessionTimedOut:
            terminal = TimedOut(request_id=request_id)
        except OciSessionCancelled:
            terminal = Cancelled(request_id=request_id)
        finally:
            done.set()
            closer.join(timeout=1)
            reader.join(timeout=2)
        if terminal is not None:
            return exit_code, terminal.model_copy(update=output.result_fields)
        if closer.is_alive() or reader.is_alive() or failures:
            return 0, _infrastructure_failure(request_id, output)
        return exit_code, None

    @staticmethod
    def _attest_stopped(
        control_plane, container_name: str, expected_exit_code: int | None
    ) -> int | None:
        """Return the attested command exit code from terminal containerd state."""
        result = control_plane.inspect_container_state(container_name)
        if result.exit_code or result.stdout_truncated or result.stderr_truncated:
            return None
        try:
            state = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            isinstance(state, dict)
            and state.get("Status") == "exited"
            and state.get("Running") is False
            and isinstance(state.get("ExitCode"), int)
            and 0 <= state["ExitCode"] <= 255
            and (expected_exit_code is None or state["ExitCode"] == expected_exit_code)
        ):
            return state["ExitCode"]
        return None
