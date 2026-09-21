"""Supervise separately leased host processes with explicitly limited guarantees."""

from __future__ import annotations

import time
from collections.abc import Callable

from ...telemetry import telemetry_audit
from ..infrastructure import (
    InfrastructureProcessCommand,
    InfrastructureProcessRunner,
    ProcessCancelledError,
    ProcessSpawnError,
    ProcessTimedOutError,
)
from .models import (
    AuthorizedHostExecution,
    HostAuditEvent,
    HostAuditEventType,
    HostAuditSink,
    HostCancelled,
    HostCompleted,
    HostExecutionResult,
    HostInfrastructureFailure,
    HostIntegrityFailure,
    HostSpawnFailure,
    HostTimedOut,
    host_request_fingerprint,
)

_FORBIDDEN_ENVIRONMENT = {
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GLOBIGNORE",
    "HOME",
    "IFS",
    "LD_PRELOAD",
    "NODE_OPTIONS",
    "PATH",
    "PERL5OPT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "RUBYOPT",
    "TMPDIR",
}


class TelemetryHostAudit:
    """Emit minimized explicit-host records through the process-wide audit facade."""

    def record(self, event: HostAuditEvent) -> None:
        """Emit one sanitized event without argv, environment values, or host paths.

        Args:
            event (HostAuditEvent): Sanitized host security event.
        """
        telemetry_audit(event.type.value, **event.model_dump(exclude={"type"}, mode="json"))


class HostExecutionSupervisor:
    """Run only exact lease-bound host commands with best-effort process-group cleanup.

    Host execution is not sandbox containment. A process may delegate work or escape ordinary
    process-group tracking after the user has approved native host authority.

    Args:
        audit (HostAuditSink | None): Sanitized security audit destination.
        clock_ns (Callable[[], int]): Trusted wall-clock source for lease validation.
        runner_factory (Callable[[int], InfrastructureProcessRunner]): Classified process runner
            factory, injectable for isolated tests.
    """

    audit: HostAuditSink
    _clock_ns: Callable[[], int]
    _runner_factory: Callable[[int], InfrastructureProcessRunner]

    def __init__(
        self,
        audit: HostAuditSink | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
        runner_factory: Callable[[int], InfrastructureProcessRunner] = InfrastructureProcessRunner,
    ) -> None:
        self.audit = audit if audit is not None else TelemetryHostAudit()
        self._clock_ns = clock_ns
        self._runner_factory = runner_factory

    def execute(
        self,
        authorized: AuthorizedHostExecution,
        cancellation: Callable[[], bool] = lambda: False,
    ) -> HostExecutionResult:
        """Execute one exact host lease and return a closed host-only result.

        Args:
            authorized (AuthorizedHostExecution): Request plus verified executable, cwd, and lease.
            cancellation (Callable[[], bool]): Predicate requesting best-effort tree termination.

        Returns:
            HostExecutionResult: Closed result that cannot be used as a sandbox outcome.
        """
        request = authorized.request
        lease = authorized.lease
        if (
            lease.request_id != request.request_id
            or lease.request_fingerprint != host_request_fingerprint(request)
            or lease.executable_sha256 != authorized.executable.sha256
            or lease.expires_at_ns <= self._clock_ns()
            or authorized.executable.path.as_posix() != request.executable
            or authorized.cwd.path.as_posix() != request.cwd
        ):
            return self._terminal(
                HostIntegrityFailure(request_id=request.request_id), lease.lease_id
            )
        try:
            environment = _host_environment(request.environment, authorized.executable.path.parent)
            command = InfrastructureProcessCommand(
                argv=request.argv,
                environment=environment,
                cwd=authorized.cwd.path,
                operation="host.execute",
                executable=authorized.executable,
                cwd_identity=authorized.cwd,
            )
            result = self._runner_factory(request.output_limit_bytes).run(
                command,
                deadline_seconds=request.deadline_seconds,
                cancellation=cancellation,
                started=lambda: self.audit.record(
                    HostAuditEvent(
                        type=HostAuditEventType.STARTED,
                        request_id=request.request_id,
                        lease_id=lease.lease_id,
                        executable_sha256=authorized.executable.sha256,
                    )
                ),
                cleanup_descendants=True,
            )
        except ValueError:
            outcome: HostExecutionResult = HostIntegrityFailure(request_id=request.request_id)
        except ProcessSpawnError:
            outcome = HostSpawnFailure(request_id=request.request_id)
        except ProcessTimedOutError:
            outcome = HostTimedOut(request_id=request.request_id)
        except ProcessCancelledError:
            outcome = HostCancelled(request_id=request.request_id)
        except RuntimeError:
            outcome = HostInfrastructureFailure(request_id=request.request_id)
        else:
            outcome = HostCompleted(
                request_id=request.request_id,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                stdout_truncated=result.stdout_truncated,
                stderr_truncated=result.stderr_truncated,
            )
        return self._terminal(outcome, lease.lease_id)

    def _terminal(self, result: HostExecutionResult, lease_id: str) -> HostExecutionResult:
        """Audit and return one host terminal outcome."""
        try:
            self.audit.record(
                HostAuditEvent(
                    type=HostAuditEventType.TERMINAL,
                    request_id=result.request_id,
                    lease_id=lease_id,
                    outcome=result.kind,
                )
            )
        except Exception:  # noqa: BLE001 - audit failure must stay a closed host result.
            return HostInfrastructureFailure(request_id=result.request_id)
        return result


def _host_environment(entries: tuple[tuple[str, str], ...], executable_dir) -> dict[str, str]:
    """Build a scrubbed host environment from fixed baseline and disclosed additions."""
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": f"{executable_dir}:/usr/bin:/bin",
    }
    for key, value in entries:
        if key in _FORBIDDEN_ENVIRONMENT or key.startswith(("DYLD_", "LD_")):
            raise ValueError("Host environment requests a process-control variable.")
        environment[key] = value
    return environment
