"""Test the public sandbox-only command tool result boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import ToolContext
from loop.execution.contracts import ExecutionLease, JobHandle, JobOperation, TerminalMode
from loop.execution.host.models import (
    HostCompleted,
    HostInfrastructureFailure,
    HostPermissionDenialReason,
    HostPermissionDenied,
)
from loop.execution.results import (
    CapabilityDenied,
    Completed,
    InfrastructureFailure,
    TimedOut,
)
from loop.instructions import InstructionsManager, RuntimeEnvironment
from loop.tools import create_default_tool_registry
from loop.tools.system import run_command


class _Executor:
    """Return a configured closed sandbox result and retain virtual inputs."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls = []
        self.cancellations = []

    def execute(
        self,
        script,
        cwd,
        timeout,
        *,
        request_id=None,
        terminal=TerminalMode.PIPE,
        terminal_columns=None,
        terminal_rows=None,
        cancellation=lambda: False,
        **effects,
    ):
        """Record the opaque shell request and return the configured result."""
        call = (script, cwd, timeout, request_id, terminal, terminal_columns, terminal_rows)
        self.calls.append((*call, effects) if effects else call)
        self.cancellations.append(cancellation)
        return self.result


class _HostBroker:
    """Record explicit host requests and return one configured host result."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def execute(self, request, cancellation):
        """Record the disjoint host request and cancellation predicate."""
        self.calls.append((request, cancellation))
        return self.result


class _DurableExecutor(_Executor):
    """Expose a fake authenticated durable lifecycle through the command service."""

    def __init__(self) -> None:
        super().__init__(Completed(request_id="foreground", exit_code=0))
        self.job_calls = []

    def start_job(self, script, cwd, timeout, **options):
        """Return one fixed authenticated handle and record the durable start."""
        self.job_calls.append(("start", script, cwd, timeout, options))
        return JobHandle(job_id="job", token="a" * 64)

    def job_manager(self):
        """Return this fake as the lifecycle facade."""
        return self

    def job_lease(self, operation, timeout):
        """Return one fresh-shaped operation lease."""
        self.job_calls.append(("lease", operation, timeout))
        return ExecutionLease(
            lease_id="lease",
            workspace_id="workspace",
            agent_run_id="agent",
            policy_version="2",
            runtime_digest="sha256:runtime",
            expires_at_ns=1,
        )

    def status(self, handle, lease):
        """Return running state without any workspace delta."""
        self.job_calls.append(("status", handle, lease))
        return SimpleNamespace(model_dump=lambda **_: {"job_id": "job", "state": "running"})

    def attach(self, handle, lease, *, deadline_seconds):
        """Record an authenticated attachment."""
        self.job_calls.append(("attach", handle, lease, deadline_seconds))

    def read(self, handle, lease, timeout):
        """Return one bounded PTY frame."""
        self.job_calls.append(("read", handle, lease, timeout))
        return SimpleNamespace(stream="pty", data=b"ready")

    def write(self, handle, lease, data):
        """Record bounded durable input."""
        self.job_calls.append(("write", handle, lease, data))

    def close_stdin(self, handle, lease):
        """Record explicit EOF."""
        self.job_calls.append(("eof", handle, lease))

    def resize(self, handle, lease, columns, rows):
        """Record a bounded PTY resize."""
        self.job_calls.append(("resize", handle, lease, columns, rows))

    def signal(self, handle, lease, signal):
        """Record one reviewed signal."""
        self.job_calls.append(("signal", handle, lease, signal))

    def detach(self, handle, lease):
        """Record management detach without stopping the job."""
        self.job_calls.append(("detach", handle, lease))

    def cancel(self, handle, lease):
        """Return a terminal cancellation status."""
        self.job_calls.append(("cancel", handle, lease))
        return SimpleNamespace(model_dump=lambda **_: {"job_id": "job", "state": "cancelled"})


def _call(executor, command="printf ok", cwd="/workspace", **kwargs):
    """Dispatch one real registered command and decode its result envelope."""
    registry = create_default_tool_registry(command_executor=executor)
    return json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": command, "cwd": cwd}),
            **kwargs,
        )
    )


def test_command_preserves_natural_shell_and_virtual_cwd_without_host_rewriting(tmp_path):
    """Forward POSIX syntax and virtual paths unchanged to the sandbox command service."""
    executor = _Executor(Completed(request_id="request", exit_code=0, stdout=b"ok\xff"))
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "scratch")
    )
    invalidate = Mock(wraps=instructions.invalidate)
    instructions.invalidate = invalidate

    payload = _call(
        executor,
        "printf x | sed s/x/y/ > /tmp/output && pwd",
        "/workspace/src",
        call_id="request",
        instructions_manager=instructions,
    )

    assert payload["result"]["stdout"]["content"] == "ok�"
    assert executor.calls == [
        (
            "printf x | sed s/x/y/ > /tmp/output && pwd",
            "/workspace/src",
            30.0,
            "request",
            TerminalMode.PIPE,
            None,
            None,
        )
    ]
    invalidate.assert_called_once_with(None)


def test_command_requests_a_bounded_merged_pty() -> None:
    """Forward explicit terminal dimensions only when PTY mode is selected."""
    executor = _Executor(Completed(request_id="request", exit_code=0))
    registry = create_default_tool_registry(command_executor=executor)

    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps(
                {
                    "command": "stty size; read value",
                    "pty": True,
                    "terminal_columns": 100,
                    "terminal_rows": 40,
                    "stdin": "value\n",
                }
            ),
        )
    )

    assert "result" in payload
    assert executor.calls[0][4:7] == (TerminalMode.PTY, 100, 40)
    assert executor.calls[0][-1]["stdin"] == b"value\n"


def test_command_schema_has_no_image_profile_selector() -> None:
    """Every language tool uses the same image without a caller-selected target."""
    executor = _Executor(Completed(request_id="request", exit_code=0))
    registry = create_default_tool_registry(command_executor=executor)

    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps({"command": "python --version", "profile": "python"}),
        )
    )

    assert payload["ok"] is False
    assert payload["problem"]["code"] == "tool.invalid_arguments"
    assert executor.calls == []


def test_command_forwards_registry_cancellation_to_the_sandbox() -> None:
    """Blocking command execution receives the application-owned cancellation predicate."""
    executor = _Executor(Completed(request_id="request", exit_code=0))
    cancellation = Mock(return_value=False)
    registry = create_default_tool_registry(
        command_executor=executor,
        cancellation=cancellation,
    )

    payload = json.loads(registry.call("run_command", json.dumps({"command": "true"})))

    assert payload["ok"] is True
    assert executor.cancellations == [cancellation]


def test_command_accepts_only_explicit_typed_effect_leases() -> None:
    """Broker capabilities enter the executor only through structured public fields."""
    executor = _Executor(Completed(request_id="request", exit_code=0))
    registry = create_default_tool_registry(command_executor=executor)
    payload = json.loads(
        registry.call(
            "run_command",
            json.dumps(
                {
                    "command": "wget https://api.example/",
                    "network_connections": [
                        {
                            "hostname": "api.example",
                            "port": 443,
                            "protocol": "https",
                            "addresses": ["93.184.216.34"],
                        }
                    ],
                    "secret_exposures": [
                        {
                            "secret_id": "token",
                            "audience": "api.example",
                            "mechanism": "request_header",
                            "target": "Authorization",
                        }
                    ],
                }
            ),
        )
    )
    assert payload["ok"] is True
    effects = executor.calls[0][-1]
    assert effects["network_connections"][0].hostname == "api.example"
    assert effects["secret_exposures"][0].secret_id == "token"

    invalid = run_command(
        ToolContext(None, "run_command", command_executor=executor),
        "true",
        network_connections=({"hostname": "localhost"},),  # type: ignore[arg-type]
    )
    assert invalid.code == "process.invalid_effect_lease"


@pytest.mark.parametrize("cwd", ["workspace", "../host", "/workspace/../host", "//host"])
def test_command_rejects_nonabsolute_or_unnormalized_virtual_cwd(cwd):
    """Reject cwd values that could confuse the guest virtual namespace."""
    executor = _Executor(Completed(request_id="request", exit_code=0))

    payload = _call(executor, cwd=cwd)

    assert payload["problem"]["code"] == "process.invalid_virtual_cwd"
    assert not executor.calls


def test_command_fails_closed_when_sandbox_service_is_unavailable():
    """Never substitute host execution for an absent sandbox composition."""
    problem = run_command(ToolContext(None, "run_command"), "printf unsafe")

    assert problem.code == "process.sandbox_unavailable"


def test_command_maps_nonzero_timeout_denial_and_infrastructure_failures():
    """Map every representative closed result without exposing privileged diagnostics."""
    nonzero = _call(
        _Executor(
            Completed(request_id="request", exit_code=7, stderr=b"bad", stderr_truncated=True)
        )
    )
    assert nonzero["problem"]["code"] == "process.nonzero_exit"
    assert nonzero["problem"]["metadata"]["stderr"]["truncated"] is True
    assert nonzero["problem"]["metadata"]["stderr"]["discarded_characters"] is None

    timeout = _call(_Executor(TimedOut(request_id="request", stdout=b"partial")))
    assert timeout["problem"]["code"] == "process.timeout"
    assert timeout["problem"]["retryable"] is True

    denied = _call(_Executor(CapabilityDenied(request_id="request", capability="workspace.write")))
    assert denied["problem"]["code"] == "process.capability_denied"

    failed = _call(_Executor(InfrastructureFailure(request_id="request", diagnostic_id="opaque")))
    assert failed["problem"]["code"] == "process.infrastructure_failure"
    assert failed["problem"]["metadata"]["diagnostic_id"] == "opaque"


def test_explicit_host_command_uses_only_the_separate_broker() -> None:
    """Expose host execution through its own request, lease broker, and result namespace."""
    broker = _HostBroker(HostCompleted(request_id="host-call", exit_code=0, stdout=b"host"))
    executor = _Executor(Completed(request_id="sandbox", exit_code=0))
    registry = create_default_tool_registry(
        command_executor=executor,
        host_execution_broker=broker,
    )

    payload = json.loads(
        registry.call(
            "run_host_command",
            json.dumps(
                {
                    "executable": "/usr/bin/true",
                    "arguments": ["--version"],
                    "cwd": "/tmp",
                    "display_cwd": "temporary host directory",
                    "reason": "Requires a macOS-only host facility.",
                    "resource_class": "macOS service",
                }
            ),
            call_id="host-call",
        )
    )

    assert payload["result"]["exit_code"] == 0
    assert payload["result"]["stdout"]["content"] == "host"
    request = broker.calls[0][0]
    assert request.boundary.value == "host"
    assert request.argv == ("/usr/bin/true", "--version")
    assert not executor.calls


def test_explicit_host_command_fails_closed_without_authority() -> None:
    """Reject absent, invalid, and denied host authority without touching sandbox execution."""
    unavailable = json.loads(
        create_default_tool_registry().call(
            "run_host_command", json.dumps({"executable": "/usr/bin/true"})
        )
    )
    assert unavailable["problem"]["code"] == "host.unavailable"

    broker = _HostBroker(
        HostPermissionDenied(
            request_id="host", reason=HostPermissionDenialReason.HOST_PROCESSES_DISABLED
        )
    )
    registry = create_default_tool_registry(host_execution_broker=broker)
    invalid = json.loads(registry.call("run_host_command", json.dumps({"executable": "relative"})))
    assert invalid["problem"]["code"] == "host.invalid_request"
    assert not broker.calls

    denied = json.loads(
        registry.call(
            "run_host_command",
            json.dumps({"executable": "/usr/bin/true"}),
            call_id="host",
        )
    )
    assert denied["problem"]["code"] == "host_permission_denied"
    assert "host processes are disabled" in denied["problem"]["detail"]
    assert "/permissions limit set workspace host-process allow" in denied["problem"]["detail"]


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        (HostPermissionDenialReason.USER_DENIED, "was not approved"),
        (HostPermissionDenialReason.APPROVAL_UNAVAILABLE, "no interactive user is available"),
        (HostPermissionDenialReason.POLICY_DENIED, "current permission policy"),
    ],
)
def test_explicit_host_command_explains_other_safe_denial_categories(
    reason: HostPermissionDenialReason, detail: str
) -> None:
    """Explain rejection, unavailable approval, and policy denial without ambiguity."""
    registry = create_default_tool_registry(
        host_execution_broker=_HostBroker(HostPermissionDenied(request_id="host", reason=reason))
    )

    denied = json.loads(
        registry.call(
            "run_host_command",
            json.dumps({"executable": "/usr/bin/true"}),
            call_id="host",
        )
    )

    assert denied["problem"]["code"] == "host_permission_denied"
    assert detail in denied["problem"]["detail"]


def test_explicit_host_command_reports_non_permission_failure() -> None:
    """Retain the generic diagnostic for sanitized host infrastructure failures."""
    registry = create_default_tool_registry(
        host_execution_broker=_HostBroker(HostInfrastructureFailure(request_id="host"))
    )

    failed = json.loads(
        registry.call(
            "run_host_command",
            json.dumps({"executable": "/usr/bin/true"}),
            call_id="host",
        )
    )

    assert failed["problem"]["code"] == "host_infrastructure_failure"
    assert failed["problem"]["detail"] == (
        "The separately authorized Host operation stopped safely."
    )


def test_public_durable_job_tools_cover_the_authenticated_interactive_lifecycle() -> None:
    """Route start, status, attach, I/O, resize, signal, detach, and cancel through one owner."""
    executor = _DurableExecutor()
    registry = create_default_tool_registry(command_executor=executor)
    started = json.loads(
        registry.call("start_command_job", json.dumps({"command": "read value"}), call_id="job")
    )
    token = started["result"]["token"]
    base = {"job_id": "job", "token": token}

    calls = (
        ("command_job_status", base),
        ("attach_command_job", base),
        ("read_command_job", base),
        ("write_command_job", {**base, "data": "input\n", "eof": True}),
        ("resize_command_job", {**base, "columns": 100, "rows": 40}),
        ("signal_command_job", {**base, "signal": "INT"}),
        ("detach_command_job", base),
        ("cancel_command_job", base),
    )
    for name, arguments in calls:
        assert json.loads(registry.call(name, json.dumps(arguments)))["ok"] is True

    operations = [entry[1] for entry in executor.job_calls if entry[0] == "lease"]
    assert operations == [
        JobOperation.STATUS,
        JobOperation.ATTACH,
        JobOperation.ATTACH,
        JobOperation.STDIN,
        JobOperation.RESIZE,
        JobOperation.SIGNAL,
        JobOperation.ATTACH,
        JobOperation.CANCEL,
    ]
    assert any(entry[0] == "eof" for entry in executor.job_calls)


def test_public_durable_job_tools_sanitize_every_failure_branch() -> None:
    """Map absent composition, denied starts, empty reads, and forged handles safely."""
    unavailable = create_default_tool_registry()
    start = json.loads(unavailable.call("start_command_job", json.dumps({"command": "true"})))
    assert start["problem"]["code"] == "process.sandbox_unavailable"
    status = json.loads(
        unavailable.call(
            "command_job_status",
            json.dumps({"job_id": "job", "token": "a" * 64}),
        )
    )
    assert status["problem"]["code"] == "process.durable_job_failed"

    executor = _DurableExecutor()
    executor.start_job = Mock(
        return_value=CapabilityDenied(request_id="job", capability="durable_job.start")
    )
    registry = create_default_tool_registry(command_executor=executor)
    denied = json.loads(
        registry.call("start_command_job", json.dumps({"command": "true"}), call_id="job")
    )
    assert denied["problem"]["code"] == "process.capability_denied"
    executor.start_job = Mock(side_effect=RuntimeError("private"))
    failed = json.loads(
        registry.call("start_command_job", json.dumps({"command": "true"}), call_id="job")
    )
    assert failed["problem"]["code"] == "process.durable_job_failed"

    executor.start_job = _DurableExecutor().start_job
    executor.read = Mock(return_value=None)
    empty = json.loads(
        registry.call(
            "read_command_job",
            json.dumps({"job_id": "job", "token": "a" * 64}),
        )
    )
    assert empty["result"]["frame"] is None
    no_input = json.loads(
        registry.call(
            "write_command_job",
            json.dumps({"job_id": "job", "token": "a" * 64}),
        )
    )
    assert no_input["result"] == {"accepted_bytes": 0, "eof": False}

    forged = {"job_id": "job", "token": "bad"}
    failing_calls = (
        ("command_job_status", forged),
        ("attach_command_job", forged),
        ("read_command_job", forged),
        ("write_command_job", forged),
        ("resize_command_job", {**forged, "columns": 80, "rows": 24}),
        ("signal_command_job", {**forged, "signal": "TERM"}),
        ("detach_command_job", forged),
        ("cancel_command_job", forged),
    )
    for name, arguments in failing_calls:
        result = json.loads(registry.call(name, json.dumps(arguments)))
        assert result["problem"]["code"] == "process.durable_job_failed"


def test_explicit_host_command_maps_nonzero_exit() -> None:
    """Keep nonzero Host outcomes distinct from sandbox program failures."""
    broker = _HostBroker(HostCompleted(request_id="host", exit_code=7, stderr=b"failed"))
    payload = json.loads(
        create_default_tool_registry(host_execution_broker=broker).call(
            "run_host_command",
            json.dumps({"executable": "/usr/bin/false"}),
            call_id="host",
        )
    )
    assert payload["problem"]["code"] == "host.nonzero_exit"
