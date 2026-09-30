"""Verify separate sandbox and exact recorded host retry authorization."""

import subprocess
import tempfile
import time
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from loop import Action, Decision, Interaction, PermissionConfiguration, PermissionManager
from loop.execution.coordinator import (
    CommandExecutionCoordinator,
    HostCommandRequest,
    LocalHostCommandExecutor,
    _denied_path_in_error_output,
)
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry


def request(tmp_path, **changes):
    """Build a bound sandbox command in the disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": tmp_path,
        "workspace": tmp_path,
        "read_roots": (tmp_path.parent,),
        "write_roots": (tmp_path,),
        "network": False,
        "environment": {"PATH": "/usr/bin:/bin"},
        "policy_version": "macos-v1",
        "deadline": time.monotonic() + 60,
        "workspace_id": "workspace-1",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


def coordinator(tmp_path, outcome, *, answer="approve", auto_approval=False, defaults=None):
    """Build the application boundary with observable, inert launch collaborators."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = answer
    backend = Mock()
    backend.run.return_value = outcome
    host = Mock()
    host.run_host_command.return_value = "host-result"
    configuration = PermissionConfiguration()
    if defaults:
        configuration.defaults.update(defaults)
    manager = PermissionManager(tmp_path, configuration=configuration)
    return (
        CommandExecutionCoordinator(
            backend, host, manager, interaction, auto_approval=auto_approval
        ),
        backend,
        host,
        interaction,
    )


def test_classified_failure_only_offers_host_retry(tmp_path):
    """A classified failure offers a concise host retry without leaking bound details."""
    service, backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=f"nested Seatbelt at {tmp_path}/private",
            possible_effects=True,
        ),
    )
    approved = request(tmp_path)

    attempt = service.run_sandboxed(approved, "run_command")

    backend.run.assert_called_once_with(approved)
    host.run_host_command.assert_not_called()
    assert attempt.offer is not None
    assert attempt.offer.warning == (
        'Command: "printf ok"\n'
        "Reason: The sandbox was unavailable. Native sandbox execution could not start. "
        "Running without the OS sandbox can read "
        "private files, change files outside the workspace, and use the network.\n"
        "The sandboxed attempt may already have had effects. Retrying may repeat those effects."
    )
    assert str(tmp_path) not in attempt.offer.warning
    assert "PATH=" not in attempt.offer.warning
    assert interaction.prompt.call_count == 1


def test_session_host_approval_is_reused_and_revocable(tmp_path):
    """An exact prelaunch host approval works on later attempts until revoked."""
    interaction = Mock(spec=Interaction)
    host_answers = iter(("session", "deny"))

    def answer(prompt, **_kwargs):
        """Approve sandbox prompts and select recorded host decisions."""
        return next(host_answers) if prompt.startswith("Run this exact command") else "approve"

    interaction.prompt.side_effect = answer
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="no profile")
    host = Mock()
    host.run_host_command.return_value = "host-result"
    manager = PermissionManager(tmp_path)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)
    first = request(tmp_path)
    offer = service.run_sandboxed(first, "renamed_command").offer
    assert service.retry_host_command(offer, first) == "host-result"
    third = request(tmp_path)
    offer = service.run_sandboxed(third, "renamed_command").offer
    assert service.retry_host_command(offer, third) == "host-result"
    assert (
        sum(
            call.args[0].startswith("Run this exact command")
            for call in interaction.prompt.call_args_list
        )
        == 1
    )
    rules = manager.host_command_rules()
    assert len(rules) == 1
    assert rules[0].source == "printf ok"
    assert manager.remove_host_command_rule(rules[0].id)
    assert not manager.remove_host_command_rule(rules[0].id)
    fourth = request(tmp_path)
    offer = service.run_sandboxed(fourth, "renamed_command").offer
    assert service.retry_host_command(offer, fourth) is None
    assert host.run_host_command.call_count == 2


def test_denied_initial_permission_starts_nothing_and_offers_nothing(tmp_path):
    """A user rejection cannot be reclassified as a sandbox inability."""
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE), answer="deny"
    )

    attempt = service.run_sandboxed(
        request(tmp_path), "run_command", prelaunch_failure="startup probe failed"
    )

    assert attempt.result is None and attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_called_once()


def test_policy_denial_and_headless_mode_start_nothing(tmp_path):
    """Explicit policy denial and absence of user interaction fail closed."""
    service, backend, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.UNAVAILABLE),
        defaults={Action.PROCESS_EXECUTE: Decision.DENY},
    )
    assert (
        service.run_sandboxed(
            request(tmp_path), "run_command", prelaunch_failure="startup probe failed"
        ).offer
        is None
    )
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_nonzero_exit_timeout_and_cancel_do_not_offer_host(tmp_path):
    """Ordinary command failures are never treated as OS sandbox denial."""
    for result in (
        CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=17),
        CommandProcessResult(SandboxOutcome.TIMED_OUT),
        CommandProcessResult(SandboxOutcome.CANCELLED),
    ):
        service, _, host, _ = coordinator(tmp_path, result)
        attempt = service.run_sandboxed(request(tmp_path), "run_command")
        assert attempt.result == result and attempt.offer is None
        host.run_host_command.assert_not_called()


def test_executable_lookup_exits_do_not_offer_host_for_incidental_denial(tmp_path):
    """Tool lookup failures stay sandboxed despite unrelated trusted OS denials."""
    for exit_code in (72, 126, 127):
        result = CommandProcessResult(
            SandboxOutcome.COMPLETED,
            exit_code=exit_code,
            observed_denial="file-read-data /Library/Preferences/unrelated.plist",
        )
        service, _, host, _ = coordinator(tmp_path, result)
        attempt = service.run_sandboxed(request(tmp_path), "run_command")
        assert attempt.result == result and attempt.offer is None
        host.run_host_command.assert_not_called()


def test_observed_denial_keeps_exit_and_requires_fresh_host_choice(tmp_path):
    """A completed shell's observed denial requires one-time host consent."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "session"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=7,
        stderr="Operation not permitted: /outside/secret",
        observed_denial="file-read-data /outside/secret",
    )
    host = Mock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    manager = PermissionManager(tmp_path)
    approved = request(tmp_path)
    service = CommandExecutionCoordinator(backend, host, manager, interaction)

    attempt = service.run_sandboxed(approved, "run_command")

    assert attempt.result.exit_code == 7
    assert attempt.offer is not None
    assert "may be unrelated" in attempt.offer.warning
    assert "Retrying may repeat those effects" in attempt.offer.warning
    assert service.retry_host_command(attempt.offer, approved) is None
    assert interaction.prompt.call_args.kwargs["choices"] == {
        "deny": "Deny",
        "approve": "Approve this one host run",
    }
    host.run_host_command.assert_not_called()

    interaction.prompt.return_value = "approve"
    second = service.run_sandboxed(request(tmp_path), "run_command")
    assert (
        service.retry_host_command(second.offer, second.offer.request)
        == host.run_host_command.return_value
    )
    host.run_host_command.assert_called_once()


def test_observed_denial_cannot_use_automatic_host_approval(tmp_path):
    """Automatic approval cannot launch an uncertain post-launch host retry."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr="connect to 127.0.0.1:443 failed: Operation not permitted",
        observed_denial="network-outbound remote:*:443",
    )
    service, _, host, interaction = coordinator(tmp_path, result, auto_approval=True)
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    assert offer is not None
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_not_called()
    interaction.prompt.assert_called_once()


@pytest.mark.parametrize(
    ("stderr", "observed_denial"),
    (
        (
            "Permission denied: unrelated operation",
            "network-outbound remote:*:443",
        ),
        (
            "connect failed: Operation not permitted",
            "network-outbound remote:*:443",
        ),
        (
            "connect to 127.0.0.1:444 failed: Operation not permitted",
            "network-outbound remote:*:443",
        ),
        (
            "Operation not permitted: unrelated operation\nconnect to 127.0.0.1:443 failed",
            "network-outbound remote:*:443",
        ),
        (
            "connect to 127.0.0.1:443 failed: Operation not permitted",
            "network-outbound remote:*:*",
        ),
    ),
)
def test_network_denial_requires_correlated_child_failure(tmp_path, stderr, observed_denial):
    """Unrelated or unscoped permission text cannot justify a network host offer."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=stderr,
        observed_denial=observed_denial,
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.reset_mock()
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_unix_socket_denial_matches_the_reported_connection_path(tmp_path):
    """A native denial for a Unix socket matches its traceback connection path."""
    target = tmp_path.parent / "service.sock"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=(
            f'  File "<string>", line 1, in <module>\n    s.connect("{target}")\n'
            "PermissionError: [Errno 1] Operation not permitted\n"
        ),
        observed_denial=f"network-outbound {target}",
    )
    service, _, host, _ = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path), "run_command")

    assert attempt.result == result
    assert attempt.offer is not None
    host.run_host_command.assert_not_called()


def test_unrelated_denial_never_offers_host_execution(tmp_path):
    """A tagged read denial cannot turn an unrelated tool error into a host prompt."""
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=2,
        stdout="error: invalid command option",
        observed_denial="Sandbox: uv(100) deny(1) file-read-data /opt/homebrew/etc/openssl.cnf",
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path, read_roots=()), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_read_only_workspace_write_denial_stays_in_sandbox(tmp_path):
    """A requested write restriction cannot be escaped through host retry."""
    target = tmp_path / ".coverage"
    result = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stderr=f"PermissionError: [Errno 1] Operation not permitted: '{target}'",
        observed_denial=f"Sandbox: Python(100) deny(1) file-write-unlink {target}",
    )
    service, _, host, interaction = coordinator(tmp_path, result)

    attempt = service.run_sandboxed(request(tmp_path, read_roots=(), write_roots=()), "run_command")

    assert attempt.result == result
    assert attempt.offer is None
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_denied_path_matches_relative_case_alias_but_not_unrelated_errors(tmp_path):
    """A genuine path error matches safe relative aliases without trusting unrelated text."""
    target = tmp_path / ".ssh" / "secret"
    target.parent.mkdir()
    target.write_text("marker")
    (tmp_path / "alias").symlink_to(target)
    approved = request(tmp_path, read_roots=())

    assert _denied_path_in_error_output(
        approved, str(target), "noise\ncat: alias: Operation not permitted"
    )
    assert _denied_path_in_error_output(
        approved, str(target), "cat: .SSH/secret: Permission denied"
    )
    assert not _denied_path_in_error_output(
        approved, str(target), "cat: /unrelated: Operation not permitted"
    )
    assert not _denied_path_in_error_output(
        approved,
        str(target),
        f"cat: /unrelated: Operation not permitted\nchecked {target} earlier",
    )


def test_denied_path_match_ignores_unresolvable_alias(tmp_path, monkeypatch):
    """A raced or unresolvable error-path alias cannot justify a host retry."""
    approved = request(tmp_path, read_roots=())
    original_resolve = Path.resolve

    def unavailable(self, *args, **kwargs):
        """Simulate an alias becoming unavailable while checking the child error."""
        if self == tmp_path / "loop":
            raise OSError("changed")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", unavailable)
    assert not _denied_path_in_error_output(
        approved, str(tmp_path / "secret"), "cat: loop: Operation not permitted"
    )


def test_host_retry_decline_and_replay_start_none(tmp_path):
    """A declined offer is consumed and cannot later authorize host execution."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    interaction.prompt.return_value = "deny"

    assert service.retry_host_command(offer, approved) is None
    interaction.prompt.return_value = "approve"
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_not_called()


def test_host_retry_requires_fresh_one_off_approval_and_runs_once(tmp_path):
    """Only the explicit host prompt launches one audited, exact host request."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    assert service.retry_host_command(offer, approved) == "host-result"
    assert "Reason: The sandbox denied this command." in interaction.info.call_args.args[0]
    host.run_host_command.assert_called_once_with(HostCommandRequest.from_sandbox(approved))
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        {"source": "different"},
        {"environment": {"PATH": "/different"}},
        {"workspace_id": "different-workspace"},
    ],
)
def test_changed_host_identity_starts_none(tmp_path, change):
    """A changed source, environment, or workspace identity invalidates the offer."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    service, _, host, _ = coordinator(tmp_path, failure)
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    assert service.retry_host_command(offer, request(tmp_path, **change)) is None
    host.run_host_command.assert_not_called()


def test_fresh_scratch_requires_a_new_host_decision(tmp_path):
    """A pending offer cannot authorize a different TMPDIR."""
    service, _, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial"),
        answer="approve",
    )
    with (
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as first_directory,
        tempfile.TemporaryDirectory(prefix="loop-seatbelt-") as second_directory,
    ):
        first = request(tmp_path, environment={"TMPDIR": str(Path(first_directory).resolve())})
        second = request(tmp_path, environment={"TMPDIR": str(Path(second_directory).resolve())})
        offer = service.run_sandboxed(first, "run_command").offer
        assert service.retry_host_command(offer, first) == "host-result"
        offer = service.run_sandboxed(second, "run_command").offer
        assert service.retry_host_command(offer, first) is None
        offer = service.run_sandboxed(second, "run_command").offer
        assert service.retry_host_command(offer, second) == "host-result"
        assert interaction.prompt.call_count == 5
        assert host.run_host_command.call_count == 2


def test_auto_approval_and_changed_root_start_none(tmp_path):
    """Automatic mode and a replaced workspace cannot approve host launch."""
    failure = CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    approved = request(tmp_path)
    automatic, automatic_backend, automatic_host, _ = coordinator(
        tmp_path, failure, auto_approval=True
    )
    offer = automatic.run_sandboxed(
        approved, "run_command", prelaunch_failure="launcher missing"
    ).offer
    assert automatic.retry_host_command(offer, approved) is None
    automatic_backend.run.assert_not_called()
    automatic_host.run_host_command.assert_not_called()

    service, _, host, interaction = coordinator(tmp_path, failure)
    offer = service.run_sandboxed(
        approved, "run_command", prelaunch_failure="launcher missing"
    ).offer
    tmp_path.rename(tmp_path.with_name("old-workspace"))
    tmp_path.mkdir()
    assert service.retry_host_command(offer, approved) is None
    interaction.prompt.assert_called_once()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("execution_timeout", [None, 30.0])
def test_changed_root_during_sandbox_approval_produces_no_host_offer(tmp_path, execution_timeout):
    """Renewing an execution budget never accepts a root replaced during authorization."""
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path)

    def answer(*_args, **_kwargs):
        """Replace the workspace while the user considers the approval."""
        tmp_path.rename(tmp_path.with_name("old-workspace"))
        tmp_path.mkdir()
        return "approve"

    interaction.prompt.side_effect = answer
    attempt = service.run_sandboxed(approved, "run_command", execution_timeout=execution_timeout)

    assert attempt.result.outcome is SandboxOutcome.STALE
    assert attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


def test_sandbox_approval_wait_does_not_consume_execution_budget(tmp_path, monkeypatch):
    """A long interactive wait leaves a full bounded sandbox budget after approval."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: now[0])
    service, backend, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path, deadline=130.0)

    def answer(*_args, **_kwargs):
        """Return approval after the original execution deadline would have expired."""
        now[0] = 243.0
        return "approve"

    interaction.prompt.side_effect = answer
    attempt = service.run_sandboxed(approved, "run_command", execution_timeout=30.0)
    launched = backend.run.call_args.args[0]
    assert launched.deadline == 273.0
    assert launched.source == approved.source
    assert launched.attempt_id == approved.attempt_id
    assert launched.paths_are_current()
    assert attempt.result.outcome is SandboxOutcome.COMPLETED
    assert attempt.offer is None
    host.run_host_command.assert_not_called()


def test_expired_absolute_budget_never_offers_host_execution(tmp_path, monkeypatch):
    """An already expired execution budget remains a timeout rather than an escalation."""
    service, backend, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    )
    approved = request(tmp_path, deadline=100.0)
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: 101.0)
    attempt = service.run_sandboxed(approved, "run_command")
    assert attempt.result.outcome is SandboxOutcome.TIMED_OUT
    assert attempt.offer is None
    backend.run.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("outcome", [SandboxOutcome.INVALID, SandboxOutcome.STALE])
def test_unsafe_or_stale_native_preparation_never_offers_host_execution(tmp_path, outcome):
    """Native safety validation failures cannot become unrestricted host authority."""
    service, _, host, _ = coordinator(tmp_path, CommandProcessResult(outcome))
    attempt = service.run_sandboxed(request(tmp_path), "run_command")
    assert attempt.result.outcome is outcome
    assert attempt.offer is None
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("execution_timeout", [None, 30.0])
def test_host_retry_rechecks_root_after_approval(tmp_path, execution_timeout):
    """A root replacement during the one-off prompt invalidates the approved host launch."""
    service, _, host, interaction = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer

    def replace_root(*_args, **_kwargs):
        """Simulate a changed workspace before the user's answer returns."""
        tmp_path.rename(tmp_path.with_name("old-workspace"))
        tmp_path.mkdir()
        return "approve"

    interaction.prompt.side_effect = replace_root
    assert service.retry_host_command(offer, approved, execution_timeout=execution_timeout) is None
    host.run_host_command.assert_not_called()


def test_host_retry_uses_fresh_deadline_with_the_same_command_identity(tmp_path):
    """The host retry can receive a fresh deadline without changing reviewed authority."""
    service, _, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.UNAVAILABLE, detail="launcher missing")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    renewed = request(tmp_path, deadline=approved.deadline + 60)

    assert service.retry_host_command(offer, renewed) == "host-result"
    host.run_host_command.assert_called_once_with(HostCommandRequest.from_sandbox(renewed))


def test_host_approval_wait_does_not_consume_execution_budget(tmp_path, monkeypatch):
    """The separately approved host attempt starts its budget after the host decision."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.coordinator.time.monotonic", lambda: now[0])
    service, _, host, interaction = coordinator(
        tmp_path,
        CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail=f"launcher failed at {tmp_path}",
            failure_context="Native sandbox launch check failed.",
        ),
    )
    approved = request(tmp_path, deadline=130.0)
    offer = service.run_sandboxed(approved, "run_command").offer

    def answer(*_args, **_kwargs):
        """Delay the host decision without changing approved authority."""
        now[0] = 300.0
        return "approve"

    interaction.prompt.side_effect = answer
    assert service.retry_host_command(offer, approved, execution_timeout=30.0) == "host-result"
    assert host.run_host_command.call_args.args[0].deadline == 330.0
    assert "Native sandbox launch check failed." in interaction.info.call_args.args[0]
    assert str(tmp_path) not in interaction.info.call_args.args[0]


def test_host_executor_failure_is_audited_and_not_replayed(tmp_path):
    """A failed approved host start consumes the offer before reporting the failure."""
    service, _, host, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path)
    offer = service.run_sandboxed(approved, "run_command").offer
    host.run_host_command.side_effect = OSError("spawn failed")

    with pytest.raises(OSError, match="spawn failed"):
        service.retry_host_command(offer, approved)
    assert service.retry_host_command(offer, approved) is None
    host.run_host_command.assert_called_once()


def test_offer_approval_and_host_execution_have_distinct_sanitized_audit_events(tmp_path):
    """Audit records separate the offer, user decision, and host launch without shell text."""
    service, _, _, _ = coordinator(
        tmp_path, CommandProcessResult(SandboxOutcome.DENIED, detail="policy denial")
    )
    approved = request(tmp_path, source="printf private-command")
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        offer = service.run_sandboxed(approved, "run_command").offer
        assert service.retry_host_command(offer, approved) == "host-result"
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    events = [record.event_name for record in adapter.records]
    assert events == [
        "sandbox.permission_decided",
        "sandbox.completed",
        "host_command.offered",
        "host_command.approved",
        "host_command.executed",
    ]
    assert all("private-command" not in str(record.attributes) for record in adapter.records)
    assert all(record.attributes["attempt_id"] == approved.attempt_id for record in adapter.records)


def test_local_host_executor_does_not_launch_after_deadline(tmp_path, monkeypatch):
    """An expired host retry cannot start a process after its bound deadline."""
    launch = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", launch)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=time.monotonic() - 1))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.TIMED_OUT
    launch.assert_not_called()


def test_local_host_executor_allows_completion_near_deadline(tmp_path, monkeypatch):
    """A completed child in the final budget interval remains successful without real waits."""
    clock = SimpleNamespace(monotonic=lambda: 100.09)
    monkeypatch.setattr("loop.execution.coordinator.time", clock)
    monkeypatch.setattr("loop.utils.process.time", clock)
    process = Mock(stdout=StringIO("finished"), stderr=StringIO(), returncode=0)
    launch = Mock(return_value=process)
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", launch)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    def reader_thread(*, target, daemon):
        """Drain a fixture stream synchronously without depending on thread scheduling."""
        reader = Mock()
        reader.start.side_effect = target
        reader.is_alive.return_value = False
        return reader

    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", reader_thread)
    approved = HostCommandRequest.from_sandbox(request(tmp_path, deadline=100.1))

    result = LocalHostCommandExecutor().run_host_command(approved)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "finished"
    assert process.wait.call_args_list[0].kwargs["timeout"] == pytest.approx(0.01)
    launch.assert_called_once()
    kill.assert_called_once_with(process)


def test_local_host_executor_cancels_and_cleans_up(tmp_path, monkeypatch):
    """Interrupting an approved host child returns cancellation after group cleanup."""
    process = Mock(stdout=StringIO(), stderr=StringIO())
    process.wait.side_effect = [KeyboardInterrupt(), 0]
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    kill.assert_called_once_with(process)


def test_local_host_executor_fails_if_child_pipes_are_missing(tmp_path, monkeypatch):
    """A malformed host child is terminated rather than treated as completed."""
    process = Mock(stdout=None, stderr=None)
    kill = Mock()
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", kill)

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    kill.assert_called_once_with(process)


def test_host_pipe_failure_does_not_wait_past_deadline(tmp_path, monkeypatch):
    """A broken host child is reported even when post-kill reaping times out."""
    process = Mock(stdout=None, stderr=None)
    process.wait.side_effect = subprocess.TimeoutExpired("sh", 0)
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", Mock())
    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )
    assert result.outcome is SandboxOutcome.UNAVAILABLE


def test_local_host_executor_detects_stuck_reader_and_bounded_cleanup(tmp_path, monkeypatch):
    """A reader that outlives the deadline does not hold the host call open."""

    class StuckReader:
        """Simulate an output reader that cannot finish within the deadline."""

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return True

    process = Mock()
    process.stdout = Mock()
    process.stderr = Mock()
    process.wait.side_effect = [0, subprocess.TimeoutExpired("sh", 1)]
    monkeypatch.setattr("loop.execution.coordinator.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr("loop.execution.coordinator.threading.Thread", StuckReader)
    monkeypatch.setattr("loop.execution.coordinator.kill_process_group", Mock())

    result = LocalHostCommandExecutor().run_host_command(
        HostCommandRequest.from_sandbox(request(tmp_path))
    )

    assert result.outcome is SandboxOutcome.TIMED_OUT
    process.stdout.close.assert_not_called()
    process.stderr.close.assert_not_called()
