"""Verify the native facade rejects unbound tool contexts."""

from unittest.mock import Mock

from loop.execution import CommandExecutionService
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, UnavailableSandboxBackend
from loop.permissions import Action, Operation, PermissionManager, ProcessBoundary, ProcessTarget
from loop.tooling import ToolContext


def test_execution_requires_a_bound_sandbox_target():
    """A context without a native process operation cannot launch a child."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())

    result = service.run_command(ToolContext(None, "renamed_command"), "pwd")

    assert result.code == "sandbox.unavailable"
    assert result.detail == "Bound sandbox request is missing."
    backend.run.assert_not_called()


def test_execution_requires_a_permission_authority(tmp_path):
    """A bound process still needs the registry permission service."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="renamed_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "pwd"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )

    result = service.run_command(
        ToolContext(None, "renamed_command", operations=(operation,)), "pwd"
    )

    assert result.code == "sandbox.unavailable"
    assert result.detail == "Native command authorization is unavailable."
    backend.run.assert_not_called()


def test_unsupported_native_backend_offers_host_retry_without_launching_it(tmp_path):
    """An unavailable native backend reports a separate offer but starts no host child."""
    host = Mock()
    service = CommandExecutionService(UnavailableSandboxBackend("Unqualified host."), host)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "pwd"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(tmp_path),
    )

    result = service.run_command(context, "pwd")

    assert result.code == "sandbox.unavailable"
    assert result.detail == "Unqualified host."
    assert "host_offer" in result.metadata
    host.run_host_command.assert_not_called()


def test_public_command_budget_starts_after_sandbox_approval(tmp_path, monkeypatch):
    """The public facade preserves a full execution budget after a slow permission decision."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.facade.time.monotonic", lambda: now[0])
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()

    def approve(*_args, **_kwargs):
        """Return a user decision after the initial deadline has elapsed."""
        now[0] = 243.0
        return "approve"

    interaction.prompt.side_effect = approve
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf text >victim"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    host = Mock()
    service = CommandExecutionService(backend, host)
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(tmp_path),
    )
    result = service.run_command(context, "printf text >victim")
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].deadline == 273.0
    host.run_host_command.assert_not_called()


def test_execution_service_closes_optional_backend_lifecycle():
    """The facade forwards teardown only when its backend exposes close."""
    backend = Mock()
    CommandExecutionService(backend, Mock()).close()
    backend.close.assert_called_once()
    CommandExecutionService(object(), Mock()).close()


def test_trusted_search_requires_an_installed_helper_and_native_runner(monkeypatch, tmp_path):
    """Search launches only an identity-bound helper under an available native backend."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())
    capture = object()
    backend.run_read_only_argv.return_value = capture
    executable = tmp_path / "rg"
    executable.write_text("installed", encoding="utf-8")
    backend.system_read_roots = (tmp_path,)
    monkeypatch.setattr(
        "loop.execution.facade.ripgrep_path",
        lambda: str(executable),
    )

    assert service.run_trusted_search(("--json",), (3,), 7.0) is capture
    backend.run_read_only_argv.assert_called_once_with(executable, ("--json",), (3,), 7.0)
    (tmp_path / "alias").hardlink_to(executable)
    assert service.run_trusted_search((), (), 7.0) is None
    monkeypatch.setattr(
        "loop.execution.facade.ripgrep_path",
        Mock(side_effect=FileNotFoundError("missing")),
    )
    assert service.run_trusted_search((), (), 7.0) is None
    assert CommandExecutionService(object(), Mock()).run_trusted_search((), (), 7.0) is None


def test_trusted_search_rejects_unmanaged_path_before_native_launch(monkeypatch, tmp_path):
    """A hostile PATH program cannot become the native search helper."""
    executable = tmp_path / "rg"
    executable.write_text("untrusted", encoding="utf-8")
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = ((tmp_path,), ())
    backend.managed_tool_root.return_value = False
    service = CommandExecutionService(backend, Mock())
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))

    assert service.run_trusted_search((), (), 7.0) is None
    backend.run_read_only_argv.assert_not_called()
    backend.managed_tool_root.return_value = True
    assert service.run_trusted_search((), (), 7.0) is backend.run_read_only_argv.return_value
    backend.installed_tool_roots.return_value = ((tmp_path, tmp_path / "unverified"), ())
    backend.managed_tool_root.side_effect = [True, False]
    assert service.run_trusted_search((), (), 7.0) is None


def test_trusted_search_rejects_workspace_writable_installation_bin(monkeypatch, tmp_path):
    """A virtual environment path alone cannot establish executable provenance."""
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "rg"
    executable.write_text("installed", encoding="utf-8")
    backend = Mock()
    backend.system_read_roots = ()
    service = CommandExecutionService(backend, Mock())
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))

    backend.installed_tool_roots.return_value = ((bin_dir,), ())
    backend.managed_tool_root.return_value = False
    assert service.run_trusted_search((), (), 7.0) is None
    backend.run_read_only_argv.assert_not_called()
