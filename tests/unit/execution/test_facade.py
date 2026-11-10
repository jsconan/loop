"""Verify the native facade rejects unbound tool contexts."""

import json
import os
import shlex
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest

from loop.execution import CommandExecutionService
from loop.execution import facade as execution_facade
from loop.execution import jobs as command_jobs
from loop.execution.facade import _automatic_executable_identities
from loop.execution.jobs import CommandJob
from loop.execution.sandbox import (
    CommandProcessResult,
    SandboxCapability,
    SandboxOutcome,
    UnavailableSandboxBackend,
)
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.instructions import InstructionsManager, RuntimeEnvironment, Skill, SkillManager
from loop.models import ToolCall
from loop.permissions import (
    Action,
    FileTarget,
    Operation,
    PermissionManager,
    PolicyScope,
    ProcessBoundary,
    ProcessTarget,
)
from loop.session import SessionManager
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.tooling import ToolContext
from loop.utils import PathHolder, VirtualPath, cached_path

from .sandbox.helpers import logical_private_name


@pytest.fixture(autouse=True)
def logical_policy_names(monkeypatch, tmp_path):
    """Keep fixture storage ancestry separate from protected policy components."""

    def logical(path):
        """Preserve protected components inside this case's logical roots."""
        return logical_private_name(path, tmp_path.parent)

    monkeypatch.setattr(execution_facade, "is_protected_external_read", logical)
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.is_protected_external_read", logical)


@pytest.fixture
def shebang_policy_root(tmp_path, shebang_root, monkeypatch):
    """Keep private policy names logical when physical fixtures use the workspace."""
    if shebang_root == tmp_path:
        return shebang_root, ".ssh"
    private_name = "fixture-private"

    def logical(path):
        """Apply the real private-name policy to the synthetic component."""
        mapped = Path(*(".ssh" if part == private_name else part for part in path.parts))
        return logical_private_name(mapped, shebang_root)

    monkeypatch.setattr(execution_facade, "is_protected_external_read", logical)
    monkeypatch.setattr("loop.execution.sandbox.linux.paths.is_protected_external_read", logical)
    return shebang_root, private_name


@pytest.fixture
def background_session(tmp_path):
    """Bind one real selected session to permission scratch for detached job checks."""
    sessions = SessionManager()
    permissions = PermissionManager(tmp_path)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    return sessions, permissions


def test_executable_lookup_resolves_project_selected_tools_independently(tmp_path, monkeypatch):
    """Each named tool receives only its own identity-bound installation reference."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, name in ((first, "builder"), (second, "formatter")):
        executable = directory / name
        executable.write_text("tool", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(first), str(second))))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.side_effect = lambda path: ((path.parent,), ())
    backend.managed_tool_root.return_value = False

    result = CommandExecutionService(backend, Mock()).resolve_executable(
        ToolContext(None, "resolve_executable"), ["builder", "formatter", "missing"], str(tmp_path)
    )

    builder, formatter, missing = result["executables"]
    assert [builder["name"], formatter["name"], missing["name"]] == [
        "builder",
        "formatter",
        "missing",
    ]
    assert builder["candidate_read_roots"] == [str(first)]
    assert formatter["candidate_read_roots"] == [str(second)]
    assert builder["grant_reference"] != formatter["grant_reference"]
    assert missing["status"] == "absent_from_host_path"
    backend.run.assert_not_called()


@pytest.mark.parametrize("case", ["unavailable", "protected"])
def test_executable_lookup_rejects_unavailable_or_protected_package_roots(
    tmp_path, monkeypatch, case
):
    """Invalid metadata and protected support roots cannot create an executable grant."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "builder"
    executable.write_text("fixture")
    executable.chmod(0o755)
    private = tmp_path / ".ssh"
    private.mkdir()
    monkeypatch.setenv("PATH", str(tools))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = None if case == "unavailable" else ((private,), ())
    result = CommandExecutionService(backend, Mock()).resolve_executable(
        ToolContext(None, "resolve_executable"), "builder", str(tmp_path)
    )
    assert result["sandbox_access"] == (
        "external_grant_unavailable" if case == "unavailable" else "protected_external_path"
    )
    assert "grant_reference" not in result


@pytest.mark.parametrize("names", [[], ["tool"] * 129, ["tool", 1], ["tool", "../unsafe"]])
def test_executable_lookup_rejects_invalid_batches_before_issuing_grants(
    tmp_path, monkeypatch, names
):
    """Malformed or oversized batches cannot partially create executable grants."""
    executable = tmp_path / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    backend = Mock()
    backend.system_read_roots = ()

    with pytest.raises(ValueError, match="Executable lookup"):
        CommandExecutionService(backend, Mock()).resolve_executable(
            ToolContext(None, "resolve_executable"), names, str(tmp_path)
        )

    backend.installed_tool_roots.assert_not_called()


@pytest.mark.parametrize("outside", [False, True])
def test_executable_lookup_requires_a_directory_inside_the_workspace(tmp_path, outside):
    """A file or an outside directory cannot supply executable lookup authority."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    target = tmp_path / "outside" if outside else workspace / "file"
    if outside:
        target.mkdir()
    else:
        target.write_text("not a directory", encoding="utf-8")
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, None))
    context = ToolContext(None, "resolve_executable", instructions_manager=instructions)
    backend = Mock()

    with pytest.raises(ValueError, match="inside the workspace"):
        CommandExecutionService(backend, Mock()).resolve_executable(context, "tool", str(target))

    backend.installed_tool_roots.assert_not_called()


@pytest.mark.parametrize("managed", [False, True])
def test_automatic_discovery_preserves_existing_executable_references(
    tmp_path, monkeypatch, managed
):
    """Automatic preparation cannot evict a reference needed by a later command."""
    workspace = tmp_path / "project"
    tools = tmp_path / "tools"
    workspace.mkdir()
    tools.mkdir()
    executable = tools / "builder"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.side_effect = lambda path: ((path.parent,), ())
    backend.managed_tool_root.return_value = managed
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, Mock())
    lookup = ToolContext(None, "resolve_executable")
    references = [
        service.resolve_executable(lookup, "builder", str(workspace))["grant_reference"]
        for _ in range(128)
    ]
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "builder"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    first = service.run_command(context, "builder")
    result = service.run_command(context, "builder", executable_grant=references[0])

    assert first["boundary"] == result["boundary"] == "sandbox"
    assert backend.run.call_count == 2
    assert interaction.prompt.call_count == (0 if managed else 1)


def test_backend_refresh_during_discovery_preserves_current_execution(tmp_path, monkeypatch):
    """Refresh is refused while a command binds a previously issued executable reference."""
    workspace = tmp_path / "project"
    tools = tmp_path / "tools"
    workspace.mkdir()
    tools.mkdir()
    executable = tools / "builder"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = ((tools,), ())
    backend.managed_tool_root.return_value = True
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    replacement = Mock()
    replacement.system_read_roots = ()
    replacement.managed_tool_root.return_value = True
    monkeypatch.setattr(execution_facade, "select_sandbox_backend", lambda: replacement)
    service = CommandExecutionService(backend, Mock())
    reference = service.resolve_executable(
        ToolContext(None, "resolve_executable"), "builder", str(workspace)
    )["grant_reference"]

    def refresh(_path):
        """Attempt replacement while a new automatic candidate is being discovered."""
        with pytest.raises(RuntimeError, match="active"):
            service.refresh_backend()
        return ((tools,), ())

    backend.installed_tool_roots.side_effect = refresh
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "builder"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = service.run_command(context, "builder", executable_grant=reference)

    assert result["boundary"] == "sandbox"
    backend.run.assert_called_once()
    replacement.run.assert_not_called()


def test_concurrent_executable_lookup_cannot_reuse_evicted_reference(tmp_path, monkeypatch):
    """A lookup that evicts a bound grant during preparation forces a fresh reference."""
    workspace = tmp_path / "project"
    tools = tmp_path / "tools"
    workspace.mkdir()
    tools.mkdir()
    for name in ("builder", "other"):
        executable = tools / name
        executable.write_text("tool", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(execution_facade, "_MAX_EXECUTABLE_GRANTS", 1)
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.return_value = ((tools,), ())
    backend.managed_tool_root.return_value = True
    service = CommandExecutionService(backend, Mock())
    reference = service.resolve_executable(
        ToolContext(None, "resolve_executable"), "builder", str(workspace)
    )["grant_reference"]

    def discover(path):
        """Issue a second public reference while the first command binds its candidate."""
        if path.name == "builder":
            service.resolve_executable(
                ToolContext(None, "resolve_executable"), "other", str(workspace)
            )
        return ((tools,), ())

    backend.installed_tool_roots.side_effect = discover
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "builder"),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = service.run_command(context, "builder", executable_grant=reference)

    assert result.code == "sandbox.unavailable"
    assert "unknown or expired" in result.detail
    backend.run.assert_not_called()


def test_execution_requires_a_bound_sandbox_target():
    """A context without a native process operation cannot launch a child."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())

    result = service.run_command(ToolContext(None, "renamed_command"), "pwd")

    assert result.code == "sandbox.unavailable"
    assert result.detail == "Bound sandbox request is missing."
    backend.run.assert_not_called()


def test_qualified_mediator_reports_unenforceable_stronger_origin_rule(tmp_path, monkeypatch):
    """A destination-only proxy cannot silently bypass a URL-specific hard policy."""
    backend = Mock(capabilities=frozenset({SandboxCapability.MEDIATED_EGRESS}))
    manager = PermissionManager(tmp_path)
    limit = Mock(return_value="HTTPS request rules require unavailable inspection.")
    monkeypatch.setattr(manager, "sandbox_egress_limit", limit)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "pwd"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(None, "run_command", operations=(operation,), permission_manager=manager)
    result = CommandExecutionService(backend, Mock()).run_command(
        context, "pwd", egress_origins=["https://example.com"]
    )
    assert result.code == "sandbox.unavailable"
    assert result.detail == "HTTPS request rules require unavailable inspection."
    limit.assert_called_once_with("run_command", ("https://example.com",))
    backend.run.assert_not_called()


@pytest.mark.parametrize(
    "mismatch", ["manager", "operation", "action", "tool", "boundary", "source", "cwd", "reason"]
)
def test_explicit_host_request_rejects_unbound_context(tmp_path, mismatch):
    """The service accepts only the exact host operation authorized by the registry."""
    manager = PermissionManager(tmp_path)
    target = ProcessTarget(
        argv=("/bin/sh", "-c", "other" if mismatch == "source" else "printf host"),
        cwd=str(tmp_path / "other") if mismatch == "cwd" else str(tmp_path),
        boundary=ProcessBoundary.SANDBOXED if mismatch == "boundary" else ProcessBoundary.HOST,
    )
    operation = Operation(
        tool_id="other" if mismatch == "tool" else "request_host_command",
        action=Action.FILESYSTEM_READ if mismatch == "action" else Action.PROCESS_EXECUTE,
        target=FileTarget(path=str(tmp_path)) if mismatch == "action" else target,
    )
    context = ToolContext(
        None,
        "request_host_command",
        operations=() if mismatch == "operation" else (operation,),
        permission_manager=None if mismatch == "manager" else manager,
    )
    host = Mock()
    result = CommandExecutionService(Mock(), host).request_host_command(
        context, "printf host", str(tmp_path), "  " if mismatch == "reason" else "Needed"
    )
    assert result.code == "host_command.invalid_request"
    host.run_host_command.assert_not_called()


def test_explicit_host_request_rejects_missing_working_directory(tmp_path):
    """An absent bound working directory cannot be approved or executed."""
    manager = PermissionManager(tmp_path)
    cwd = str(tmp_path / "missing")
    operation = Operation(
        tool_id="request_host_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(argv=("/bin/sh", "-c", "pwd"), cwd=cwd, boundary=ProcessBoundary.HOST),
    )
    context = ToolContext(
        None, "request_host_command", operations=(operation,), permission_manager=manager
    )
    host = Mock()
    result = CommandExecutionService(Mock(), host).request_host_command(
        context, "pwd", cwd, "Needed"
    )
    assert result.code == "host_command.invalid_request"
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "exit_code", "possible_effects", "expected"),
    [
        (SandboxOutcome.COMPLETED, 7, False, "process.nonzero_exit"),
        (SandboxOutcome.CANCELLED, None, False, "host_command.cancelled"),
        (SandboxOutcome.UNAVAILABLE, None, True, "host_command.unavailable"),
        (SandboxOutcome.TIMED_OUT, None, True, "host_command.timed_out"),
    ],
)
def test_explicit_host_request_reports_results_and_invalidates_effects(
    tmp_path, outcome, exit_code, possible_effects, expected
):
    """Host outcomes stay separate and possible effects invalidate instruction state."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    instructions = Mock()
    instructions.virtual_paths = VirtualPath(
        workspace=tmp_path, temporary_directory=manager.temporary_directory
    )
    operation = Operation(
        tool_id="request_host_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "pwd"), cwd=str(tmp_path), boundary=ProcessBoundary.HOST
        ),
    )
    context = ToolContext(
        interaction,
        "request_host_command",
        operations=(operation,),
        permission_manager=manager,
        instructions_manager=instructions,
    )
    host = Mock()
    host.run_host_command.return_value = CommandProcessResult(
        outcome,
        exit_code=exit_code,
        stdout=f"{tmp_path}-extra/file",
        stderr=f"https://example.test{tmp_path}/file",
        detail="host stopped",
        possible_effects=possible_effects,
    )

    result = CommandExecutionService(Mock(), host).request_host_command(
        context, "pwd", str(tmp_path), "Needed"
    )

    assert result.code == expected
    assert result.metadata["boundary"] == "host"
    assert result.metadata["possible_effects"] is (
        outcome is SandboxOutcome.COMPLETED or possible_effects
    )
    if outcome is not SandboxOutcome.COMPLETED and possible_effects:
        assert "host attempt" in result.metadata["recovery_hint"]
        assert "not retried" in result.metadata["recovery_hint"]
        assert "may repeat effects" in result.metadata["recovery_hint"]
    else:
        assert "recovery_hint" not in result.metadata
    assert result.metadata["stdout"]["content"] == f"{tmp_path}-extra/file"
    assert result.metadata["stderr"]["content"] == f"https://example.test{tmp_path}/file"
    if outcome is SandboxOutcome.COMPLETED or possible_effects:
        instructions.invalidate.assert_called_once_with(None)
    else:
        instructions.invalidate.assert_not_called()


@pytest.mark.parametrize("invalidation_fails", [False, True])
def test_explicit_host_exception_invalidates_instructions(tmp_path, invalidation_fails):
    """A host execution error invalidates possible effects without hiding the original error."""
    manager = PermissionManager(tmp_path)
    manager.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    interaction = Mock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    instructions = Mock()
    instructions.virtual_paths = VirtualPath(
        workspace=tmp_path, temporary_directory=manager.temporary_directory
    )
    if invalidation_fails:
        instructions.invalidate.side_effect = ValueError("invalidation failed")
    operation = Operation(
        tool_id="request_host_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "pwd"), cwd=str(tmp_path), boundary=ProcessBoundary.HOST
        ),
    )
    context = ToolContext(
        interaction,
        "request_host_command",
        operations=(operation,),
        permission_manager=manager,
        instructions_manager=instructions,
    )
    host = Mock()
    host.run_host_command.side_effect = RuntimeError("host failed")

    with pytest.raises(RuntimeError, match="host failed") as raised:
        CommandExecutionService(Mock(), host).request_host_command(
            context, "pwd", str(tmp_path), "Needed"
        )
    instructions.invalidate.assert_called_once_with(None)
    if invalidation_fails:
        assert isinstance(raised.value.__cause__, ValueError)


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
    diagnostics = result.metadata["sandbox_diagnostics"]
    assert diagnostics["requested_operation"] == "process_execute"
    assert diagnostics["requested_cwd"] == "$WORKSPACE_ROOT"
    assert diagnostics["required_capability"] == "native_sandbox"
    assert diagnostics["failure_source"] == "capability_check"
    assert diagnostics["cause_confidence"] == "unknown"
    assert diagnostics["sandbox_approval_supported"] is False
    assert diagnostics["next_request"] == "request_host_command"
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    "scenario", ["sandbox_effects", "host_capture", "host_cancelled", "host_error"]
)
def test_execution_invalidates_instructions_after_possible_effects(tmp_path, scenario):
    """Possible effects invalidate instructions even when an attempt is interrupted."""
    host = Mock()
    if scenario != "host_capture":
        backend = Mock()
        backend.run.return_value = CommandProcessResult(
            SandboxOutcome.DENIED,
            detail="The sandbox could not admit the command.",
            possible_effects=scenario == "sandbox_effects",
        )
        host.run_host_command.return_value = CommandProcessResult(
            SandboxOutcome.CANCELLED, possible_effects=True
        )
    else:
        backend = UnavailableSandboxBackend("Unqualified host.")
        host.run_host_command.return_value = CommandProcessResult(
            SandboxOutcome.UNAVAILABLE,
            detail="Host child pipes unavailable.",
            possible_effects=True,
        )
    interaction = Mock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    instructions = Mock()
    instructions.agents_filenames = ()
    events = []
    host_result = host.run_host_command.return_value

    def run_host(*_args, **_kwargs):
        """Record host execution before returning or propagating its failure."""
        events.append("host")
        if scenario == "host_error":
            raise RuntimeError("host execution failed")
        return host_result

    host.run_host_command.side_effect = run_host
    instructions.invalidate.side_effect = lambda _path: events.append("invalidate")
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    instructions.virtual_paths = VirtualPath(
        workspace=tmp_path, temporary_directory=permissions.temporary_directory
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
        instructions_manager=instructions,
    )

    if scenario == "host_error":
        with pytest.raises(RuntimeError, match="host execution failed"):
            CommandExecutionService(backend, host).run_command(context, "pwd")
    elif scenario == "sandbox_effects":
        result = CommandExecutionService(backend, host).run_command(context, "pwd")
        assert result.code == "sandbox.denied"
        assert result.metadata["boundary"] == "sandbox"
        assert result.metadata["sandbox_diagnostics"]["possible_effects"] is True
        host.run_host_command.assert_not_called()
    else:
        result = CommandExecutionService(backend, host).run_command(context, "pwd")
        assert result.code == (
            "host_command.unavailable" if scenario == "host_capture" else "host_command.cancelled"
        )
        assert result.metadata["boundary"] == "host"
        assert result.metadata["possible_effects"] is True
        assert "host attempt" in result.metadata["recovery_hint"]
    instructions.invalidate.assert_called_once_with(None)
    assert events == (["invalidate"] if scenario == "sandbox_effects" else ["host", "invalidate"])


@pytest.mark.parametrize("invalidation_fails", [False, True])
def test_execution_invalidates_instructions_before_rendering_output(
    tmp_path, monkeypatch, invalidation_fails
):
    """A completed command invalidates once before rendering or propagates that failure."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout="a" * execution_facade.constants.MAX_TOOL_CONTENT_BYTES,
    )
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    instructions = Mock()
    instructions.agents_filenames = ()
    invalidation_error = ValueError("invalidation failed")
    if invalidation_fails:
        instructions.invalidate.side_effect = invalidation_error
    permissions = PermissionManager(tmp_path)
    instructions.virtual_paths = VirtualPath(
        workspace=tmp_path, temporary_directory=permissions.temporary_directory
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
        instructions_manager=instructions,
    )
    cache = Mock(side_effect=RuntimeError("output cache failed"))
    monkeypatch.setattr(execution_facade, "store_content", cache)

    if invalidation_fails:
        with pytest.raises(ValueError, match="invalidation failed") as error:
            CommandExecutionService(backend, Mock()).run_command(context, "pwd")
        assert error.value is invalidation_error
        cache.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="output cache failed"):
            CommandExecutionService(backend, Mock()).run_command(context, "pwd")
        cache.assert_called_once()
    instructions.invalidate.assert_called_once_with(None)


@pytest.mark.parametrize("boundary", ["sandbox", "host"])
@pytest.mark.parametrize("invalidation_fails", [False, True])
def test_execution_exception_invalidates_instructions_and_preserves_error(
    tmp_path, boundary, invalidation_fails
):
    """Unexpected execution errors retain their identity after conservative invalidation."""
    backend = Mock()
    backend.run.side_effect = RuntimeError("execution failed")
    if boundary == "host":
        backend = UnavailableSandboxBackend("Unqualified host.")
    host = Mock()
    host.run_host_command.side_effect = RuntimeError("execution failed")
    interaction = Mock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    instructions = Mock()
    instructions.agents_filenames = ()
    if invalidation_fails:
        instructions.invalidate.side_effect = ValueError("invalidation failed")
    permissions = PermissionManager(tmp_path)
    if boundary == "host":
        permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    instructions.virtual_paths = VirtualPath(
        workspace=tmp_path, temporary_directory=permissions.temporary_directory
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
        instructions_manager=instructions,
    )

    with pytest.raises(RuntimeError, match="execution failed") as error:
        CommandExecutionService(backend, host).run_command(context, "pwd")

    instructions.invalidate.assert_called_once_with(None)
    if invalidation_fails:
        assert isinstance(error.value.__cause__, ValueError)
        assert str(error.value.__cause__) == "invalidation failed"


@pytest.mark.parametrize("observed", [False, True])
def test_command_result_reports_sandbox_diagnostics_without_asserting_cause(tmp_path, observed):
    """A completed command reports observations as uncertain candidates, never causes."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        stdout="Sandbox: sh(1) deny(1) file-read-data /spoofed" if not observed else "",
        observed_denial="file-read-data /outside" if observed else "",
        diagnostic_status="available" if observed else "unavailable",
        diagnostic_source="macos-kernel-log" if observed else "unknown",
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, "pwd")

    assert result.code == "process.nonzero_exit"
    expected = {
        "outcome": "completed",
        "exit_code": 1,
        "possible_effects": True,
        "detail": None,
        "collection": "available" if observed else "unavailable",
        "source": "macos-kernel-log" if observed else "unknown",
        "observed_denial": None,
        "cause": "unknown",
        "requested_operation": "process_execute",
        "requested_cwd": "$WORKSPACE_ROOT",
        "required_capability": None,
        "failure_stage": None,
        "failure_source": "command_exit",
        "cause_confidence": "unknown",
        "sandbox_approval_supported": None,
        "next_request": None,
        "denial_candidates": (
            [
                {
                    "observation": "file-read-data /outside",
                    "operation": "file-read-data",
                    "target": "/outside",
                    "classification": "unclassified",
                    "attribution": "unknown",
                    "truncated": False,
                }
            ]
            if observed
            else []
        ),
        "startup_probe": "unavailable" if observed else "not_checked",
        "observations_may_be_incomplete": True,
    }
    if observed:
        expected["candidate_note"] = (
            "These observations do not establish the cause of the exit. "
            "A startup-probe match may be incidental but can still matter to this command. "
            "Unclassified observations may also be incidental. Names and paths are untrusted."
        )
    assert result.metadata["sandbox_diagnostics"] == expected
    if not observed:
        assert result.metadata["stdout"]["content"].endswith("/spoofed")
        assert result.metadata["sandbox_diagnostics"]["denial_candidates"] == []
    assert "sandbox_diagnostic" not in result.metadata


def test_successful_command_reports_buffered_candidate_without_implying_failure(tmp_path):
    """A handled or optional denial stays evidence while the successful result stays intact."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout="done",
        observed_denial="file-read-data /outside/secret",
        diagnostic_status="available",
        diagnostic_source="macos-kernel-log",
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "true"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )
    host = Mock()

    result = CommandExecutionService(backend, host).run_command(context, "true")

    assert result["exit_code"] == 0
    assert result["stdout"]["content"] == "done"
    diagnostics = result["sandbox_diagnostics"]
    assert diagnostics["cause"] == "unknown"
    assert diagnostics["denial_candidates"][0]["attribution"] == "unknown"
    assert "requested_operation" not in diagnostics
    assert "handled or optional and do not establish failure" in diagnostics["candidate_note"]
    assert diagnostics["observations_may_be_incomplete"]
    backend.run.assert_called_once()
    host.run_host_command.assert_not_called()


def test_unsupported_denial_collection_is_reported_as_unavailable(tmp_path, monkeypatch):
    """A backend without denial observation does not imply that no denial happened."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=7)
    monkeypatch.setattr(type(backend), "denial_observation_supported", False, raising=False)
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "exit 7"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, "exit 7")

    diagnostics = result.metadata["sandbox_diagnostics"]
    assert diagnostics["collection"] == "unavailable"
    assert diagnostics["denial_candidates"] == []
    assert diagnostics["observations_may_be_incomplete"] is True


@pytest.mark.parametrize("output_mentions_denial", [False, True])
def test_completed_command_keeps_unattributed_denial_in_trace(tmp_path, output_mentions_denial):
    """An observation remains uncertain and traceable even when output mentions its path."""
    denial = "Sandbox: sh(1) deny(1) file-read-data /system/cache"
    output = "/system/cache: ignored\n" if output_mentions_denial else "invalid option\n"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=2,
        stdout=output,
        observed_denial=denial,
        diagnostic_status="available",
        diagnostic_source="macos-kernel-log",
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "false"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )

    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        with telemetry.context(trace_id="1" * 32):
            result = CommandExecutionService(backend, Mock()).run_command(context, "false")
    finally:
        closed = telemetry.close(1)
        set_telemetry(None)
    assert closed

    assert result.code == "process.nonzero_exit"
    assert result.metadata["stdout"]["content"] == output
    assert result.metadata["sandbox_diagnostics"]["observed_denial"] is None
    assert result.metadata["sandbox_diagnostics"]["denial_candidates"] == [
        {
            "observation": denial,
            "operation": "file-read-data",
            "target": "/system/cache",
            "classification": "unclassified",
            "attribution": "unknown",
            "truncated": False,
        }
    ]
    assert "sandbox_diagnostic" not in result.metadata
    records = [
        record for record in adapter.records if record.event_name == "sandbox.unattributed_denial"
    ]
    assert len(records) == 1
    assert records[0].trace_id == "1" * 32
    assert records[0].attributes["attempt_id"] == backend.run.call_args.args[0].attempt_id
    assert records[0].payload == {
        "observed_denials": (denial,),
        "source": "macos-kernel-log",
    }


def test_command_candidates_keep_startup_matches_and_unclassified_denials(tmp_path, monkeypatch):
    """Both observed resources reach the original result with explicit uncertainty."""
    startup = "Sandbox: sh(1) deny(1) file-read-data /system/cache"
    denied = "Sandbox: cat(2) deny(1) file-read-data /outside/secret"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=7,
        observed_denial=f"{startup}\n{denied}",
        diagnostic_status="available",
        diagnostic_source="macos-kernel-log",
    )
    monkeypatch.setattr(
        type(backend),
        "classify_denial_candidates",
        lambda _self, _observed: (((denied, False), (startup, True)), "observed"),
        raising=False,
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "exit 7"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )
    host = Mock()

    result = CommandExecutionService(backend, host).run_command(context, "exit 7")

    diagnostics = result.metadata["sandbox_diagnostics"]
    assert result.code == "process.nonzero_exit"
    assert diagnostics["cause"] == "unknown"
    assert diagnostics["startup_probe"] == "observed"
    assert diagnostics["observations_may_be_incomplete"]
    assert diagnostics["denial_candidates"] == [
        {
            "observation": denied,
            "operation": "file-read-data",
            "target": "/outside/secret",
            "classification": "unclassified",
            "attribution": "unknown",
            "truncated": False,
        },
        {
            "observation": startup,
            "operation": "file-read-data",
            "target": "/system/cache",
            "classification": "also_observed_in_successful_startup_probe",
            "attribution": "unknown",
            "truncated": False,
        },
    ]
    assert "may be incidental but can still matter" in diagnostics["candidate_note"]
    backend.run.assert_called_once()
    host.run_host_command.assert_not_called()


def test_linux_journal_candidates_are_redacted_unclassified_and_bounded(tmp_path, monkeypatch):
    """Linux candidates retain uncertainty, truncation, and protected-path redaction."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=7,
        observed_denial="encoded records",
        diagnostic_status="available",
        diagnostic_source="linux-kernel-journal",
    )
    records = (("apparmor-open /outside/.ssh/private", False), ("apparmor-open /other…", False))
    monkeypatch.setattr(
        type(backend),
        "classify_denial_candidates",
        lambda *_args: (records, "unavailable"),
        raising=False,
    )
    trace = Mock()
    monkeypatch.setattr(execution_facade, "telemetry_trace_event", trace)
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "exit 7"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, "exit 7")

    diagnostic = result.metadata["sandbox_diagnostics"]
    assert diagnostic["startup_probe"] == "unavailable"
    assert diagnostic["source"] == "linux-kernel-journal"
    assert diagnostic["denial_candidates"] == [
        {
            "observation": "apparmor-open <protected>",
            "operation": "open",
            "target": "<protected>",
            "classification": "unclassified",
            "attribution": "unknown",
            "truncated": False,
        },
        {
            "observation": "apparmor-open /other…",
            "operation": "open",
            "classification": "unclassified",
            "attribution": "unknown",
            "truncated": True,
        },
    ]
    assert "private" not in json.dumps(trace.call_args.kwargs["payload_factory"]())


@pytest.mark.parametrize("component", [".ssh", ".SSH", "Library"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_denial_candidates_hide_protected_resources_from_results_and_traces(
    tmp_path, monkeypatch, component, ambiguous
):
    """A kernel-resolved protected target cannot leak through diagnostic metadata."""
    resource = tmp_path.parent / component / "private-target"
    denial = f"Sandbox: tool(2) deny(1) file-read-metadata {resource}"
    if ambiguous:
        denial = f"Sandbox: deny(1) network-(2) deny(1) file-read-metadata {resource}"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=7,
        observed_denial=denial,
        diagnostic_status="available",
        diagnostic_source="macos-kernel-log",
    )
    trace = Mock()
    monkeypatch.setattr(execution_facade, "telemetry_trace_event", trace)
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "exit 7"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, "exit 7")

    candidate = result.metadata["sandbox_diagnostics"]["denial_candidates"][0]
    expected = (
        "<ambiguous sandbox denial>"
        if ambiguous
        else "Sandbox: tool(2) deny(1) file-read-metadata <protected>"
    )
    assert candidate["observation"] == expected
    assert "private-target" not in json.dumps(result.metadata)
    assert trace.call_args.kwargs["payload_factory"]()["observed_denials"] == [expected]


@pytest.mark.parametrize(
    "outcome",
    [
        SandboxOutcome.DENIED,
        SandboxOutcome.UNAVAILABLE,
        SandboxOutcome.UNSUPPORTED,
        SandboxOutcome.INVALID,
        SandboxOutcome.STALE,
        SandboxOutcome.TIMED_OUT,
        SandboxOutcome.CANCELLED,
    ],
)
@pytest.mark.parametrize("possible_effects", [False, True])
def test_interrupted_sandbox_result_warns_only_when_effects_are_possible(
    tmp_path, outcome, possible_effects
):
    """Interrupted outcomes do not imply replay safety or recommend an unrelated host run."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(outcome, possible_effects=possible_effects)
    host = Mock()
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None, "run_command", operations=(operation,), permission_manager=PermissionManager(tmp_path)
    )

    result = CommandExecutionService(backend, host).run_command(context, ":")

    assert result.code == f"sandbox.{outcome.value}"
    diagnostics = result.metadata["sandbox_diagnostics"]
    assert diagnostics["possible_effects"] is possible_effects
    assert diagnostics["failure_source"] == "native_backend"
    assert diagnostics["failure_stage"] == (
        None if possible_effects else "Native sandbox execution could not start."
    )
    assert diagnostics["cause_confidence"] == "unknown"
    assert diagnostics["sandbox_approval_supported"] is None
    assert diagnostics["next_request"] == (
        "request_host_command"
        if not possible_effects and outcome in {SandboxOutcome.DENIED, SandboxOutcome.UNAVAILABLE}
        else None
    )
    if possible_effects:
        assert "not retried" in result.metadata["recovery_hint"]
        assert "inspect" in result.metadata["recovery_hint"].lower()
        assert "may repeat" in result.metadata["recovery_hint"]
        assert "host_offer" not in result.metadata
        if outcome not in {SandboxOutcome.DENIED, SandboxOutcome.UNAVAILABLE}:
            assert "request_host_command" not in result.metadata["recovery_hint"]
    else:
        assert "recovery_hint" not in result.metadata
    host.run_host_command.assert_not_called()


def test_sandbox_reader_failure_preserves_partial_output_without_host_retry(tmp_path):
    """A failed stream drain cannot appear complete or trigger an unrestricted retry."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNAVAILABLE,
        stdout="partial",
        stdout_discarded=None,
        detail="Command output capture failed.",
        possible_effects=True,
        failure_context="Native command output could not be fully captured.",
    )
    host = Mock()
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction, "run_command", operations=(operation,), permission_manager=permissions
    )

    result = CommandExecutionService(backend, host).run_command(context, ":")

    assert result.code == "sandbox.unavailable"
    assert result.metadata["stdout"]["content"] == "partial"
    assert result.metadata["stdout"]["capture_complete"] is False
    assert result.metadata["stdout"]["discarded_characters"] is None
    assert result.metadata["sandbox_diagnostics"]["possible_effects"] is True
    assert result.metadata["sandbox_diagnostics"]["failure_stage"] == (
        "Native command output could not be fully captured."
    )
    assert "host_offer" not in result.metadata
    host.run_host_command.assert_not_called()


def test_unsupported_capability_never_offers_host_retry(tmp_path):
    """An incompatible installed feature cannot turn into an unrestricted host execution."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNSUPPORTED, detail="Request schema is unavailable."
    )
    host = Mock()
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
    )
    result = CommandExecutionService(backend, host).run_command(context, "pwd")
    assert result.code == "sandbox.unsupported"
    assert "host_offer" not in result.metadata
    interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_host_retry_preserves_diagnostic_from_boundary_failure(tmp_path):
    """A separately approved host run retains any observed boundary diagnostic."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.UNAVAILABLE,
        detail="Admission failed.",
        observed_denial="file-read-data /outside",
    )
    host = Mock()
    host.run_host_command.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.supports_fresh_host_approval = True
    interaction.prompt.return_value = "approve"
    permissions = PermissionManager(tmp_path)
    permissions.set_limit("allow_host_processes", True, scope=PolicyScope.SESSION)
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
    )

    result = CommandExecutionService(backend, host).run_command(context, "pwd")

    assert result["boundary"] == "host"
    assert "repeat effects" in result["authority_notice"]
    assert result["possible_effects"] is True
    assert result["sandbox_attempt"]["observed_denial"] == "file-read-data /outside"
    host.run_host_command.assert_called_once()


def test_sandbox_diagnostic_metadata_redacts_external_paths(tmp_path, monkeypatch):
    """Unattributed denial traces and command metadata redact registered external roots."""
    policy = execution_facade.is_protected_external_read

    def logical_record(path):
        """Remove the operation prefix before applying the logical fixture policy."""
        value = str(path)
        if value.startswith(("file-read-data ", "file-write-data ")):
            path = Path(value.split(" ", 1)[1])
        return policy(path)

    monkeypatch.setattr(execution_facade, "is_protected_external_read", logical_record)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=1,
        detail=f"Cannot inspect {external}/file",
        observed_denial=f"file-read-data {external}/secret",
        diagnostic_status="available",
        diagnostic_source="macos-kernel-log",
    )
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    trace = Mock()
    monkeypatch.setattr(execution_facade, "telemetry_trace_event", trace)
    monkeypatch.setattr(
        type(backend),
        "classify_denial_candidates",
        lambda _self, _observed: (
            (
                (f"file-read-data {external}/secret", False),
                (f"file-write-data {external}/other", False),
            ),
            "not_observed",
        ),
        raising=False,
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "pwd"),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(
        context, "pwd", read_roots=[str(external)]
    )

    assert result.code == "process.nonzero_exit"
    diagnostic = result.metadata["sandbox_diagnostics"]
    assert diagnostic["detail"] == "Cannot inspect <external>/file"
    assert diagnostic["observed_denial"] is None
    assert diagnostic["denial_candidates"][0]["observation"] == "file-read-data <external>/secret"
    assert diagnostic["denial_candidates"][1]["observation"] == "file-write-data <external>/other"
    assert trace.call_args.kwargs["payload_factory"]()["observed_denials"] == [
        "file-read-data <external>/secret",
        "file-write-data <external>/other",
    ]
    assert str(external) not in json.dumps(diagnostic)


@pytest.mark.parametrize("path_denial", [False, True])
def test_model_diagnostic_hides_ungranted_external_denial_path(tmp_path, monkeypatch, path_denial):
    """Model diagnostics hide ungranted paths and preserve non-path observations."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    resource = tmp_path.parent / "outside-tool" / "sibling-secret"
    denial = f"apparmor-open {resource}" if path_denial else "apparmor-connect tcp:443"
    expected = "apparmor-open <external>" if path_denial else denial
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout="exact-file-only",
        observed_denial=denial,
        diagnostic_status="available",
        diagnostic_source="linux-kernel-journal",
    )
    monkeypatch.setattr(
        type(backend),
        "classify_denial_candidates",
        lambda *_args: (((denial, False),), "unavailable"),
        raising=False,
    )
    trace = Mock()
    monkeypatch.setattr(execution_facade, "telemetry_trace_event", trace)
    permissions = PermissionManager(workspace)
    instructions = Mock()
    instructions.agents_filenames = ()
    instructions.virtual_paths = VirtualPath(
        workspace=workspace, temporary_directory=permissions.temporary_directory
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "true"),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
        instructions_manager=instructions,
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, "true")

    assert result["stdout"]["content"] == "exact-file-only"
    assert result["sandbox_diagnostics"]["denial_candidates"][0]["observation"] == expected
    assert trace.call_args.kwargs["payload_factory"]()["observed_denials"] == [expected]
    assert str(resource) not in json.dumps(result)


@pytest.mark.parametrize(
    "scope",
    [
        "approved",
        "other",
        "workspace",
        "protected",
        "limited",
        "inside",
        "direct",
        "system",
        "relative",
        "absolute",
        "not-executable",
        "parent",
        "long",
    ],
)
def test_selected_executable_alias_uses_only_approved_scope(tmp_path, monkeypatch, scope):
    """Bind only selected directory aliases that point into an approved read root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lookup = workspace / "lookup"
    lookup.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    links = (
        workspace
        if scope == "workspace"
        else tmp_path / (".ssh" if scope == "protected" else "links")
    )
    if links != workspace:
        links.mkdir()
    executable = installation / ("fixture-tool" if scope == "direct" else "actual")
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o644 if scope == "not-executable" else 0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    target = alias / executable.name
    (lookup / "fixture-tool").symlink_to(
        os.path.relpath(target, lookup) if scope == "relative" else target
    )
    monkeypatch.setenv(
        "PATH", str(installation if scope == "direct" else other if scope == "long" else lookup)
    )
    if scope == "limited":
        monkeypatch.setattr(execution_facade, "_MAX_EXECUTABLE_SYMLINKS", 1)
    command = {
        "relative": "./lookup/fixture-tool",
        "absolute": shlex.quote(str(lookup / "fixture-tool")),
        "not-executable": "./lookup/fixture-tool",
        "parent": "./lookup/../lookup/fixture-tool",
        "long": "./" + "a" * execution_facade.constants.MAX_PATH_LENGTH,
    }.get(scope, "fixture-tool")
    backend = Mock()
    backend.system_read_roots = (links,) if scope == "system" else ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    requested = workspace if scope == "inside" else other if scope == "other" else installation
    result = CommandExecutionService(backend, Mock()).run_command(
        context, command, read_roots=[str(requested)]
    )

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_aliases == (
        (alias,) if scope in {"approved", "relative", "absolute", "parent"} else ()
    )
    assert request.read_roots == (() if scope == "inside" else (requested,))
    assert interaction.prompt.call_count == (0 if scope == "inside" else 1)


@pytest.mark.parametrize(
    "case",
    [
        "selected",
        "unicode",
        "punctuation",
        "shadowed",
        "nonexecutable",
        "outside",
        "sibling",
        "protected",
        "bounded",
        "bounded-grant",
        "candidates",
        "target-nonexecutable",
        "error",
    ],
)
def test_indirect_executable_alias_respects_approved_scope_and_path_order(
    tmp_path, monkeypatch, case
):
    """Prepare only aliases for the first executable that points into the approved root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launcher = workspace / "launcher"
    launcher.write_text("launcher", encoding="utf-8")
    launcher.chmod(0o755)
    installation = tmp_path / "installation"
    installation.mkdir()
    other = tmp_path / ("installation-other" if case == "sibling" else "other")
    other.mkdir()
    lookup = tmp_path / (".ssh" if case == "protected" else "lookup")
    lookup.mkdir()
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    executable = (other if case in {"outside", "sibling"} else installation) / "actual"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o644 if case == "target-nonexecutable" else 0o755)
    alias = links / "alias"
    alias.symlink_to(executable.parent, target_is_directory=True)
    name = {"unicode": "加工-tool", "punctuation": "@fixture"}.get(case, "fixture-tool")
    (lookup / name).symlink_to(alias / executable.name)
    (lookup / "broken").symlink_to(lookup / "missing")
    if case == "candidates":
        monkeypatch.setattr(execution_facade.constants, "MAX_COMMAND_PATH_COUNT", 3)
        for index in range(2):
            (lookup / f"candidate-{index}").symlink_to(alias / executable.name)
    if case in {"shadowed", "nonexecutable", "bounded", "bounded-grant"}:
        shadow = earlier / ("filler" if case.startswith("bounded") else "fixture-tool")
        shadow.write_text("shadow", encoding="utf-8")
        shadow.chmod(0o644 if case == "nonexecutable" else 0o755)
    if case.startswith("bounded"):
        monkeypatch.setattr(execution_facade, "_MAX_INDIRECT_PATH_ENTRIES", 1)
    if case == "error":
        original_scandir = os.scandir

        def scandir(path):
            """Make one selected directory unreadable while preserving other scans."""
            if not isinstance(path, int) and Path(path) == lookup:
                raise PermissionError("fixture path unavailable")
            return original_scandir(path)

        monkeypatch.setattr(execution_facade.os, "scandir", scandir)
    monkeypatch.setenv("PATH", os.pathsep.join((str(earlier), str(lookup))))
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "./launcher"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    service = CommandExecutionService(backend, Mock())
    grant = (
        service.resolve_executable(
            ToolContext(None, "resolve_executable"), "fixture-tool", str(workspace)
        )["grant_reference"]
        if case == "bounded-grant"
        else None
    )
    result = service.run_command(
        context, "./launcher", read_roots=[str(installation)], executable_grant=grant
    )

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_roots == (installation,)
    assert request.read_aliases == (
        (alias,)
        if case
        in {"selected", "unicode", "punctuation", "nonexecutable", "bounded-grant", "candidates"}
        else ()
    )
    assert interaction.prompt.call_count == 1


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "no-newline",
        "plain",
        "partial",
        "relative",
        "nul",
        "missing",
        "protected",
        "hardlink",
        "nonexecutable",
        "open-error",
        "changed-type",
        "changed-links",
        "launcher",
        "launcher-quoted",
        "launcher-name",
        "launcher-name-hardlink",
        "launcher-name-nonexec",
        "launcher-name-unapproved",
        "launcher-name-no-root",
        "nested",
        "launcher-nested",
        "launcher-duplicate",
        "nested-cycle",
        "nested-hardlink",
        "nested-protected",
        "nested-limit",
        "nested-extra-approved",
        "nested-extra-denied",
        "launcher-variable",
        "launcher-protected",
        "launcher-hardlink",
        "launcher-nonexec",
        "launcher-unapproved",
        "launcher-no-root",
    ],
)
def test_workspace_script_interpreter_discovery_respects_existing_scope(
    shebang_policy_root, monkeypatch, case
):
    """Selected scripts request any missing runtime scope before exposing its alias."""
    tmp_path, private_name = shebang_policy_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    executable = installation / "actual"
    executable.write_text("engine", encoding="utf-8")
    executable.chmod(0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    environment = workspace / "arbitrary-tools"
    environment.mkdir()
    interpreter = environment / "engine"
    interpreter.symlink_to(alias / executable.name)
    if case in {"protected", "launcher-protected"}:
        directory = workspace / private_name
        directory.mkdir()
        script = directory / "task"
        command = (
            f"./{private_name}/task" if case == "protected" else f"./launcher ./{private_name}/task"
        )
    elif case == "launcher-variable":
        directory = workspace / "$VALUE"
        directory.mkdir()
        script = directory / "task"
        command = "./launcher $VALUE/task"
    elif case == "launcher-quoted":
        script = workspace / "task with space"
        command = "./launcher './task with space'"
    elif case == "launcher-duplicate":
        script = workspace / "task"
        command = f"./launcher ./task {workspace / 'nested'}"
    else:
        script = workspace / "task"
        command = (
            "./launcher task"
            if case.startswith("launcher-name")
            else "./launcher ./task"
            if case.startswith("launcher")
            else "./task"
        )
    if case.startswith("launcher"):
        launcher = workspace / "launcher"
        launcher.write_text("launcher", encoding="utf-8")
        launcher.chmod(0o755)
    line = f"#! \t{interpreter} --fixture-option".encode()
    if case.startswith("nested") or case in {"launcher-nested", "launcher-duplicate"}:
        nested = workspace / (f"{private_name}/nested" if case == "nested-protected" else "nested")
        nested.parent.mkdir(exist_ok=True)
        nested.write_bytes(line + b"\nfixture body\n")
        nested.chmod(0o755)
        if case == "nested-cycle":
            nested.write_text(f"#!{script}\n", encoding="utf-8")
        elif case == "nested-hardlink":
            os.link(nested, workspace / "nested-link")
        elif case == "nested-limit":
            previous = nested
            for index in range(128):
                following = workspace / f"nested-{index}"
                previous.write_text(f"#!{following}\n", encoding="utf-8")
                following.write_bytes(line + b"\n")
                following.chmod(0o755)
                previous = following
        line = f"#!{nested}".encode()
    content = {
        "plain": b"ordinary file\n",
        "partial": b"#!" + b"x" * execution_facade._MAX_INTERPRETER_LINE_BYTES,
        "relative": b"#!relative/engine\n",
        "nul": b"#!/fixture\x00engine\n",
        "no-newline": line,
    }.get(case, line + b"\nfixture body\n")
    if case != "missing":
        script.write_bytes(content)
        script.chmod(
            0o644
            if case in {"nonexecutable", "launcher-nonexec", "launcher-name-nonexec"}
            else 0o755
        )
    if case in {"hardlink", "launcher-hardlink", "launcher-name-hardlink"}:
        os.link(script, workspace / "other-link")
    if case in {"protected", "hardlink", "nonexecutable", "missing"}:
        monkeypatch.setattr(
            execution_facade,
            "open_selected_file",
            Mock(side_effect=AssertionError("unsafe script must not be opened")),
        )
    if case in {
        "launcher-variable",
        "launcher-protected",
        "launcher-hardlink",
        "launcher-nonexec",
        "launcher-name-hardlink",
        "launcher-name-nonexec",
        "launcher-name-no-root",
        "launcher-no-root",
    }:
        original_open = execution_facade.open_selected_file

        def checked_open(root, path, identity):
            """Allow the launcher but reject opening an unsafe argument candidate."""
            assert path != script
            return original_open(root, path, identity)

        monkeypatch.setattr(execution_facade, "open_selected_file", checked_open)
    if case == "open-error":
        monkeypatch.setattr(
            execution_facade, "open_selected_file", Mock(side_effect=OSError("changed"))
        )
    if case in {"changed-type", "changed-links"}:
        original_open = execution_facade.open_selected_file
        original_fstat = os.fstat
        opened = []

        def open_file(*args):
            """Record the selected descriptor after the anchored open completes."""
            descriptor = original_open(*args)
            opened.append(descriptor)
            return descriptor

        def changed_fstat(descriptor):
            """Simulate changed selected-file metadata after the safe open."""
            details = original_fstat(descriptor)
            if descriptor not in opened:
                return details
            opened.remove(descriptor)
            return SimpleNamespace(
                st_mode=0 if case == "changed-type" else details.st_mode,
                st_nlink=2 if case == "changed-links" else details.st_nlink,
            )

        monkeypatch.setattr(execution_facade, "open_selected_file", open_file)
        monkeypatch.setattr(execution_facade.os, "fstat", changed_fstat)
    monkeypatch.setenv("PATH", str(empty_path))
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "deny" if case == "nested-extra-denied" else "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    requested = (
        empty_path
        if case
        in {
            "launcher-unapproved",
            "launcher-name-unapproved",
            "nested-extra-approved",
            "nested-extra-denied",
        }
        else installation
    )
    no_root = case in {"launcher-no-root", "launcher-name-no-root"}
    result = CommandExecutionService(backend, Mock()).run_command(
        context, command, read_roots=None if no_root else [str(requested)]
    )

    if case == "nested-extra-denied":
        assert result.code == "tool.denied"
        interaction.prompt.assert_called_once()
        backend.run.assert_not_called()
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert scope["read_roots"] == [str(requested), str(installation)]
        return
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_roots == (
        (requested, installation)
        if case == "nested-extra-approved"
        else ()
        if no_root
        else (requested,)
    )
    assert request.read_aliases == (
        (alias,)
        if case
        in {
            "valid",
            "no-newline",
            "launcher",
            "launcher-quoted",
            "launcher-name",
            "nested",
            "launcher-nested",
            "launcher-duplicate",
            "nested-extra-approved",
        }
        else ()
    )
    assert interaction.prompt.call_count == (0 if no_root else 1)
    if case == "nested-extra-approved":
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert scope["read_roots"] == [str(requested), str(installation)]


@pytest.mark.parametrize(
    ("outer", "inner", "target", "expected"),
    [
        ('exec "$@"', 'exec "$@"', "./tools/fixture", True),
        ('exec "$@"', "repeat", "./tools/fixture", True),
        ('exec "${@}"', 'exec "$1"', "./tools/fixture", True),
        ('exec "$1"', 'exec "$@"', "./tools/fixture", False),
        ('exec "$@"', 'exec "$@"', "$TARGET", False),
        ('exec "$@"', 'printf ignored; exec "$@"', "./tools/fixture", False),
    ],
)
def test_nested_forwarders_discover_only_literal_selected_targets(
    shebang_policy_root, monkeypatch, outer, inner, target, expected
):
    """Forwarding chains expose only a deterministically selected installation."""
    root, _ = shebang_policy_root
    workspace = root / "workspace"
    workspace.mkdir()
    installation = root / "installation"
    installation.mkdir()
    executable = installation / "actual"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    tools = workspace / "tools"
    tools.mkdir()
    (tools / "fixture").symlink_to(executable)
    for name, body in (("outer", outer), ("inner", outer if inner == "repeat" else inner)):
        launcher = workspace / name
        launcher.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        launcher.chmod(0o755)
    empty = root / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    backend = Mock()
    backend.system_read_roots = (Path("/bin"), Path("/usr/bin"))
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    command = f"./outer {'./outer' if inner == 'repeat' else './inner'} {target}"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert result["boundary"] == "sandbox"
    assert backend.run.call_count == 1
    assert backend.run.call_args.args[0].cleanup_proof is True
    assert backend.run.call_args.args[0].read_roots == ((installation,) if expected else ())
    assert interaction.prompt.call_count == int(expected)


@pytest.mark.parametrize(
    ("retarget", "available", "fault", "selection"),
    [
        (False, True, None, "project"),
        (True, True, None, "project"),
        (False, False, None, "project"),
        (False, True, ValueError, "project"),
        (False, True, OSError, "project"),
        (False, True, TimeoutError, "project"),
        (False, True, "irrelevant", "project"),
        (False, True, "inside", "project"),
        (False, True, "contextless", "project"),
        (False, True, None, "direct"),
        (True, True, None, "direct"),
        (False, False, None, "direct"),
    ],
)
def test_selected_project_runtime_is_validated_and_approved_in_original_call(
    shebang_policy_root, monkeypatch, retarget, available, fault, selection
):
    """An identified off-PATH runtime is scoped and checked again after approval."""
    tmp_path, _ = shebang_policy_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / ".venv"
    commands = environment / "bin"
    commands.mkdir(parents=True)
    (workspace / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(runtime, target_is_directory=True)
    (environment / "pyvenv.cfg").write_text(
        f"home = {alias}\nimplementation = CPython\nversion_info = 3.14.2\n", encoding="utf-8"
    )
    selected = runtime / "engine"
    selected.write_text("fixture", encoding="utf-8")
    selected.chmod(0o755)
    interpreter = commands / "python"
    interpreter.symlink_to(selected)
    alternative = runtime / "other"
    alternative.write_text("fixture", encoding="utf-8")
    alternative.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "uv"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    inspected = []

    def runtime_reader(_self, _workspace, roots, identities):
        """Return a candidate only for the selected off-PATH environment."""
        inspected.append((roots, identities))
        identified = any(identity[1] == interpreter for identity in identities)
        if identified and isinstance(fault, type):
            raise fault("fixture metadata unavailable")
        if identified and fault == "irrelevant":
            return (tmp_path / "unrelated",)
        if identified and fault == "inside":
            return (workspace,)
        return (runtime,) if identified else ()

    if available:
        monkeypatch.setattr(
            type(backend), "automatic_tool_read_roots", runtime_reader, raising=False
        )
    interaction = Mock()

    def approve(*_args, **_kwargs):
        """Change the selected interpreter only in the retargeting control."""
        if retarget:
            interpreter.unlink()
            interpreter.symlink_to(alternative)
        return "approve"

    interaction.prompt.side_effect = approve
    command = "uv run --no-sync pytest" if selection == "project" else "./.venv/bin/python -c pass"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )
    if fault == "contextless":
        monkeypatch.setattr(PermissionManager, "describe_sandbox_reads", lambda *_args: None)

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert bool(inspected) is available
    if available:
        assert (commands, commands) in inspected[-1][0]
    approved = not available or fault in {None, "contextless"}
    assert interaction.prompt.call_count == int(approved)
    if retarget:
        assert result.code == "sandbox.stale"
        backend.run.assert_not_called()
    elif fault is TimeoutError:
        assert result.code == "sandbox.unavailable"
        backend.run.assert_not_called()
    elif approved:
        assert result["boundary"] == "sandbox"
        request = backend.run.call_args.args[0]
        assert request.read_roots == (runtime,)
        assert request.read_aliases == (alias,)
        assert request.workspace_executable_identities[0][1:3] == (interpreter, selected)
        assert dict(request.environment)["PATH"] == str(tools)
        assert request.shell_environment()["HOME"] == str(
            context.permission_manager.temporary_directory
        )
        assert "UV_CACHE_DIR" not in request.shell_environment()
        assert "including neighboring files" in request.read_context
    else:
        assert result["boundary"] == "sandbox"
        assert backend.run.call_args.args[0].read_roots == ()


@pytest.mark.parametrize(
    "case",
    [
        "approved",
        "denied",
        "stale",
        "bound-method",
        "single",
        "wrong-root",
        "too-many",
        "unmanaged",
        "protected",
        "bad-alias",
        "wrong-target",
        "missing-alias",
        "invalid-metadata",
        "metadata-error",
        "veto-direct",
        "veto-wrapped",
    ],
)
def test_selected_runtime_package_dependencies_require_approval(tmp_path, monkeypatch, case):
    """Only verified dependencies of a selected runtime join its reviewed scope."""
    workspace = tmp_path / "workspace"
    commands = workspace / ".venv/bin"
    commands.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(runtime, target_is_directory=True)
    selected = runtime / "engine"
    selected.write_text("fixture")
    selected.chmod(0o755)
    interpreter = commands / "python"
    interpreter.symlink_to(selected)
    (commands.parent / "pyvenv.cfg").write_text(
        f"home = {alias}\nimplementation = CPython\nversion_info = 3.14\n"
    )
    dependency = tmp_path / (".ssh" if case == "protected" else "dependency")
    dependency.mkdir()
    dependency_alias = tmp_path / "dependency-alias"
    dependency_alias.symlink_to(dependency, target_is_directory=True)
    details = dependency_alias.lstat()
    target = runtime if case == "wrong-target" else dependency
    binding = (dependency_alias, details.st_dev, details.st_ino, target)
    if case == "bad-alias":
        binding = (dependency_alias, details.st_dev, details.st_ino + 1, target)
    if case == "missing-alias":
        dependency_alias.unlink()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    def runtime_reader(_self, _workspace, _roots, identities):
        """Select only the workspace interpreter's linked runtime."""
        return (runtime,) if any(identity[1] == interpreter for identity in identities) else ()

    def package_roots(_selected):
        """Return managed package metadata or a selected invalid form."""
        if case == "metadata-error":
            raise OSError("package metadata unavailable")
        if case == "invalid-metadata":
            return None
        if case == "veto-direct":
            raise PermissionError("package metadata veto")
        if case == "veto-wrapped":
            raise ValueError("package metadata unavailable") from PermissionError(
                "package metadata veto"
            )
        if case == "single":
            return (runtime,), ()
        if case == "wrong-root":
            return (dependency, runtime), ()
        if case == "too-many":
            return (runtime,) * 129, ()
        return (runtime, dependency), (binding,)

    monkeypatch.setattr(type(backend), "automatic_tool_read_roots", runtime_reader, raising=False)
    monkeypatch.setattr(
        type(backend), "installed_tool_roots", staticmethod(package_roots), raising=False
    )
    monkeypatch.setattr(
        type(backend),
        "managed_tool_root",
        staticmethod(lambda root: case != "unmanaged"),
        raising=False,
    )
    if case == "bound-method":
        monkeypatch.setattr(
            type(backend), "installed_tool_roots", lambda _self, selected: package_roots(selected)
        )
        monkeypatch.setattr(type(backend), "managed_tool_root", lambda _self, _root: True)
    interaction = Mock()

    def decide(*_args, **_kwargs):
        """Retarget a reviewed package alias only in the stale-selection control."""
        if case == "stale":
            dependency_alias.unlink()
            dependency_alias.symlink_to(runtime, target_is_directory=True)
        return "deny" if case == "denied" else "approve"

    interaction.prompt.side_effect = decide
    command = "./.venv/bin/python -c pass"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    if case in {"veto-direct", "veto-wrapped", "metadata-error", "missing-alias"}:
        assert result.code == "sandbox.unavailable"
        backend.run.assert_not_called()
        return
    if case in {"denied", "stale"}:
        assert result.code == ("tool.denied" if case == "denied" else "sandbox.stale")
        backend.run.assert_not_called()
        return
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert interaction.prompt.call_count == 1
    assert (dependency in request.read_roots) is (case in {"approved", "bound-method"})
    assert (dependency_alias in request.read_aliases) is (case in {"approved", "bound-method"})
    assert dependency not in request.automatic_tool_reads


def test_runtime_package_discovery_requires_a_known_home(tmp_path, monkeypatch):
    """Package hints cannot broaden scope when the account home cannot be determined."""
    backend = Mock()
    monkeypatch.setattr(
        type(backend), "installed_tool_roots", staticmethod(lambda _: None), raising=False
    )
    monkeypatch.setattr(
        type(backend), "managed_tool_root", staticmethod(lambda _: True), raising=False
    )
    monkeypatch.setattr(
        execution_facade,
        "pwd",
        SimpleNamespace(getpwuid=Mock(side_effect=KeyError("unknown account"))),
    )

    assert execution_facade._installed_runtime_packages(backend, (object(),), tmp_path, ()) == (
        (),
        (),
    )


def test_protected_runtime_is_rejected_before_package_inspection(tmp_path, monkeypatch):
    """A forbidden runtime cannot cause host-side package metadata reads."""
    workspace = tmp_path / "workspace"
    commands = workspace / ".venv/bin"
    commands.mkdir(parents=True)
    private = tmp_path / ".ssh"
    private.mkdir()
    selected = private / "python"
    selected.write_text("fixture")
    selected.chmod(0o755)
    interpreter = commands / "python"
    interpreter.symlink_to(selected)
    details = selected.stat()
    candidate = execution_facade.RuntimeCandidate(
        (str(interpreter), interpreter, selected, details.st_dev, details.st_ino), selected
    )
    monkeypatch.setattr(execution_facade, "discover_workspace_runtimes", lambda *_: (candidate,))
    backend = Mock()
    backend.system_read_roots = ()
    resolver = Mock(side_effect=AssertionError("protected package must not be inspected"))
    monkeypatch.setattr(
        type(backend), "installed_tool_roots", staticmethod(resolver), raising=False
    )
    monkeypatch.setattr(
        type(backend), "managed_tool_root", staticmethod(lambda _: True), raising=False
    )
    command = "./.venv/bin/python -c pass"
    context = ToolContext(
        Mock(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(
        context, command, read_roots=[str(private)]
    )
    assert result.code == "sandbox.unavailable"
    resolver.assert_not_called()
    backend.run.assert_not_called()


@pytest.mark.parametrize(
    ("body", "argument", "expected"),
    [
        ('exec "$1"', "./task", True),
        ('exec "$@"', "./task", True),
        ('exec "./$1"', "task", True),
        ('exec  "$1"', "./task", True),
        ('exec\t"$@"', "./task", True),
        ('exec \t"./${1}"', "task", True),
        ('exec\n"$@"', "./task", False),
        ('exec\t"$@"; :', "./task", False),
        ('exec "$1"', "'./$VALUE/task'", True),
        ('exec "$1"', "$TASK", False),
        ('exec "$1"', "-./task", False),
        ('exec "$1"', "task", False),
        ("path-forward", "task", True),
        ("path-earlier", "task", False),
        ('exec ./task "$@"', "", True),
        ("exec ./task", "ignored", True),
        ("exec ./task --fixed=literal", "ignored", True),
        ("exec ./task $OPTION", "ignored", False),
        ("exec ./task --output=*.txt", "ignored", False),
        ('exec ./task -I -B "$@"', "ignored", True),
        ("exec './task' --flag='two words' \"$@\"", "ignored", True),
        ('exec "./task" "$@"', "ignored", True),
        ("exec './task' \"$@\"", "ignored", True),
        ('exec "./task" "${@}"', "ignored", True),
        ("exec './$VALUE/task' \"$@\"", "ignored", True),
        ('exec ./@task "$@"', "", True),
        ('exec ./tâsk "$@"', "", True),
        ('exec "" "$@"', "ignored", False),
        ('exec "-option" "$@"', "ignored", False),
        ("static-path-override", "ignored", False),
        ("forwarded-path-override", "task", False),
        ("static-direct-override", "ignored", True),
        ('exec $TASK "$@"', "ignored", False),
        ('exec ./task "$OPTION" "$@"', "ignored", False),
        ('exec ./task --output=*.txt "$@"', "ignored", False),
        ('exec "./$TASK" "$@"', "ignored", False),
        ('exec ./task "$*"', "ignored", False),
        ('exec ./task "$@"; :', "ignored", False),
        ('exec ./task "$@" > output', "ignored", False),
        ('exec "$1"; :', "./task", False),
        ('printf unrelated; exec "$1"', "./task", False),
        ("ordinary launcher", "./task", False),
        ("oversized", "./task", False),
        ("changed", "./task", False),
        ("forward-with-extra", "./task", True),
        ("legacy", "./task", True),
        ("env-shell", "./task", True),
        ("env-split-shell", "./task", True),
        ("env-safe-shell-options", "./task", True),
        ("env-glob-shell-options", "./task", True),
        ("env-multiple-shell-options", "./task", True),
        ("env-end-shell-options", "./task", True),
        ("env-explicit-shell", "./task", True),
        ("env-shadow-shell", "./task", False),
        ("env-shell-options", "./task", False),
        ("env-noexec-shell-options", "./task", False),
        ("env-after-end-shell-options", "./task", False),
        ("env-long-shell-options", "./task", False),
        ("env-shell-path-override", "./task", False),
        ("env-explicit-path-override", "./task", True),
        ("env-missing-shell", "./task", False),
        ("env-unavailable-shell", "./task", False),
        ("env-no-newline", "./task", False),
        ("direct-safe-shell-options", "./task", True),
        ("direct-multiple-shell-options", "./task", False),
        ("direct-long-shell-options", "./task", False),
        ("direct-unsafe-shell-options", "./task", False),
    ],
)
def test_forwarded_workspace_script_requests_identified_runtime_scope(
    shebang_policy_root, monkeypatch, body, argument, expected
):
    """Only a deterministic literal forwarded script can suggest a runtime read scope."""
    tmp_path, _ = shebang_policy_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    executable = installation / "actual"
    executable.write_text("engine", encoding="utf-8")
    executable.chmod(0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    interpreter = workspace / "engine"
    interpreter.symlink_to(alias / executable.name)
    script_root = (
        workspace / "$VALUE"
        if argument == "'./$VALUE/task'" or "'./$VALUE/task'" in body
        else workspace
    )
    script_root.mkdir(exist_ok=True)
    script = script_root / "task"
    script.write_text(f"#!{interpreter}\n", encoding="utf-8")
    script.chmod(0o755)
    if body in {'exec ./@task "$@"', 'exec ./tâsk "$@"'}:
        (workspace / ("@task" if "@" in body.split()[1] else "tâsk")).symlink_to(script)
    launcher = workspace / "launcher"
    launcher.write_text(
        '#!/bin/sh -eu\nexec "$1"\n'
        if body == "direct-safe-shell-options"
        else "#!/bin/sh -" + "e" * 130 + '\nexec "$1"\n'
        if body == "direct-long-shell-options"
        else '#!/bin/sh -e -u\nexec "$1"\n'
        if body == "direct-multiple-shell-options"
        else '#!/bin/sh -c\nexec "$1"\n'
        if body == "direct-unsafe-shell-options"
        else "#!/usr/bin/env fixture-shell"
        if body == "env-no-newline"
        else (
            "#!/usr/bin/env -S fixture-shell\n"
            if body == "env-split-shell"
            else "#!/usr/bin/env -S fixture-shell -eu\n"
            if body == "env-safe-shell-options"
            else "#!/usr/bin/env -S fixture-shell -f\n"
            if body == "env-glob-shell-options"
            else "#!/usr/bin/env -S fixture-shell -e -u\n"
            if body == "env-multiple-shell-options"
            else "#!/usr/bin/env -S fixture-shell -n\n"
            if body == "env-noexec-shell-options"
            else "#!/usr/bin/env -S fixture-shell -- -e\n"
            if body == "env-after-end-shell-options"
            else "#!/usr/bin/env -S fixture-shell" + " -e" * 60 + "\n"
            if body == "env-long-shell-options"
            else "#!/usr/bin/env -S fixture-shell --\n"
            if body == "env-end-shell-options"
            else "#!/usr/bin/env /bin/sh\n"
            if body in {"env-explicit-shell", "env-explicit-path-override"}
            else "#!/usr/bin/env -S fixture-shell -c\n"
            if body == "env-shell-options"
            else "#!/usr/bin/env fixture-shell\n"
        )
        + 'exec "$1"\n'
        if body.startswith("env-")
        else '#!/bin/sh\nexec "$1"\n' + " " * execution_facade._MAX_INTERPRETER_LINE_BYTES
        if body == "oversized"
        else '#!/bin/sh\nexec task "$@"\n'
        if body == "static-path-override"
        else '#!/bin/sh\nexec ./task "$@"\n'
        if body == "static-direct-override"
        else '#!/bin/sh\nexec "$1"\n'
        if body
        in {
            "forward-with-extra",
            "changed",
            "legacy",
            "path-forward",
            "path-earlier",
            "forwarded-path-override",
        }
        else f"#!/bin/sh\n{body}\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    if body == "changed":
        original_open = execution_facade.open_selected_file
        original_fstat = os.fstat
        observed = {}

        def open_launcher(root, path, identity):
            """Track the bound launcher descriptor while retaining the real safe open."""
            descriptor = original_open(root, path, identity)
            if path == launcher:
                observed[descriptor] = 0
            return descriptor

        def changed_launcher(descriptor):
            """Report changed launcher content metadata after its bounded read."""
            details = original_fstat(descriptor)
            if descriptor in observed:
                observed[descriptor] += 1
                if observed.pop(descriptor) == 2:
                    return SimpleNamespace(
                        st_mode=details.st_mode,
                        st_nlink=details.st_nlink,
                        st_size=details.st_size,
                        st_mtime_ns=details.st_mtime_ns + 1,
                        st_ctime_ns=details.st_ctime_ns,
                    )
                observed[descriptor] = 1
            return details

        monkeypatch.setattr(execution_facade, "open_selected_file", open_launcher)
        monkeypatch.setattr(execution_facade.os, "fstat", changed_launcher)
    empty = tmp_path / "empty"
    empty.mkdir()
    if body.startswith("env-") and body != "env-missing-shell":
        shell = empty / "fixture-shell"
        if body == "env-shadow-shell":
            shell.write_text("fixture interpreter", encoding="utf-8")
            shell.chmod(0o755)
        else:
            shell.symlink_to("/bin/sh")
    if body == "env-unavailable-shell":
        original_selected = execution_facade._selected_executable

        def selected(name, path_value, cwd):
            """Simulate unavailable shell lookup while retaining other selections."""
            if name == "fixture-shell":
                raise OSError("fixture shell unavailable")
            return original_selected(name, path_value, cwd)

        monkeypatch.setattr(execution_facade, "_selected_executable", selected)
    if body == "path-earlier":
        earlier = empty / "task"
        earlier.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        earlier.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(empty), str(workspace)))
        if body.startswith("path-") or body.endswith("-override")
        else str(empty),
    )
    command = f"./launcher {argument}"
    if body.endswith("-override"):
        command = f"PATH={shlex.quote(str(empty))} " + command
    if body == "forward-with-extra":
        second_alias = links / "second-alias"
        second_alias.symlink_to(installation, target_is_directory=True)
        second_interpreter = workspace / "second-engine"
        second_interpreter.symlink_to(second_alias / executable.name)
        second_script = workspace / "other"
        second_script.write_text(f"#!{second_interpreter}\n", encoding="utf-8")
        second_script.chmod(0o755)
        command += " ./other"
    backend = Mock()
    backend.system_read_roots = (Path("/bin"), Path("/usr/bin"))
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )
    if body == "legacy":
        analysis = replace(
            context.permission_manager.analyze_command(command), first_executable_arguments=()
        )
        monkeypatch.setattr(PermissionManager, "analyze_command", lambda _self, _source: analysis)

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].source == command
    assert backend.run.call_args.args[0].read_roots == ((installation,) if expected else ())
    assert backend.run.call_args.args[0].read_aliases == (
        (alias, second_alias) if body == "forward-with-extra" else (alias,) if expected else ()
    )
    assert interaction.prompt.call_count == int(expected)


@pytest.mark.parametrize(
    "case",
    [
        "direct",
        "path",
        "path-override",
        "path-override-later",
        "path-assignment",
        "path-env-reset",
        "quoted-assignment",
        "expanded",
        "external",
        "outside-relative",
        "covered",
        "deny",
        "direct-file",
        "retarget",
        "covered-retarget",
        "removed-during-approval",
        "forwarded-binary",
        "forwarded-many",
    ],
)
def test_workspace_executable_link_requests_identified_scope(tmp_path, monkeypatch, case):
    """Only a selected workspace executable can suggest its linked installation scope."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    program = installation / "actual"
    program.write_text("binary", encoding="utf-8")
    program.chmod(0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    tools = workspace / "tools"
    tools.mkdir()
    selected = tools / "fixture"
    selected.symlink_to(
        program
        if case in {"direct-file", "retarget", "covered-retarget", "removed-during-approval"}
        else alias / program.name
    )
    if case in {"forwarded-binary", "forwarded-many"}:
        launcher = workspace / "launcher"
        launcher.write_text('#!/bin/sh\nexec "$1"\n', encoding="utf-8")
        launcher.chmod(0o755)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty if case.startswith("forwarded") else tools))
    command = (
        "./launcher ./tools/fixture"
        if case.startswith("forwarded")
        else "$PROGRAM"
        if case == "expanded"
        else shlex.quote(str(alias / program.name))
        if case == "external"
        else "../links/alias/actual"
        if case == "outside-relative"
        else f"PATH={shlex.quote(str(empty))} fixture"
        if case == "path-override"
        else f":; PATH={shlex.quote(str(empty))} fixture"
        if case == "path-override-later"
        else f"PATH={shlex.quote(str(empty))}; fixture"
        if case == "path-assignment"
        else "env -i fixture"
        if case == "path-env-reset"
        else "'FOO=bar' ./tools/fixture"
        if case == "quoted-assignment"
        else "fixture"
        if case == "path"
        else "./tools/fixture"
    )
    if case == "forwarded-many":
        command += "; " + "; ".join(f"other{index}" for index in range(128))
    backend = Mock()
    backend.system_read_roots = (Path("/bin"), Path("/usr/bin"))
    backend.managed_tool_root.return_value = case in {
        "path-override",
        "path-override-later",
        "path-assignment",
    }
    if backend.managed_tool_root.return_value:
        backend.installed_tool_roots.return_value = ((installation,), ())
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "deny" if case == "deny" else "approve"
    if case in {"retarget", "covered-retarget"}:
        replacement = installation / "replacement"
        replacement.write_text("replacement", encoding="utf-8")
        replacement.chmod(0o755)

        def approve_changed_selection(*_args, **_kwargs):
            """Change the selected file while the original scope is being approved."""
            selected.unlink()
            selected.symlink_to(replacement)
            return "approve"

        interaction.prompt.side_effect = approve_changed_selection
    elif case == "removed-during-approval":

        def approve_removed_selection(*_args, **_kwargs):
            """Remove the selected executable while its scope is being approved."""
            program.unlink()
            return "approve"

        interaction.prompt.side_effect = approve_removed_selection
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(
        context,
        command,
        read_roots=[str(installation)] if case in {"covered", "covered-retarget"} else None,
    )

    if case in {"deny", "retarget", "covered-retarget", "removed-during-approval"}:
        assert result.code == ("tool.denied" if case == "deny" else "sandbox.stale")
        backend.run.assert_not_called()
        interaction.prompt.assert_called_once()
        return
    assert result["boundary"] == "sandbox"
    expected = case in {
        "direct",
        "direct-file",
        "path",
        "covered",
        "forwarded-binary",
        "forwarded-many",
    }
    assert backend.run.call_args.args[0].read_roots == ((installation,) if expected else ())
    assert backend.run.call_args.args[0].read_aliases == (
        (alias,) if expected and case != "direct-file" else ()
    )
    if case in {"path-override", "path-override-later", "path-assignment"}:
        assert backend.run.call_args.args[0].automatic_tool_reads == ()
        assert backend.run.call_args.args[0].executable_identities == ()
    assert interaction.prompt.call_count == int(expected)
    if expected and case != "covered":
        assert (
            "identified executable folder, including neighboring files"
            in (interaction.info.call_args.args[0])
        )


@pytest.mark.parametrize("change", ["removed", "replaced", "hardlinked"])
def test_identified_runtime_rechecks_selected_executable(tmp_path, monkeypatch, change):
    """A stale or newly hardlinked executable cannot establish a proposed runtime scope."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    link = workspace / "selected"
    link.symlink_to(executable)

    def candidates(*_args, **_kwargs):
        """Change the selected inode after the first candidate was inspected."""
        yield link, installation, executable
        if change == "removed":
            executable.unlink()
        elif change == "replaced":
            executable.rename(installation / "original")
            executable.write_text("replacement", encoding="utf-8")
        else:
            os.link(executable, installation / "hardlink")

    monkeypatch.setattr(execution_facade, "_executable_directory_links", candidates)
    monkeypatch.setattr(execution_facade, "is_protected_external_read", lambda _path: False)
    monkeypatch.setattr(
        execution_facade.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(tmp_path))
    )

    assert (
        execution_facade._interpreter_runtime_roots((str(link),), "", workspace, workspace, (), ())
        == ()
    )


@pytest.mark.parametrize("error", [OSError, RuntimeError, ValueError])
def test_unavailable_executable_link_does_not_create_a_scope(tmp_path, monkeypatch, error):
    """A failed host lookup contributes no alias or inferred read scope."""
    monkeypatch.setattr(execution_facade, "_selected_executable", Mock(side_effect=error("gone")))

    assert tuple(execution_facade._executable_directory_links(("fixture",), "", tmp_path)) == ()


def test_unavailable_managed_executable_does_not_create_a_grant(tmp_path, monkeypatch):
    """An unusable host lookup cannot create automatic read authority."""
    service = CommandExecutionService(Mock(), Mock())
    monkeypatch.setattr(service, "_discover_executable", Mock(side_effect=ValueError("gone")))

    assert (
        service._automatic_tool_grants(ToolContext(None, "run_command"), ("fixture",), tmp_path)
        == ()
    )


def test_invalid_automatic_tool_name_does_not_create_a_grant(tmp_path, monkeypatch):
    """A shell command that is not a valid lookup name does not infer installed authority."""
    monkeypatch.setenv("PATH", str(tmp_path))
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    command = "@fixture"
    context = ToolContext(
        Mock(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command)
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].read_roots == ()


def test_makefile_candidate_already_in_system_scope_needs_no_extra_grant(tmp_path, monkeypatch):
    """A build tool already readable as system data does not request another scope."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "system-tools"
    tools.mkdir()
    executable = tools / "fixture-tool"
    executable.write_text("fixture")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        execution_facade, "discover_makefile_commands", lambda *_: ("fixture-tool",)
    )
    original_selection = execution_facade._selected_executable

    def selected(name, path, cwd):
        """Use a platform-independent system make selection and real fixture selection."""
        if name == "/usr/bin/make":
            return Path(name), Path(name).resolve()
        return original_selection(name, path, cwd)

    monkeypatch.setattr(execution_facade, "_selected_executable", selected)
    backend = Mock()
    backend.system_read_roots = (tools,)
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kw: True, raising=False
    )
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    command = "/usr/bin/make test"
    context = ToolContext(
        Mock(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command)
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].read_roots == ()


@pytest.mark.parametrize("decision", ["approve", "deny"])
@pytest.mark.parametrize("policy_case", ["new", "default_saved", "custom_saved"])
@pytest.mark.parametrize("command", ["/usr/bin/make test", "/usr/bin/make test 2>&1 | tail -200"])
def test_makefile_candidate_requests_scoped_authority_in_original_call(
    tmp_path, monkeypatch, decision, policy_case, command
):
    """Makefile scope needs one approval unless a customized legacy ceiling needs review."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if policy_case != "new":
        policy = workspace / ".loop" / "permissions.yaml"
        policy.parent.mkdir()
        payload = (
            "version: 1\nlimits:\n  readable_roots: []\n"
            if policy_case == "custom_saved"
            else "version: 1\n"
        )
        policy.write_text(payload, encoding="utf-8")
    (workspace / "Makefile").write_text(
        ".PHONY: test\ntest:\n\tfixture-tool run\n", encoding="utf-8"
    )
    installation = tmp_path / "installation"
    installation.mkdir()
    tool = installation / "fixture-tool"
    tool.write_text("fixture", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(installation))
    monkeypatch.setattr(
        "loop.execution.runtime_discovery._system_make_selected", lambda *_args: True
    )
    backend = Mock()
    backend.system_read_roots = ()
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kw: True, raising=False
    )
    backend.installed_tool_roots.return_value = ((installation,), ())
    backend.managed_tool_root.return_value = True
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.side_effect = (
        ["review", decision] if policy_case == "custom_saved" else [decision]
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert interaction.prompt.call_count == (2 if policy_case == "custom_saved" else 1)
    if policy_case == "custom_saved":
        assert interaction.prompt.call_args_list[0].kwargs["index"] == {
            "deny": "N",
            "review": "Y",
        }
        assert (
            context.permission_manager.configuration.limit_classifications.sources["readable_roots"]
            == "user"
        )
    elif policy_case == "default_saved":
        assert (
            context.permission_manager.configuration.limit_classifications.sources["readable_roots"]
            == "legacy"
        )
    if decision == "deny":
        assert result.code == "tool.denied"
        backend.run.assert_not_called()
    else:
        assert result["boundary"] == "sandbox"
        request = backend.run.call_args.args[0]
        assert request.read_roots == (installation,)
        assert request.automatic_tool_reads == ()
        assert request.source == command
        assert "including neighboring files" in request.read_context


@pytest.mark.parametrize("decision", ["approve", "deny", "retarget", "retarget-wrapper"])
def test_timeout_child_requires_scoped_approval_and_stable_identity(
    tmp_path, monkeypatch, decision
):
    """A timeout child gains only explicitly approved, identity-bound tool authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    wrapper_dir = workspace / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "timeout"
    wrapper.write_text("fixture", encoding="utf-8")
    wrapper.chmod(0o755)
    installation = tmp_path / "installation"
    installation.mkdir()
    child = installation / "fixture-tool"
    child.write_text("fixture", encoding="utf-8")
    child.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(wrapper_dir), str(installation))))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.side_effect = lambda executable: ((executable.parent,), ())
    backend.managed_tool_root.return_value = True
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    if decision in {"retarget", "retarget-wrapper"}:

        def replace_child(*_args, **_kwargs):
            """Replace an executable while its read scope is awaiting approval."""
            selected = wrapper if decision == "retarget-wrapper" else child
            selected.rename(selected.with_name("old-" + selected.name))
            selected.write_text("replacement", encoding="utf-8")
            selected.chmod(0o755)
            return "approve"

        interaction.prompt.side_effect = replace_child
    else:
        interaction.prompt.return_value = decision
    command = "timeout 180 fixture-tool run"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        instructions_manager=InstructionsManager(
            runtime_environment=RuntimeEnvironment(workspace, None)
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command)
    assert interaction.prompt.call_count == 1, (
        result,
        backend.run.call_args.args[0].read_roots if backend.run.called else None,
    )
    if decision == "deny":
        assert result.code == "tool.denied"
        backend.run.assert_not_called()
    elif decision in {"retarget", "retarget-wrapper"}:
        assert result.code == "sandbox.stale"
        backend.run.assert_not_called()
    else:
        assert result["boundary"] == "sandbox"
        request = backend.run.call_args.args[0]
        assert request.read_roots == (installation,)
        assert request.automatic_tool_reads == ()
        assert request.source == command


@pytest.mark.parametrize(
    "command",
    [
        "timeout --preserve-status 180 fixture-tool run",
        "timeout 180 PATH=/untrusted fixture-tool run",
        "timeout $DEADLINE fixture-tool run",
    ],
)
def test_timeout_ambiguous_arguments_do_not_request_child_authority(tmp_path, monkeypatch, command):
    """Options, assignments and expansion cannot suggest an executable grant."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    wrapper_dir = workspace / "bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "timeout"
    wrapper.write_text("fixture", encoding="utf-8")
    wrapper.chmod(0o755)
    installation = tmp_path / "installation"
    installation.mkdir()
    child = installation / "fixture-tool"
    child.write_text("fixture", encoding="utf-8")
    child.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(wrapper_dir), str(installation))))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.side_effect = lambda executable: ((executable.parent,), ())
    backend.managed_tool_root.return_value = True
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        instructions_manager=InstructionsManager(
            runtime_environment=RuntimeEnvironment(workspace, None)
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command)
    assert result["boundary"] == "sandbox"
    assert installation not in backend.run.call_args.args[0].read_roots
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize(("direct", "retarget"), [(False, False), (True, False), (True, True)])
def test_makefile_prepares_selected_off_path_runtime_in_original_call(
    shebang_policy_root, monkeypatch, direct, retarget
):
    """A build recipe and its selected runtime share the same authorization and launch."""
    root, _ = shebang_policy_root
    workspace = root / "workspace"
    workspace.mkdir()
    recipe = (
        './.venv/bin/python -c "print(1)"' if direct else "PYTHONDONTWRITEBYTECODE=1 uv run pytest"
    )
    (workspace / "Makefile").write_text(f".PHONY: test\ntest:\n\t{recipe}\n")
    (workspace / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    commands = workspace / ".venv/bin"
    commands.mkdir(parents=True)
    runtime = root / "runtime"
    runtime.mkdir()
    interpreter = runtime / "engine"
    interpreter.write_text("fixture")
    interpreter.chmod(0o755)
    (commands / "python").symlink_to(interpreter)
    alternative = runtime / "other"
    alternative.write_text("other")
    alternative.chmod(0o755)
    (commands.parent / "pyvenv.cfg").write_text(
        f"home = {runtime}\nimplementation = CPython\nversion_info = 3.14.2\n"
    )
    tools = root / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("fixture")
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.runtime_discovery._system_make_selected", lambda *_args: True
    )
    backend = Mock()
    backend.system_read_roots = ()
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kw: True, raising=False
    )
    backend.managed_tool_root.return_value = False
    backend.installed_tool_roots.return_value = ((tools,), ())
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    def runtime_reader(_self, _workspace, _roots, identities):
        """Recognize only the selected workspace interpreter."""
        return (runtime,) if any(item[1] == commands / "python" for item in identities) else ()

    monkeypatch.setattr(type(backend), "automatic_tool_read_roots", runtime_reader, raising=False)
    interaction = Mock()

    def approve(*_args, **_kwargs):
        """Retarget the selected interpreter during approval when requested."""
        if retarget:
            (commands / "python").unlink()
            (commands / "python").symlink_to(alternative)
        return "approve"

    interaction.prompt.side_effect = approve
    command = "/usr/bin/make test"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    if retarget:
        assert result.code == "sandbox.stale"
        backend.run.assert_not_called()
        interaction.prompt.assert_called_once()
        return
    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert set(request.read_roots) == ({runtime} if direct else {tools, runtime})
    assert request.source == command
    assert request.shell_environment()["HOME"] == str(
        context.permission_manager.temporary_directory
    )
    assert "UV_CACHE_DIR" not in request.shell_environment()
    interaction.prompt.assert_called_once()


def test_repeated_make_recipes_share_bounded_workspace_script_discovery(tmp_path, monkeypatch):
    """Repeated literal selections do not repeatedly inspect a chained workspace launcher."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(32):
        script = workspace / f"fixture-{index}"
        child = f"./fixture-{index + 1}" if index < 31 else "/bin/true"
        script.write_text(f"#!/bin/sh\nexec {child}\n")
        script.chmod(0o755)
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n" + "\t./fixture-0\n" * 96)
    monkeypatch.setattr(
        "loop.execution.runtime_discovery._system_make_selected", lambda *_args: True
    )
    inspected = []
    original_lstat = Path.lstat

    def count_script_inspections(path, *args, **kwargs):
        """Count selected fixture metadata reads at the filesystem boundary."""
        if path.parent == workspace and path.name.startswith("fixture-"):
            inspected.append(path)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", count_script_inspections)
    backend = Mock()
    backend.system_read_roots = (Path("/usr/bin"), Path("/bin"))
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kwargs: True, raising=False
    )
    command = "/usr/bin/make test"
    context = ToolContext(
        Mock(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, command, read_only=True)
    assert result["boundary"] == "sandbox"
    assert len(inspected) <= 64
    assert len(set(inspected)) == 32


def test_unavailable_backend_does_not_request_makefile_candidates(tmp_path, monkeypatch):
    """A known unavailable backend does not request unrelated build-tool scopes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        execution_facade,
        "discover_makefile_commands",
        Mock(side_effect=AssertionError("unavailable backend inspected the makefile")),
    )
    backend = Mock()
    backend.system_read_roots = ()
    backend.capability_failure.return_value = CommandProcessResult(
        SandboxOutcome.INVALID, detail="fixture unavailable"
    )
    interaction = Mock()
    command = "/usr/bin/make test"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert result.code == "sandbox.invalid"
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()


@pytest.mark.parametrize(
    "case", ["safe", "workspace", "missing-workspace", "new-earlier", "retarget"]
)
def test_makefile_discovery_binds_bare_make_path_precedence(tmp_path, monkeypatch, case):
    """Mutable or retargeted earlier PATH entries cannot select a different build tool."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "Makefile").write_text(".PHONY: test\ntest:\n\tfixture-tool\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    tool = tools / "fixture-tool"
    tool.write_text("fixture")
    tool.chmod(0o755)
    earlier = workspace / "earlier"
    if case in {"workspace", "new-earlier"}:
        earlier.mkdir()
    if case in {"workspace", "missing-workspace", "new-earlier"}:
        path = os.pathsep.join((str(earlier), "/usr/bin", str(tools)))
    elif case == "retarget":
        path = os.pathsep.join(("/usr/bin", str(tools)))
    else:
        path = os.pathsep.join(("/usr/bin", str(tools)))
    monkeypatch.setenv("PATH", path)
    backend = Mock()
    backend.system_read_roots = (Path("/usr/bin"),)
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kw: True, raising=False
    )
    backend.installed_tool_roots.return_value = ((tools,), ())
    backend.managed_tool_root.return_value = False
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()

    def approve(*_args, **_kwargs):
        """Simulate a selected system executable being changed during authorization."""
        if case == "retarget":
            monkeypatch.setattr(execution_facade, "resolve_host_executable", lambda *_: None)
            monkeypatch.setattr(
                "loop.execution.sandbox.contracts.resolve_host_executable", lambda *_: None
            )
        elif case == "new-earlier":
            replacement = earlier / "make"
            replacement.write_text("fixture")
            replacement.chmod(0o755)
        return "approve"

    interaction.prompt.side_effect = approve
    command = "make test"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    if case in {"retarget", "new-earlier"}:
        assert result.code == "sandbox.stale"
        backend.run.assert_not_called()
    else:
        assert result["boundary"] == "sandbox"
        request = backend.run.call_args.args[0]
        assert request.read_roots == (tools,)
        assert request.executable_identities
        interaction.prompt.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [
        "missing-selection",
        "changed-selection",
        "untrusted-file",
        "untrusted-alias",
        "relative-path",
        "empty-path",
        "parent-path",
        "untrusted-directory",
        "unlisted-path",
        "path-resolution-error",
        "absolute-candidate",
        "missing-candidate",
        "invalid-candidate",
        "too-many-candidates",
        "missing-recipe-selection",
        "untrusted-recipe",
        "untrusted-recipe-error",
        "absolute-unmatched",
        "absolute-alternative-unmatched",
        "absolute-same-target-different-spelling",
        "absolute-multihop-unmatched",
        "absolute-within-runtime-unmatched",
        "absolute-parent-unmatched",
        "absolute-checker-error",
        "combined-identity-limit",
    ],
)
def test_makefile_discovery_falls_back_without_unrelated_authority(tmp_path, monkeypatch, case):
    """Ambiguous selection or candidates never create an unrelated read grant."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    make = Path("/usr/bin/make").resolve()
    recipes = ("missing-tool",)
    if case == "absolute-candidate":
        recipes = ("/usr/bin/true",)
    elif case in {
        "absolute-unmatched",
        "absolute-alternative-unmatched",
        "absolute-same-target-different-spelling",
        "absolute-multihop-unmatched",
        "absolute-within-runtime-unmatched",
        "absolute-parent-unmatched",
        "absolute-checker-error",
    }:
        recipes = ("/usr/bin/awk",)
    elif case == "invalid-candidate":
        recipes = ("x" * 128,)
    elif case == "too-many-candidates":
        recipes = tuple(f"sh -c candidate-{number}" for number in range(128))
    elif case == "combined-identity-limit":
        recipes = tuple(f"tool-{number}" for number in range(128))
    monkeypatch.setattr(execution_facade, "discover_makefile_commands", lambda *_args: recipes)
    backend = Mock()
    backend.system_read_roots = (Path("/usr/bin"),)
    checks = []

    def trusted(_self, path, *, directory=False):
        """Refuse the requested system identity at the selected validation point."""
        checks.append((path, directory))
        if case == "untrusted-recipe-error" and len(checks) > 2 and not directory:
            raise OSError("system executable changed")
        return not (
            case == "untrusted-file"
            and len(checks) == 1
            or case == "untrusted-alias"
            and len(checks) == 2
            or case == "untrusted-directory"
            and directory
            or case == "untrusted-recipe"
            and len(checks) > 2
            and not directory
        )

    monkeypatch.setattr(type(backend), "trusted_system_path", trusted, raising=False)
    selected = (Path("/usr/bin/make"), make)
    if case == "missing-selection":
        selected = None
    elif case == "changed-selection":
        selected = (Path("/usr/bin/make"), workspace / "changed")
    elif case == "unlisted-path":
        selected = (Path("/bin/make"), make)
    alternative = workspace / "alternative"
    if case in {
        "absolute-alternative-unmatched",
        "absolute-same-target-different-spelling",
        "absolute-multihop-unmatched",
        "absolute-within-runtime-unmatched",
        "absolute-checker-error",
    }:
        alternative.symlink_to(
            workspace / "resolved"
            if case == "absolute-within-runtime-unmatched"
            else "/etc/alternatives/awk"
        )
    if case == "absolute-within-runtime-unmatched":
        monkeypatch.setattr(
            type(backend), "system_executable_requires_alias", lambda *_args: False, raising=False
        )
    if case in {
        "absolute-alternative-unmatched",
        "absolute-same-target-different-spelling",
        "absolute-multihop-unmatched",
        "absolute-parent-unmatched",
    }:
        monkeypatch.setattr(
            type(backend), "system_executable_requires_alias", lambda *_args: True, raising=False
        )
    if case == "absolute-checker-error":

        def fail_alias_check(*_args):
            """A failed protected-path check cannot permit the build to start."""
            raise OSError("alias authority changed")

        monkeypatch.setattr(
            type(backend), "system_executable_requires_alias", fail_alias_check, raising=False
        )
    if case == "absolute-parent-unmatched":
        parent = workspace / "parent-link"
        parent.symlink_to("/etc/alternatives/selected", target_is_directory=True)
        alternative = parent / "awk"
    other = workspace / "other"
    if case == "absolute-same-target-different-spelling":
        other.mkdir()
        (other / "awk").symlink_to(workspace / "resolved")
    if case == "absolute-multihop-unmatched":
        middle = workspace / "middle"
        middle.symlink_to(alternative)
        alternative = middle

    def select(name, *_args):
        """Supply controlled make and recipe selections without granting their paths."""
        if name == "missing-tool" and case == "missing-recipe-selection":
            return None
        if name == "/usr/bin/awk" and case in {
            "absolute-alternative-unmatched",
            "absolute-same-target-different-spelling",
            "absolute-multihop-unmatched",
            "absolute-within-runtime-unmatched",
            "absolute-parent-unmatched",
            "absolute-checker-error",
        }:
            return alternative, workspace / "resolved"
        if name == "/usr/bin/awk" and case == "absolute-unmatched":
            return Path("/usr/bin/true"), workspace / "resolved"
        if name == "awk" and case == "absolute-same-target-different-spelling":
            return other / "awk", workspace / "resolved"
        if name == "awk" and case in {
            "absolute-unmatched",
            "absolute-alternative-unmatched",
            "absolute-multihop-unmatched",
            "absolute-within-runtime-unmatched",
            "absolute-parent-unmatched",
            "absolute-checker-error",
        }:
            return None
        return selected

    monkeypatch.setattr(execution_facade, "_selected_executable", select)
    if case == "path-resolution-error":
        original_resolve = Path.resolve

        def resolve(path, *args, **kwargs):
            """Reject the first PATH entry without affecting other path identities."""
            if path == Path("/bin"):
                raise OSError("PATH resolution failed")
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
    path_value = {
        "relative-path": "relative:/usr/bin",
        "empty-path": ":/usr/bin",
        "parent-path": "/usr/bin/../bin:/usr/bin",
        "path-resolution-error": "/bin:/usr/bin",
    }.get(case, "/usr/bin")
    monkeypatch.setenv("PATH", path_value)
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    command = "make test"
    context = ToolContext(
        interaction,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    if case in {
        "absolute-alternative-unmatched",
        "absolute-same-target-different-spelling",
        "absolute-multihop-unmatched",
        "absolute-parent-unmatched",
        "absolute-checker-error",
        "combined-identity-limit",
        "too-many-candidates",
    }:
        assert result.code == "sandbox.unsupported"
        backend.run.assert_not_called()
    elif case == "unlisted-path":
        assert result.code == "sandbox.unavailable"
        backend.run.assert_not_called()
    else:
        assert result["boundary"] == "sandbox"
        assert backend.run.call_args.args[0].read_roots == ()
        interaction.prompt.assert_not_called()


@pytest.mark.parametrize("virtual_root", [False, True])
def test_system_alias_preflight_uses_the_same_requested_roots(tmp_path, monkeypatch, virtual_root):
    """Relative and virtual requested roots reach the same preflight and authorization path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    requested = workspace / "requested"
    requested.mkdir()
    permission = PermissionManager(workspace)
    instructions = Mock()
    instructions.agents_filenames = ()
    instructions.virtual_paths = VirtualPath(
        workspace=workspace, temporary_directory=permission.temporary_directory
    )
    backend = Mock()
    backend.system_read_roots = (Path("/usr/bin"),)
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(
        type(backend), "trusted_system_path", lambda *_args, **_kwargs: True, raising=False
    )
    checker = Mock(return_value=False)
    monkeypatch.setattr(type(backend), "system_executable_requires_alias", checker, raising=False)
    monkeypatch.setattr(
        execution_facade, "discover_makefile_commands", lambda *_args: ("/usr/bin/awk",)
    )
    make = Path("/usr/bin/make").resolve()

    def select(name, *_args):
        """Keep the absolute system selection distinct from the unavailable PATH name."""
        if name == "/usr/bin/awk":
            return Path("/usr/bin/awk"), Path("/usr/bin/awk")
        if name == "awk":
            return None
        return Path("/usr/bin/make"), make

    monkeypatch.setattr(execution_facade, "_selected_executable", select)
    monkeypatch.setenv("PATH", "/usr/bin")
    context = ToolContext(
        Mock(),
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", "make test"),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permission,
        instructions_manager=instructions,
    )
    root = "$WORKSPACE_ROOT/requested" if virtual_root else os.path.relpath(requested, Path.cwd())
    result = CommandExecutionService(backend, Mock()).run_command(
        context, "make test", read_roots=[root]
    )
    assert result["boundary"] == "sandbox"
    assert checker.call_args.args[-1] == (requested,)


@pytest.mark.parametrize("absolute_path", [True, False])
def test_managed_executable_lookup_after_directory_change(tmp_path, monkeypatch, absolute_path):
    """An absolute PATH remains usable after cd, while a relative PATH is not misresolved."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "child").mkdir()
    tools = workspace / "tools"
    tools.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "fixture"
    executable.write_text("binary", encoding="utf-8")
    executable.chmod(0o755)
    (tools / "fixture").symlink_to(executable)
    monkeypatch.setenv("PATH", str(tools) if absolute_path else "tools")
    backend = Mock()
    backend.system_read_roots = ()
    backend.managed_tool_root.return_value = True
    backend.installed_tool_roots.return_value = ((installation,), ())
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    command = "cd child; fixture"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.automatic_tool_reads == ((installation,) if absolute_path else ())
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize(
    ("argument", "path_override", "shadowed", "decision", "expected"),
    [
        ("fixture-engine", False, False, "approve", True),
        ("-S fixture-engine", False, False, "approve", True),
        ("-S fixture-engine --fixture", False, False, "approve", True),
        ("@selected", False, False, "approve", True),
        ("-S @selected --fixture", False, False, "approve", True),
        ("@selected", True, False, "approve", True),
        ("fixture-engine", False, False, "deny", True),
        ("fixture-engine", True, False, "approve", False),
        ("fixture-engine", False, True, "approve", False),
        ("-i fixture-engine", False, False, "approve", False),
        ("-S PATH=/elsewhere fixture-engine", False, False, "approve", False),
        ("-S ${ENGINE}", False, False, "approve", False),
        ("-S fixture-engine ${OPTIONS}", False, False, "approve", False),
        ("-S " + "x" * 120, False, False, "approve", False),
    ],
)
def test_workspace_env_shebang_uses_only_unambiguous_inherited_selection(
    shebang_policy_root, monkeypatch, argument, path_override, shadowed, decision, expected
):
    """A literal env interpreter requests only its selected installation before launch."""
    root, _ = shebang_policy_root
    workspace = root / "workspace"
    workspace.mkdir()
    installation = root / "installation"
    installation.mkdir()
    executable = installation / "actual"
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o755)
    links = root / "links"
    links.mkdir()
    alias = links / "runtime"
    alias.symlink_to(installation, target_is_directory=True)
    tools = workspace / "tools"
    tools.mkdir()
    name = "x" * 120 if argument == "-S " + "x" * 120 else "fixture-engine"
    (tools / name).symlink_to(alias / executable.name)
    argument = argument.replace("@selected", f"./tools/{name}")
    earlier = workspace / "earlier"
    earlier.mkdir()
    if shadowed:
        first = earlier / "fixture-engine"
        first.write_text("other executable", encoding="utf-8")
        first.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(earlier), str(tools))))
    script = workspace / "task"
    script.write_text(f"#!/usr/bin/env {argument}\n", encoding="utf-8")
    script.chmod(0o755)
    command = "PATH=/other ./task" if path_override else "./task"
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = decision
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(context, command)

    if expected and decision == "deny":
        assert result.code == "tool.denied"
        backend.run.assert_not_called()
    else:
        assert result["exit_code"] == 0
        request = backend.run.call_args.args[0]
        assert request.read_roots == ((installation,) if expected else ())
        assert request.read_aliases == ((alias,) if expected else ())
    assert interaction.prompt.call_count == int(expected)


@pytest.mark.parametrize(
    "case",
    [
        "approve",
        "deny",
        "changed",
        "argument",
        "multiple",
        "first-chain",
        "first-wrapper-chain",
        "first-assignment-chain",
        "first-prefix-redirect-chain",
        "first-redirect",
        "first-quoted",
        "first-double-quoted",
        "first-double-apostrophe",
        "first-quoted-space",
        "first-single-dollar",
        "first-double-dollar",
        "first-quoted-tilde",
        "first-quoted-glob",
        "first-function",
        "first-dynamic",
        "first-tilde",
        "first-glob",
        "first-braces",
        "empty-analysis",
        "single-dynamic",
        "single-double-dollar",
        "single-quoted-dollar",
        "single-escaped-dollar",
        "single-wrapper",
        "single-redirect",
        "single-dynamic-then-literal",
        "single-literal-then-dynamic",
        "single-glob",
        "single-quoted-glob",
        "system",
        "covered",
        "workspace-requested",
        "extra-requested",
        "extra-denied",
        "extra-broad-context",
        "runtime-overlap",
        "protected",
        "home",
        "home-unavailable",
        "home-file",
        "home-foreign",
        "file-link",
        "interpreter-hardlink",
        "interpreter-protected",
        "multiple-links",
        "workspace-link",
        "interpreter-disappeared",
        "alias-error",
    ],
)
def test_direct_workspace_script_requests_identified_runtime_scope(
    shebang_policy_root, monkeypatch, case
):
    """A selected script may request its interpreter's directory, never an argument's."""
    tmp_path, private_name = shebang_policy_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / (private_name if case == "protected" else "installation")
    installation.mkdir()
    tools = workspace / "tools"
    tools.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    engine_directory = (
        installation / private_name
        if case == "interpreter-protected"
        else installation / "package"
        if case == "multiple-links"
        else installation
    )
    engine_directory.mkdir(exist_ok=True)
    executable = engine_directory / "engine"
    executable.write_text("engine", encoding="utf-8")
    executable.chmod(0o755)
    alias = workspace / "selected" if case == "workspace-link" else links / "selected"
    alias.symlink_to(installation, target_is_directory=True)
    interpreter = tools / "engine"
    if case == "multiple-links":
        (installation / "current").symlink_to(engine_directory, target_is_directory=True)
    interpreter.symlink_to(
        executable
        if case == "file-link"
        else alias / "current" / executable.name
        if case == "multiple-links"
        else alias / executable.relative_to(installation)
    )
    if case == "interpreter-hardlink":
        os.link(executable, tmp_path / "hardlink")
    if case == "interpreter-disappeared":
        original_is_dir = Path.is_dir
        removed = False

        def is_dir(path, *args, **kwargs):
            """Remove the interpreter after its directory link has been resolved."""
            nonlocal removed
            result = original_is_dir(path, *args, **kwargs)
            if path == installation and not removed:
                executable.unlink()
                removed = True
            return result

        monkeypatch.setattr(Path, "is_dir", is_dir)
    if case == "alias-error":
        original_protection = execution_facade.is_protected_external_read
        checked = []

        def protected(path):
            """Simulate a directory alias becoming unavailable during validation."""
            if path == alias:
                checked.append(path)
                raise OSError("alias unavailable")
            return original_protection(path)

        monkeypatch.setattr(execution_facade, "is_protected_external_read", protected)
    script = workspace / "task"
    if case in {
        "first-dynamic",
        "first-tilde",
        "first-single-dollar",
        "first-double-dollar",
        "first-quoted-tilde",
        "single-dynamic",
        "single-double-dollar",
        "single-quoted-dollar",
        "single-escaped-dollar",
        "single-wrapper",
        "single-redirect",
        "single-dynamic-then-literal",
        "single-literal-then-dynamic",
    }:
        directory = workspace / (
            "$NAME"
            if case
            in {
                "first-dynamic",
                "first-single-dollar",
                "first-double-dollar",
                "single-dynamic",
                "single-double-dollar",
                "single-quoted-dollar",
                "single-escaped-dollar",
                "single-wrapper",
                "single-redirect",
                "single-dynamic-then-literal",
                "single-literal-then-dynamic",
            }
            else "~"
        )
        directory.mkdir()
        script = directory / "task"
    elif case in {"first-glob", "first-quoted-glob", "single-glob", "single-quoted-glob"}:
        script = workspace / "ta?k"
    elif case == "first-braces":
        script = workspace / "{task,other}"
    elif case == "first-quoted-space":
        script = workspace / "task with space"
    elif case == "first-double-apostrophe":
        script = workspace / "task's"
    script.write_text(f"#!{interpreter}\n", encoding="utf-8")
    script.chmod(0o755)
    launcher = workspace / "launcher"
    launcher.write_text("launcher", encoding="utf-8")
    launcher.chmod(0o755)
    empty = tmp_path / "empty"
    empty.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    command = (
        "./launcher ./task"
        if case == "argument"
        else ":; ./task"
        if case == "multiple"
        else "./task && :"
        if case == "first-chain"
        else "exec ./task; :"
        if case == "first-wrapper-chain"
        else "OPTION=value ./task && :"
        if case == "first-assignment-chain"
        else ">/dev/null ./task; :"
        if case == "first-prefix-redirect-chain"
        else "./task >/dev/null; :"
        if case == "first-redirect"
        else "'./task' && :"
        if case == "first-quoted"
        else '"./task" && :'
        if case == "first-double-quoted"
        else '"./task\'s" && :'
        if case == "first-double-apostrophe"
        else "'./task with space' && :"
        if case == "first-quoted-space"
        else "'./$NAME/task' && :"
        if case == "first-single-dollar"
        else '"$NAME/task" && :'
        if case == "first-double-dollar"
        else "'~/task' && :"
        if case == "first-quoted-tilde"
        else '"./ta?k" && :'
        if case == "first-quoted-glob"
        else "./task() { :; }"
        if case == "first-function"
        else "$NAME/task && :"
        if case == "first-dynamic"
        else "~/task && :"
        if case == "first-tilde"
        else "./ta?k && :"
        if case == "first-glob"
        else "./{task,other} && :"
        if case == "first-braces"
        else "'"
        if case == "empty-analysis"
        else "$NAME/task"
        if case == "single-dynamic"
        else '"$NAME/task"'
        if case == "single-double-dollar"
        else "'$NAME/task'"
        if case == "single-quoted-dollar"
        else r"\$NAME/task"
        if case == "single-escaped-dollar"
        else "exec '$NAME/task'"
        if case == "single-wrapper"
        else ">/dev/null '$NAME/task'"
        if case == "single-redirect"
        else "$NAME/task; '$NAME/task'"
        if case == "single-dynamic-then-literal"
        else "'$NAME/task'; $NAME/task"
        if case == "single-literal-then-dynamic"
        else "./ta?k"
        if case == "single-glob"
        else "'./ta?k'"
        if case == "single-quoted-glob"
        else "./task"
    )
    backend = Mock()
    backend.system_read_roots = (installation,) if case == "system" else ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    if case == "runtime-overlap":
        runtime = installation / "other-runtime"
        runtime.mkdir()
        monkeypatch.setattr(
            type(backend), "automatic_tool_read_roots", lambda *_args: (runtime,), raising=False
        )
    interaction = Mock()

    def decide(*_args, **_kwargs):
        """Deny or change the alias before the sandbox can launch."""
        if case == "changed":
            replacement = tmp_path / "replacement"
            replacement.mkdir()
            alias.unlink()
            alias.symlink_to(replacement, target_is_directory=True)
        return "deny" if case in {"deny", "extra-denied"} else "approve"

    interaction.prompt.side_effect = decide
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )
    if case == "extra-broad-context":
        monkeypatch.setattr(
            PermissionManager,
            "describe_sandbox_reads",
            staticmethod(lambda *_args: "Read a broad area outside the workspace"),
        )
    if case == "home-unavailable":
        unavailable = tmp_path / "looping-home"
        unavailable.symlink_to(unavailable, target_is_directory=True)
        monkeypatch.setattr(
            execution_facade.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(unavailable))
        )
    if case in {"home", "home-file", "home-foreign"}:
        home = executable if case == "home-file" else installation
        monkeypatch.setattr(
            execution_facade.pwd, "getpwuid", lambda _uid: SimpleNamespace(pw_dir=str(home))
        )
    if case == "home-foreign":
        original_stat = Path.stat

        def stat_foreign(path, *args, **kwargs):
            """Present the account home as owned by another user."""
            details = original_stat(path, *args, **kwargs)
            if path == installation:
                return SimpleNamespace(st_mode=details.st_mode, st_uid=os.getuid() + 1)
            return details

        monkeypatch.setattr(Path, "stat", stat_foreign)

    requested = (
        [str(installation)]
        if case in {"covered", "alias-error"}
        else [str(workspace)]
        if case == "workspace-requested"
        else [str(extra)]
        if case in {"extra-requested", "extra-denied", "extra-broad-context"}
        else None
    )
    result = CommandExecutionService(backend, Mock()).run_command(
        context, command, read_roots=requested
    )

    if case in {"deny", "extra-denied", "changed"}:
        assert result.code == ("sandbox.stale" if case == "changed" else "tool.denied")
        backend.run.assert_not_called()
        interaction.prompt.assert_called_once()
    elif case in {
        "argument",
        "multiple",
        "first-function",
        "first-dynamic",
        "first-double-dollar",
        "first-tilde",
        "first-glob",
        "first-braces",
        "empty-analysis",
        "single-dynamic",
        "single-double-dollar",
        "single-dynamic-then-literal",
        "single-glob",
        "system",
        "protected",
        "home",
        "home-unavailable",
        "home-file",
        "home-foreign",
        "interpreter-hardlink",
        "interpreter-protected",
        "interpreter-disappeared",
    }:
        assert result["exit_code"] == 0
        assert backend.run.call_args.args[0].read_roots == ()
        interaction.prompt.assert_not_called()
        if case == "interpreter-disappeared":
            assert removed
    elif case == "alias-error":
        assert result["exit_code"] == 0
        assert backend.run.call_args.args[0].read_roots == (installation,)
        assert backend.run.call_args.args[0].read_aliases == ()
        interaction.prompt.assert_called_once()
        assert checked == [alias]
    elif case == "runtime-overlap":
        assert result["exit_code"] == 0
        assert backend.run.call_args.args[0].read_roots == (runtime,)
        interaction.prompt.assert_called_once()
    else:
        assert result["exit_code"] == 0
        request = backend.run.call_args.args[0]
        assert request.read_roots == (
            (extra, installation)
            if case in {"extra-requested", "extra-broad-context"}
            else (installation,)
        )
        assert request.read_aliases == (() if case in {"workspace-link", "file-link"} else (alias,))
        interaction.prompt.assert_called_once()
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert scope["read_roots"] == (
            [str(extra), str(installation)]
            if case in {"extra-requested", "extra-broad-context"}
            else [str(installation)]
        )
        if case in {"approve", "extra-requested"}:
            assert "including neighboring files" in interaction.info.call_args.args[0]
        if case == "extra-requested":
            assert "additional folder" in interaction.info.call_args.args[0]
        if case == "extra-broad-context":
            assert "Read a broad area outside the workspace" in interaction.info.call_args.args[0]
            assert "including neighboring files" in interaction.info.call_args.args[0]


def test_indirect_alias_order_preserves_a_remembered_scope(tmp_path, monkeypatch):
    """Reuse unchanged approvals and require review when the available alias scope changes."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    lookup = tmp_path / "lookup"
    lookup.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    for name in ("second", "first"):
        package = installation / name
        package.mkdir()
        executable = package / "actual"
        executable.write_text("tool", encoding="utf-8")
        executable.chmod(0o755)
        alias = links / name
        alias.symlink_to(package, target_is_directory=True)
        (lookup / name).symlink_to(alias / executable.name)
    launcher = workspace / "launcher"
    launcher.write_text("launcher", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(lookup))
    backend = Mock()
    backend.system_read_roots = ()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "session"
    manager = PermissionManager(workspace)
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "./launcher"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction, "run_command", operations=(operation,), permission_manager=manager
    )
    service = CommandExecutionService(backend, Mock())

    first = service.run_command(context, "./launcher", read_roots=[str(installation)])
    original_scandir = os.scandir

    def reversed_scandir(path):
        """Reverse only the selected PATH directory without changing its contents."""
        entries = original_scandir(path)
        if isinstance(path, int) or Path(path) != lookup:
            return entries
        with entries:
            return nullcontext(iter(reversed(list(entries))))

    monkeypatch.setattr(execution_facade.os, "scandir", reversed_scandir)
    second = service.run_command(context, "./launcher", read_roots=[str(installation)])
    extra = installation / "extra"
    extra.mkdir()
    executable = extra / "actual"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    alias = links / "extra"
    alias.symlink_to(extra, target_is_directory=True)
    (lookup / "extra").symlink_to(alias / executable.name)
    third = service.run_command(context, "./launcher", read_roots=[str(installation)])

    assert first["boundary"] == second["boundary"] == third["boundary"] == "sandbox"
    assert backend.run.call_args_list[0].args[0].read_aliases == (
        backend.run.call_args_list[1].args[0].read_aliases
    )
    assert len(backend.run.call_args_list[2].args[0].read_aliases) == 3
    assert interaction.prompt.call_count == 2


def test_selected_executable_alias_uses_an_existing_grant(tmp_path, monkeypatch):
    """An identity-bound executable grant also supplies its already-approved alias scope."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lookup = workspace / "lookup"
    lookup.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    executable = installation / "actual"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    (lookup / "fixture-tool").symlink_to(alias / executable.name)
    monkeypatch.setenv("PATH", str(lookup))
    backend = Mock()
    backend.system_read_roots = ()
    backend.managed_tool_root.return_value = False
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, Mock())
    grant = service.resolve_executable(
        ToolContext(None, "resolve_executable"), "fixture-tool", str(workspace)
    )
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "fixture-tool"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = service.run_command(context, "fixture-tool", executable_grant=grant["grant_reference"])

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_roots == (installation,)
    assert request.read_aliases == (alias,)
    assert interaction.prompt.call_count == 1


def test_public_command_budget_starts_after_sandbox_approval(tmp_path, monkeypatch):
    """The public facade preserves a full execution budget after a slow permission decision."""
    now = [100.0]
    monkeypatch.setattr("loop.execution.facade.time.monotonic", lambda: now[0])
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()

    def approve(*_args, **_kwargs):
        """Return a user decision after the initial deadline has elapsed."""
        now[0] = 543.0
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
        cancellation_event=Event(),
    )
    result = service.run_command(context, "printf text >victim")
    assert result["boundary"] == "sandbox"
    assert backend.run.call_args.args[0].deadline == 4203.0
    assert backend.run.call_args.args[0].runtime_timeout == 3600.0
    assert backend.run.call_args.kwargs["cancellation_event"] is context.cancellation_event
    host.run_host_command.assert_not_called()


def test_facade_forwards_capability_explanation_to_user(tmp_path):
    """The execution facade retains the model explanation through sandbox authorization."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf safe"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    manager = PermissionManager(tmp_path)
    manager.set_limit("deny_private_networks", False)
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=manager,
    )

    result = CommandExecutionService(backend, Mock()).run_command(
        context, "printf safe", network=True, reason="Fetch the package index"
    )

    assert result["boundary"] == "sandbox"
    assert 'Agent explanation: "Fetch the package index"' in interaction.info.call_args.args[0]
    assert backend.run.call_args.args[0].network is True


def test_facade_reports_unenforceable_network_limit_without_prompt(tmp_path):
    """A broad network request returns policy guidance before any native or human attempt."""
    backend = Mock()
    interaction = Mock()
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf safe"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(tmp_path),
    )

    result = CommandExecutionService(backend, Mock()).run_command(
        context, "printf safe", network=True
    )

    assert result.code == "sandbox.unavailable"
    assert "private-network limit" in result.detail
    interaction.prompt.assert_not_called()
    backend.run.assert_not_called()


def test_facade_binds_scoped_external_write_and_rejects_broad_control_root(tmp_path):
    """Public dispatch approves one external directory and refuses control path overlap."""
    external = tmp_path.parent / "external-write"
    external.mkdir()
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf safe"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(tmp_path, protected_host_roots=(tmp_path / ".loop",)),
    )
    service = CommandExecutionService(backend, Mock())

    result = service.run_command(context, "printf safe", write_roots=[str(external)])

    assert result["boundary"] == "sandbox"
    assert external in backend.run.call_args.args[0].write_roots
    assert str(external) in interaction.info.call_args.args[0]

    denied = service.run_command(context, "printf safe", write_roots=[str(tmp_path.parent)])
    assert denied.code == "sandbox.unavailable"
    assert "external write root" in denied.detail.lower()
    read_only = service.run_command(
        context, "printf safe", read_only=True, write_roots=[str(external)]
    )
    assert read_only.code == "sandbox.unavailable"
    assert "Read-only workspace commands" in read_only.detail
    backend.run.assert_called_once()


@pytest.mark.parametrize("interactive", [False, True])
def test_facade_reports_hard_saved_root_before_asking_for_scope(tmp_path, interactive):
    """Headless requests and declined legacy migration keep external reads hard-denied."""
    path = tmp_path / ".loop" / "permissions.yaml"
    path.parent.mkdir()
    path.write_text("version: 1\nlimits:\n  readable_roots: []\n", encoding="utf-8")
    external = tmp_path.parent / "legacy-external"
    external.mkdir()
    backend = Mock()
    interaction = Mock() if interactive else None
    if interaction is not None:
        interaction.prompt.return_value = "deny"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "printf safe"),
            cwd=str(tmp_path),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(tmp_path),
    )

    result = CommandExecutionService(backend, Mock()).run_command(
        context, "printf safe", read_roots=[str(external)]
    )

    assert result.code == "sandbox.unavailable"
    assert "readable_roots limit is hard" in result.detail
    if interaction is not None:
        interaction.prompt.assert_called_once()
    backend.run.assert_not_called()


def test_empty_executable_grant_does_not_block_sandbox_execution(tmp_path):
    """Treat an empty optional tool reference as absent without granting host reads."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"),
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

    result = service.run_command(context, ":", executable_grant="")

    assert result["boundary"] == "sandbox"
    request = backend.run.call_args.args[0]
    assert request.read_roots == ()
    backend.run.assert_called_once()


def test_execution_service_closes_backend_lifecycle():
    """The facade releases its selected backend through the shared lifecycle contract."""
    backend = Mock()
    CommandExecutionService(backend, Mock()).close()
    backend.close.assert_called_once()


def test_refresh_backend_reselects_authority_and_closes_previous_backend(monkeypatch):
    """Setup changes replace the cached sandbox backend and close its prior resources."""
    previous = Mock()
    replacement = Mock()
    select = Mock(return_value=replacement)
    monkeypatch.setattr(execution_facade, "select_sandbox_backend", select)
    service = CommandExecutionService(previous, Mock())

    service.refresh_backend()
    service.close()

    select.assert_called_once_with()
    previous.close.assert_called_once_with()
    replacement.close.assert_called_once_with()


def test_refresh_waits_before_admitting_a_new_terminal_command(monkeypatch):
    """A command entering during native replacement sees the new qualified backend."""
    old = Mock(capabilities=frozenset())
    replacement = Mock(capabilities=frozenset({SandboxCapability.TERMINAL}))
    service = CommandExecutionService(old, Mock())
    selecting = Event()
    release = Event()
    results = []

    def select():
        """Hold native selection until another caller begins command admission."""
        selecting.set()
        assert release.wait(5)
        return replacement

    monkeypatch.setattr(execution_facade, "select_sandbox_backend", select)
    refresher = Thread(target=service.refresh_backend)
    refresher.start()
    assert selecting.wait(5)
    command = Thread(
        target=lambda: results.append(
            service.run_command(ToolContext(None, "run_command"), ":", interactive=True)
        )
    )
    command.start()
    release.set()
    refresher.join(timeout=5)
    command.join(timeout=5)

    assert not refresher.is_alive() and not command.is_alive()
    assert len(results) == 1 and results[0].code == "command_job.unavailable"
    old.close.assert_called_once_with()
    service.close()


def test_active_trusted_search_keeps_backend_open_through_refresh_and_close(monkeypatch, tmp_path):
    """A native search retains its backend until it finishes after a shutdown request."""
    backend = Mock()
    backend.system_read_roots = (tmp_path,)
    executable = tmp_path / "rg"
    executable.write_text("installed", encoding="utf-8")
    service = CommandExecutionService(backend, Mock())
    capture = object()
    logged = Mock()
    monkeypatch.setattr(execution_facade, "ripgrep_path", lambda: str(executable))
    monkeypatch.setattr(execution_facade, "JOB_CLOSE_SECONDS", 0)
    monkeypatch.setattr(execution_facade, "log_problem", logged)

    def run(*_args):
        """Prove refresh and close cannot retire the backend during native search."""
        with pytest.raises(RuntimeError, match="active"):
            service.refresh_backend()
        service.close()
        backend.close.assert_not_called()
        return capture

    backend.run_read_only_argv.side_effect = run
    assert service.run_trusted_search((), (), 7.0) is capture
    assert logged.call_args.args[1].code == "command_job.cleanup_uncertain"
    backend.close.assert_called_once_with()
    assert service.run_trusted_search((), (), 7.0) is None
    assert service.run_command(ToolContext(None, "run_command"), ":").code == (
        "sandbox.unavailable"
    )
    with pytest.raises(RuntimeError, match="closed"):
        service.resolve_executable(ToolContext(None, "resolve_executable"), "rg")
    with pytest.raises(RuntimeError, match="active"):
        service.refresh_backend()
    service.close()
    backend.close.assert_called_once_with()


def test_deferred_backend_close_failure_preserves_native_search_result(monkeypatch, tmp_path):
    """A delayed close error is reported without replacing the completed native result."""
    backend = Mock()
    backend.system_read_roots = (tmp_path,)
    backend.close.side_effect = [OSError("eject failed"), None]
    executable = tmp_path / "rg"
    executable.write_text("installed", encoding="utf-8")
    service = CommandExecutionService(backend, Mock())
    logged = Mock()
    monkeypatch.setattr(execution_facade, "ripgrep_path", lambda: str(executable))
    monkeypatch.setattr(execution_facade, "JOB_CLOSE_SECONDS", 0)
    monkeypatch.setattr(execution_facade, "log_problem", logged)

    def run(*_args):
        """Request shutdown while the native helper still holds its backend lease."""
        service.close()
        return "searched"

    backend.run_read_only_argv.side_effect = run
    assert service.run_trusted_search((), (), 7.0) == "searched"
    assert [call.args[1].code for call in logged.call_args_list] == [
        "command_job.cleanup_uncertain",
        "sandbox.backend_cleanup_failed",
    ]
    service.close()
    assert backend.close.call_count == 2


def test_close_waits_for_native_search_lease_before_retiring_backend(monkeypatch, tmp_path):
    """Shutdown waits for an admitted native search before closing its backend."""
    backend = Mock()
    backend.system_read_roots = (tmp_path,)
    executable = tmp_path / "rg"
    executable.write_text("installed", encoding="utf-8")
    service = CommandExecutionService(backend, Mock())
    started = Event()
    release = Event()
    closing = Event()
    results = []
    monkeypatch.setattr(execution_facade, "ripgrep_path", lambda: str(executable))
    monkeypatch.setattr(command_jobs.CommandJobs, "close", lambda _self: closing.set() or True)

    def run(*_args):
        """Hold the native backend until the test releases the selected search."""
        started.set()
        assert release.wait(5)
        return "searched"

    backend.run_read_only_argv.side_effect = run
    search = Thread(target=lambda: results.append(service.run_trusted_search((), (), 7.0)))
    search.start()
    assert started.wait(5)
    shutdown = Thread(target=service.close)
    shutdown.start()
    assert closing.wait(5)
    backend.close.assert_not_called()
    release.set()
    search.join(timeout=5)
    shutdown.join(timeout=5)

    assert not search.is_alive() and not shutdown.is_alive()
    assert results == ["searched"]
    backend.close.assert_called_once_with()


def test_automatic_executable_identities_bind_one_live_path_file(tmp_path):
    """A safe PATH executable is bound by file identity without granting its directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "uv"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()

    identities = _automatic_executable_identities(
        ("uv", "missing", "uv", "tools/uv"), str(tools), workspace
    )

    assert identities == (("uv", executable, executable, details.st_dev, details.st_ino),)


def test_automatic_executable_identities_enforce_backend_owner_authority(tmp_path):
    """An executable owned outside the backend's trusted IDs cannot gain automatic access."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    assert (
        _automatic_executable_identities(("tool",), str(tools), workspace, trusted_owners=(-1,))
        == ()
    )


def test_run_command_binds_an_unmanaged_path_tool_as_one_exact_file(tmp_path, monkeypatch):
    """Each platform binds exact tool access without inferring directory reads or a custom mask."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    monkeypatch.setenv("PATH", str(tools))
    backend = Mock()
    backend.system_read_roots = ()
    backend.installed_tool_roots.side_effect = lambda path: ((path.parent,), ())
    backend.managed_tool_root.return_value = False
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "tool --version"),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = service.run_command(context, "tool --version", cwd=str(workspace))

    request = backend.run.call_args.args[0]
    assert result["exit_code"] == 0
    assert request.executable_identities == (
        ("tool", executable, executable, details.st_dev, details.st_ino),
    )
    assert request.read_roots == ()
    assert request.automatic_tool_reads == ()
    assert request.file_creation_mask == 0o022


@pytest.mark.parametrize("decision", ["approve", "deny", "unselected", "changed"])
@pytest.mark.parametrize(
    "launcher_name", ["fixture-builder", "fixture-analyzer", "fixture-compiler"]
)
def test_run_command_requires_approval_for_selected_workspace_interpreter_runtime(
    tmp_path, monkeypatch, decision, launcher_name
):
    """Arbitrary launchers need a selected interpreter before its runtime is reviewed."""
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    venv_bin = workspace / ".venv/bin"
    venv_bin.mkdir(parents=True)
    runtime_parent = home / "runtime-store"
    runtime_parent.mkdir()
    runtime_alias = runtime_parent / "cpython-3.14"
    runtime_root = runtime_parent / "cpython-3.14.1"
    (runtime_root / "bin").mkdir(parents=True)
    (runtime_root / "lib/python3.14").mkdir(parents=True)
    interpreter = runtime_root / "bin/python3.14"
    interpreter.write_text("interpreter", encoding="utf-8")
    interpreter.chmod(0o755)
    runtime_alias.symlink_to(runtime_root, target_is_directory=True)
    (venv_bin / "python").symlink_to(runtime_alias / "bin/python3.14")
    (venv_bin.parent / "pyvenv.cfg").write_text(
        f"home = {runtime_alias / 'bin'}\nimplementation = CPython\nversion_info = 3.14\n",
        encoding="utf-8",
    )
    unrelated_entry = venv_bin / "other-command"
    unrelated_entry.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    unrelated_entry.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    launcher = tools / launcher_name
    launcher.write_text("launcher", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", f"{venv_bin}{os.pathsep}{tools}")
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    monkeypatch.setattr("loop.execution.sandbox.linux.backend.runtime_read_roots", lambda: ())
    backend = LinuxNativeBackend(tmp_path / "management.json")
    monkeypatch.setattr(type(backend), "file_creation_mask", 0o077)
    backend.capability_failure = Mock(return_value=None)
    backend.run = Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    service = CommandExecutionService(backend, Mock())
    command = f"{launcher_name} execute fixture-task"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    interaction = Mock()

    def decide(*_args, **_kwargs):
        """Change the interpreter during review only for the stale-identity control."""
        if decision == "changed":
            replacement = runtime_root / "bin/replacement"
            replacement.write_text("replacement", encoding="utf-8")
            replacement.chmod(0o755)
            (venv_bin / "python").unlink()
            (venv_bin / "python").symlink_to(replacement)
            return "approve"
        return decision

    interaction.prompt.side_effect = decide
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    lookup = (
        service.resolve_executable(context, "python", cwd=str(workspace))
        if decision != "unselected"
        else None
    )
    result = service.run_command(
        context,
        command,
        cwd=str(workspace),
        executable_grant=lookup["grant_reference"] if lookup is not None else None,
    )

    if decision == "unselected":
        request = backend.run.call_args.args[0]
        assert result["exit_code"] == 0
        assert request.read_roots == () and request.automatic_tool_reads == ()
        interaction.prompt.assert_not_called()
    elif decision == "approve":
        interaction.prompt.assert_called_once()
        request = backend.run.call_args.args[0]
        assert result["exit_code"] == 0
        assert set(request.read_roots) == {runtime_root, interpreter.parent}
        assert request.automatic_tool_reads == ()
        assert request.file_creation_mask == 0o077
        scope = json.loads(
            interaction.info.call_args.args[0].split("\nScope: ", 1)[1].split("\n", 1)[0]
        )
        assert str(runtime_root) in scope["read_roots"]
    elif decision == "deny":
        interaction.prompt.assert_called_once()
        assert result.code == "tool.denied"
        backend.run.assert_not_called()
    else:
        interaction.prompt.assert_called_once()
        assert result.code == "sandbox.stale"
        backend.run.assert_not_called()


def test_run_command_ignores_unidentified_environment_metadata(tmp_path, monkeypatch):
    """An unrelated workspace executable keeps its identity without gaining runtime reads."""
    workspace = tmp_path / "project"
    tools = workspace / "bin"
    tools.mkdir(parents=True)
    executable = tools / "fixture-processor"
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o755)
    (workspace / "pyvenv.cfg").symlink_to(workspace / "unrelated")
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr("loop.execution.sandbox.linux.backend.runtime_read_roots", lambda: ())
    monkeypatch.setattr(LinuxNativeBackend, "file_creation_mask", 0o077)
    backend = LinuxNativeBackend(tmp_path / "management.json")
    backend.capability_failure = Mock(return_value=None)
    backend.run = Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    interaction = Mock()
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", executable.name),
            cwd=str(workspace),
            boundary=ProcessBoundary.SANDBOXED,
        ),
    )
    context = ToolContext(
        interaction,
        "run_command",
        operations=(operation,),
        permission_manager=PermissionManager(workspace),
    )

    result = CommandExecutionService(backend, Mock()).run_command(
        context, executable.name, cwd=str(workspace), read_only=True
    )

    assert result["exit_code"] == 0
    request = backend.run.call_args.args[0]
    assert request.read_roots == ()
    assert request.executable_identities[0][:3] == (executable.name, executable, executable)
    interaction.prompt.assert_not_called()


@pytest.mark.parametrize("unsafe_mode", ["root", "file"])
def test_automatic_executable_identities_reject_group_or_world_writable_paths(
    tmp_path, unsafe_mode
):
    """A writable PATH directory or executable cannot receive automatic launch authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o757 if unsafe_mode == "file" else 0o755)
    if unsafe_mode == "root":
        tools.chmod(0o777)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


def test_automatic_executable_identities_accept_primary_group_writable_path_root(
    tmp_path, monkeypatch
):
    """A group-writable PATH directory owned through the primary group can pin an exact file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    tools.chmod(0o775)
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    details = executable.stat()
    monkeypatch.setattr("loop.execution.facade.os.getgid", lambda: tools.stat().st_gid)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == (
        ("tool", executable, executable, details.st_dev, details.st_ino),
    )


def test_automatic_executable_identities_rejects_foreign_group_writable_path_root(
    tmp_path, monkeypatch
):
    """A group-writable PATH directory is unsafe when its group is not the primary group."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    tools.chmod(0o775)
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("loop.execution.facade.os.getgid", lambda: tools.stat().st_gid + 1)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


@pytest.mark.parametrize("foreign", ["directory", "file"])
def test_automatic_executable_identities_reject_foreign_owners(tmp_path, monkeypatch, foreign):
    """Foreign-owned search roots and executables cannot receive automatic launch authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    original_stat = Path.stat

    def foreign_owner(path, *args, **kwargs):
        """Report an untrusted owner for the selected search root or executable inode."""
        details = original_stat(path, *args, **kwargs)
        if path == (tools if foreign == "directory" else executable):
            return type(details)(
                (
                    details.st_mode,
                    details.st_ino,
                    details.st_dev,
                    details.st_nlink,
                    details.st_uid + 1,
                    *details[5:],
                )
            )
        return details

    monkeypatch.setattr(Path, "stat", foreign_owner)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


@pytest.mark.parametrize("private_target", [False, True])
def test_automatic_executable_identities_reject_private_paths(tmp_path, private_target):
    """Private components in either PATH spelling or target cannot be auto-executed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / ("tools" if private_target else ".ssh/bin")
    tools.mkdir(parents=True)
    executable = tools / "tool"
    if private_target:
        private = tools / ".ssh"
        private.mkdir()
        target = private / "tool"
        target.write_text("tool", encoding="utf-8")
        target.chmod(0o755)
        executable.symlink_to(target)
    else:
        executable.write_text("tool", encoding="utf-8")
        executable.chmod(0o755)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


def test_automatic_executable_identities_reject_unbound_targets_and_hardlinks(tmp_path):
    """Targets outside PATH roots and files with aliases cannot get exact-file grants."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    target = tmp_path / "target"
    target.write_text("tool", encoding="utf-8")
    target.chmod(0o755)
    external_link = tools / "external"
    external_link.symlink_to(target)
    assert _automatic_executable_identities(("external",), str(tools), workspace) == ()

    hardlinked = tools / "hardlinked"
    hardlinked.hardlink_to(target)
    assert _automatic_executable_identities(("hardlinked",), str(tools), workspace) == ()


def test_automatic_executable_identities_ignore_lookup_and_metadata_errors(tmp_path, monkeypatch):
    """A disappearing or unreadable PATH candidate fails closed without aborting binding."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)

    monkeypatch.setattr(
        "loop.execution.facade.resolve_host_executable", Mock(side_effect=OSError("changed"))
    )
    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()

    monkeypatch.setattr(
        "loop.execution.facade.resolve_host_executable", lambda *_args: (executable, executable)
    )
    assert execution_facade.path_search_roots({"PATH": str(tools)}, workspace) == ((tools, tools),)
    monkeypatch.setattr("loop.execution.facade.path_search_roots", lambda *_args: ((tools, tools),))
    original_stat = Path.stat

    def failing_stat(path, *args, **kwargs):
        """Fail only when the discovered PATH directory is revalidated."""
        if path == tools:
            raise OSError("changed")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


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


def test_trusted_search_reports_invalid_runtime_authority(monkeypatch, tmp_path):
    """A changed native runtime cannot silently switch search to the Python matcher."""
    executable = tmp_path / "rg"
    executable.write_text("installed", encoding="utf-8")
    backend = Mock()
    monkeypatch.setattr(
        type(backend),
        "system_read_roots",
        PropertyMock(side_effect=ValueError("changed runtime")),
        raising=False,
    )
    monkeypatch.setattr("loop.execution.facade.ripgrep_path", lambda: str(executable))

    with pytest.raises(RuntimeError, match="Native search runtime authority is invalid"):
        CommandExecutionService(backend, Mock()).run_trusted_search((), (), 7.0)
    backend.run_read_only_argv.assert_not_called()


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
    backend.installed_tool_roots.return_value = None
    assert service.run_trusted_search((), (), 7.0) is None
    backend.installed_tool_roots.side_effect = OSError("helper unavailable")
    with pytest.raises(OSError, match="helper unavailable"):
        service.run_trusted_search((), (), 7.0)


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


def test_command_preparation_failure_before_path_binding_has_no_native_exception_text(tmp_path):
    """Early preparation errors cannot disclose native paths before a redactor exists."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())
    missing = tmp_path / "synthetic-home-canary" / "missing"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(missing), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    result = service.run_command(
        ToolContext(
            None,
            "run_command",
            operations=(operation,),
            permission_manager=PermissionManager(tmp_path),
        ),
        ":",
    )
    assert result.code == "sandbox.unavailable"
    assert result.detail == "Command preparation failed."
    backend.run.assert_not_called()


@pytest.mark.parametrize("native_output", [False, True])
def test_command_output_is_preserved_in_preview_and_continuation_storage(
    tmp_path, monkeypatch, native_output
):
    """Command streams preserve path-like text in previews and cached continuations."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    scratch = tmp_path / "Library/Application Support/synthetic é ;$'"
    scratch.mkdir(parents=True, mode=0o700)
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    variants = [
        str(scratch),
        shlex.quote(str(scratch)),
        json.dumps(str(scratch), ensure_ascii=True)[1:-1],
        str(scratch.parent),
        f"{workspace}-extra/file",
        f"https://example.test{workspace}/file",
    ]
    raw = "\r\n".join(variants) * 300
    backend = Mock()
    if native_output:
        raw += "/native-private-scratch/result\n" * 1000
        monkeypatch.setattr(
            type(backend),
            "output_path_roots",
            Mock(return_value=(("/native-private-scratch", str(scratch)),)),
            raising=False,
        )
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout=raw,
        stderr=raw,
    )
    command = '''printf '%s' '/tmp/data'; cat "$TMPDIR/report"'''
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", command), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    service = CommandExecutionService(backend, Mock())
    result = service.run_command(
        ToolContext(
            None,
            "run_command",
            instructions_manager=instructions,
            operations=(operation,),
            permission_manager=PermissionManager(workspace),
        ),
        command,
    )
    assert backend.run.call_args.args[0].execution_source == command
    for name in ("stdout", "stderr"):
        stream = result[name]
        assert raw.startswith(stream["content"])
        assert stream["decoding_loss"] == "none"
        cached = cached_path(stream["handle"])[0].read_bytes().decode("utf-8")
        assert cached == raw


@pytest.mark.parametrize("discarded", [0, 1, None])
def test_large_command_output_includes_a_truthful_final_excerpt(tmp_path, discarded):
    """Complete long streams expose their ending without losing the original continuation."""
    output = "prefix\n" + "é" * 5000 + "\n72 passed\n"
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout=output,
        stdout_discarded=discarded,
    )
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    result = CommandExecutionService(backend, Mock()).run_command(
        ToolContext(
            None,
            "run_command",
            operations=(operation,),
            permission_manager=PermissionManager(tmp_path),
        ),
        ":",
    )
    stream = result["stdout"]
    encoded = output.encode()
    assert cached_path(stream["handle"])[0].read_bytes() == encoded
    if discarded == 0:
        tail = stream["tail"]
        assert tail["content"].endswith("72 passed\n")
        assert tail["end_byte"] == len(encoded)
        assert encoded[tail["start_byte"] : tail["end_byte"]].decode() == tail["content"]
        assert tail["start_byte"] >= stream["included_bytes"]
    else:
        assert "tail" not in stream


@pytest.mark.parametrize("overflow", [False, True])
def test_scratch_shell_script_prepares_its_selected_managed_tool(tmp_path, monkeypatch, overflow):
    """A session script and its literal tool are prepared in the original sandbox call."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "managed-tools"
    tools.mkdir()
    executable = tools / "fixture-tool"
    executable.write_text("fixture")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(tools), os.defpath)))
    permissions = PermissionManager(workspace)
    if overflow:
        analyze = permissions.analyze_command

        def many_candidates(source):
            analysis = analyze(source)
            if "fixture-tool" in source:
                return replace(
                    analysis,
                    inherited_path_executables=tuple(
                        (f"tool-{index}", True)
                        for index in range(execution_facade.constants.MAX_COMMAND_PATH_COUNT + 1)
                    ),
                )
            return analysis

        monkeypatch.setattr(permissions, "analyze_command", many_candidates)
    script = permissions.temporary_directory / "task.sh"
    script.write_text('#!/bin/sh\ncd "$WORKSPACE_ROOT"\nfixture-tool run\n')
    backend = Mock()
    backend.system_read_roots = (Path("/bin").resolve(), Path("/usr/bin").resolve())
    monkeypatch.setattr(type(backend), "trusted_system_path", lambda *_: True, raising=False)
    backend.installed_tool_roots.side_effect = lambda selected: (
        ((tools,), ()) if selected == executable else None
    )
    backend.managed_tool_root.return_value = True
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    command = 'sh "$TMPDIR/task.sh"'
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", command),
                    cwd=str(workspace),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=permissions,
    )
    host = Mock()

    result = CommandExecutionService(backend, host).run_command(context, command)

    assert result["exit_code"] == 0
    request = backend.run.call_args.args[0]
    assert request.execution_source == command
    assert request.automatic_tool_reads == (() if overflow else (tools,))
    assert request.read_roots == (() if overflow else (tools,))
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize(
    ("content", "discarded", "expected"),
    [
        ("valid é", 0, "none"),
        ("literal �", 0, "unknown"),
        ("valid prefix", 1, "unknown"),
        ("valid prefix", None, "unknown"),
    ],
)
def test_command_output_reports_conservative_decoding_loss(tmp_path, content, discarded, expected):
    """Only complete output without a replacement character proves lossless decoding."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        stdout=content,
        stderr=content,
        stdout_discarded=discarded,
        stderr_discarded=discarded,
    )
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    result = CommandExecutionService(backend, Mock()).run_command(
        ToolContext(
            None,
            "run_command",
            operations=(operation,),
            permission_manager=PermissionManager(tmp_path),
        ),
        ":",
    )

    for name in ("stdout", "stderr"):
        assert result[name]["content"] == content
        assert result[name]["decoding_loss"] == expected


@pytest.mark.parametrize("cleanup_verified", [None, True])
def test_command_result_reports_only_verified_descendant_cleanup(tmp_path, cleanup_verified):
    """A completed shell exit and native descendant proof remain distinct claims."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=0,
        possible_effects=True,
        cleanup_verified=cleanup_verified,
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(
            Operation(
                tool_id="run_command",
                action=Action.PROCESS_EXECUTE,
                target=ProcessTarget(
                    argv=("/bin/sh", "-c", ":"),
                    cwd=str(tmp_path),
                    boundary=ProcessBoundary.SANDBOXED,
                ),
            ),
        ),
        permission_manager=PermissionManager(tmp_path),
    )
    result = CommandExecutionService(backend, Mock()).run_command(context, ":")
    assert result["lifecycle"] == {
        "cancel_requested": False,
        "main_process": "exited",
        "descendant_cleanup": "verified_absent" if cleanup_verified else "uncertain",
    }


@pytest.mark.parametrize("same_root", [False, True])
def test_scratch_data_classification_cannot_transfer_to_a_different_runtime_root(
    tmp_path, same_root
):
    """The trusted generation flag grants ordinary names only to its exact bound root."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    scratch = tmp_path / "session"
    scratch.mkdir(mode=0o700)
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(
        PathHolder(scratch if same_root else permissions.temporary_directory),
        ordinary_names=lambda: True,
    )
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(backend, Mock())
    result = service.run_command(
        ToolContext(
            None,
            "run_command",
            operations=(operation,),
            instructions_manager=instructions,
            permission_manager=permissions,
        ),
        ":",
    )
    assert result["exit_code"] == 0
    approved = backend.run.call_args.args[0]
    assert approved.scratch_root == scratch
    assert approved.scratch_data is same_root


def test_workspace_environment_keeps_its_mounted_path_beneath_restricted_scratch(tmp_path):
    """Workspace paths are not remapped as generated temporary-file environment values."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    workspace = scratch / "project"
    workspace.mkdir()
    permissions = PermissionManager(workspace)
    permissions.bind_temporary_directory(PathHolder(scratch))
    instructions = InstructionsManager(runtime_environment=RuntimeEnvironment(workspace, scratch))
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    result = CommandExecutionService(backend, Mock()).run_command(
        ToolContext(
            None,
            "run_command",
            operations=(operation,),
            instructions_manager=instructions,
            permission_manager=permissions,
        ),
        ":",
    )
    assert result["exit_code"] == 0
    approved = backend.run.call_args.args[0]
    assert approved.shell_environment()["WORKSPACE_ROOT"] == str(workspace)
    assert "WORKSPACE_ROOT" not in approved.generated_environment
    assert {"TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HOME"} <= set(
        approved.generated_environment
    )
    assert "UV_CACHE_DIR" not in approved.shell_environment()


@pytest.mark.parametrize("referenced", [False, True])
def test_only_referenced_active_skill_roots_enter_command_authority(tmp_path, referenced):
    """Concrete skill paths preserve their identity without broadening unrelated commands."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    skill_root = tmp_path / "guide"
    skill_root.mkdir()
    location = skill_root / "SKILL.md"
    location.write_text("---\nname: guide\ndescription: Guide\n---\nGuide instructions")
    permissions = PermissionManager(workspace)
    instructions = InstructionsManager(
        runtime_environment=RuntimeEnvironment(workspace, permissions.temporary_directory),
        skill_manager=SkillManager([Skill("guide", "Guide", location)]),
    )
    instructions.activate_skill("guide")
    source = "cat " + shlex.quote(str(skill_root / "data")) if referenced else ":"
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", source), cwd=str(workspace), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout=str(skill_root) if referenced else ""
    )
    interaction = Mock()
    interaction.prompt.return_value = "approve"
    result = CommandExecutionService(backend, Mock()).run_command(
        ToolContext(
            interaction,
            "run_command",
            operations=(operation,),
            instructions_manager=instructions,
            permission_manager=permissions,
        ),
        source,
    )
    assert result["stdout"]["content"] == (str(skill_root) if referenced else "")
    approved = backend.run.call_args.args[0]
    assert approved.read_roots == ((skill_root,) if referenced else ())
    assert approved.execution_source == source
    if not referenced:
        interaction.prompt.assert_not_called()


def test_command_job_service_binds_reads_and_cancellation_to_session(tmp_path):
    """Only a matching scratch identity can inspect, read, and cancel a retained job."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    service = CommandExecutionService(Mock(), Mock())
    first_context = ToolContext(
        None, "read_command_job", permission_manager=PermissionManager(first)
    )
    second_context = ToolContext(
        None, "read_command_job", permission_manager=PermissionManager(second)
    )
    owner = service._job_owner(first_context)
    job = CommandJob(owner, None)
    service._jobs._jobs[job.handle] = job
    job.publish("stdout", "ready")

    assert service.inspect_command_jobs(first_context)["jobs"][0]["handle"] == job.handle
    assert service.inspect_command_jobs(second_context)["jobs"] == []
    assert service.read_command_job(first_context, job.handle)["events"][0]["content"] == "ready"
    assert service.read_command_job(second_context, job.handle).code == "command_job.unavailable"
    assert (
        service.read_command_job(first_context, job.handle, cursor=-1).code
        == "command_job.invalid_read"
    )
    assert (
        service.read_command_job(first_context, job.handle, cursor=2).code
        == "command_job.invalid_read"
    )
    assert service.cancel_command_job(second_context, job.handle).code == "command_job.unavailable"
    assert service.cancel_command_job(first_context, job.handle)["cancel_requested"]
    job.complete({"exit_code": 0})
    assert service.read_command_job(first_context, job.handle, cursor=1)["result"] == {
        "exit_code": 0
    }


def test_command_job_service_rejects_missing_session_scratch_identity(tmp_path):
    """Job tools and launch refuse absent scratch authority without leaking handles."""
    service = CommandExecutionService(Mock(), Mock())
    context = ToolContext(None, "run_command")
    assert service.inspect_command_jobs(context).code == "command_job.unavailable"
    assert service.read_command_job(context, "unknown").code == "command_job.unavailable"
    assert service.cancel_command_job(context, "unknown").code == "command_job.unavailable"
    assert service.run_command(context, ":", background=True).code == "command_job.unavailable"


def test_terminal_controls_require_the_original_session_and_live_terminal(tmp_path):
    """Controls cannot cross sessions, act on ordinary jobs, or use missing scratch authority."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    service = CommandExecutionService(Mock(), Mock())
    owner_context = ToolContext(
        None, "control_command_terminal", permission_manager=PermissionManager(first)
    )
    foreign_context = ToolContext(
        None, "control_command_terminal", permission_manager=PermissionManager(second)
    )
    owner = service._job_owner(owner_context)
    ordinary = CommandJob(owner, None)
    terminal = CommandJob(owner, None, terminal=True)
    service._jobs._jobs[ordinary.handle] = ordinary
    service._jobs._jobs[terminal.handle] = terminal

    assert (
        service.control_command_terminal(
            ToolContext(None, "control_command_terminal"), terminal.handle, "eof"
        ).code
        == "command_terminal.unavailable"
    )
    assert (
        service.control_command_terminal(foreign_context, terminal.handle, "eof").code
        == "command_job.unavailable"
    )
    assert (
        service.control_command_terminal(owner_context, ordinary.handle, "eof").code
        == "command_terminal.unavailable"
    )
    assert service.control_command_terminal(owner_context, terminal.handle, "write", "abc")[
        "terminal"
    ]
    assert terminal.next_terminal_frame() == b"I\x00\x03abc"
    assert service.control_command_terminal(
        owner_context, terminal.handle, "resize", rows=4, columns=9
    )["terminal"]
    assert terminal.next_terminal_frame() == b"Z\x00\x04\x00\x09"
    assert service.control_command_terminal(owner_context, terminal.handle, "eof")["terminal"]
    assert terminal.next_terminal_frame() == b"F"
    assert service.control_command_terminal(owner_context, terminal.handle, "interrupt")["terminal"]
    assert terminal.next_terminal_frame() == b"X"
    for action, content, rows, columns in (
        ("write", None, None, None),
        ("resize", None, None, None),
        ("eof", "extra", None, None),
        ("unknown", None, None, None),
        ("write", "", None, None),
    ):
        assert (
            service.control_command_terminal(
                owner_context, terminal.handle, action, content, rows, columns
            ).code
            == "command_terminal.invalid_control"
        )
    terminal.complete({"exit_code": 0})
    assert service.control_command_terminal(owner_context, terminal.handle, "close")["closed"]
    assert (
        service.control_command_terminal(owner_context, terminal.handle, "eof").code
        == "command_job.unavailable"
    )


def test_interactive_command_refuses_unsupported_backend_and_egress(tmp_path):
    """Interactive admission fails before approval on unsupported or shared-control paths."""
    backend = Mock(capabilities=frozenset())
    service = CommandExecutionService(backend, Mock())
    context = ToolContext(None, "run_command")
    assert (
        service.run_command(context, ":", interactive=True).code == "command_terminal.unavailable"
    )
    backend.capabilities = frozenset({SandboxCapability.TERMINAL})
    assert (
        service.run_command(
            context, ":", interactive=True, egress_origins=["https://example.com"]
        ).code
        == "command_terminal.unavailable"
    )
    backend.run.assert_not_called()


def test_background_command_refuses_a_stopped_origin_before_reserving_a_worker(background_session):
    """A stopped model run cannot create a late detached attempt or consume capacity."""
    sessions, permissions = background_session
    stopped = Event()
    stopped.set()
    service = CommandExecutionService(Mock(), Mock())
    context = ToolContext(
        None,
        "run_command",
        permission_manager=permissions,
        session_manager=sessions,
        cancellation_event=stopped,
    )

    assert service.run_command(context, ":", background=True).code == "sandbox.cancelled"
    assert not service._jobs.has_active()


def test_background_command_refuses_a_scratch_owner_mismatch(tmp_path, background_session):
    """A direct registry call cannot substitute a different scratch identity for its session."""
    sessions, permissions = background_session
    other = tmp_path / "other-session-scratch"
    other.mkdir(mode=0o700)
    permissions.bind_temporary_directory(PathHolder(other))
    service = CommandExecutionService(Mock(), Mock())
    context = ToolContext(
        None, "run_command", permission_manager=permissions, session_manager=sessions
    )

    assert service.run_command(context, ":", background=True).code == "command_job.unavailable"
    assert not service._jobs.has_active()


def test_background_command_propagates_interrupted_lease_entry(monkeypatch, background_session):
    """An interrupted parent lease before reservation creates no detached attempt."""
    sessions, permissions = background_session
    monkeypatch.setattr(sessions, "interaction_execution", Mock(side_effect=KeyboardInterrupt))
    service = CommandExecutionService(Mock(), Mock())
    context = ToolContext(
        None, "run_command", permission_manager=permissions, session_manager=sessions
    )

    with pytest.raises(KeyboardInterrupt):
        service.run_command(context, ":", background=True)
    assert not service._jobs.has_active()


def test_background_command_resumes_one_authorized_attempt_without_replay(
    tmp_path, monkeypatch, background_session
):
    """Recovery reuses a call while later calls with the same provider ID run separately."""
    backend = Mock()
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="done"
    )
    service = CommandExecutionService(backend, Mock())
    sessions, permissions = background_session
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
        call_id="one-call",
    )

    class Worker:
        """Run the retained worker synchronously for a deterministic unit check."""

        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            """Finish the command in the current test thread."""
            self.target()

        def join(self, _timeout):
            """A synchronous worker needs no cleanup wait."""

    monkeypatch.setattr(command_jobs.threading, "Thread", Worker)
    sessions.add_message(ToolCall(call_id="one-call", name="run_command", arguments="{}"))
    handle = service.run_command(context, ":", background=True)
    assert handle["status"] == "complete"
    assert "content" not in handle["result"]["stdout"]
    again = service.run_command(context, ":", background=True)
    assert again["handle"] == handle["handle"]
    result = service.read_command_job(context, handle["handle"], include_output=True)
    assert result["result"]["stdout"]["content"] == "done"
    assert backend.run.call_count == 1
    sessions.add_message(ToolCall(call_id="one-call", name="run_command", arguments="{}"))
    later = service.run_command(context, ":", background=True)
    assert later["handle"] != handle["handle"]
    assert service.run_command(context, ":", background=True)["handle"] == later["handle"]
    assert backend.run.call_count == 2
    assert backend.run.call_args.args[0].cleanup_proof is True
    assert callable(backend.run.call_args.kwargs["output_observer"])
    assert backend.run.call_args.kwargs["cancellation_event"].is_set() is False


def test_interactive_command_uses_one_owned_job_and_escapes_terminal_output(
    tmp_path, monkeypatch, background_session
):
    """The original approved run yields a terminal handle with inert control text."""
    backend = Mock(capabilities=frozenset({SandboxCapability.TERMINAL}))
    backend.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=0, stdout="\x1b]52;c;bad\x07done"
    )
    service = CommandExecutionService(backend, Mock())
    sessions, permissions = background_session
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", ":"), cwd=str(tmp_path), boundary=ProcessBoundary.SANDBOXED
        ),
    )
    context = ToolContext(
        None,
        "run_command",
        operations=(operation,),
        permission_manager=permissions,
        session_manager=sessions,
        call_id="interactive-call",
    )

    class Worker:
        """Settle the owned attempt synchronously for deterministic output checks."""

        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            """Run the captured target once."""
            self.target()

        def join(self, _timeout):
            """The synchronous worker has already settled."""

    monkeypatch.setattr(command_jobs.threading, "Thread", Worker)
    handle = service.run_command(context, ":", interactive=True)

    assert handle["terminal"] and handle["status"] == "complete"
    assert handle["result"]["terminal_output"] == "escaped-controls"
    result = service.read_command_job(context, handle["handle"], include_output=True)
    assert result["result"]["stdout"]["content"] == "␛]52;c;bad␇done"
    assert backend.run.call_count == 1
    approved = backend.run.call_args.args[0]
    assert approved.terminal
    assert callable(backend.run.call_args.kwargs["terminal_input"])


def test_command_jobs_keep_backend_authority_until_workers_settle(monkeypatch):
    """Refresh rejects reserved jobs; closing an idle backend reports unsettled workers."""
    backend = Mock()
    service = CommandExecutionService(backend, Mock())
    monkeypatch.setattr(service._jobs, "has_active", lambda: True)
    with pytest.raises(RuntimeError, match="active"):
        service.refresh_backend()
    monkeypatch.setattr(service._jobs, "close", lambda: False)
    service.close()
    backend.close.assert_called_once()


def test_background_command_reports_capacity_refusal(monkeypatch, background_session):
    """A full job registry refuses the request before another shell can launch."""
    service = CommandExecutionService(Mock(), Mock())
    refusal = execution_facade.Problem(code="command_job.unavailable", title="Full", detail="Full")
    monkeypatch.setattr(service._jobs, "start", lambda *_args, **_kwargs: refusal)
    sessions, permissions = background_session
    context = ToolContext(
        None, "run_command", permission_manager=permissions, session_manager=sessions
    )
    assert service.run_command(context, ":", background=True) is refusal


@pytest.mark.parametrize("settled", [False, True])
def test_background_handle_returns_without_requiring_a_second_result_turn(
    monkeypatch, settled, background_session
):
    """A live handle returns promptly, while an already-settled one includes its result."""
    service = CommandExecutionService(Mock(), Mock())
    sessions, permissions = background_session
    context = ToolContext(
        None, "run_command", permission_manager=permissions, session_manager=sessions
    )
    job = CommandJob("owner", None)
    if settled:
        job.complete({"exit_code": 0})
        monkeypatch.setattr(
            job,
            "read",
            lambda _cursor: execution_facade.Problem(
                code="command_job.cursor_expired", title="Expired", detail="Expired"
            ),
        )
    else:
        job.admitted()
    monkeypatch.setattr(service._jobs, "start", lambda *_args, **_kwargs: job)

    result = service.run_command(context, ":", background=True)

    assert result["handle"] == job.handle
    assert result["status"] == ("complete" if settled else "running")
    assert "result" not in result


@pytest.mark.parametrize("failure", [KeyboardInterrupt, ValueError])
def test_interrupted_background_handoff_cancels_its_unreturned_job(
    monkeypatch, background_session, failure
):
    """An interrupted wait cannot leave an active command without a usable handle."""
    sessions, permissions = background_session
    service = CommandExecutionService(Mock(), Mock())
    context = ToolContext(
        None, "run_command", permission_manager=permissions, session_manager=sessions
    )
    job = CommandJob("owner", None)
    monkeypatch.setattr(service._jobs, "start", lambda *_args, **_kwargs: job)
    monkeypatch.setattr(job.ready, "wait", Mock(side_effect=failure))

    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            service.run_command(context, ":", background=True)
    else:
        assert service.run_command(context, ":", background=True).code == "command_job.unavailable"

    assert job.cancellation_event.is_set()
