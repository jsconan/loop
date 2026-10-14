"""Verify the native facade rejects unbound tool contexts."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import pytest

from loop.execution import CommandExecutionService
from loop.execution import facade as execution_facade
from loop.execution.facade import _automatic_executable_identities
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, UnavailableSandboxBackend
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
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
    assert backend.run.call_args.args[0].deadline == 303.0
    host.run_host_command.assert_not_called()


def test_execution_service_closes_optional_backend_lifecycle():
    """The facade forwards teardown only when its backend exposes close."""
    backend = Mock()
    CommandExecutionService(backend, Mock()).close()
    backend.close.assert_called_once()
    CommandExecutionService(object(), Mock()).close()


@pytest.mark.parametrize("has_close", [False, True])
def test_refresh_backend_reselects_authority_and_closes_previous_backend(monkeypatch, has_close):
    """Setup changes replace the cached sandbox backend and close its prior resources."""
    previous = Mock() if has_close else object()
    replacement = Mock()
    select = Mock(return_value=replacement)
    monkeypatch.setattr(execution_facade, "select_sandbox_backend", select)
    service = CommandExecutionService(previous, Mock())

    service.refresh_backend()
    service.close()

    select.assert_called_once_with()
    if has_close:
        previous.close.assert_called_once_with()
    replacement.close.assert_called_once_with()


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


def test_automatic_executable_identity_does_not_mistake_mapped_uid_for_host_root(
    tmp_path, monkeypatch
):
    """A user-namespace UID zero file is not administrator-owned host authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    original_read_text = Path.read_text
    monkeypatch.setattr(execution_facade.sys, "platform", "linux")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "0 1000 1\n"
            if path == Path("/proc/self/uid_map")
            else original_read_text(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(execution_facade.os, "getuid", lambda: -1)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


@pytest.mark.parametrize(
    ("mapping", "uid", "expected"),
    [
        ("0 0 4294967295\n", 1000, (0, 1000)),
        ("0 1000 1\n", 1000, (1000,)),
        (OSError("missing proc status"), 1000, (1000,)),
    ],
)
def test_trusted_file_owners_follow_live_user_namespace_mapping(
    monkeypatch, mapping, uid, expected
):
    """Only an initial namespace can treat file owner zero as host administrator."""
    original_read_text = Path.read_text
    monkeypatch.setattr(execution_facade.sys, "platform", "linux")
    monkeypatch.setattr(execution_facade.os, "getuid", lambda: uid)

    def read_text(path, *args, **kwargs):
        """Provide the chosen live mapping while leaving all other path reads untouched."""
        if path == Path("/proc/self/uid_map"):
            if isinstance(mapping, OSError):
                raise mapping
            return mapping
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert execution_facade._trusted_file_owner_ids() == expected


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_run_command_binds_an_unmanaged_path_tool_as_one_exact_file(
    tmp_path, monkeypatch, platform
):
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
    monkeypatch.setattr(execution_facade.sys, "platform", platform)
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


def test_run_command_automatically_binds_the_workspace_venv_python_runtime(tmp_path, monkeypatch):
    """Run uv with its workspace venv runtime in the default sandbox request without a prompt."""
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
    pytest_executable = venv_bin / "pytest"
    pytest_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pytest_executable.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("uv", encoding="utf-8")
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{venv_bin}{os.pathsep}{tools}")
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    monkeypatch.setattr("loop.execution.sandbox.linux.backend.runtime_read_roots", lambda: ())
    backend = LinuxNativeBackend(tmp_path / "management.json")
    monkeypatch.setattr(execution_facade.sys, "platform", "linux")
    monkeypatch.setattr(type(backend), "file_creation_mask", 0o077)
    backend.capability_failure = Mock(return_value=None)
    backend.run = Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    service = CommandExecutionService(backend, Mock())
    operation = Operation(
        tool_id="run_command",
        action=Action.PROCESS_EXECUTE,
        target=ProcessTarget(
            argv=("/bin/sh", "-c", "uv run pytest"),
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

    result = service.run_command(context, "uv run pytest", cwd=str(workspace))

    request = backend.run.call_args.args[0]
    assert result["exit_code"] == 0
    assert request.read_roots == (runtime_root,)
    assert request.automatic_tool_reads == (runtime_root,)
    assert request.file_creation_mask == 0o077


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


def test_automatic_executable_identities_reject_foreign_owners(tmp_path, monkeypatch):
    """Files owned by another account cannot receive automatic launch authority."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("loop.execution.facade.os.getuid", lambda: -1)

    assert _automatic_executable_identities(("tool",), str(tools), workspace) == ()


def test_automatic_executable_identities_reject_foreign_file_owner(tmp_path, monkeypatch):
    """A current-user PATH cannot automatically authorize a foreign-owned executable."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    original_stat = Path.stat

    def foreign_file_owner(path, *args, **kwargs):
        """Report a foreign owner only for the executable inode."""
        details = original_stat(path, *args, **kwargs)
        if path == executable:
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

    monkeypatch.setattr(Path, "stat", foreign_file_owner)

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
    backend.installed_tool_roots.side_effect = OSError("helper unavailable")
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
