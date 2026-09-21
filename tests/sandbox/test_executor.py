"""Tests for platform-native sandbox process compilation."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.sandbox import (
    HostProcessPlan,
    SandboxPlan,
    SandboxUnavailableError,
    sandbox_status,
    spawn_host,
    spawn_sandboxed,
)
from loop.sandbox import executor as executor_module


def executable(tmp_path, name="tool"):
    """Create an executable fixture without relying on a host command."""
    path = tmp_path / name
    path.write_text("tool", encoding="utf-8")
    path.chmod(0o755)
    return path


def temporary_descriptor():
    """Return an owned descriptor backed by an anonymous temporary file."""
    descriptor, path = tempfile.mkstemp()
    os.unlink(path)
    return descriptor


def plan(tmp_path):
    """Return a realistic immutable sandbox plan."""
    return SandboxPlan.create((str(executable(tmp_path)),), tmp_path, tmp_path)


@pytest.fixture(autouse=True)
def available_backend(monkeypatch):
    """Keep compiler unit tests independent from the host sandbox capability."""
    monkeypatch.setattr(executor_module, "sandbox_status", lambda: (True, None))


@pytest.fixture
def linux_backend(tmp_path, monkeypatch):
    """Provide trusted mocked Linux wrapper and seccomp descriptors."""
    launcher = tmp_path / "trusted-bwrap"
    monkeypatch.setattr("loop.sandbox.executor._trusted_bubblewrap", lambda _roots=(): launcher)
    monkeypatch.setattr(
        "loop.sandbox.executor._linux_seccomp_descriptor",
        temporary_descriptor,
    )
    return launcher


def test_macos_spawn_uses_a_closed_seatbelt_profile(tmp_path, monkeypatch):
    """macOS commands deny default and network while protecting control paths."""
    sandbox_plan = plan(tmp_path)
    popen = MagicMock()
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    spawn_sandboxed(sandbox_plan, popen_options={"text": True})

    command = popen.call_args.args[0]
    profile = command[2]
    assert command[0] == "/usr/bin/sandbox-exec"
    assert "(deny default)" in profile
    assert "allow network" not in profile
    assert f'(deny file-read* (subpath "{tmp_path}/.loop"))' in profile
    assert f'(deny file-write* (subpath "{tmp_path}/.git"))' in profile
    assert command[-1] == str(sandbox_plan.executable)
    assert command[-5:-1] == [sys.executable, "-m", "loop.sandbox.launcher", "--"]


def test_macos_profile_exposes_python_runtime_and_private_temporary_root(tmp_path, monkeypatch):
    """Python and private temporary execution receive only their explicitly planned runtime roots."""
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    sandbox_plan = SandboxPlan.create(
        (sys.executable, "-c", "pass"),
        tmp_path,
        tmp_path,
        temporary_directory=temporary,
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    spawn_sandboxed(sandbox_plan, popen_options={})

    profile = popen.call_args.args[0][2]
    assert str(Path(sys.base_prefix).resolve()) in profile
    assert f'(allow file-write* (subpath "{temporary}"))' in profile
    assert sandbox_plan.environment["TMPDIR"] == str(temporary)


def test_macos_spawn_fails_closed_when_seatbelt_disappears(tmp_path, monkeypatch):
    """A missing system Seatbelt launcher never falls back to the host."""
    sandbox_plan = plan(tmp_path)
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: False)

    with pytest.raises(SandboxUnavailableError, match="Seatbelt"):
        spawn_sandboxed(sandbox_plan, popen_options={})


def test_spawn_reports_a_nonfunctional_backend_before_creating_a_process(tmp_path, monkeypatch):
    """A failed capability probe produces a sandbox incompatibility without launching anything."""
    popen = MagicMock()
    monkeypatch.setattr(executor_module, "sandbox_status", lambda: (False, "not permitted"))
    monkeypatch.setattr(executor_module.subprocess, "Popen", popen)

    with pytest.raises(SandboxUnavailableError, match="not permitted"):
        spawn_sandboxed(plan(tmp_path), popen_options={})

    popen.assert_not_called()


def test_linux_spawn_uses_isolated_namespaces_and_protected_overlays(
    tmp_path, monkeypatch, linux_backend
):
    """Linux commands receive minimal mounts, no network, no capabilities, and protected Git state."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".loop").mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    command = executable(tmp_path)
    sandbox_plan = SandboxPlan.create(
        (str(command),),
        tmp_path,
        tmp_path,
        temporary_directory=temporary,
        read_only_roots=(reference,),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    spawn_sandboxed(sandbox_plan, popen_options={"text": True})

    command = popen.call_args.args[0]
    assert command[0] == str(linux_backend)
    assert "--unshare-user" in command
    assert "--unshare-pid" in command
    assert "--unshare-net" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--seccomp" in command
    assert popen.call_args.kwargs["pass_fds"]
    assert command[command.index("--tmpfs", command.index(str(tmp_path))) + 1] == str(
        tmp_path / ".loop"
    )


def test_linux_spawn_fails_closed_without_bubblewrap(tmp_path, monkeypatch):
    """Linux never falls back to an unrestricted host process."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    sandbox_plan = plan(tmp_path)
    monkeypatch.setattr("loop.sandbox.executor._BUBBLEWRAP_LOCATIONS", ())

    with pytest.raises(SandboxUnavailableError, match="Bubblewrap"):
        spawn_sandboxed(sandbox_plan, popen_options={})


def test_linux_path_launcher_is_never_executed_by_preflight_or_spawn(tmp_path, monkeypatch):
    """An attacker-controlled PATH launcher is ignored by readiness and execution."""
    attacker = executable(tmp_path, "bwrap")
    run = MagicMock()
    popen = MagicMock()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor._BUBBLEWRAP_LOCATIONS", ())
    monkeypatch.setattr("loop.sandbox.executor.subprocess.run", run)
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    available, _detail = sandbox_status()
    with pytest.raises(SandboxUnavailableError, match="Bubblewrap"):
        spawn_sandboxed(plan(tmp_path), popen_options={})

    assert available is False
    assert attacker.exists()
    run.assert_not_called()
    popen.assert_not_called()


def test_linux_spawn_rejects_bubblewrap_from_a_writable_root(tmp_path, monkeypatch):
    """An agent-controlled PATH entry cannot replace the outer sandbox launcher."""
    launcher = tmp_path / "bwrap"
    launcher.write_text("fake", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setattr("loop.sandbox.executor._BUBBLEWRAP_LOCATIONS", (launcher,))
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor.platform.machine", lambda: "aarch64")

    with pytest.raises(SandboxUnavailableError, match="agent-writable"):
        spawn_sandboxed(plan(tmp_path), popen_options={})


def test_linux_spawn_builds_a_real_seccomp_filter_for_a_trusted_launcher(tmp_path, monkeypatch):
    """The Linux wrapper receives a generated seccomp program through an inherited descriptor."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launcher = tmp_path / "bwrap"
    launcher.write_text("trusted", encoding="utf-8")
    launcher.chmod(0o755)
    popen = MagicMock()
    monkeypatch.setattr("loop.sandbox.executor._BUBBLEWRAP_LOCATIONS", (launcher,))
    monkeypatch.setattr("loop.sandbox.executor._agent_writable", lambda _path, _roots: False)
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor.platform.machine", lambda: "aarch64")
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    spawn_sandboxed(plan(workspace), popen_options={})

    command = popen.call_args.args[0]
    assert command[0] == str(launcher.resolve())
    assert "--seccomp" in command


def test_linux_spawn_rejects_non_executable_and_unsupported_backends(tmp_path, monkeypatch):
    """Linux helper and seccomp capability validation fail before process creation."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    launcher = tmp_path / "bwrap"
    launcher.write_text("not executable", encoding="utf-8")
    sandbox_plan = plan(workspace)
    monkeypatch.setattr("loop.sandbox.executor._BUBBLEWRAP_LOCATIONS", (launcher,))
    monkeypatch.setattr("loop.sandbox.executor._agent_writable", lambda _path, _roots: False)

    with pytest.raises(SandboxUnavailableError, match="not executable"):
        spawn_sandboxed(sandbox_plan, popen_options={})

    launcher.chmod(0o755)
    monkeypatch.setattr("loop.sandbox.executor.platform.machine", lambda: "unsupported")
    with pytest.raises(SandboxUnavailableError, match="architecture"):
        spawn_sandboxed(sandbox_plan, popen_options={})


def test_linux_descriptor_and_command_failures_close_owned_descriptors(tmp_path, monkeypatch):
    """Linux setup failures close temporary filter descriptors before propagating."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor.platform.machine", lambda: "aarch64")
    monkeypatch.setattr(
        "loop.sandbox.executor._trusted_bubblewrap", lambda _roots=(): tmp_path / "trusted-bwrap"
    )
    monkeypatch.setattr("loop.sandbox.executor.os.write", MagicMock(side_effect=OSError("write")))
    with pytest.raises(OSError, match="write"):
        spawn_sandboxed(plan(tmp_path), popen_options={})

    descriptor = temporary_descriptor()
    monkeypatch.setattr("loop.sandbox.executor._linux_seccomp_descriptor", lambda: descriptor)
    monkeypatch.setattr(
        "loop.sandbox.executor._linux_command", MagicMock(side_effect=ValueError("compile"))
    )
    with pytest.raises(ValueError, match="compile"):
        spawn_sandboxed(plan(tmp_path), popen_options={})
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_spawn_rejects_authority_replaced_after_planning(tmp_path):
    """Executable replacement invalidates an already authorized immutable plan."""
    executable = tmp_path / "tool"
    executable.write_text("before", encoding="utf-8")
    executable.chmod(0o755)
    sandbox_plan = SandboxPlan.create((str(executable),), tmp_path, tmp_path)
    executable.write_text("after replacement", encoding="utf-8")

    with pytest.raises(ValueError, match="authority changed"):
        spawn_sandboxed(sandbox_plan, popen_options={})


def test_spawn_rejects_authority_removed_after_planning(tmp_path):
    """A disappearing executable invalidates its already authorized plan."""
    executable = tmp_path / "tool"
    executable.write_text("before", encoding="utf-8")
    executable.chmod(0o755)
    sandbox_plan = SandboxPlan.create((str(executable),), tmp_path, tmp_path)
    executable.unlink()

    with pytest.raises(ValueError, match="authority disappeared"):
        spawn_sandboxed(sandbox_plan, popen_options={})


def test_linux_minimal_plan_masks_absent_control_paths(tmp_path, monkeypatch, linux_backend):
    """Absent and symlinked control paths remain unavailable inside Bubblewrap."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    (tmp_path / ".git").symlink_to(tmp_path / "external")
    sandbox_plan = plan(tmp_path)
    popen = MagicMock()
    monkeypatch.setattr("loop.sandbox.executor.subprocess.Popen", popen)

    spawn_sandboxed(sandbox_plan, popen_options={})

    command = popen.call_args.args[0]
    for protected in (".git", ".gitignore", ".agentignore", ".loop"):
        target = str(tmp_path / protected)
        index = command.index(target)
        assert command[index - 1] == "/dev/null"


@pytest.mark.parametrize("system", ["Windows", "FreeBSD"])
def test_unsupported_backend_fails_closed(tmp_path, monkeypatch, system):
    """Platforms without a complete backend cannot launch agent commands."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: system)

    with pytest.raises(SandboxUnavailableError):
        spawn_sandboxed(plan(tmp_path), popen_options={})


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_status_reports_functional_backend_readiness(tmp_path, monkeypatch, system):
    """Tool preflight probes the recognized native sandbox instead of its mere presence."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: system)
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "loop.sandbox.executor._trusted_bubblewrap", lambda _roots=(): tmp_path / "trusted-bwrap"
    )
    monkeypatch.setattr(
        "loop.sandbox.executor._linux_seccomp_descriptor",
        temporary_descriptor,
    )
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("loop.sandbox.executor.subprocess.run", run)

    result, detail = sandbox_status()

    assert result is True
    assert detail is None
    assert run.called


@pytest.mark.parametrize("system", ["Darwin", "Linux"])
def test_status_rejects_a_backend_that_cannot_enforce(tmp_path, monkeypatch, system):
    """A present backend that fails its launch probe is reported unavailable."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: system)
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "loop.sandbox.executor._trusted_bubblewrap", lambda _roots=(): tmp_path / "trusted-bwrap"
    )
    monkeypatch.setattr(
        "loop.sandbox.executor._linux_seccomp_descriptor",
        temporary_descriptor,
    )
    monkeypatch.setattr(
        "loop.sandbox.executor.subprocess.run",
        MagicMock(return_value=SimpleNamespace(returncode=1)),
    )

    result, detail = sandbox_status()

    assert result is False
    assert "cannot" in detail


def test_status_rejects_missing_or_failing_native_helpers(monkeypatch):
    """Readiness reports both absent Seatbelt and Linux probe setup failures."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Darwin")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: False)
    assert sandbox_status() == (False, "Seatbelt sandbox-exec is unavailable.")

    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: "Linux")
    monkeypatch.setattr("loop.sandbox.executor.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "loop.sandbox.executor._trusted_bubblewrap", MagicMock(side_effect=OSError("blocked"))
    )
    available, detail = sandbox_status()
    assert available is False
    assert detail == "blocked"


@pytest.mark.parametrize("system", ["Windows", "FreeBSD"])
def test_status_rejects_unsupported_backends(monkeypatch, system):
    """Preflight remains fail-closed on platforms without a complete launcher."""
    monkeypatch.setattr("loop.sandbox.executor.platform.system", lambda: system)

    result, detail = sandbox_status()

    assert result is False
    assert detail is not None


def test_linux_status_requires_a_trusted_probe_shell(monkeypatch):
    """Linux readiness fails closed before launch when its active probe cannot run."""
    monkeypatch.setattr(executor_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(executor_module.Path, "is_file", lambda _path: False)

    assert sandbox_status() == (
        False,
        "A trusted Bash executable is required for the Linux sandbox probe.",
    )


def test_spawn_host_revalidates_and_uses_shell_free_bounded_options(tmp_path, monkeypatch):
    """Confirmed host execution forwards one exact immutable plan without a shell."""
    sandbox_plan = plan(tmp_path)
    host_plan = HostProcessPlan.create(sandbox_plan, "unsupported sandbox feature")
    popen = MagicMock()
    monkeypatch.setattr(executor_module.subprocess, "Popen", popen)

    spawn_host(host_plan, popen_options={"stdout": subprocess.PIPE})

    assert popen.call_args.args == (executor_module._limited_command(host_plan),)
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["close_fds"] is True
    assert popen.call_args.kwargs["stdout"] is subprocess.PIPE


@pytest.mark.parametrize(
    ("mode", "uid", "gid", "expected"),
    [(0o777, 0, 0, True), (0o700, 501, 0, True), (0o070, 0, 20, True), (0o755, 0, 0, False)],
)
def test_linux_launcher_parent_permission_validation(
    tmp_path, monkeypatch, mode, uid, gid, expected
):
    """Trusted-launcher validation rejects owner, group, and world-writable components."""
    metadata = SimpleNamespace(st_mode=mode, st_uid=uid, st_gid=gid)
    monkeypatch.setattr(executor_module.os, "geteuid", lambda: 501)
    monkeypatch.setattr(executor_module.os, "getegid", lambda: 20)
    monkeypatch.setattr(executor_module.os, "getgroups", lambda: [20])
    monkeypatch.setattr(executor_module.Path, "stat", lambda _path: metadata)

    result = executor_module._agent_writable(tmp_path, ())  # pylint: disable=protected-access

    assert result is expected
