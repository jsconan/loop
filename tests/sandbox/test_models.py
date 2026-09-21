"""Tests for immutable sandbox execution plans."""

import os

import pytest

from loop.sandbox import SANDBOX_POLICY_VERSION, HostProcessPlan, SandboxPlan


def executable(tmp_path, name="tool"):
    """Create an executable fixture without relying on a host command."""
    path = tmp_path / name
    path.write_text("tool", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_policy_version_is_the_initial_unreleased_version():
    """Fresh sandbox policy development remains version one without migration history."""
    assert SANDBOX_POLICY_VERSION == "1"


def test_plan_canonicalizes_authority_and_binds_it_to_a_stable_digest(tmp_path, monkeypatch):
    """A plan fixes executable, roots, environment, and policy identity before approval."""
    executable = tmp_path / "tool"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    read_only = tmp_path / "reference"
    read_only.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("SECRET", "not inherited")

    first = SandboxPlan.create(("tool", "arg"), tmp_path, tmp_path, read_only_roots=(read_only,))
    second = SandboxPlan.create(("tool", "arg"), tmp_path, tmp_path, read_only_roots=(read_only,))

    assert first == second
    assert first.argv == (str(executable.resolve()), "arg")
    assert first.environment["PATH"].split(os.pathsep)[0] == str(tmp_path)
    assert first.environment["HOME"] == str(tmp_path)
    assert first.environment["TMPDIR"] == str(tmp_path)
    assert "SECRET" not in first.environment
    assert len(first.policy_digest) == 64
    with pytest.raises(TypeError):
        first.environment["NEW"] = "value"


def test_plan_digest_changes_with_granted_authority(tmp_path):
    """Changing command arguments or writable roots invalidates remembered authorization."""
    command = executable(tmp_path)
    child = tmp_path / "child"
    child.mkdir()

    base = SandboxPlan.create((str(command),), tmp_path, tmp_path)
    changed_arguments = SandboxPlan.create((str(command), "value"), tmp_path, tmp_path)
    changed_root = SandboxPlan.create((str(command),), child, child)

    assert base.policy_digest != changed_arguments.policy_digest
    assert base.policy_digest != changed_root.policy_digest


def test_plan_ignores_untrusted_host_environment_changes(tmp_path, monkeypatch):
    """Host environment changes cannot alter the deterministic child environment."""
    command = executable(tmp_path)
    first = SandboxPlan.create((str(command),), tmp_path, tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "host-home"))

    second = SandboxPlan.create((str(command),), tmp_path, tmp_path)

    assert first.policy_digest == second.policy_digest


@pytest.mark.parametrize(
    ("argv", "cwd_name", "workspace_name", "message"),
    [
        ((), ".", ".", "include an executable"),
        (("missing-command",), ".", ".", "executable is unavailable"),
    ],
)
def test_plan_rejects_commands_without_a_complete_containment_root(
    tmp_path, argv, cwd_name, workspace_name, message
):
    """Invalid executable and working-directory authority fails before authorization."""
    cwd = tmp_path / cwd_name
    workspace = tmp_path / workspace_name
    cwd.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match=message):
        SandboxPlan.create(argv, cwd, workspace)


def test_plan_rejects_a_working_directory_outside_readable_roots(tmp_path):
    """Working-directory authority must remain within an exposed policy root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside readable roots"):
        SandboxPlan.create((str(executable(tmp_path)),), outside, workspace)


def test_plan_rejects_missing_read_only_roots(tmp_path):
    """A disappearing read grant cannot silently weaken or redirect a plan."""
    command = executable(tmp_path)
    with pytest.raises(FileNotFoundError):
        SandboxPlan.create(
            (str(command),),
            tmp_path,
            tmp_path,
            read_only_roots=(tmp_path / "missing",),
        )


def test_plan_accepts_an_explicit_relative_executable_path(tmp_path):
    """A relative executable path is resolved against the authorized working directory."""
    executable = tmp_path / "bin" / "tool"
    executable.parent.mkdir()
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    result = SandboxPlan.create(("bin/tool",), tmp_path, tmp_path)

    assert result.executable == executable.resolve()


def test_plan_accepts_an_explicit_absolute_executable_path(tmp_path):
    """An absolute executable remains usable when PATH lookup supplies no result."""
    executable_path = executable(tmp_path)
    result = SandboxPlan.create((str(executable_path),), tmp_path, tmp_path)

    assert result.executable == executable_path.resolve()


def test_plan_rejects_a_non_executable_file(tmp_path):
    """Planning refuses a regular file that the operating system cannot execute."""
    executable = tmp_path / "tool"
    executable.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="not executable"):
        SandboxPlan.create((str(executable),), tmp_path, tmp_path)


def test_host_plan_binds_reason_environment_cwd_and_executable(tmp_path):
    """Fresh host authority binds every displayed field and rejects later replacement."""
    sandbox_plan = SandboxPlan.create((str(executable(tmp_path)), "argument"), tmp_path, tmp_path)

    first = HostProcessPlan.create(sandbox_plan, "unsupported syscall")
    second = HostProcessPlan.create(sandbox_plan, "different incompatibility")

    assert first.confirmation_digest != second.confirmation_digest
    assert first.argv == sandbox_plan.argv
    assert first.cwd == sandbox_plan.cwd
    assert first.environment == sandbox_plan.environment
    sandbox_plan.executable.write_text("replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="authority changed"):
        first.validate_authority()


def test_plan_rejects_empty_path_entries_and_backend_changes(tmp_path, monkeypatch):
    """Executable lookup skips empty PATH entries and backend changes invalidate authority."""
    command = executable(tmp_path)
    monkeypatch.setenv("PATH", f"{os.pathsep}{tmp_path}")
    sandbox_plan = SandboxPlan.create((command.name,), tmp_path, tmp_path)
    monkeypatch.setattr("loop.sandbox.models.platform.system", lambda: "Linux")

    with pytest.raises(ValueError, match="backend changed"):
        sandbox_plan.validate_authority()


def test_host_plan_requires_a_specific_incompatibility_reason(tmp_path):
    """Host authority cannot be confirmed without a displayed incompatibility."""
    sandbox_plan = SandboxPlan.create((str(executable(tmp_path)),), tmp_path, tmp_path)

    with pytest.raises(ValueError, match="specific sandbox incompatibility"):
        HostProcessPlan.create(sandbox_plan, "  ")
