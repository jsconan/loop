"""Verify host qualification selects only a working built-in sandbox."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox import CommandProcessResult, SandboxOutcome, selection


def test_unsupported_host_fails_closed_without_instantiating_seatbelt(monkeypatch):
    """An unsupported host never constructs a Seatbelt backend."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Windows")
    factory = Mock()
    monkeypatch.setattr(selection, "_macos_backend", factory)
    backend = selection.select_sandbox_backend()

    assert "No native command sandbox" in backend.capability_failure()
    assert backend.run(Mock()).outcome is SandboxOutcome.UNAVAILABLE
    assert backend.system_read_roots == ()
    assert backend.policy_version == "unavailable-native-v1"
    assert backend.scratch_prefix == "loop-sandbox-"
    assert backend.trusted_file_owner_ids == (os.getuid(),)
    assert backend.automatic_tool_read_roots(Path("/work"), (), ()) == ()
    assert not backend.managed_tool_root(Mock())
    assert backend.installed_tool_roots(Path("/outside/bin/tool")) == (
        (Path("/outside/bin"),),
        (),
    )
    factory.assert_not_called()


@pytest.mark.parametrize(
    "outcome", [SandboxOutcome.INVALID, SandboxOutcome.STALE, SandboxOutcome.UNAVAILABLE]
)
def test_linux_selection_retains_typed_preflight_failure(monkeypatch, outcome):
    """Initial selection preserves integrity outcomes without running the probe or command."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Linux")
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    failure = CommandProcessResult(outcome, detail="Sanitized native authority failure.")
    native.capability_failure.return_value = failure
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.LinuxNativeBackend", Mock(return_value=native)
    )
    selected = selection.select_sandbox_backend()
    assert selected.capability_failure() == failure
    assert selected.run(Mock()) == failure
    native.run.assert_not_called()


@pytest.mark.parametrize("enforced", [False, True])
def test_linux_selection_requires_operational_enforcement(monkeypatch, enforced):
    """Intact metadata alone cannot select a Linux backend that fails native enforcement."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Linux")
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.capability_failure.return_value = None

    def probe(request):
        """Model enforcement effects on disposable probe files without starting processes."""
        assert request.shell_environment()["TMPDIR"]
        (request.cwd / "allowed").write_text("allowed")
        if not enforced:
            (request.cwd / "leaked").write_text("probe-secret")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.run.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.LinuxNativeBackend", Mock(return_value=native)
    )
    selected = selection.select_sandbox_backend()
    if enforced:
        assert selected is native
    else:
        assert selected.capability_failure().outcome is SandboxOutcome.INVALID
    native.run.assert_called_once()


def test_linux_selection_reuses_stable_probe_policy_across_calls(monkeypatch, tmp_path):
    """Keep Linux startup probes on one owned workspace policy across app restarts."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selection.tempfile, "tempdir", str(tmp_path))
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.capability_failure.return_value = None

    def probe(request):
        """Model the allowed probe write without launching a child process."""
        (request.cwd / "allowed").write_text("allowed")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.run.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.LinuxNativeBackend", Mock(return_value=native)
    )

    first = selection.select_sandbox_backend()
    second = selection.select_sandbox_backend()

    requests = [call.args[0] for call in native.run.call_args_list]
    assert first is native and second is native
    assert requests[0].workspace == requests[1].workspace
    assert requests[0].write_roots == requests[1].write_roots == (requests[0].workspace,)
    assert requests[0].cwd != requests[1].cwd
    assert requests[0].workspace == tmp_path / f"loop-sandbox-probe-{os.getuid()}"
    assert list(requests[0].workspace.iterdir()) == []


def test_linux_selection_refuses_unprotected_probe_parent(monkeypatch, tmp_path):
    """Fail closed when a stable Linux probe root would live under a writable unsafe parent."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Linux")
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    monkeypatch.setattr(selection.tempfile, "tempdir", str(parent))
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.capability_failure.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.LinuxNativeBackend", Mock(return_value=native)
    )

    selected = selection.select_sandbox_backend()

    failure = selected.capability_failure()
    assert isinstance(failure, CommandProcessResult)
    assert failure.outcome is SandboxOutcome.INVALID
    assert failure.detail == "Native sandbox capability probe could not be prepared."
    native.run.assert_not_called()


@pytest.mark.parametrize("root_kind", ["symlink", "permissions"])
def test_linux_selection_refuses_untrusted_existing_probe_root(monkeypatch, tmp_path, root_kind):
    """Reject a preexisting probe root that is a symlink or has broader permissions."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Linux")
    monkeypatch.setattr(selection.tempfile, "tempdir", str(tmp_path))
    root = tmp_path / f"loop-sandbox-probe-{os.getuid()}"
    if root_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        root.chmod(0o755)

    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.capability_failure.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.LinuxNativeBackend", Mock(return_value=native)
    )

    selected = selection.select_sandbox_backend()

    failure = selected.capability_failure()
    assert isinstance(failure, CommandProcessResult)
    assert failure.outcome is SandboxOutcome.INVALID
    assert failure.detail == "Native sandbox capability probe could not be prepared."
    native.run.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "exit_code", "outside_changed", "private_leaked", "detail"),
    [
        (SandboxOutcome.COMPLETED, 1, False, False, ""),
        (SandboxOutcome.COMPLETED, 0, False, False, ""),
        (SandboxOutcome.COMPLETED, 1, True, False, ""),
        (SandboxOutcome.COMPLETED, 1, False, True, ""),
        (SandboxOutcome.UNAVAILABLE, None, False, False, ""),
        (SandboxOutcome.UNAVAILABLE, None, False, False, "sandbox_apply: Operation not permitted"),
    ],
)
def test_darwin_selection_requires_enforced_native_write_boundary(
    monkeypatch, outcome, exit_code, outside_changed, private_leaked, detail
):
    """Darwin selection probes enforcement without consulting version or architecture."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        selection.platform, "machine", Mock(side_effect=AssertionError("machine gate"))
    )
    monkeypatch.setattr(
        selection.platform, "mac_ver", Mock(side_effect=AssertionError("version gate"))
    )
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"

    def run(request):
        """Simulate allowed workspace output and optional outside overwrite."""
        (request.workspace / "allowed").write_text("allowed", encoding="utf-8")
        if outside_changed:
            (request.workspace.parent / "outside" / "canary").write_text(
                "overwritten", encoding="utf-8"
            )
        if private_leaked:
            (request.workspace / "leaked").write_text("leaked", encoding="utf-8")
        return CommandProcessResult(outcome, exit_code=exit_code, detail=detail)

    native.run.side_effect = run
    monkeypatch.setattr(selection, "_macos_backend", Mock(return_value=native))

    selected = selection.select_sandbox_backend()

    native.run.assert_called_once()
    if (
        outcome is SandboxOutcome.COMPLETED
        and exit_code == 1
        and not outside_changed
        and not private_leaked
    ):
        assert selected is native
    else:
        assert selected is not native
        native.close.assert_called_once()
        assert (
            "Nested sandbox" if detail else "capability probe failed"
        ) in selected.capability_failure()


def test_darwin_selection_closes_backend_when_probe_cannot_bind(monkeypatch):
    """A failed probe setup leaves no usable backend or retained native service."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Darwin")
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    monkeypatch.setattr(selection, "_macos_backend", Mock(return_value=native))
    monkeypatch.setattr(
        selection.SandboxRequest, "create", Mock(side_effect=ValueError("private path"))
    )

    selected = selection.select_sandbox_backend()

    assert "could not be prepared" in selected.capability_failure()
    native.close.assert_called_once()
    native.run.assert_not_called()


def test_darwin_selection_constructs_builtin_backend_without_external_probe(monkeypatch):
    """The native factory constructs the built-in adapter before its mocked qualification."""
    monkeypatch.setattr(selection.platform, "system", lambda: "Darwin")
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.run.return_value = CommandProcessResult(SandboxOutcome.UNAVAILABLE)
    factory = Mock(return_value=native)
    monkeypatch.setattr("loop.execution.sandbox.macos.MacOSSeatbeltBackend", factory)

    selected = selection.select_sandbox_backend()

    factory.assert_called_once_with()
    native.run.assert_called_once()
    native.close.assert_called_once()
    assert "capability probe failed" in selected.capability_failure()
