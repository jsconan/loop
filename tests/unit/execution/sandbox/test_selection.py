"""Verify host qualification selects only a working built-in sandbox."""

import os
import platform
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox import (
    CommandProcessResult,
    SandboxCapability,
    SandboxOutcome,
    selection,
)
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend


def _native() -> Mock:
    """Configure a mocked backend with its real probe layout and failure semantics."""
    native = Mock()
    native.scratch_prefix = "loop-sandbox-"
    native.capability_failure.return_value = None
    if platform.system() == "Linux":
        native.qualification_layout.side_effect = LinuxNativeBackend().qualification_layout
        native.qualification_failure.side_effect = lambda result, reason: (
            LinuxNativeBackend.qualification_failure(native, result, reason)
        )
    else:
        native.qualification_layout.return_value = selection.QualificationProbeLayout(None, None)
        native.qualification_failure.side_effect = lambda result, reason: (
            MacOSSeatbeltBackend.qualification_failure(native, result, reason)
        )
    return native


def test_unsupported_host_fails_closed_without_instantiating_seatbelt(monkeypatch):
    """An unsupported host never constructs a Seatbelt backend."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    factory = Mock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.create_backend", factory)
    backend = selection.select_sandbox_backend()

    assert "No native command sandbox" in backend.capability_failure()
    assert backend.run(Mock()).outcome is SandboxOutcome.UNAVAILABLE
    assert backend.qualification_probe(Mock()).outcome is SandboxOutcome.UNAVAILABLE
    backend.complete_qualification(Mock())
    assert backend.qualification_layout() == selection.QualificationProbeLayout(None, None)
    assert backend.qualification_failure(None, "other") == backend.capability_failure()
    backend.close()
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
    "outcome",
    [
        SandboxOutcome.INVALID,
        SandboxOutcome.STALE,
        SandboxOutcome.UNAVAILABLE,
        SandboxOutcome.UNSUPPORTED,
    ],
)
def test_linux_selection_retains_typed_preflight_failure(monkeypatch, outcome):
    """Initial selection preserves integrity outcomes without running the probe or command."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    failure = CommandProcessResult(outcome, detail="Sanitized native authority failure.")
    native.capability_failure.return_value = failure
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )
    selected = selection.select_sandbox_backend()
    assert selected.capability_failure() == failure
    assert selected.run(Mock()) == failure
    native.qualification_probe.assert_not_called()


@pytest.mark.parametrize("enforced", [False, True])
def test_linux_selection_requires_operational_enforcement(monkeypatch, enforced):
    """Intact metadata alone cannot select a Linux backend that fails native enforcement."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None

    def probe(request):
        """Model enforcement effects on disposable probe files without starting processes."""
        assert request.shell_environment()["TMPDIR"]
        assert request.local_network
        (request.cwd / "allowed").write_text("allowed")
        if not enforced:
            (request.cwd / "leaked").write_text("probe-secret")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.qualification_probe.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )
    selected = selection.select_sandbox_backend()
    if enforced:
        assert selected is native
    else:
        assert selected.capability_failure().outcome is SandboxOutcome.INVALID
    native.qualification_probe.assert_called_once()


def test_linux_selection_retains_unsupported_probe_result(monkeypatch):
    """A feature unavailable during probing retains its no-retry classification."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None
    failure = CommandProcessResult(SandboxOutcome.UNSUPPORTED, detail="Unsupported request.")
    native.qualification_probe.return_value = failure
    monkeypatch.setattr("loop.execution.sandbox.linux.backend.LinuxNativeBackend", lambda: native)
    selected = selection.select_sandbox_backend()
    assert selected.capability_failure() == failure


def test_linux_selection_isolates_probe_workspaces_across_calls(monkeypatch, tmp_path):
    """Keep each Linux startup probe from scanning another probe's changing workspace."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(selection.tempfile, "tempdir", str(tmp_path))
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None

    def probe(request):
        """Model the allowed probe write without launching a child process."""
        (request.cwd / "allowed").write_text("allowed")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.qualification_probe.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )

    first = selection.select_sandbox_backend()
    second = selection.select_sandbox_backend()

    requests = [call.args[0] for call in native.qualification_probe.call_args_list]
    assert first is native and second is native
    assert requests[0].workspace != requests[1].workspace
    assert all(request.write_roots == (request.workspace,) for request in requests)
    assert requests[0].cwd != requests[1].cwd
    parent = tmp_path / f"loop-sandbox-probe-{os.getuid()}"
    assert all(request.workspace.parent.parent == parent for request in requests)
    assert list(parent.iterdir()) == []


def test_linux_selection_binds_scratch_for_a_receiver_with_a_different_temp_root(
    monkeypatch, tmp_path
):
    """Restore probe scratch even when the receiver has a different temporary directory."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    client_root = tmp_path / "client"
    receiver_root = tmp_path / "receiver"
    client_root.mkdir(mode=0o700)
    receiver_root.mkdir(mode=0o700)
    monkeypatch.setattr(selection.tempfile, "tempdir", str(client_root))
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None

    def probe(request):
        """Restore the bound request as an independently configured receiver would."""
        wire = request.to_wire()
        with monkeypatch.context() as receiver:
            receiver.setattr(selection.tempfile, "tempdir", str(receiver_root))
            restored = selection.SandboxRequest.from_wire(wire)
        assert restored.scratch_root == Path(request.shell_environment()["TMPDIR"])
        assert restored.scratch_data is False
        (request.cwd / "allowed").write_text("allowed")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.qualification_probe.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )

    assert selection.select_sandbox_backend() is native
    native.qualification_probe.assert_called_once()


def test_linux_selection_refuses_unprotected_probe_parent(monkeypatch, tmp_path):
    """Fail closed when a stable Linux probe root would live under a writable unsafe parent."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    monkeypatch.setattr(selection.tempfile, "tempdir", str(parent))
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )

    selected = selection.select_sandbox_backend()

    failure = selected.capability_failure()
    assert isinstance(failure, CommandProcessResult)
    assert failure.outcome is SandboxOutcome.INVALID
    assert failure.detail == "Native sandbox capability probe could not be prepared."
    native.qualification_probe.assert_not_called()


@pytest.mark.parametrize("root_kind", ["symlink", "permissions"])
def test_linux_selection_refuses_untrusted_existing_probe_root(monkeypatch, tmp_path, root_kind):
    """Reject a preexisting probe root that is a symlink or has broader permissions."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(selection.tempfile, "tempdir", str(tmp_path))
    root = tmp_path / f"loop-sandbox-probe-{os.getuid()}"
    if root_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    else:
        root.mkdir(mode=0o700)
        root.chmod(0o755)

    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset({SandboxCapability.LOCAL_NETWORK})
    native.capability_failure.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.backend.LinuxNativeBackend", Mock(return_value=native)
    )

    selected = selection.select_sandbox_backend()

    failure = selected.capability_failure()
    assert isinstance(failure, CommandProcessResult)
    assert failure.outcome is SandboxOutcome.INVALID
    assert failure.detail == "Native sandbox capability probe could not be prepared."
    native.qualification_probe.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "exit_code", "outside_changed", "private_leaked", "detail"),
    [
        (SandboxOutcome.COMPLETED, 1, False, False, ""),
        (SandboxOutcome.COMPLETED, 0, False, False, ""),
        (SandboxOutcome.COMPLETED, 1, True, False, ""),
        (SandboxOutcome.COMPLETED, 1, False, True, ""),
        (SandboxOutcome.UNAVAILABLE, None, False, False, ""),
        (SandboxOutcome.INVALID, None, False, False, ""),
        (SandboxOutcome.STALE, None, False, False, ""),
        (SandboxOutcome.UNSUPPORTED, None, False, False, ""),
        (SandboxOutcome.UNAVAILABLE, None, False, False, "sandbox_apply: Operation not permitted"),
    ],
)
def test_darwin_selection_requires_enforced_native_write_boundary(
    monkeypatch, outcome, exit_code, outside_changed, private_leaked, detail
):
    """Darwin selection probes enforcement without consulting version or architecture."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", Mock(side_effect=AssertionError("machine gate")))
    monkeypatch.setattr(platform, "mac_ver", Mock(side_effect=AssertionError("version gate")))
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset()

    def run(request):
        """Simulate allowed workspace output and optional outside overwrite."""
        assert not request.local_network
        (request.workspace / "allowed").write_text("allowed", encoding="utf-8")
        if outside_changed:
            (request.workspace.parent / "outside" / "canary").write_text(
                "overwritten", encoding="utf-8"
            )
        if private_leaked:
            (request.workspace / "leaked").write_text("leaked", encoding="utf-8")
        return CommandProcessResult(outcome, exit_code=exit_code, detail=detail)

    native.qualification_probe = Mock(side_effect=run)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.create_backend", Mock(return_value=native)
    )

    selected = selection.select_sandbox_backend()

    native.qualification_probe.assert_called_once()
    if (
        outcome is SandboxOutcome.COMPLETED
        and exit_code == 1
        and not outside_changed
        and not private_leaked
    ):
        assert selected is native
        native.complete_qualification.assert_called_once()
    else:
        assert selected is not native
        native.complete_qualification.assert_not_called()
        native.close.assert_called_once()
        failure = selected.capability_failure()
        if outcome in {
            SandboxOutcome.INVALID,
            SandboxOutcome.STALE,
            SandboxOutcome.UNSUPPORTED,
        }:
            assert failure.outcome is outcome
            assert "capability probe failed" in failure.detail
        else:
            assert ("Nested sandbox" if detail else "capability probe failed") in failure


def test_darwin_selection_keeps_private_temporary_ancestry_out_of_probe_workspace(
    monkeypatch, tmp_path
):
    """Probe a safe workspace while binding private scratch under the configured temp root."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    private = tmp_path / "Library" / "scratch"
    private.mkdir(mode=0o700, parents=True)
    safe = tmp_path / "safe"
    safe.mkdir(mode=0o700)
    monkeypatch.setattr(selection.tempfile, "tempdir", str(private))
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset()
    native.qualification_layout.return_value = selection.QualificationProbeLayout(safe, None)

    def probe(request):
        """Model enforcement and verify the distinct workspace and scratch authorities."""
        assert request.workspace.is_relative_to(safe)
        assert request.scratch_root.parent == private
        assert request.scratch_data is False
        (request.workspace / "allowed").write_text("allowed")
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1)

    native.qualification_probe.side_effect = probe
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.create_backend", Mock(return_value=native)
    )

    assert selection.select_sandbox_backend() is native
    native.qualification_probe.assert_called_once()
    native.complete_qualification.assert_called_once()


def test_darwin_selection_closes_backend_when_probe_cannot_bind(monkeypatch):
    """A failed probe setup leaves no usable backend or retained native service."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.create_backend", Mock(return_value=native)
    )
    monkeypatch.setattr(
        selection.SandboxRequest, "create", Mock(side_effect=ValueError("private path"))
    )

    selected = selection.select_sandbox_backend()

    assert "could not be prepared" in selected.capability_failure()
    native.close.assert_called_once()
    native.qualification_probe.assert_not_called()


def test_darwin_selection_constructs_builtin_backend_without_external_probe(monkeypatch):
    """The native factory constructs the built-in adapter before its mocked qualification."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    native = _native()
    native.scratch_prefix = "loop-sandbox-"
    native.qualification_capabilities = frozenset()
    native.qualification_probe = Mock(return_value=CommandProcessResult(SandboxOutcome.UNAVAILABLE))
    factory = Mock(return_value=native)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.MacOSSeatbeltBackend", factory)

    selected = selection.select_sandbox_backend()

    factory.assert_called_once_with()
    native.qualification_probe.assert_called_once()
    native.close.assert_called_once()
    assert "capability probe failed" in selected.capability_failure()
