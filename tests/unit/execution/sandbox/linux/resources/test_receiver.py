"""Verify protected receiver admission and complete deadline-bound result delivery."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from loop.execution.sandbox.contracts import CommandProcessResult, SandboxOutcome, SandboxRequest
from loop.execution.sandbox.linux import runtime as linux_runtime
from loop.execution.sandbox.linux.resources import receiver as module


@pytest.fixture
def wire_restorer():
    """Retain the real public parser before receiver collaborator isolation."""
    return SandboxRequest.from_wire


@pytest.fixture
def receiver(tmp_path, monkeypatch, wire_restorer):
    """Isolate immutable installed ancestry and native admission at their public boundaries."""
    entry = tmp_path.resolve() / "loop_receiver.py"
    entry.write_bytes(b"receiver")
    entry.chmod(0o555)
    monkeypatch.setattr(module, "__file__", str(entry))
    monkeypatch.setattr(module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(module.sys, "path", list(module.sys.path))
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    original = Path.lstat
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: SimpleNamespace(
            st_uid=0,
            st_mode=original(path).st_mode & ~0o6022,
            st_dev=original(path).st_dev,
            st_ino=original(path).st_ino,
        ),
    )
    admitted = Mock(return_value=b"opaque private request")
    monkeypatch.setattr(linux_runtime, "receive_installed_request", admitted)
    restored = Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=17))
    monkeypatch.setattr(
        SandboxRequest,
        "from_wire",
        Mock(
            return_value=SimpleNamespace(
                attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7", deadline=20
            )
        ),
    )
    monkeypatch.setattr(linux_runtime, "execute_installed_request", restored)
    return entry, admitted, restored


def test_receiver_forwards_typed_result_after_protected_admission(receiver, monkeypatch):
    """Socket sendall forwards the entire result within the original remaining deadline."""
    entry, admitted, restored = receiver
    connection = MagicMock()
    connection.__enter__.return_value = connection
    factory = Mock(return_value=connection)
    monkeypatch.setattr(module.socket, "fromfd", factory)
    restored.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED, exit_code=17, stdout="x" * 1000000, stderr="y" * 1000000
    )
    module.main()
    admitted.assert_called_once_with(entry.parent / "management.json", 15)
    restored.assert_called_once_with(entry.parent / "management.json", b"opaque private request")
    factory.assert_called_once_with(1, module.socket.AF_UNIX, module.socket.SOCK_STREAM)
    connection.settimeout.assert_called_once_with(15)
    data = connection.sendall.call_args.args[0]
    assert json.loads(data)["result"]["exit_code"] == 17
    assert len(json.loads(data)["result"]["stdout"]) == 1000000
    connection.__exit__.assert_called_once()


def test_receiver_expired_delivery_refuses_before_socket_send(receiver, monkeypatch):
    """A finished command cannot extend the fixed cleanup/result-delivery budget."""
    monkeypatch.setattr(module.time, "monotonic", lambda: 25)
    factory = Mock()
    monkeypatch.setattr(module.socket, "fromfd", factory)
    with pytest.raises(TimeoutError, match="result delivery deadline expired"):
        module.main()
    factory.assert_not_called()


@pytest.mark.parametrize(
    "change", ["name", "alias", "owner", "mode", "symlink", "root", "admission"]
)
def test_receiver_invalid_authority_never_reports_admission(receiver, monkeypatch, change, capsys):
    """Unsafe ancestry, root entry and failed native admission refuse without a success response."""
    entry, admitted, restored = receiver
    if change == "name":
        monkeypatch.setattr(module, "__file__", str(entry.with_name("other.py")))
    elif change == "alias":
        monkeypatch.setattr(Path, "resolve", lambda *args, **kw: Path("/other"))
    elif change == "root":
        monkeypatch.setattr(module.os, "getuid", lambda: 0)
    elif change == "admission":
        admitted.side_effect = ValueError("unsafe original receipt")
    else:
        original = Path.lstat

        def details(path):
            """Change the selected ancestry invariant without modifying real host metadata."""
            info = original(path)
            if path == entry:
                return SimpleNamespace(
                    st_uid=1000 if change == "owner" else 0,
                    st_mode=0o120777
                    if change == "symlink"
                    else info.st_mode | (0o022 if change == "mode" else 0),
                )
            return info

        monkeypatch.setattr(Path, "lstat", details)
    category = ValueError if change in {"name", "alias", "admission"} else PermissionError
    message = (
        "fixed canonical"
        if change in {"name", "alias"}
        else "ordinary service identity"
        if change == "root"
        else "unsafe original receipt"
        if change == "admission"
        else "administrator protected"
    )
    with pytest.raises(category, match=message):
        module.main()
    assert admitted.call_count == (change == "admission")
    assert capsys.readouterr().out == ""
    restored.assert_not_called()


def test_receiver_invalid_request_never_reports_readiness(receiver, capsys):
    """A malformed or stale admitted request refuses before the fixed unavailable response."""
    entry, admitted, restored = receiver
    restored.side_effect = ValueError("request snapshot changed")
    with pytest.raises(ValueError, match="request snapshot changed"):
        module.main()
    admitted.assert_called_once()
    restored.assert_called_once_with(entry.parent / "management.json", b"opaque private request")
    assert capsys.readouterr().out == ""


def test_receiver_script_entry_refuses_uninstalled_code(monkeypatch):
    """Direct resource-script execution cannot adopt an application source path as retained code."""
    with pytest.raises(ValueError, match="fixed canonical"):
        runpy.run_path(module.__file__, run_name="__main__")


def test_receiver_delivers_timeout_capture_after_execution_expiry(receiver, monkeypatch):
    """Timeout stdout/stderr survive cleanup and delivery without changing execution authority."""
    _, _, restored = receiver
    clock = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    expected = CommandProcessResult(
        SandboxOutcome.TIMED_OUT,
        stdout="before-timeout-out",
        stderr="before-timeout-err",
        stdout_discarded=None,
        stderr_discarded=None,
        possible_effects=True,
    )

    def execute(*_):
        """Consume execution and cleanup time before returning captured timeout streams."""
        clock[0] = 21.0
        return expected

    restored.side_effect = execute
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(module.socket, "fromfd", Mock(return_value=connection))
    module.main()
    connection.settimeout.assert_called_once_with(4.0)
    assert (
        CommandProcessResult.from_wire(
            connection.sendall.call_args.args[0], "1248c892-42dc-4b68-a3fb-30d2a70a93a7"
        )
        == expected
    )


def test_receiver_disconnect_does_not_retry_delivery(receiver, monkeypatch):
    """Client disconnect closes the response channel and never repeats execution."""
    _, _, executed = receiver
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.sendall.side_effect = BrokenPipeError("client disconnected")
    monkeypatch.setattr(module.socket, "fromfd", Mock(return_value=connection))
    with pytest.raises(BrokenPipeError, match="client disconnected"):
        module.main()
    executed.assert_called_once()
    connection.sendall.assert_called_once()
    connection.__exit__.assert_called_once()


@pytest.mark.parametrize("phase", ["restoration", "preparation"])
def test_receiver_delivers_preparation_expiry_with_original_identity(receiver, monkeypatch, phase):
    """Validated expiry produces a timeout envelope rather than an empty transport response."""
    from loop.execution.sandbox.contracts import SandboxRequestExpired

    _, _, executed = receiver
    request = SimpleNamespace(attempt_id="1248c892-42dc-4b68-a3fb-30d2a70a93a7", deadline=20)
    if phase == "restoration":
        SandboxRequest.from_wire.side_effect = SandboxRequestExpired(request)
    else:
        executed.side_effect = TimeoutError("compilation deadline expired")
    monkeypatch.setattr(module.time, "monotonic", lambda: 21)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(module.socket, "fromfd", Mock(return_value=connection))
    module.main()
    result = CommandProcessResult.from_wire(
        connection.sendall.call_args.args[0], request.attempt_id
    )
    assert result.outcome is SandboxOutcome.TIMED_OUT
    connection.settimeout.assert_called_once_with(4)
    assert executed.call_count == (phase == "preparation")
    if phase == "preparation":
        assert result.stdout_discarded is None
        assert result.stderr_discarded is None


@pytest.mark.parametrize("malformed", [False, True])
def test_receiver_real_expired_snapshot_preserves_validated_authority(
    receiver, wire_restorer, monkeypatch, malformed
):
    """Real restoration authenticates expiry results and refuses malformed expired frames."""
    entry, admitted, executed = receiver
    request = SandboxRequest.create(
        source="printf never-entered",
        cwd=entry.parent,
        workspace=entry.parent,
        read_roots=(),
        write_roots=(),
        network=False,
        environment={"PATH": "/usr/bin:/bin"},
        policy_version="fixture",
        workspace_id="fixture",
        deadline=20,
    )
    payload = request.to_wire()
    if malformed:
        value = json.loads(payload)
        value["request"]["network"] = 1
        payload = json.dumps(value).encode()
    admitted.return_value = payload
    monkeypatch.setattr(SandboxRequest, "from_wire", wire_restorer)
    monkeypatch.setattr(module.time, "monotonic", lambda: 21)
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(module.socket, "fromfd", Mock(return_value=connection))
    if malformed:
        with pytest.raises(ValueError, match="boolean authority"):
            module.main()
        connection.sendall.assert_not_called()
    else:
        module.main()
        result = CommandProcessResult.from_wire(
            connection.sendall.call_args.args[0], request.attempt_id
        )
        assert result.outcome is SandboxOutcome.TIMED_OUT
        assert result.stdout_discarded == result.stderr_discarded == 0
        assert not result.possible_effects
        connection.settimeout.assert_called_once_with(4)
    executed.assert_not_called()
