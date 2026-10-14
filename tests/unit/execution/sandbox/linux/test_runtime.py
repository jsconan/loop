"""Verify Linux request admission, transport and confined execution."""

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.contracts import SandboxOutcome
from loop.execution.sandbox.linux import host as linux_host
from loop.execution.sandbox.linux import runtime as linux_runtime

from .helpers import publish


@pytest.fixture(autouse=True)
def runtime_revision(installation):
    """Include current native namespace semantics in each isolated retained installation."""
    record, arguments = installation
    path = record.parent / "loop/execution/sandbox/linux/constants.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(Path(linux_runtime.__file__).with_name("constants.py").read_bytes())
    path.chmod(0o444)
    arguments[-1]["loop/execution/sandbox/linux/constants.py"] = linux_host.bind_host_file(path)


@pytest.mark.parametrize("change", [None, "receipt", "record"])
def test_installed_admission_checks_original_receipt_and_late_record(
    installation, monkeypatch, change
):
    """Bind installed roles to saved originals and revalidate after receiving."""
    record, arguments = installation
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    if change == "receipt":
        receipt["receiver"] = arguments[6]
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))

    def receive(*args):
        """Return opaque content and optionally simulate late original record loss."""
        if change == "record":
            record.unlink()
        return b"opaque request"

    received = Mock(side_effect=receive)
    monkeypatch.setattr(linux_runtime, "receive_request", received)
    if change:
        with pytest.raises(
            ValueError if change == "receipt" else FileNotFoundError,
            match="Receipt admission differs" if change == "receipt" else "management.json",
        ):
            linux_runtime.receive_installed_request(record, 50)
        assert received.call_count == (change == "record")
    else:
        assert linux_runtime.receive_installed_request(record, 50) == b"opaque request"
        received.assert_called_once_with(arguments[0], receipt, 50)
    linux_runtime.installation_files.assert_called_once_with(
        record.parent / "receipt.json", deadline=50
    )


@pytest.mark.parametrize("change", [None, "receipt", "record", "deadline", "late_deadline"])
def test_installed_transport_retains_original_receipt_record_and_deadline(
    installation, monkeypatch, change
):
    """Ordinary exchange uses saved original authority and refuses drift or expired budgets."""
    record, arguments = installation
    publish(installation)
    receipt = {**arguments[1], "receipt": linux_host.bind_host_file(record)}
    if change == "receipt":
        receipt["receiver"] = arguments[6]
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    clock = [10]
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])

    def exchange(*args):
        """Return opaque response and optionally invalidate original late authority or budget."""
        if change == "record":
            record.unlink()
        if change == "late_deadline":
            clock[0] = 55
        return b"unavailable response"

    exchanged = Mock(side_effect=exchange)
    monkeypatch.setattr(linux_runtime, "exchange_request", exchanged)
    if change:
        with pytest.raises(
            FileNotFoundError
            if change == "record"
            else TimeoutError
            if "deadline" in change
            else ValueError,
            match="management.json"
            if change == "record"
            else "transport deadline"
            if "deadline" in change
            else "Receipt transport differs",
        ):
            linux_runtime.exchange_installed_request(
                record, b"opaque request", 10 if change == "deadline" else 50
            )
        assert exchanged.call_count == (change in {"record", "late_deadline"})
    else:
        assert (
            linux_runtime.exchange_installed_request(record, b"opaque request", 50)
            == b"unavailable response"
        )
        exchanged.assert_called_once_with(arguments[0], receipt, b"opaque request", 50)


def test_installed_transport_refuses_deadline_expiring_after_record_check(
    installation, monkeypatch
):
    """The final record verification cannot return a response after its delivery budget."""
    from types import SimpleNamespace

    record, arguments = installation
    clock = [10]

    @contextmanager
    def verified(*, deadline):
        """Expire only after original record verification has completed."""
        yield
        clock[0] = deadline

    authority = Mock(open_verified=Mock(side_effect=verified))
    owner = SimpleNamespace(config=arguments[0], files=arguments[1], artifacts=arguments[-1])
    receipt = {**arguments[1], "receipt": authority}
    monkeypatch.setattr(linux_runtime, "read_generation", Mock(return_value=(owner, authority)))
    monkeypatch.setattr(linux_runtime, "installation_files", Mock(return_value=receipt))
    exchange = Mock(return_value=b"opaque response")
    monkeypatch.setattr(linux_runtime, "exchange_request", exchange)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="transport deadline"):
        linux_runtime.exchange_installed_request(record, b"opaque request", 50)
    exchange.assert_called_once()
    authority.open_verified.assert_called_once_with(deadline=55)


@pytest.mark.parametrize("missing", [False, True])
def test_installed_authority_refuses_incompatible_namespace_revision(
    installation, monkeypatch, missing
):
    """A package upgrade cannot send new namespace semantics to an older retained receiver."""
    record, arguments = installation
    name = "loop/execution/sandbox/linux/constants.py"
    if missing:
        path = arguments[-1][name].path
        path.unlink()
        for parent in path.parents:
            if parent == record.parent:
                break
            parent.rmdir()
        del arguments[-1][name]
    else:
        path = arguments[-1][name].path
        path.chmod(0o644)
        path.write_bytes(b"old native revision")
        path.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(path)
    publish(installation)
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    transport = Mock()
    monkeypatch.setattr(linux_runtime, "exchange_request", transport)
    for operation in (
        lambda: linux_runtime.execution_authority(record),
        lambda: linux_runtime.exchange_installed_request(record, b"request", 50),
    ):
        with pytest.raises(ValueError, match="runtime revision changed"):
            operation()
    transport.assert_not_called()


@pytest.fixture
def execution(installation, monkeypatch):
    """Bind a complete runtime while controlling privileged launch collaborators."""
    from types import SimpleNamespace

    from loop.utils.process import ProcessCapture, ProcessCaptureStatus

    record, arguments = installation
    for name, data in (
        ("cache.lock", b"test\n"),
        ("mask.empty", b""),
        ("loop_payload.py", b"entry"),
    ):
        path = record.parent / name
        path.write_bytes(data)
        path.chmod(0o444)
        arguments[-1][name] = linux_host.bind_host_file(path)
    bootstrap_path = record.parent / "policies/root/bootstrap.bin"
    bootstrap_path.parent.mkdir(parents=True)
    bootstrap_path.write_bytes(b"bootstrap")
    bootstrap_path.chmod(0o444)
    arguments[-1]["policies/root/bootstrap.bin"] = linux_host.bind_host_file(bootstrap_path)
    publish(installation)
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(linux_runtime.os, name, lambda: 1000)
    request = Mock(
        deadline=50,
        attempt_id="11111111-1111-4111-8111-111111111111",
        environment=(),
        aliases=(),
        policy_version=linux_runtime.NATIVE_POLICY_VERSION,
    )
    request.shell_environment.return_value = {"TMPDIR": "/tmp/private"}
    request.paths_are_current.return_value = True
    request.host_command_runtime_signature.return_value = "a" * 64
    monkeypatch.setattr(linux_runtime.SandboxRequest, "from_wire", Mock(return_value=request))
    monkeypatch.setattr(
        linux_runtime,
        "installation_files",
        Mock(return_value={**arguments[1], "apparmor": arguments[6]}),
    )
    original_bind = linux_runtime.bind_host_file
    monkeypatch.setattr(
        linux_runtime,
        "bind_host_file",
        lambda path, **_kwargs: (
            arguments[6] if path == Path("/usr/bin/python3") else original_bind(path, **_kwargs)
        ),
    )
    monkeypatch.setattr(
        linux_runtime,
        "namespace_arguments",
        Mock(return_value=nullcontext((("--unshare-pid",), (77,)))),
    )
    monkeypatch.setattr(linux_runtime, "runtime_read_roots", Mock(return_value=(Path("/usr"),)))
    monkeypatch.setattr(linux_runtime, "payload_profile", Mock(return_value="payload_profile"))
    monkeypatch.setattr(linux_runtime, "compile_payload", Mock(return_value=b"compiled"))
    monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 10)
    original_open, original_close = os.open, os.close
    monkeypatch.setattr(linux_runtime.os, "O_PATH", 0, raising=False)
    monkeypatch.setattr(
        linux_runtime.os,
        "open",
        lambda path, *args: (
            999
            if path in (record.parent, Path("/sys/kernel/security/apparmor"))
            else original_open(path, *args)
        ),
    )
    monkeypatch.setattr(
        linux_runtime.os, "close", lambda fd: None if fd == 999 else original_close(fd)
    )
    launch = Mock()
    monkeypatch.setattr(linux_runtime.subprocess, "Popen", launch)
    capture = ProcessCapture(
        ProcessCaptureStatus.COMPLETED,
        exit_code=17,
        stdout="LOOP_PAYLOAD_" + request.attempt_id + "\noutput",
        stderr="error",
        stdout_discarded=3,
    )
    supervisor = Mock(return_value=capture)
    monkeypatch.setattr(linux_runtime, "supervise_payload", supervisor)
    return SimpleNamespace(
        record=record,
        arguments=arguments,
        request=request,
        launch=launch,
        capture=capture,
        supervisor=supervisor,
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "caller",
        "revision",
        "scratch",
        "cache",
        "mask",
        "entry",
        "nonempty",
        "receipt",
        "bootstrap",
        "write",
        "size",
        "stale",
        "expired",
        "late-expired",
        "no-header",
        "no-header-overflow",
        "unknown-discard",
        "overflow",
        "timeout",
        "pre-entry-timeout",
        "pre-entry-cancel",
    ],
)
def test_execute_installed_request_authenticates_launch_and_result(execution, monkeypatch, change):
    """A complete original runtime returns only a bounded attempt-bound result or fails closed."""
    from dataclasses import replace

    from loop.execution.sandbox.contracts import SandboxOutcome
    from loop.utils.process import ProcessCaptureStatus

    state = execution
    if change == "caller":
        monkeypatch.setattr(linux_runtime.os, "geteuid", lambda: 0)
    elif change == "revision":
        state.request.policy_version = "linux-native-unsupported"
    elif change == "scratch":
        state.request.shell_environment.return_value = {}
    elif change in {"cache", "mask", "entry", "bootstrap", "nonempty"}:
        name = {
            "cache": "cache.lock",
            "mask": "mask.empty",
            "entry": "loop_payload.py",
            "bootstrap": "policies/root/bootstrap.bin",
            "nonempty": "mask.empty",
        }[change]
        if change == "nonempty":
            path = state.arguments[-1][name].path
            path.chmod(0o644)
            path.write_bytes(b"unsafe")
            path.chmod(0o444)
            state.arguments[-1][name] = linux_host.bind_host_file(path)
        else:
            state.arguments[-1][name].path.unlink()
            if change == "bootstrap":
                (state.record.parent / "policies/root").rmdir()
                (state.record.parent / "policies").rmdir()
            del state.arguments[-1][name]
        state.record.unlink()
        publish((state.record, state.arguments))
    elif change == "receipt":
        linux_runtime.installation_files.return_value["receiver"] = state.arguments[6]
    elif change == "stale":
        state.request.paths_are_current.return_value = False
    elif change == "expired":
        monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: 50)
    elif change == "late-expired":
        clock = [10]
        monkeypatch.setattr(linux_runtime.time, "monotonic", lambda: clock[0])
        state.request.paths_are_current.side_effect = lambda: clock.__setitem__(0, 50) or True
    elif change in {"write", "size"}:
        if change == "size":
            linux_runtime.compile_payload.return_value = b"x" * 1_600_000
        else:
            stream = Mock()
            stream.write.return_value = 0
            monkeypatch.setattr(
                linux_runtime.tempfile, "TemporaryFile", lambda: nullcontext(stream)
            )
    elif change in {"no-header", "no-header-overflow"}:
        state.supervisor.return_value = replace(
            state.capture, stdout="forged" if change == "no-header" else "x" * 1_000_080
        )
    elif change == "unknown-discard":
        state.supervisor.return_value = replace(state.capture, stdout_discarded=None)
    elif change == "overflow":
        state.supervisor.return_value = replace(
            state.capture, stdout=state.capture.stdout + "x" * 1_000_000
        )
    elif change in {"pre-entry-timeout", "pre-entry-cancel"}:
        state.supervisor.return_value = replace(
            state.capture,
            status=ProcessCaptureStatus.TIMED_OUT
            if change == "pre-entry-timeout"
            else ProcessCaptureStatus.CANCELLED,
            exit_code=None,
            stdout="bootstrap-out",
            stderr="bootstrap-error",
            stdout_discarded=None,
            stderr_discarded=13,
        )
    elif change == "timeout":
        state.supervisor.return_value = replace(
            state.capture, status=ProcessCaptureStatus.TIMED_OUT, exit_code=None
        )
    failures = {
        "caller",
        "revision",
        "scratch",
        "cache",
        "mask",
        "entry",
        "nonempty",
        "receipt",
        "bootstrap",
        "write",
        "size",
        "stale",
        "expired",
        "late-expired",
    }
    if change in failures:
        with pytest.raises(TimeoutError if change in {"expired", "late-expired"} else ValueError):
            linux_runtime.execute_installed_request(state.record, b"snapshot")
        state.launch.assert_not_called()
    else:
        result = linux_runtime.execute_installed_request(state.record, b"snapshot")
        assert result.outcome is (
            SandboxOutcome.INVALID
            if change in {"no-header", "no-header-overflow"}
            else SandboxOutcome.TIMED_OUT
            if change in {"timeout", "pre-entry-timeout"}
            else SandboxOutcome.CANCELLED
            if change == "pre-entry-cancel"
            else SandboxOutcome.COMPLETED
        )
        assert result.possible_effects
        assert len(result.stdout) <= 1_000_000
        if change in {"pre-entry-timeout", "pre-entry-cancel"}:
            assert (result.stdout, result.stderr) == ("bootstrap-out", "bootstrap-error")
            assert result.stdout_discarded is None
            assert result.stderr_discarded == 13
            assert result.exit_code is None
        elif change == "no-header":
            assert (result.stdout, result.stderr) == ("forged", "error")
        elif change == "no-header-overflow":
            assert result.stdout == "x" * 1_000_000
            assert result.stdout_discarded == state.capture.stdout_discarded + 80
        elif change == "unknown-discard":
            assert result.stdout_discarded is None
        elif change == "overflow":
            assert result.stdout_discarded == 9
        elif change is None:
            assert (result.stdout, result.stderr, result.exit_code) == ("output", "error", 17)
        assert state.launch.call_args.kwargs["pass_fds"] == (77, 999, 999)
        state.supervisor.assert_called_once_with(
            state.launch.return_value, state.arguments[0], 50, connection=0
        )


@pytest.mark.parametrize("unknown", [False, True])
def test_execute_installed_request_restores_alias_output_with_bounded_capture(execution, unknown):
    """Preserve original alias spellings, numeric suffixes and loss metadata in both streams."""
    from dataclasses import replace

    state = execution
    state.request.aliases = tuple(
        (f"/root{index}", Path(f"/tmp/host/r{index}"), Path("/work")) for index in range(11)
    )
    text = "/.loop-private-paths/r10/file /.loop-private-paths/r1/file /.loop-private-paths"
    state.supervisor.return_value = replace(
        state.capture,
        stdout="LOOP_PAYLOAD_" + state.request.attempt_id + "\n" + text,
        stderr=text + "x" * 1_000_000,
        stderr_discarded=None if unknown else 3,
    )
    result = linux_runtime.execute_installed_request(state.record, b"snapshot")
    assert result.stdout == "/tmp/host/r10/file /tmp/host/r1/file /tmp/host"
    assert result.stderr.startswith(result.stdout)
    assert len(result.stderr) == 1_000_000
    assert (
        result.stderr_discarded is None
        if unknown
        else result.stderr_discarded == 3 + len(result.stdout)
    )


def test_execute_installed_request_returns_typed_path_preparation_refusal(execution):
    """Return an authenticated invalid result when path preparation refuses before launch."""
    state = execution
    linux_runtime.namespace_arguments.side_effect = ValueError(
        "Linux hardlinked file aliases protected or private data."
    )

    result = linux_runtime.execute_installed_request(state.record, b"snapshot")

    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "Linux hardlinked file aliases protected or private data."
    assert not result.possible_effects
    assert result.failure_context == "Native sandbox path preparation was refused."
    state.launch.assert_not_called()
    state.supervisor.assert_not_called()


@pytest.mark.parametrize("entry", [False, True])
def test_execution_authority_requires_complete_trusted_entry(installation, monkeypatch, entry):
    """Ordinary preflight authenticates loaded authority without reconstructing or launching it."""
    record, arguments = installation
    if entry:
        entry_path = record.parent / "loop_payload.py"
        entry_path.write_bytes(b"entry")
        entry_path.chmod(0o444)
        arguments[-1]["loop_payload.py"] = linux_host.bind_host_file(entry_path)
        path = record.parent / "policies/root/bootstrap.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bootstrap")
        path.chmod(0o444)
        arguments[-1]["policies/root/bootstrap.bin"] = linux_host.bind_host_file(path)
    publish(installation)
    if entry:
        linux_runtime.execution_authority(record)
    else:
        with pytest.raises(ValueError, match="no trusted payload"):
            linux_runtime.execution_authority(record)
