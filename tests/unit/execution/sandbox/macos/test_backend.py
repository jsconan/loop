"""Test the macOS Seatbelt process backend and its fail-closed outcomes."""

from __future__ import annotations

import errno
import io
import json
import os
import platform
import plistlib
import re
import stat
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.execution.sandbox import SandboxCapability, SandboxOutcome, SandboxRequest, selection
from loop.execution.sandbox.contracts import CommandProcessResult, DirectoryIdentity
from loop.execution.sandbox.macos import backend as macos_backend
from loop.execution.sandbox.macos.backend import MacOSSeatbeltBackend
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import ProcessCapture, ProcessCaptureStatus

from ..helpers import logical_private_name

_MOUNT_LOCAL_SOCKET_VOLUME = macos_backend._mount_local_socket_volume
_EJECT_LOCAL_SOCKET_VOLUME = macos_backend._eject_local_socket_volume


@pytest.fixture(autouse=True)
def macos_environment(monkeypatch, tmp_path, request):
    """Keep macOS backend unit tests independent of native host services."""
    local_root = tmp_path.parent / "local-sockets"
    local_root.mkdir(mode=0o700)
    original_stat = Path.stat

    def fake_volume_stat(path, *args, **kwargs):
        """Represent the unit fixture's socket root as a different device."""
        details = original_stat(path, *args, **kwargs)
        if path == local_root:
            values = list(details)
            values[2] = -1
            return os.stat_result(values)
        return details

    monkeypatch.setattr(Path, "stat", fake_volume_stat)
    monkeypatch.setattr(
        macos_backend,
        "_mount_local_socket_volume",
        lambda _deadline: macos_backend.LocalSocketVolume(
            "/dev/disk0", local_root, -1, "unit", "unit-uuid"
        ),
    )
    monkeypatch.setattr(macos_backend, "_eject_local_socket_volume", lambda _volume: None)
    monkeypatch.setattr(
        "loop.utils.process.os.waitid",
        lambda _kind, pid, _options: SimpleNamespace(si_pid=pid),
        raising=False,
    )
    for name, value in (("P_PID", 1), ("WEXITED", 1), ("WNOHANG", 2), ("WNOWAIT", 4)):
        monkeypatch.setattr("loop.utils.process.os." + name, value, raising=False)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.platform", SimpleNamespace(system=lambda: "Darwin")
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.time",
        SimpleNamespace(**{**vars(time), "sleep": lambda seconds: None}),
    )
    home = tmp_path.with_name(tmp_path.name + "-home")
    home.mkdir()
    if getattr(request, "param", None) != "real_home":
        monkeypatch.setattr("loop.execution.sandbox.macos.backend._trusted_home", lambda: home)
    monitor = MagicMock()
    monitor.denial.return_value = None
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda self, deadline: monitor,
    )
    opener = macos_backend.open_selected_file

    def open_owned_file(root, path, identity):
        """Anchor fixture reads at their owned root when host root reads are unavailable."""
        if root == Path("/") and path.is_relative_to(tmp_path):
            return opener(tmp_path, path, identity)
        return opener(root, path, identity)

    monkeypatch.setattr(macos_backend, "open_selected_file", open_owned_file)


def request(root: Path, **changes) -> SandboxRequest:
    """Create a current Seatbelt request rooted in one disposable workspace."""
    values = {
        "source": "printf ok",
        "cwd": root,
        "workspace": root,
        "read_roots": (),
        "write_roots": (root,),
        "network": False,
        "environment": {"PATH": os.environ["PATH"], "LANG": "C"},
        "policy_version": "macos-seatbelt-v2",
        "deadline": time.monotonic() + 30,
        "workspace_id": "test",
    }
    values.update(changes)
    return SandboxRequest.create(**values)


@pytest.mark.parametrize("protected", [False, True])
def test_backend_selects_probe_parent_outside_private_temporary_paths(
    monkeypatch, tmp_path, protected
):
    """Place a selector probe outside protected scratch ancestry when required."""
    monkeypatch.setattr(macos_backend.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(macos_backend, "is_protected_external_read", lambda _path: protected)
    backend = MacOSSeatbeltBackend(diagnostics=MagicMock())

    layout = backend.qualification_layout()

    assert layout.probe_parent == (macos_backend._PRIVATE_TEMP_ROOT if protected else None)
    assert layout.outside_parent is None
    assert backend.capability_failure() is None
    assert backend.qualification_failure(None, "unavailable") == "unavailable"
    invalid = CommandProcessResult(SandboxOutcome.INVALID)
    assert backend.qualification_failure(invalid, "bad policy") == CommandProcessResult(
        SandboxOutcome.INVALID, detail="bad policy"
    )
    for outcome in (SandboxOutcome.STALE, SandboxOutcome.UNSUPPORTED):
        assert backend.qualification_failure(
            CommandProcessResult(outcome), "changed authority"
        ) == CommandProcessResult(outcome, detail="changed authority")


@pytest.fixture
def fake_terminal(tmp_path):
    """Supply inert descriptors for trusted terminal orchestration unit tests."""
    master = os.open(tmp_path / "master", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with open(tmp_path / "slave", "w+b") as slave:
            yield macos_backend.OwnedTerminal(master, slave, Path("/dev/ttys123"))
    finally:
        os.close(master)


@pytest.mark.parametrize("invalid", ["path", "type", "identity", "owner"])
def test_owned_terminal_rejects_unverified_slave(monkeypatch, tmp_path, invalid):
    """Trusted PTY ownership requires a matching character device and exact path."""
    master = os.open(tmp_path / "master", os.O_RDWR | os.O_CREAT, 0o600)
    slave = os.open(tmp_path / "slave", os.O_RDWR | os.O_CREAT, 0o600)
    real_fstat = os.fstat
    real_stat = Path.stat
    monkeypatch.setattr(macos_backend.os, "openpty", lambda: (master, slave))
    path = "/dev/ttyhost" if invalid == "path" else "/dev/ttys123"
    monkeypatch.setattr(macos_backend.os, "ttyname", lambda _fd: path)
    details = SimpleNamespace(
        st_mode=stat.S_IFREG if invalid == "type" else stat.S_IFCHR,
        st_dev=1,
        st_ino=2,
        st_rdev=3,
        st_uid=os.getuid() + (1 if invalid == "owner" else 0),
    )
    bound = SimpleNamespace(st_dev=1, st_ino=2, st_rdev=4 if invalid == "identity" else 3)
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, *args, **kwargs: (
            details if str(self) == path else real_stat(self, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        macos_backend.os, "fstat", lambda fd: bound if fd == slave else real_fstat(fd)
    )
    with pytest.raises(OSError, match="could not be verified"), macos_backend._owned_terminal():
        pass
    with pytest.raises(OSError):
        real_fstat(master)


def test_owned_terminal_retains_master_until_context_exit(monkeypatch, tmp_path):
    """A verified owned PTY keeps its master live for the entire command scope."""
    master = os.open(tmp_path / "master", os.O_RDWR | os.O_CREAT, 0o600)
    slave = os.open(tmp_path / "slave", os.O_RDWR | os.O_CREAT, 0o600)
    real_fstat = os.fstat
    real_stat = Path.stat
    details = SimpleNamespace(
        st_mode=stat.S_IFCHR, st_dev=1, st_ino=2, st_rdev=3, st_uid=os.getuid()
    )
    monkeypatch.setattr(macos_backend.os, "openpty", lambda: (master, slave))
    monkeypatch.setattr(macos_backend.os, "ttyname", lambda _fd: "/dev/ttys123")
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, *args, **kwargs: (
            details if str(self) == "/dev/ttys123" else real_stat(self, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        macos_backend.os, "fstat", lambda fd: details if fd == slave else real_fstat(fd)
    )
    with macos_backend._owned_terminal() as terminal:
        assert terminal.master == master
        assert terminal.path == Path("/dev/ttys123")
        real_fstat(master)
    with pytest.raises(OSError):
        real_fstat(master)


def test_terminal_bootstrap_uses_fixed_interpreter_and_pipe_environment(
    monkeypatch, tmp_path, fake_terminal
):
    """Trusted launch applies the command environment only after bootstrap is ready."""
    expected = SimpleNamespace(
        stdout=None, stderr=io.StringIO(), kill=MagicMock(), wait=MagicMock()
    )
    calls = []

    def launch(arguments, **options):
        """Signal a successful trusted bootstrap without starting a process."""
        calls.append((arguments, options))
        expected.environment_reader = os.dup(int(arguments[5]))
        os.write(int(arguments[6]), b"R")
        return expected

    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)
    approved = replace(request(tmp_path), runtime_timeout=2)
    monkeypatch.setattr(macos_backend.time, "monotonic", lambda: approved.deadline - 10)
    monkeypatch.setattr(
        macos_backend.select, "select", lambda readers, writers, *_: (readers, writers, [])
    )
    result = macos_backend._spawn_terminal(
        approved, "(version 1)", {"PATH": "/custom", "SECRET": "value"}, fake_terminal
    )
    assert result is expected
    assert calls[0][0][:4] == ["/usr/bin/python3", "-I", "-S", "-c"]
    assert calls[0][1]["env"] == {"PATH": macos_backend.MINIMAL_SYSTEM_PATH, "LANG": "C"}
    assert calls[0][1]["pass_fds"] == (int(calls[0][0][5]), int(calls[0][0][6]))
    assert calls[0][1]["close_fds"] is True
    result.stdout.close()
    os.close(expected.environment_reader)


@pytest.mark.parametrize(
    "failure",
    [
        "too_large",
        "second_pipe",
        "launch",
        "write_timeout",
        "write_zero",
        "ready_timeout",
        "no_ready",
        "no_stderr",
        "dup",
    ],
)
def test_terminal_bootstrap_fails_closed_and_reaps_child(
    monkeypatch, tmp_path, fake_terminal, failure
):
    """Every failed bootstrap phase closes inherited channels and reaps its child."""
    approved = request(tmp_path)
    monkeypatch.setattr(macos_backend.time, "monotonic", lambda: approved.deadline - 10)
    environment = {"PATH": "/usr/bin"}
    if failure == "too_large":
        environment["OVERSIZED"] = "x" * (macos_backend._TERMINAL_ENVIRONMENT_LIMIT + 1)
    real_pipe = os.pipe
    real_dup = os.dup
    real_write = os.write
    child = SimpleNamespace(stderr=io.StringIO(), kill=MagicMock(), wait=MagicMock())
    if failure == "no_stderr":
        child.stderr = None
    pipe_calls = []

    def pipe():
        """Fail only the second pipe after recording the first pair for closure checks."""
        pair = real_pipe()
        pipe_calls.append(pair)
        if len(pipe_calls) == 2:
            os.close(pair[0])
            os.close(pair[1])
            raise OSError("ready pipe unavailable")
        return pair

    def launch(arguments, **_options):
        """Model pre-exec success or a failed trusted interpreter spawn."""
        if failure == "launch":
            raise OSError("interpreter unavailable")
        child.environment_reader = real_dup(int(arguments[5]))
        if failure == "dup" and hasattr(child, "stderr"):
            real_write(int(arguments[6]), b"R")
        return child

    if failure == "second_pipe":
        monkeypatch.setattr(macos_backend.os, "pipe", pipe)
    if failure == "write_zero":
        monkeypatch.setattr(macos_backend.os, "write", lambda _fd, _data: 0)
    if failure in {"write_timeout", "ready_timeout"}:
        calls = 0

        def select_once(readers, writers, errors, timeout):
            """Refuse the selected launch phase while leaving the other phase writable."""
            nonlocal calls
            calls += 1
            if failure == "write_timeout" or calls == 2:
                return ([], [], [])
            return (readers, writers, [])

        monkeypatch.setattr(macos_backend.select, "select", select_once)
    if failure in {"write_zero", "no_ready", "no_stderr"}:
        monkeypatch.setattr(
            macos_backend.select, "select", lambda readers, writers, *_: (readers, writers, [])
        )
    if failure == "dup":
        monkeypatch.setattr(macos_backend.os, "dup", MagicMock(side_effect=OSError("dup")))
        monkeypatch.setattr(
            macos_backend.select, "select", lambda readers, writers, *_: (readers, writers, [])
        )
    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)
    with pytest.raises((OSError, TimeoutError, ValueError)):
        macos_backend._spawn_terminal(approved, "(version 1)", environment, fake_terminal)
    if failure == "second_pipe":
        for descriptor in pipe_calls[0]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    elif failure not in {"too_large", "launch"}:
        child.kill.assert_called_once() if failure != "dup" else child.kill.assert_not_called()
        child.wait.assert_called_once()
    if failure == "dup":
        macos_backend.kill_process_group.assert_called_once_with(child)
    if hasattr(child, "environment_reader"):
        os.close(child.environment_reader)


@pytest.mark.parametrize(
    "frame",
    [b"", b"Q", b"I", b"I\x00\x00", b"I\x00\x02x", b"Z\x00\x00\x00\x01", b"Z\x00\x01\x01\xf5"],
)
def test_terminal_control_rejects_malformed_frames(monkeypatch, frame):
    """Native control parsing rejects unknown, truncated, and out-of-range frames."""
    write = MagicMock()
    monkeypatch.setattr(macos_backend.os, "write", write)
    with pytest.raises(ValueError):
        macos_backend._write_terminal_control(8, frame)
    write.assert_not_called()


@pytest.mark.parametrize("frame", [b"I\x00\x01x", b"F", b"X", b"Z\x00\x28\x00\x64"])
def test_terminal_control_applies_bounded_frames(monkeypatch, frame):
    """Write, EOF, interrupt, and resize reach only the owned master descriptor."""
    write = MagicMock(return_value=1)
    ioctl = MagicMock()
    monkeypatch.setattr(macos_backend.os, "write", write)
    monkeypatch.setattr(macos_backend.fcntl, "ioctl", ioctl)
    monkeypatch.setattr(macos_backend.select, "select", lambda *_args: ([], [8], []))
    macos_backend._write_terminal_control(8, frame)
    if frame.startswith(b"Z"):
        ioctl.assert_called_once()
        write.assert_not_called()
    else:
        write.assert_called_once_with(
            8, frame[3:] if frame.startswith(b"I") else (b"\x04" if frame == b"F" else b"\x03")
        )


@pytest.mark.parametrize("failure", ["timeout", "zero"])
def test_terminal_control_stops_on_unwritable_or_stalled_master(monkeypatch, failure):
    """A stalled PTY never leaves an unbounded control write behind."""
    monkeypatch.setattr(
        macos_backend.select, "select", lambda *_args: ([], [] if failure == "timeout" else [8], [])
    )
    monkeypatch.setattr(macos_backend.os, "write", lambda *_args: 0)
    with pytest.raises(TimeoutError if failure == "timeout" else OSError):
        macos_backend._write_terminal_control(8, b"I\x00\x01x")


@pytest.mark.parametrize(
    ("outcome", "exit_code"),
    [
        (SandboxOutcome.COMPLETED, 0),
        (SandboxOutcome.COMPLETED, 1),
        (SandboxOutcome.UNAVAILABLE, None),
    ],
)
def test_terminal_capability_requires_successful_native_isolation_probe(
    monkeypatch, tmp_path, outcome, exit_code
):
    """Only an owned TTY with an inner PTY and denied foreign access is advertised."""
    backend = MacOSSeatbeltBackend()
    foreign = macos_backend.OwnedTerminal(8, io.BytesIO(), Path("/dev/ttys123"))
    monkeypatch.setattr(macos_backend, "_owned_terminal", lambda: nullcontext(foreign))
    result = CommandProcessResult(
        outcome,
        exit_code=exit_code,
        stdout="INNER_PTY_READY" if exit_code == 0 else "",
    )
    probe = MagicMock(return_value=result)
    monkeypatch.setattr(backend, "_run", probe)
    backend._qualify_terminal(request(tmp_path))
    assert (SandboxCapability.TERMINAL in backend.capabilities) is (
        outcome is SandboxOutcome.COMPLETED and exit_code == 0
    )
    launched = probe.call_args.args[0]
    assert launched.terminal and "/usr/bin/script -q /dev/null" in launched.source
    assert ": </dev/tty" in launched.source
    assert str(foreign.path) in launched.source


def test_terminal_capability_requires_inner_slave_marker(monkeypatch, tmp_path):
    """A successful launcher without a usable inner TTY cannot advertise terminals."""
    backend = MacOSSeatbeltBackend()
    foreign = macos_backend.OwnedTerminal(8, io.BytesIO(), Path("/dev/ttys123"))
    monkeypatch.setattr(macos_backend, "_owned_terminal", lambda: nullcontext(foreign))
    monkeypatch.setattr(
        backend,
        "_run",
        MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)),
    )

    backend._qualify_terminal(request(tmp_path))

    assert SandboxCapability.TERMINAL not in backend.capabilities


def test_terminal_capability_probe_failure_stays_unadvertised(monkeypatch, tmp_path):
    """A native PTY setup error cannot advertise interactive execution."""
    backend = MacOSSeatbeltBackend()
    monkeypatch.setattr(
        macos_backend, "_owned_terminal", MagicMock(side_effect=OSError("PTY unavailable"))
    )
    backend._qualify_terminal(request(tmp_path))
    assert SandboxCapability.TERMINAL not in backend.capabilities


def test_terminal_request_requires_matching_controls_and_separate_egress(monkeypatch, tmp_path):
    """Invalid terminal plumbing and incompatible egress fail before native launch."""
    backend = MacOSSeatbeltBackend()
    launch = MagicMock()
    monkeypatch.setattr(backend, "_run", launch)
    approved = request(tmp_path)
    mismatch = backend.run(approved, terminal_input=lambda: None)
    assert mismatch.outcome is SandboxOutcome.INVALID
    backend.capabilities = backend.capabilities | {SandboxCapability.TERMINAL}
    terminal = replace(approved, terminal=True, egress_origins=("https://example.com",))
    refused = backend.run(terminal, terminal_input=lambda: None)
    assert refused.outcome is SandboxOutcome.UNSUPPORTED
    launch.assert_not_called()


@pytest.mark.parametrize("control_failure", [False, True])
@pytest.mark.parametrize("pending_stderr", [0, 1, 2])
def test_terminal_execute_gates_output_and_controls_on_verified_shell(
    monkeypatch, tmp_path, control_failure, pending_stderr
):
    """Trusted terminal control starts only after entry authentication and reports failures."""
    approved = replace(request(tmp_path), terminal=True)
    process = SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO())
    terminal = macos_backend.OwnedTerminal(8, io.BytesIO(), Path("/dev/ttys123"))
    monkeypatch.setattr(macos_backend, "_spawn_terminal", lambda *_args: process)
    read = MagicMock(return_value=0)
    monkeypatch.setattr(macos_backend, "read_bounded_stream", read)
    processed = Event()
    frames = iter([b"I\x00\x01x"])
    observed = []
    monkeypatch.setattr(macos_backend.constants, "MAX_OUTPUT_CHARS", 10)

    def terminal_input():
        """Return one transient frame and then no further input."""
        return next(frames, None)

    def write(master, frame):
        """Record the one owned frame and optionally model a native write failure."""
        assert master == terminal.master and frame == b"I\x00\x01x"
        processed.set()
        if control_failure:
            raise OSError("PTY write failed")

    def supervise(_process, _deadline, **options):
        """Publish a split entry marker before permitting the controller to consume input."""
        publish = options["output_observer"]
        if pending_stderr:
            publish("stderr", "pre-entry")
        if pending_stderr == 2:
            publish("stderr", "overflow")
            publish("stderr", "discarded")
        publish("stdout", "LOOP_SHELL_")
        assert not processed.is_set()
        publish("stdout", "STARTED\r\nvisible")
        assert processed.wait(2)
        assert options["reader"](process.stdout, [], maximum=10) == 0
        return ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\r\nvisible")

    monkeypatch.setattr(macos_backend, "_write_terminal_control", write)
    monkeypatch.setattr(macos_backend, "supervise_process", supervise)
    result = MacOSSeatbeltBackend._execute(
        approved,
        "profile",
        {},
        "tag",
        None,
        terminal=terminal,
        terminal_input=terminal_input,
        output_observer=lambda stream, data: observed.append((stream, data)),
    )
    assert result.outcome is (
        SandboxOutcome.UNAVAILABLE if control_failure else SandboxOutcome.COMPLETED
    )
    if not control_failure:
        assert result.stdout == "visible"
        assert observed == (
            [("stderr", "pre-entry")]
            + ([("stderr", "o")] if pending_stderr == 2 else [])
            + [("stdout", "visible")]
            if pending_stderr
            else [("stdout", "visible")]
        )
    read.assert_called_once()
    assert read.call_args.kwargs["eof_errno"] == errno.EIO


def test_terminal_execute_reports_controller_that_cannot_stop(monkeypatch, tmp_path):
    """A terminal controller that outlives supervision cannot report success."""
    approved = replace(request(tmp_path), terminal=True)
    process = SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO())
    terminal = macos_backend.OwnedTerminal(8, io.BytesIO(), Path("/dev/ttys123"))
    monkeypatch.setattr(macos_backend, "_spawn_terminal", lambda *_args: process)
    controller = MagicMock()
    controller.is_alive.return_value = True
    monkeypatch.setattr(macos_backend.threading, "Thread", MagicMock(return_value=controller))
    monkeypatch.setattr(
        macos_backend,
        "supervise_process",
        MagicMock(
            return_value=ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\r\n")
        ),
    )
    result = MacOSSeatbeltBackend._execute(
        approved,
        "profile",
        {},
        "tag",
        None,
        terminal=terminal,
        terminal_input=lambda: None,
    )
    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert "control failed" in result.detail
    controller.start.assert_called_once()
    controller.join.assert_called_once()


def test_terminal_execute_discards_frame_after_supervision_stops(monkeypatch, tmp_path):
    """A control callback returning after stop cannot touch the closing master."""
    approved = replace(request(tmp_path), terminal=True)
    process = SimpleNamespace(stdout=io.StringIO(), stderr=io.StringIO())
    terminal = macos_backend.OwnedTerminal(8, io.BytesIO(), Path("/dev/ttys123"))
    monkeypatch.setattr(macos_backend, "_spawn_terminal", lambda *_args: process)
    events = []
    called = Event()

    def event_factory():
        """Record the execution-owned ready and stop events for the callback race."""
        event = Event()
        events.append(event)
        return event

    def terminal_input():
        """Set stop after a queued frame is consumed but before its master write."""
        events[1].set()
        called.set()
        return b"I\x00\x01x"

    def supervise(_process, _deadline, **options):
        """Release the controller only after the shell marker is verified."""
        options["output_observer"]("stdout", "LOOP_SHELL_STARTED\r\n")
        assert called.wait(2)
        return ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\r\n")

    write = MagicMock()
    monkeypatch.setattr(macos_backend.threading, "Event", event_factory)
    monkeypatch.setattr(macos_backend, "_write_terminal_control", write)
    monkeypatch.setattr(macos_backend, "supervise_process", supervise)
    result = MacOSSeatbeltBackend._execute(
        approved,
        "profile",
        {},
        "tag",
        None,
        terminal=terminal,
        terminal_input=terminal_input,
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    write.assert_not_called()


@pytest.fixture
def local_volume_simulator(monkeypatch, tmp_path):
    """Model one owned RAM volume and native disk responses without touching host disks."""
    parent = tmp_path / "Volumes"
    parent.mkdir()
    monkeypatch.setattr(macos_backend, "_PRIVATE_TEMP_ROOT", parent)
    name = "LoopIPC-1234-deadbeef"
    mount = parent / name
    state = SimpleNamespace(case="ok", calls=[], info_calls=0, registry_calls=0, pin_active=False)
    real_stat = Path.stat
    real_resolve = Path.resolve
    real_is_mount = Path.is_mount
    real_is_symlink = Path.is_symlink
    real_open, real_close, real_fstat, real_os_stat = os.open, os.close, os.fstat, os.stat
    raw_descriptor = 999999

    def device_open(path, flags, *args, **kwargs):
        """Pin the fixture's raw device while preserving ordinary filesystem operations."""
        if path == "/dev/rdisk99":
            if state.case == "pin_failure":
                raise OSError("Cannot open the raw device")
            state.pin_active = True
            return raw_descriptor
        return real_open(path, flags, *args, **kwargs)

    def device_close(descriptor):
        """Release only the fixture's synthetic pin."""
        if descriptor == raw_descriptor:
            state.pin_active = False
        else:
            real_close(descriptor)

    def device_fstat(descriptor):
        """Return a character device identity for the fixture's pin."""
        if descriptor == raw_descriptor:
            mode = stat.S_IFREG if state.case == "pin_type" else stat.S_IFCHR
            return SimpleNamespace(st_mode=mode, st_dev=1, st_ino=2, st_rdev=3)
        return real_fstat(descriptor)

    def device_stat(path, *args, **kwargs):
        """Expose changed path identity without opening another device."""
        if path == "/dev/rdisk99":
            return SimpleNamespace(
                st_dev=1, st_ino=4 if state.case == "pin_changed" else 2, st_rdev=3
            )
        return real_os_stat(path, *args, **kwargs)

    monkeypatch.setattr(macos_backend.os, "open", device_open)
    monkeypatch.setattr(macos_backend.os, "close", device_close)
    monkeypatch.setattr(macos_backend.os, "fstat", device_fstat)
    monkeypatch.setattr(macos_backend.os, "stat", device_stat)
    monkeypatch.setattr(macos_backend.os, "getpid", lambda: 1234)
    monkeypatch.setattr(macos_backend.secrets, "token_hex", lambda _size: "deadbeef")
    monkeypatch.setattr(
        macos_backend, "Path", lambda value: parent if value == "/Volumes" else Path(value)
    )

    def volume_stat(path, *args, **kwargs):
        """Give the fixture mount an independent device and selectable unsafe metadata."""
        details = real_stat(path, *args, **kwargs)
        if path != mount:
            return details
        values = list(details)
        values[2] = parent.stat().st_dev if state.case == "same_device" else 999
        if state.case == "wrong_owner":
            values[4] = 42
        if state.case == "wrong_mode":
            values[0] = (values[0] & ~0o777) | 0o755
        return os.stat_result(values)

    def volume_resolve(path, *args, **kwargs):
        """Expose an unexpected canonical path for the alias control."""
        if path == mount and state.case == "alias":
            return parent
        return real_resolve(path, *args, **kwargs)

    def volume_is_mount(path):
        """Treat the fresh fixture directory as a mounted disk unless denied."""
        if path == mount:
            return state.case != "not_mounted" and mount.exists()
        return real_is_mount(path)

    def volume_is_symlink(path):
        """Exercise a late path substitution independently of the initial check."""
        if path == mount and state.case == "late_symlink" and mount.exists():
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "stat", volume_stat)
    monkeypatch.setattr(Path, "resolve", volume_resolve)
    monkeypatch.setattr(Path, "is_mount", volume_is_mount)
    monkeypatch.setattr(Path, "is_symlink", volume_is_symlink)

    def disk_run(argv, **_kwargs):
        """Return controlled attach, format, identity, and eject outcomes."""
        state.calls.append(tuple(argv))
        if argv[0] == "/usr/sbin/ioreg":
            state.registry_calls += 1
            marker = argv[-1]
            if (
                state.case == "registry_failure"
                or (state.case == "cleanup_registry_changed" and state.registry_calls > 3)
                or (state.case == "format_interrupt" and state.registry_calls > 2)
                or (state.case == "pin_unenforced" and state.registry_calls > 1)
                or (state.case == "registry_changes_after_pin_test" and state.registry_calls > 1)
            ):
                return SimpleNamespace(returncode=1, stdout=b"")
            if state.case == "registry_oversized":
                return SimpleNamespace(returncode=0, stdout=b"x" * (256 * 1024 + 1))
            if state.case == "registry_bad_plist":
                return SimpleNamespace(returncode=0, stdout=b"invalid")
            if state.case == "pin_unenforced_removed" and state.registry_calls > 1:
                return SimpleNamespace(returncode=0, stdout=plistlib.dumps([]))
            media = {
                "BSD Name": "disk99",
                "IOObjectClass": "IOMedia",
                "Whole": True,
                "Writable": True,
                "Size": 16384 * 512,
            }
            root = {
                marker: 1,
                "owner-uid": os.getuid(),
                "Physical Interconnect Location": "RAM",
                "IORegistryEntryChildren": [media],
            }
            changes = {
                "registry_marker": (root, marker, 0),
                "registry_owner": (root, "owner-uid", -1),
                "registry_not_ram": (root, "Physical Interconnect Location", "USB"),
                "registry_wrong_device": (media, "BSD Name", "disk98"),
                "registry_class": (media, "IOObjectClass", "other"),
                "registry_not_whole": (media, "Whole", False),
                "registry_not_writable": (media, "Writable", False),
                "registry_wrong_size": (media, "Size", 1),
                "registry_children": (root, "IORegistryEntryChildren", "invalid"),
                "registry_non_node": (root, "IORegistryEntryChildren", ["invalid"]),
                "registry_many_nodes": (root, "IORegistryEntryChildren", [{}] * 129),
                "registry_duplicate_media": (root, "IORegistryEntryChildren", [media, media]),
            }
            if state.case in changes:
                target, key, value = changes[state.case]
                target[key] = value
            entries = [root]
            if state.case == "registry_not_list":
                entries = {}
            elif state.case == "registry_non_dict":
                entries = ["invalid"]
            elif state.case == "registry_duplicate":
                entries = [root, root]
            return SimpleNamespace(returncode=0, stdout=plistlib.dumps(entries))
        if argv[0] == "/sbin/newfs_hfs":
            assert state.pin_active
            if state.case == "format_interrupt":
                raise KeyboardInterrupt
            return SimpleNamespace(
                returncode=1
                if state.case
                in {
                    "format_failure",
                    "no_cleanup_identity",
                    "cleanup_registry_changed",
                    "cleanup_wrong_bus",
                    "cleanup_not_whole",
                    "cleanup_eject_failure",
                }
                else 0
            )
        verb = argv[1]
        if verb == "detach":
            assert state.pin_active
            return SimpleNamespace(
                returncode=0
                if state.case in {"pin_unenforced", "pin_unenforced_removed"}
                else errno.EBUSY
            )
        if verb == "attach":
            if state.case == "attach_exception":
                raise OSError("Attachment failed before returning its device")
            return SimpleNamespace(
                returncode=1 if state.case == "attach_failure" else 0,
                stdout="bad device\n" if state.case == "bad_device" else "/dev/disk99\n",
            )
        if verb == "mount":
            assert state.pin_active
            if state.case != "mount_failure":
                mount.mkdir(exist_ok=True)
            return SimpleNamespace(returncode=1 if state.case == "mount_failure" else 0)
        if verb == "info":
            state.info_calls += 1
            if state.case == "no_cleanup_identity":
                return SimpleNamespace(returncode=1, stdout=b"")
            if state.case == "missing_info" and state.info_calls == 1:
                return SimpleNamespace(returncode=1, stdout=b"")
            if state.case == "bad_plist" and state.info_calls == 1:
                return SimpleNamespace(returncode=0, stdout=b"bad plist")
            identity = {
                "DeviceNode": "/dev/disk99",
                "MountPoint": str(mount),
                "VolumeName": name,
                "VolumeUUID": "owned-uuid",
                "FilesystemType": "hfs",
                "BusProtocol": "Disk Image",
                "WholeDisk": True,
            }
            changes = {
                "wrong_node": ("DeviceNode", "/dev/disk98"),
                "wrong_mount": ("MountPoint", "/Volumes/other"),
                "wrong_name": ("VolumeName", "other"),
                "wrong_fs": ("FilesystemType", "apfs"),
                "wrong_bus": ("BusProtocol", "USB"),
                "not_whole": ("WholeDisk", False),
                "cleanup_wrong_bus": ("BusProtocol", "USB"),
                "cleanup_not_whole": ("WholeDisk", False),
            }
            if state.case in changes and state.info_calls == 1:
                key, value = changes[state.case]
                identity[key] = value
            if state.case == "missing_uuid" and state.info_calls == 1:
                del identity["VolumeUUID"]
            if state.case == "non_mapping" and state.info_calls == 1:
                identity = ["unexpected"]
            return SimpleNamespace(returncode=0, stdout=plistlib.dumps(identity))
        if verb == "eject":
            return SimpleNamespace(
                returncode=1 if state.case in {"eject_failure", "cleanup_eject_failure"} else 0
            )
        raise AssertionError(f"Unexpected native disk command: {argv}")

    def registry_popen(argv, **_kwargs):
        """Model the registry process without launching host tools."""
        assert argv[0] == "/usr/sbin/ioreg"
        return disk_run(argv)

    def registry_supervise(process, _deadline, *, stdout_limit, stderr_limit):
        """Return bounded registry output through the process supervision boundary."""
        assert stderr_limit > 0
        if state.case == "registry_timeout":
            return ProcessCapture(ProcessCaptureStatus.TIMED_OUT)
        value = process.stdout.decode("utf-8")
        return ProcessCapture(
            ProcessCaptureStatus.COMPLETED,
            process.returncode,
            value[:stdout_limit],
            "",
            stdout_discarded=max(0, len(value) - stdout_limit),
            stderr_discarded=1 if state.case == "registry_stderr_discarded" else 0,
        )

    monkeypatch.setattr(macos_backend.subprocess, "run", disk_run)
    monkeypatch.setattr(macos_backend.subprocess, "Popen", registry_popen)
    monkeypatch.setattr(macos_backend, "supervise_process", registry_supervise)
    return SimpleNamespace(state=state, mount=mount, name=name)


def test_local_socket_volume_setup_time_is_bounded():
    """RAM-volume preparation refuses an expired request and caps its own setup time."""
    now = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded its deadline"):
        macos_backend._local_socket_setup_timeout(now - 1)
    assert 0 < macos_backend._local_socket_setup_timeout(now + 5) <= 5
    assert macos_backend._local_socket_setup_timeout(now + 100) == 30


def test_local_socket_volume_mount_and_safe_eject(local_volume_simulator):
    """A formatted RAM volume is accepted and later ejected only under its bound identity."""
    volume = _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)
    assert volume.device == "/dev/disk99"
    assert volume.mount == local_volume_simulator.mount
    assert volume.device_id == 999
    assert volume.volume_uuid == "owned-uuid"
    assert not local_volume_simulator.state.pin_active
    assert (
        "/sbin/newfs_hfs",
        "-U",
        str(os.getuid()),
        "-G",
        str(os.getgid()),
        "-M",
        "0700",
        "-v",
        local_volume_simulator.name,
        "/dev/disk99",
    ) in local_volume_simulator.state.calls
    assert ("/usr/sbin/diskutil", "mount", "/dev/disk99") in local_volume_simulator.state.calls
    _EJECT_LOCAL_SOCKET_VOLUME(volume)
    assert local_volume_simulator.state.calls[-1][1] == "eject"


@pytest.mark.parametrize(
    "case",
    [
        "name_exists",
        "name_symlink",
        "attach_failure",
        "attach_exception",
        "bad_device",
        "pin_failure",
        "pin_type",
        "pin_changed",
        "pin_unenforced",
        "registry_changes_after_pin_test",
        "registry_failure",
        "registry_timeout",
        "registry_stderr_discarded",
        "registry_oversized",
        "registry_bad_plist",
        "registry_not_list",
        "registry_non_dict",
        "registry_duplicate",
        "registry_marker",
        "registry_owner",
        "registry_not_ram",
        "registry_wrong_device",
        "registry_class",
        "registry_not_whole",
        "registry_not_writable",
        "registry_wrong_size",
        "registry_children",
        "registry_non_node",
        "registry_many_nodes",
        "registry_duplicate_media",
        "format_failure",
        "mount_failure",
        "cleanup_registry_changed",
        "cleanup_wrong_bus",
        "cleanup_not_whole",
        "cleanup_eject_failure",
        "no_cleanup_identity",
        "missing_info",
        "bad_plist",
        "non_mapping",
        "wrong_node",
        "wrong_mount",
        "wrong_name",
        "missing_uuid",
        "wrong_fs",
        "not_mounted",
        "late_symlink",
        "alias",
        "wrong_owner",
        "wrong_mode",
        "same_device",
    ],
)
def test_local_socket_volume_rejects_unsafe_mounts(local_volume_simulator, case):
    """RAM-volume setup fails closed on missing, altered, or unsafe native state."""
    local_volume_simulator.state.case = case
    if case == "name_exists":
        local_volume_simulator.mount.mkdir()
    elif case == "name_symlink":
        local_volume_simulator.mount.symlink_to("missing")
    with pytest.raises(OSError):
        _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)
    if case.startswith(("pin_", "registry_")):
        assert not any(call[0] == "/sbin/newfs_hfs" for call in local_volume_simulator.state.calls)
        assert not any(call[1] == "eject" for call in local_volume_simulator.state.calls)
    if case == "cleanup_registry_changed":
        assert not any(call[1] == "eject" for call in local_volume_simulator.state.calls)


def test_unverified_local_socket_attachment_reports_recovery_identity(
    local_volume_simulator, monkeypatch
):
    """An uncertain attachment is retained and records its exact checked-recovery identity."""
    local_volume_simulator.state.case = "pin_failure"
    audit = MagicMock()
    monkeypatch.setattr(macos_backend, "telemetry_audit", audit)

    with pytest.raises(macos_backend.LocalSocketAttachmentError) as caught:
        _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)

    assert caught.value.device == "/dev/disk99"
    assert caught.value.marker == "LoopOwned4d2deadbeef"
    audit.assert_called_once_with(
        "sandbox.local_socket_volume.cleanup_pending",
        device="/dev/disk99",
        attachment="LoopOwned4d2deadbeef",
    )


def test_detached_socket_attachment_needs_no_pending_cleanup(local_volume_simulator, monkeypatch):
    """A successful probe detach with an absent marker does not leave false pending cleanup."""
    local_volume_simulator.state.case = "pin_unenforced_removed"
    audit = MagicMock()
    monkeypatch.setattr(macos_backend, "telemetry_audit", audit)
    monkeypatch.setattr(macos_backend, "_mount_local_socket_volume", _MOUNT_LOCAL_SOCKET_VOLUME)
    backend = MacOSSeatbeltBackend(MagicMock())
    monkeypatch.setattr(backend, "_qualify_terminal", MagicMock())

    backend.complete_qualification(
        request(local_volume_simulator.mount.parent, local_network=False)
    )
    assert SandboxCapability.LOCAL_NETWORK not in backend.capabilities
    backend.close()

    audit.assert_not_called()
    assert not any(call[0] == "/sbin/newfs_hfs" for call in local_volume_simulator.state.calls)
    assert not any(call[1] == "eject" for call in local_volume_simulator.state.calls)


def test_interrupted_local_socket_setup_preserves_the_interrupt(
    local_volume_simulator, monkeypatch
):
    """An interruption remains visible even when its attachment cannot be safely removed."""
    local_volume_simulator.state.case = "format_interrupt"
    audit = MagicMock()
    monkeypatch.setattr(macos_backend, "telemetry_audit", audit)

    with pytest.raises(KeyboardInterrupt):
        _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)

    audit.assert_called_once_with(
        "sandbox.local_socket_volume.cleanup_pending",
        device="/dev/disk99",
        attachment="LoopOwned4d2deadbeef",
    )


@pytest.mark.parametrize("case", ["attach_failure", "attach_exception", "bad_device"])
def test_uncertain_local_socket_attach_reports_its_marker(
    local_volume_simulator, monkeypatch, case
):
    """An attach failure reports its marker even when no device name was returned."""
    local_volume_simulator.state.case = case
    audit = MagicMock()
    monkeypatch.setattr(macos_backend, "telemetry_audit", audit)

    with pytest.raises(OSError):
        _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)

    audit.assert_called_once_with(
        "sandbox.local_socket_volume.cleanup_pending", attachment="LoopOwned4d2deadbeef"
    )
    assert not any(call[1] == "eject" for call in local_volume_simulator.state.calls)


@pytest.mark.parametrize(
    "case",
    [
        "missing_info",
        "bad_plist",
        "non_mapping",
        "wrong_node",
        "wrong_mount",
        "wrong_name",
        "missing_uuid",
        "wrong_fs",
        "not_mounted",
        "same_device",
        "eject_failure",
    ],
)
def test_local_socket_volume_eject_revalidates_identity(local_volume_simulator, case):
    """Eject refuses changed media or native failure while retaining the owned identity."""
    volume = _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)
    local_volume_simulator.state.case = case
    local_volume_simulator.state.info_calls = 0
    with pytest.raises(OSError):
        _EJECT_LOCAL_SOCKET_VOLUME(volume)
    assert (
        not any(call[1] == "eject" for call in local_volume_simulator.state.calls[3:])
        or case == "eject_failure"
    )


def test_local_socket_volume_missing_after_detach_needs_no_eject(local_volume_simulator):
    """A detached owned volume cannot cause the backend to eject a reused disk number."""
    volume = _MOUNT_LOCAL_SOCKET_VOLUME(time.monotonic() + 30)
    local_volume_simulator.mount.rmdir()
    local_volume_simulator.state.case = "missing_info"
    local_volume_simulator.state.info_calls = 0
    _EJECT_LOCAL_SOCKET_VOLUME(volume)
    assert local_volume_simulator.state.calls[-1][1] == "info"


def test_optional_local_socket_qualification_runs_its_own_command(monkeypatch, tmp_path):
    """The optional native probe executes the Unix exchange instead of replaying the core probe."""
    backend = MacOSSeatbeltBackend(MagicMock())
    qualified = MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    monkeypatch.setattr(backend, "qualification_probe", qualified)
    core = request(tmp_path, local_network=False, source="printf core")

    backend._qualify_local_network(core)

    assert SandboxCapability.LOCAL_NETWORK in backend.capabilities
    probe = qualified.call_args.args[0]
    assert probe.local_network is True
    assert probe.source == probe.execution_source
    assert "pack_sockaddr_un" in probe.execution_source
    assert probe.source != core.source
    backend.close()


def test_uncertain_socket_attachment_retains_identity_for_checked_close(monkeypatch, tmp_path):
    """A failed attachment remains available for checked cleanup and a later retry."""
    backend = MacOSSeatbeltBackend(MagicMock())
    pending = macos_backend.LocalSocketAttachmentError("/dev/disk99", "LoopOwnedfixture")
    mount = MagicMock(side_effect=pending)
    monkeypatch.setattr(macos_backend, "_mount_local_socket_volume", mount)
    remove = MagicMock(side_effect=[OSError("still uncertain"), None])
    monkeypatch.setattr(macos_backend, "_remove_local_socket_attachment", remove)
    monkeypatch.setattr(backend, "_qualify_terminal", MagicMock())

    backend.complete_qualification(request(tmp_path, local_network=False))
    backend.complete_qualification(request(tmp_path, local_network=False))

    assert SandboxCapability.LOCAL_NETWORK not in backend.capabilities
    with (
        pytest.raises(macos_backend.LocalSocketAttachmentError, match="checked cleanup"),
        backend._local_socket_directory(request(tmp_path), tmp_path),
    ):
        pytest.fail("A pending attachment cannot be replaced.")
    mount.assert_called_once()
    with pytest.raises(macos_backend.LocalSocketAttachmentError) as caught:
        backend.close()
    assert caught.value is pending
    backend.close()
    assert remove.call_count == 2
    assert all(call.args[:2] == (pending.device, pending.marker) for call in remove.call_args_list)


def test_uncertain_socket_attachment_during_use_is_retried_on_close(monkeypatch, tmp_path):
    """An uncertain attachment created during use retains its original cleanup identity."""
    backend = MacOSSeatbeltBackend(MagicMock())
    pending = macos_backend.LocalSocketAttachmentError("/dev/disk99", "LoopOwnedfixture")
    monkeypatch.setattr(macos_backend, "_mount_local_socket_volume", MagicMock(side_effect=pending))
    remove = MagicMock()
    monkeypatch.setattr(macos_backend, "_remove_local_socket_attachment", remove)

    with (
        pytest.raises(macos_backend.LocalSocketAttachmentError, match="checked cleanup"),
        backend._local_socket_directory(request(tmp_path), tmp_path),
    ):
        pytest.fail("An uncertain attachment cannot be used.")
    backend.close()
    assert remove.call_args.args[:2] == (pending.device, pending.marker)


def test_uncertain_socket_qualification_is_retained_before_concurrent_close(monkeypatch, tmp_path):
    """Concurrent close cannot miss an attachment whose setup is still completing."""
    backend = MacOSSeatbeltBackend(MagicMock())
    pending = macos_backend.LocalSocketAttachmentError("/dev/disk99", "LoopOwnedfixture")
    entered, release, closed = Event(), Event(), Event()

    def interrupted_mount(_deadline):
        """Hold the setup lock until the close worker is ready to proceed."""
        entered.set()
        if not release.wait(5):
            raise TimeoutError("The test did not release setup")
        raise pending

    remove = MagicMock()
    monkeypatch.setattr(macos_backend, "_mount_local_socket_volume", interrupted_mount)
    monkeypatch.setattr(macos_backend, "_remove_local_socket_attachment", remove)
    qualification = Thread(
        target=backend._qualify_local_network, args=(request(tmp_path, local_network=False),)
    )

    def close():
        """Close the backend and signal when checked cleanup finishes."""
        backend.close()
        closed.set()

    closer = Thread(target=close)
    qualification.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not closed.is_set()
    finally:
        release.set()
        qualification.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert not qualification.is_alive() and not closer.is_alive()
    assert closed.is_set()
    assert remove.call_args.args[:2] == (pending.device, pending.marker)


@pytest.mark.parametrize("case", ["available", "refused", "exception", "no-local", "no-scratch"])
def test_optional_macos_egress_qualification_requires_exact_native_socket(
    monkeypatch, tmp_path, case
):
    """Only a successful exact-path native probe advertises mediated egress."""
    backend = MacOSSeatbeltBackend(MagicMock())
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    environment = {"PATH": os.environ["PATH"], "LANG": "C", "TMPDIR": str(scratch)}
    if case == "no-local":
        backend.capabilities = frozenset()
    if case == "no-scratch":
        environment.pop("TMPDIR")
    native = request(
        tmp_path,
        local_network=False,
        environment=environment,
        scratch_root=scratch if case != "no-scratch" else None,
    )
    listener = MagicMock()
    monkeypatch.setattr(macos_backend.socket, "socket", MagicMock(return_value=listener))
    result = MagicMock(returncode=0 if case == "available" else 1)
    run = MagicMock(return_value=result)
    if case == "exception":
        run.side_effect = OSError("native probe unavailable")
    monkeypatch.setattr(macos_backend.subprocess, "run", run)

    backend._qualify_mediated_egress(native)

    assert (SandboxCapability.MEDIATED_EGRESS in backend.capabilities) is (case == "available")
    if case in {"available", "refused"}:
        probe = run.call_args.args[0]
        assert probe[0] == "/usr/bin/sandbox-exec"
        assert "(remote unix-socket (literal " in probe[2]
        assert "cross-use" in probe[-1]
    else:
        assert not run.called if case in {"no-local", "no-scratch"} else run.called
    backend.close()


@pytest.mark.parametrize(
    "failure", [OSError("mount failed"), subprocess.TimeoutExpired("diskutil", 1)]
)
def test_optional_local_socket_mount_failure_keeps_core_commands(monkeypatch, tmp_path, failure):
    """An unavailable local volume removes only optional IPC from the native backend."""
    backend = MacOSSeatbeltBackend(MagicMock())
    monkeypatch.setattr(macos_backend, "_mount_local_socket_volume", MagicMock(side_effect=failure))
    backend._qualify_local_network(request(tmp_path, local_network=False))
    assert backend.capabilities == frozenset()
    assert backend.run(request(tmp_path)).outcome is SandboxOutcome.UNAVAILABLE
    monkeypatch.setattr(
        macos_backend.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0))
    )
    monkeypatch.setattr(macos_backend.subprocess, "Popen", MagicMock(return_value=Process()))
    assert backend.run(request(tmp_path, local_network=False)).outcome is SandboxOutcome.COMPLETED
    backend.close()


@pytest.mark.parametrize("probe_failure", ["result", "exception", "eject"])
def test_optional_local_socket_probe_failure_reclaims_or_retains_volume(
    monkeypatch, tmp_path, probe_failure
):
    """A failed optional probe disables IPC and retains failed cleanup for a later retry."""
    backend = MacOSSeatbeltBackend(MagicMock())
    probe = MagicMock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=1))
    if probe_failure == "exception":
        probe.side_effect = OSError("probe failed")
    monkeypatch.setattr(backend, "qualification_probe", probe)
    eject = MagicMock(side_effect=OSError("eject failed") if probe_failure == "eject" else None)
    monkeypatch.setattr(macos_backend, "_eject_local_socket_volume", eject)

    backend._qualify_local_network(request(tmp_path, local_network=False))

    assert backend.capabilities == frozenset()
    eject.assert_called_once()
    if probe_failure == "eject":
        assert backend._local_volume is not None
        eject.side_effect = None
        backend.close()
        assert eject.call_count == 2
    else:
        assert backend._local_volume is None
        backend.close()


def test_local_socket_volume_close_waits_for_active_attempt(monkeypatch, tmp_path):
    """Backend close defers volume ejection until the final local attempt releases it."""
    backend = MacOSSeatbeltBackend(MagicMock())
    eject = MagicMock()
    monkeypatch.setattr(macos_backend, "_eject_local_socket_volume", eject)
    with backend._local_socket_directory(request(tmp_path), tmp_path) as directory:
        assert directory.is_dir()
        backend.close()
        eject.assert_not_called()
    eject.assert_called_once()
    with (
        pytest.raises(OSError, match="closed"),
        backend._local_socket_directory(request(tmp_path), tmp_path),
    ):
        pass


def test_local_socket_volume_close_retries_failed_eject(monkeypatch, tmp_path):
    """A failed eject keeps its verified identity available for the next close attempt."""
    backend = MacOSSeatbeltBackend(MagicMock())
    with backend._local_socket_directory(request(tmp_path), tmp_path):
        pass
    eject = MagicMock(side_effect=[OSError("busy"), None])
    monkeypatch.setattr(macos_backend, "_eject_local_socket_volume", eject)
    with pytest.raises(OSError, match="busy"):
        backend.close()
    assert backend._local_volume is not None
    backend.close()
    assert backend._local_volume is None
    assert eject.call_count == 2


@pytest.mark.parametrize("case", ["changed", "ancestor", "same_device"])
def test_local_socket_volume_rejects_changed_or_shared_device(monkeypatch, tmp_path, case):
    """A local volume cannot be shared with an approved writable root or change devices."""
    backend = MacOSSeatbeltBackend(MagicMock())
    local_root = tmp_path.parent / "local-sockets"
    writable = tmp_path
    if case == "changed":
        monkeypatch.setattr(
            macos_backend,
            "_mount_local_socket_volume",
            lambda _deadline: macos_backend.LocalSocketVolume(
                "/dev/disk0", local_root, 998, "unit", "unit-uuid"
            ),
        )
    elif case == "ancestor":
        writable = tmp_path.parent
    else:
        writable = tmp_path.parent / "sibling"
        writable.mkdir()
        original_stat = Path.stat

        def shared_stat(path, *args, **kwargs):
            """Model another writable root on the same native volume."""
            details = original_stat(path, *args, **kwargs)
            if path == writable:
                values = list(details)
                values[2] = -1
                return os.stat_result(values)
            return details

        monkeypatch.setattr(Path, "stat", shared_stat)
    with (
        pytest.raises(OSError, match="changed|shares"),
        backend._local_socket_directory(request(tmp_path, write_roots=(writable,)), tmp_path),
    ):
        pass
    backend.close()


@pytest.mark.parametrize(
    "case",
    [
        "file",
        "directory",
        "wrong-type",
        "relative",
        "alias",
        "owner",
        "writable",
        "mount",
        "ancestor-type",
        "acl",
        "changed",
        "veto",
    ],
)
def test_trusted_system_path_requires_read_only_acl_free_stable_ancestry(
    tmp_path, monkeypatch, case
):
    """Only root-owned, immutable, no-follow system paths can justify inference."""
    path = (tmp_path / "system-file").resolve()
    path.write_text("fixture")
    if case == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(path)
        path = alias
    if case == "relative":
        path = Path("relative")
    paths = {path, *path.parents}
    mode = stat.S_IFDIR | 0o755 if case in {"directory", "wrong-type"} else stat.S_IFREG | 0o755

    def metadata(entry):
        """Provide stable root-owned test identities for selected ancestry."""
        entry_mode = mode if entry == path else stat.S_IFDIR | 0o755
        if case == "owner" and entry == path:
            owner = 1000
        else:
            owner = 0
        if case == "writable" and entry == path:
            entry_mode |= 0o020
        if case == "ancestor-type" and entry == path.parent:
            entry_mode = stat.S_IFREG | 0o755
        return SimpleNamespace(st_mode=entry_mode, st_uid=owner, st_gid=0, st_dev=1, st_ino=2)

    original_lstat = Path.lstat
    current = {}

    def opened(entry, *_args):
        """Bind the mocked descriptor to its selected path."""
        current["entry"] = entry
        return 7

    def bound(_fd):
        """Return the selected descriptor identity or simulate replacement."""
        value = metadata(current["entry"])
        if case == "changed":
            value.st_ino += 1
        return value

    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "lstat",
            lambda entry: metadata(entry) if entry in paths else original_lstat(entry),
        )
        patch.setattr(
            "loop.execution.sandbox.macos.backend.os.statvfs",
            lambda _path: SimpleNamespace(f_flag=0 if case == "mount" else os.ST_RDONLY),
        )
        patch.setattr("loop.execution.sandbox.macos.backend.os.open", opened)
        patch.setattr("loop.execution.sandbox.macos.backend.os.close", lambda _fd: None)
        patch.setattr(
            "loop.execution.sandbox.macos.backend.os.fstat",
            bound,
        )

        def acl(_fd):
            """Return a native ACL result or surface an actual access veto."""
            if case == "veto":
                raise PermissionError("ACL veto")
            return case == "acl"

        patch.setattr("loop.execution.sandbox.macos.backend.has_extended_access_acl", acl)
        backend = MacOSSeatbeltBackend()
        if case == "veto":
            with pytest.raises(PermissionError, match="ACL veto"):
                backend.trusted_system_path(path)
        else:
            assert backend.trusted_system_path(path, directory=case == "directory") is (
                case in {"file", "directory"}
            )


def test_backend_and_final_launch_gate_honor_pre_cancelled_request(monkeypatch, tmp_path):
    """Cancellation during preparation or just before spawn prevents a child launch."""
    cancellation_event = Event()
    cancellation_event.set()
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)

    result = MacOSSeatbeltBackend().run(
        request(tmp_path),
        cancellation_event=cancellation_event,
    )
    final_gate = MacOSSeatbeltBackend._execute(
        request(tmp_path),
        "",
        {},
        "",
        None,
        cancellation_event=cancellation_event,
    )

    assert result.outcome is SandboxOutcome.CANCELLED
    assert final_gate.outcome is SandboxOutcome.CANCELLED
    launch.assert_not_called()


def regex_patterns(profile):
    """Decode emitted Seatbelt regex strings for semantic policy assertions."""
    return "\n".join(
        json.loads(value) for value in re.findall(r'\(regex ("(?:\\.|[^"\\])*")\)', profile)
    )


class Process:
    """Provide the subprocess surface used by the backend capture loop."""

    def __init__(self, stdout="ok", stderr="", returncode=0, wait_error=None, started=True):
        self.stdout = (
            io.TextIOWrapper(
                io.BytesIO((("LOOP_SHELL_STARTED\n" if started else "") + stdout).encode()),
                encoding="utf-8",
            )
            if stdout is not None
            else None
        )
        self.stderr = (
            io.TextIOWrapper(io.BytesIO(stderr.encode()), encoding="utf-8")
            if stderr is not None
            else None
        )
        self.returncode = returncode
        self.pid = 123
        self.wait_error = wait_error

    def wait(self, timeout=None):
        """Return completion or raise the configured timeout."""
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode


class LogStream(io.BytesIO):
    """Represent a diagnostic descriptor and producer EOF entirely in memory."""

    eof: bool

    def __init__(self, records):
        super().__init__(records)
        self.eof = False

    def fileno(self):
        """Return this controlled descriptor for the scoped select and read doubles."""
        return self


@pytest.fixture(autouse=True)
def diagnostic_pipe_boundary(monkeypatch):
    """Model readable records and EOF without allocating or waiting on native pipes."""
    original_read = os.read

    def select_streams(readers, writers, errors, timeout):
        """Return only streams with buffered bytes or a closed producer."""
        ready = []
        for reader in readers:
            stream = reader.fileno()
            assert isinstance(stream, LogStream)
            if stream.tell() < len(stream.getvalue()) or stream.eof:
                ready.append(reader)
        return ready, [], []

    def read_stream(descriptor, count):
        """Read diagnostic bytes in memory while preserving ordinary owned file reads."""
        if isinstance(descriptor, LogStream):
            return descriptor.read(count)
        return original_read(descriptor, count)

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.select", SimpleNamespace(select=select_streams)
    )
    monkeypatch.setattr(os, "read", read_stream)


class LogProcess:
    """Expose bounded kernel-log bytes through a real readable pipe."""

    def __init__(
        self,
        records=b"",
        *,
        running=False,
        stdout=True,
        wait_error=None,
        stuck_after_kill=False,
    ):
        self.records = records
        self.running = running
        self.has_stdout = stdout
        self.killed = False
        self.wait_error = wait_error
        self.stuck_after_kill = stuck_after_kill
        self.wait_timeouts = []
        self.stdout = LogStream(records) if stdout else None
        if not running or not stuck_after_kill:
            self.finish()

    def clone(self):
        """Create a fresh in-memory stream for a later bounded log snapshot."""
        return LogProcess(
            self.records,
            running=self.running,
            stdout=self.has_stdout,
            wait_error=self.wait_error,
            stuck_after_kill=self.stuck_after_kill,
        )

    def poll(self):
        """Report whether the simulated log query still runs."""
        return None if self.running and (self.stuck_after_kill or not self.killed) else 0

    def kill(self):
        """Record termination of a simulated running log query."""
        self.killed = True
        if not self.stuck_after_kill:
            self.finish()

    def wait(self, timeout=None):
        """Complete the simulated log query."""
        self.wait_timeouts.append(timeout)
        if self.wait_error is not None:
            raise self.wait_error
        return 0

    def finish(self):
        """Close the producer after its data has been consumed."""
        if self.stdout is not None:
            self.stdout.eof = True


class CloseErrorStream:
    """Close a pipe and report a diagnostic close failure."""

    def __init__(self, stream):
        self._stream = stream

    @property
    def closed(self):
        """Report whether the underlying pipe is closed."""
        return self._stream.closed

    def fileno(self):
        """Return the underlying pipe descriptor for select and os.read."""
        return self._stream.fileno()

    def close(self):
        """Close the pipe before raising the configured I/O failure."""
        self._stream.close()
        raise OSError("diagnostic pipe close failed")


class PollErrorLogProcess(LogProcess):
    """Simulate a diagnostic child whose status cannot be queried."""

    def __init__(self):
        super().__init__()
        self.stdout = CloseErrorStream(self.stdout)
        self.kill_attempts = 0

    def poll(self):
        """Raise the process-status error used to exercise best-effort cleanup."""
        raise OSError("diagnostic process status failed")

    def kill(self):
        """Reject termination after the process-status query has failed."""
        self.kill_attempts += 1
        raise OSError("diagnostic process termination failed")


def kernel_record(message, *, process_id=0, image="/kernel", sender="/Sandbox.kext/Sandbox"):
    """Encode one structured macOS kernel log record."""
    return (
        json.dumps(
            {
                "processID": process_id,
                "processImagePath": image,
                "senderImagePath": sender,
                "eventMessage": message,
            }
        ).encode()
        + b"\n"
    )


def run_with_log(
    monkeypatch,
    tmp_path,
    log_process,
    *,
    output="Permission denied",
    log_error=None,
    expire_after_wait=False,
    later_records=None,
    returncode=1,
):
    """Run the public backend with one completed child and one log query."""
    command = Process(stdout="", stderr=output, returncode=returncode)
    launches = 0

    def launch(*args, **kwargs):
        """Return the command once and a fresh log stream on each diagnostic retry."""
        nonlocal launches
        launches += 1
        if launches == 1:
            return command
        if log_error is not None:
            raise log_error
        if launches == 2:
            return log_process
        return LogProcess(later_records) if later_records is not None else log_process.clone()

    popen = MagicMock(side_effect=launch)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.secrets.token_hex",
        MagicMock(side_effect=["a" * 32, "b" * 32]),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    value = request(tmp_path)
    if expire_after_wait:

        def expire(timeout=None):
            """Move the request deadline past before denial inspection."""
            object.__setattr__(value, "deadline", time.monotonic() - 1)
            return 1

        command.wait = expire
    result = MacOSSeatbeltBackend().run(value)
    return result, popen


def test_backend_rejects_unsupported_policy(tmp_path):
    """A policy mismatch fails before launcher preparation."""
    result = MacOSSeatbeltBackend().run(request(tmp_path, policy_version="other"))
    assert result.outcome is SandboxOutcome.UNAVAILABLE


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf"), True, "20"])
def test_backend_rejects_invalid_deadlines_before_preparation(monkeypatch, tmp_path, deadline):
    """A forged non-finite or nonnumeric deadline cannot launch the sandbox or search helper."""
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    backend = MacOSSeatbeltBackend()
    result = backend.run(replace(request(tmp_path), deadline=deadline))
    assert result.outcome is SandboxOutcome.INVALID
    assert backend.run_read_only_argv(tmp_path / "helper", (), (), deadline) is None
    launch.assert_not_called()


def test_backend_rejects_stale_initial_identity(monkeypatch, tmp_path):
    """A changed approved root fails before profile generation."""
    value = request(tmp_path)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: False)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.STALE
    assert "path identity changed" in result.detail


def test_backend_expired_budget_is_timeout_before_native_launch(monkeypatch, tmp_path):
    """An expired request remains a timeout without attempting native initialization."""
    value = request(tmp_path, deadline=100.0)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time.monotonic", lambda: 101.0)
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert result.detail == "Command deadline expired."
    popen.assert_not_called()


def test_backend_builds_profile_and_runs_child(monkeypatch, tmp_path):
    """A successful launch keeps private scratch and denies the shared directory helper."""
    probe = MagicMock(returncode=0, stderr="")
    process = Process()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(tmp_path, network=True))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.stdout == "ok"
    argv = popen.call_args.args[0]
    assert argv[-2:] == ["loop-shell", "printf ok"]
    assert argv[3:5] == ["/bin/sh", "-c"]
    assert "(allow network*)" in argv[2]
    assert '"com.apple.bsd.dirhelper"' not in argv[2]
    assert '"com.apple.system.opendirectoryd.libinfo"' in argv[2]
    assert "xcrun_db" not in argv[2]
    assert popen.call_args.kwargs["close_fds"] is True
    assert Path(popen.call_args.kwargs["env"]["TMPDIR"]).is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    )
    assert "LOOP_LOCAL_SOCKET_DIR" not in popen.call_args.kwargs["env"]


@pytest.mark.parametrize("local_network", [True, False])
def test_backend_scopes_attempt_local_unix_sockets(monkeypatch, tmp_path, local_network):
    """Offline Unix sockets use one disposable path or stay denied by a local limit."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path, local_network=local_network))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    environment = popen.call_args.kwargs["env"]
    if local_network:
        path = Path(environment["LOOP_LOCAL_SOCKET_DIR"])
        assert path.parent == tmp_path.parent / "local-sockets"
        assert f'(allow network-bind (local unix-socket (subpath "{path}")))' in profile
        assert f'(allow network-outbound (remote unix-socket (subpath "{path}")))' in profile
        assert "(allow network*)" not in profile
        assert not path.exists()
    else:
        assert "LOOP_LOCAL_SOCKET_DIR" not in environment
        assert "(allow network-" not in profile


@pytest.mark.parametrize(
    "case", ["missing-authority", "unqualified", "approved", "interrupted", "lost-authority"]
)
def test_backend_mediated_egress_uses_one_private_native_proxy(monkeypatch, tmp_path, case):
    """A mediated request needs exact policy authority and a scoped native broker path."""
    native = MacOSSeatbeltBackend()
    if case != "unqualified":
        native.capabilities = frozenset(
            {SandboxCapability.LOCAL_NETWORK, SandboxCapability.MEDIATED_EGRESS}
        )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(macos_backend.subprocess, "Popen", popen)
    monkeypatch.setattr(
        macos_backend.subprocess,
        "run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    broker = MagicMock()
    broker.interrupted = case == "interrupted"
    broker_factory = MagicMock(return_value=broker)
    broker.__enter__.return_value = broker
    monkeypatch.setattr(macos_backend, "MacOSEgressBroker", broker_factory)
    authority = MagicMock()
    value = request(
        tmp_path,
        local_network=False,
        egress_origins=("https://example.com",),
    )

    result = (
        native.run(
            value,
            **(
                {}
                if case == "missing-authority"
                else {"policy_authority": authority, "approved_revision": 7}
            ),
        )
        if case != "lost-authority"
        else native._run(
            value,
            cancellation_event=None,
            qualification=False,
        )
    )

    if case in {"missing-authority", "unqualified", "lost-authority"}:
        assert (
            result.outcome is SandboxOutcome.UNAVAILABLE or result.outcome is SandboxOutcome.INVALID
        )
        if case != "lost-authority":
            popen.assert_not_called()
    else:
        assert result.outcome is (
            SandboxOutcome.UNAVAILABLE if case == "interrupted" else SandboxOutcome.COMPLETED
        )
        if case == "interrupted":
            assert result.possible_effects
            assert result.exit_code is None
        profile = popen.call_args.args[0][2]
        environment = popen.call_args.kwargs["env"]
        socket_path = broker_factory.call_args.args[1]
        assert f'(remote unix-socket (literal "{socket_path}"))' in profile
        assert environment["https_proxy"] == f"socks5h://localhost{socket_path}"
        assert environment["NO_PROXY"] == ""
        assert "LOOP_LOCAL_SOCKET_DIR" not in environment
    native.close()


def test_backend_discovers_a_verified_system_directory_from_the_existing_probe(
    monkeypatch, tmp_path
):
    """The fixed startup probe can suggest a verified directory without changing user source."""
    directory = Path("/System/example/runtime-cache")
    identity = DirectoryIdentity(directory, 7, 11)
    monitor = MagicMock()
    monitor.register.return_value = True
    monitor.denial.return_value = f"Sandbox: true(123) deny(1) file-read-data {directory}"
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda _self, _deadline: monitor,
    )
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", lambda *_args: identity)
    probe = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(macos_backend.subprocess, "run", probe)
    monkeypatch.setattr(macos_backend.subprocess, "Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_args.args[0][-1] == "printf ok"
    assert f'(literal "{directory}")' in popen.call_args.args[0][2]
    assert f'(subpath "{directory}")' not in popen.call_args.args[0][2]
    assert probe.call_args.kwargs["cwd"] == "/"
    assert probe.call_args.kwargs["env"]["PATH"] == "/usr/bin:/bin"
    assert monitor.register.call_count == 2
    assert monitor.unregister.call_count == 2


def test_startup_matches_are_not_treated_as_proven_incidental():
    """A matching startup operation remains an uncertain candidate for the command."""
    service = macos_backend.SeatbeltDiagnosticService()
    service.parent_verified()
    startup = "Sandbox: true(1) deny(1) file-read-data /system/cache"
    service.observe_startup_denial(startup)
    backend = MacOSSeatbeltBackend(service)
    observed = "Sandbox: sh(2) deny(1) file-read-data /system/cache"

    candidates, baseline_status = backend.classify_denial_candidates(observed)

    assert baseline_status == "observed"
    assert candidates == ((observed, True),)
    other = "Sandbox: cat(3) deny(1) file-read-data /outside/secret"
    assert backend.classify_denial_candidates(other) == (((other, False),), "observed")


def test_clipped_startup_observation_cannot_establish_a_match():
    """Partial resources cannot be compared as if their full paths were known."""
    service = macos_backend.SeatbeltDiagnosticService()
    service.parent_verified()
    clipped = "Sandbox: true(1) deny(1) file-read-data /" + "x" * 460 + "…"
    service.observe_startup_denial(clipped)
    backend = MacOSSeatbeltBackend(service)

    candidates, baseline_status = backend.classify_denial_candidates(clipped)

    assert baseline_status == "not_observed"
    assert candidates == ((clipped, False),)


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "x" * 2001,
        "\x1eLOOP_DENIALS:not-json",
        "\x1eLOOP_DENIALS:[]",
        '\x1eLOOP_DENIALS:[""]',
        "\x1eLOOP_DENIALS:[1]",
        "\x1eLOOP_DENIALS:" + json.dumps(["x"] * 5),
        "\x1eLOOP_DENIALS:" + json.dumps(["x" * 501]),
    ],
)
def test_invalid_encoded_candidates_do_not_create_observations(encoded):
    """Malformed bounded collections cannot invent or split kernel observations."""
    backend = MacOSSeatbeltBackend()
    candidates, baseline_status = backend.classify_denial_candidates(encoded)
    assert candidates == ()
    assert baseline_status == "not_observed"


@pytest.mark.parametrize("error", [FileNotFoundError, NotADirectoryError])
def test_missing_probe_directory_does_not_block_commands(monkeypatch, tmp_path, error):
    """A disappearing optional loader candidate cannot stop a sandboxed command."""
    monitor = MagicMock()
    monitor.register.return_value = True
    monitor.denial.return_value = "Sandbox: true(1) deny(1) file-read-data /System/example"
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", lambda *_: monitor)
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", MagicMock(side_effect=error))
    monkeypatch.setattr(
        macos_backend.subprocess, "run", MagicMock(return_value=SimpleNamespace(returncode=0))
    )
    monkeypatch.setattr(macos_backend.subprocess, "Popen", MagicMock(return_value=Process()))

    assert MacOSSeatbeltBackend().run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED


def test_probe_directory_recovery_and_concurrent_publication(monkeypatch):
    """Stream recovery can probe later without overwriting a concurrent verified candidate."""
    service = macos_backend.SeatbeltDiagnosticService()
    assert not service.directory_probe_pending()
    assert not service.parent_verified()
    service.mark_parent_verified(directory_probe_attempted=False)
    assert service.parent_verified() and service.directory_probe_pending()
    first = DirectoryIdentity(Path("/System/example"), 1, 2)
    service.set_system_directory(first)
    service.set_system_directory(None)
    service.set_system_directory(DirectoryIdentity(Path("/System/other"), 3, 4))
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", lambda *_: first)
    assert service.system_directory(time.monotonic() + 1) == first.path
    service.mark_parent_verified()
    assert not service.directory_probe_pending()
    monkeypatch.setattr(
        macos_backend, "_trusted_probe_directory", MagicMock(side_effect=FileNotFoundError)
    )
    assert service.system_directory(time.monotonic() + 1) is None
    assert not service.parent_verified() and service.directory_probe_pending()


def test_stale_validation_does_not_clear_a_new_concurrent_candidate(monkeypatch):
    """An older validation cannot erase a candidate published by a newer probe."""
    service = macos_backend.SeatbeltDiagnosticService()
    service.parent_verified()
    first = DirectoryIdentity(Path("/System/first"), 1, 2)
    replacement = DirectoryIdentity(Path("/System/second"), 3, 4)
    service.set_system_directory(first)

    def replace_during_validation(*_args):
        """Simulate process-state reset and publication while validation is in flight."""
        service.close()
        service.parent_verified()
        service.set_system_directory(replacement)

    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", replace_during_validation)
    assert service.system_directory(time.monotonic() + 1) is None
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", lambda *_: replacement)
    assert service.system_directory(time.monotonic() + 1) == replacement.path


def test_backend_probes_again_after_diagnostic_stream_recovers(monkeypatch, tmp_path):
    """A later healthy stream can discover the directory without retrying a user command."""
    backend = MacOSSeatbeltBackend()
    monkeypatch.setattr(backend, "_qualify_local_network", MagicMock())
    monkeypatch.setattr(backend, "_qualify_mediated_egress", MagicMock())
    monkeypatch.setattr(backend, "_qualify_terminal", MagicMock())
    backend.complete_qualification(request(tmp_path))
    monitor = MagicMock()
    monitor.register.return_value = True
    monitor.denial.return_value = "Sandbox: true(1) deny(1) file-read-data /System/example"
    get_monitor = MagicMock(side_effect=[None, monitor])
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", get_monitor)
    candidate = DirectoryIdentity(Path("/System/example"), 1, 2)
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", lambda *_: candidate)
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    launch = MagicMock(side_effect=[Process(), Process()])
    monkeypatch.setattr(macos_backend.subprocess, "run", run)
    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)

    assert backend.run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED
    assert backend.run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED
    assert run.call_count == 2
    assert launch.call_count == 2
    assert f'(literal "{candidate.path}")' in launch.call_args.args[0][2]


@pytest.mark.parametrize(
    "denial",
    [
        None,
        "other diagnostic",
        "x file-read-data relative",
        "x" * 4096,
        "x file-read-data /System/clipped…",
    ],
)
def test_probe_rejects_malformed_or_truncated_diagnostics(monkeypatch, denial):
    """A partial or unrelated diagnostic cannot suggest filesystem authority."""
    validator = MagicMock(side_effect=AssertionError("unexpected filesystem inspection"))
    monkeypatch.setattr(macos_backend, "_trusted_probe_directory", validator)
    assert macos_backend._probe_system_directory(denial, time.monotonic() + 1) is None


def test_probe_directory_preserves_audit_veto(monkeypatch):
    """An optional candidate never swallows a permission veto."""
    monkeypatch.setattr(
        macos_backend, "_trusted_probe_directory", MagicMock(side_effect=PermissionError)
    )
    with pytest.raises(PermissionError):
        macos_backend._probe_system_directory(
            "x file-read-data /System/example", time.monotonic() + 1
        )


@pytest.mark.parametrize("code", [errno.ELOOP, errno.ESTALE])
def test_probe_directory_rejects_replaced_ancestor(monkeypatch, code):
    """A replaced or stale ancestor drops optional compatibility authority."""
    monkeypatch.setattr(
        macos_backend, "_trusted_probe_directory", MagicMock(side_effect=OSError(code, "changed"))
    )
    assert macos_backend._optional_trusted_probe_directory(Path("/System/example"), 1) is None


@pytest.mark.parametrize("case", ["under-home", "contains-home", "missing-system-flags"])
def test_probe_directory_rejects_home_or_missing_system_authority(monkeypatch, case):
    """Neither user data nor an unprovable system flag may become automatic authority."""
    candidate = Path("/example/cache")
    home = candidate if case == "under-home" else candidate / "user"
    if case == "missing-system-flags":
        home = Path("/other/home")
        monkeypatch.setattr(stat, "SF_IMMUTABLE", 0, raising=False)
        monkeypatch.setattr(stat, "SF_RESTRICTED", 0, raising=False)
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: home)
    monkeypatch.setattr(Path, "is_dir", lambda _self: True)
    assert macos_backend._trusted_probe_directory(candidate, time.monotonic() + 1) is None


@pytest.mark.parametrize("candidate", [Path("/home/user/cache"), Path("/example/.ssh/keys")])
def test_probe_directory_rejects_private_paths_before_stat(monkeypatch, candidate):
    """Diagnostics cannot cause host-side metadata reads under a protected path."""
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: Path("/home/user"))
    monkeypatch.setattr(
        Path, "is_dir", MagicMock(side_effect=AssertionError("private path must not be inspected"))
    )
    assert macos_backend._trusted_probe_directory(candidate, time.monotonic() + 1) is None


@pytest.mark.parametrize(
    "case",
    [
        "trusted",
        "writable-parent-protected",
        "writable-parent-unprotected",
        "writable-target",
        "nonroot-ancestor",
        "nonroot-target",
        "directory-acl",
        "missing-system-flag",
        "changed-identity",
        "other-system-root",
        "private-ancestor",
        "private-library",
        "other-library",
        "system-library",
        "missing",
        "relative",
        "traversal",
    ],
)
def test_system_directory_requires_independent_filesystem_authority(monkeypatch, case):
    """A probe hint grants only an independently authenticated system directory."""
    path = {
        "other-system-root": Path("/other/example/cache"),
        "private-ancestor": Path("/System/.ssh/cache"),
        "private-library": Path("/System/Library"),
        "other-library": Path("/other/Library/cache"),
        "system-library": Path("/other/System/Library/cache"),
        "relative": Path("System/example/cache"),
        "traversal": Path("/System/example/../cache"),
    }.get(case, Path("/System/example/cache"))
    identity = DirectoryIdentity(path, 7, 11)
    monkeypatch.setattr(stat, "SF_IMMUTABLE", 0x20000, raising=False)
    monkeypatch.setattr(stat, "SF_NOUNLINK", 0x100000, raising=False)
    monkeypatch.setattr(stat, "SF_RESTRICTED", 0x80000, raising=False)
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: Path("/Users/example"))
    original_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda value: case != "missing" if value == path else original_is_dir(value),
    )
    descriptors = {}
    original_open = macos_backend.os.open
    original_fstat = macos_backend.os.fstat
    original_close = macos_backend.os.close

    def open_directory(name, flags, mode=0o777, *, dir_fd=None):
        """Provide a no-follow descriptor for each synthetic system component."""
        if not (name == "/" and dir_fd is None or dir_fd in descriptors):
            return original_open(name, flags, mode, dir_fd=dir_fd)
        descriptor = len(descriptors) + 1_000_000
        descriptors[descriptor] = (str(name), dir_fd)
        return descriptor

    def directory_stat(descriptor):
        """Provide root-owned directories and one selected invalid authority case."""
        if descriptor not in descriptors:
            return original_fstat(descriptor)
        name, _ = descriptors[descriptor]
        mode = stat.S_IFDIR | 0o755
        if name == "example" and case.startswith("writable-parent"):
            mode |= 0o022
        if name == "cache" and case == "writable-target":
            mode |= 0o022
        owner = (
            1
            if (name == "example" and case == "nonroot-ancestor")
            or (name == "cache" and case == "nonroot-target")
            else 0
        )
        flags = 0x80000 if name == "cache" and case != "missing-system-flag" else 0
        if name == "cache" and case == "writable-parent-protected":
            flags |= 0x100000
        return SimpleNamespace(st_mode=mode, st_uid=owner, st_flags=flags, st_dev=7, st_ino=11)

    def close_directory(descriptor):
        """Close real descriptors while treating synthetic ones as already owned."""
        if descriptor not in descriptors:
            original_close(descriptor)

    monkeypatch.setattr(macos_backend.os, "open", open_directory)
    monkeypatch.setattr(macos_backend.os, "fstat", directory_stat)
    monkeypatch.setattr(macos_backend.os, "close", close_directory)
    monkeypatch.setattr(
        macos_backend, "has_extended_access_acl", lambda _descriptor: case == "directory-acl"
    )
    monkeypatch.setattr(DirectoryIdentity, "is_current", lambda _self: case != "changed-identity")
    service = macos_backend.SeatbeltDiagnosticService()
    service.parent_verified()
    service.set_system_directory(identity)

    result = service.system_directory(time.monotonic() + 10)

    assert result == (
        path
        if case in {"trusted", "writable-parent-protected", "other-system-root", "system-library"}
        else None
    )


def test_backend_guards_private_names_when_system_root_is_inside_path(monkeypatch, tmp_path):
    """An overlapping PATH root excludes its system subtree but keeps its own guard."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = tmp_path / "system"
    system = parent / "Library"
    system.mkdir(parents=True)
    monkeypatch.setattr(macos_backend, "_SYSTEM_READ_ROOTS", (system,))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(
        request(workspace, environment={"PATH": str(parent), "LANG": "C"})
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert f'(require-not (require-any (subpath "{system}")))' in profile
    patterns = regex_patterns(profile)
    assert f"^{re.escape(str(parent))}/" in patterns
    assert f"^{re.escape(str(system))}/" in patterns


def test_backend_limits_private_name_filters_to_granted_roots(monkeypatch, tmp_path):
    """An ungranted home needs no broad filter, while an approved home read does."""
    home = tmp_path.with_name(tmp_path.name + "-home")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(side_effect=lambda *_args, **_kwargs: Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    backend = MacOSSeatbeltBackend()

    assert backend.run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED
    patterns = regex_patterns(popen.call_args.args[0][2]).splitlines()
    home_prefix = f"^{re.escape(str(home))}/"
    assert not any(pattern.startswith(home_prefix) for pattern in patterns)

    result = backend.run(request(tmp_path, read_roots=(home,)))
    assert result.outcome is SandboxOutcome.COMPLETED, result.detail
    patterns = regex_patterns(popen.call_args.args[0][2]).splitlines()
    assert any(pattern.startswith(home_prefix) for pattern in patterns)


def test_backend_uses_owned_scratch_under_private_temporary_ancestry(monkeypatch, tmp_path):
    """Fresh backend scratch remains usable without granting its private parent."""
    private = tmp_path / "Library" / "temporary"
    private.mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    scratch = Path(popen.call_args.kwargs["env"]["TMPDIR"])
    profile = popen.call_args.args[0][2]
    assert scratch.parent == private
    assert f'(require-all (require-not (subpath "{scratch}")) (require-any' in profile
    assert f"^{re.escape(str(scratch))}/(.*/)?" not in regex_patterns(profile)


def test_backend_accepts_unbound_scratch_under_the_temporary_root(monkeypatch, tmp_path):
    """A caller-provided private child of the ordinary temporary root stays usable."""
    scratch = Path(tempfile.gettempdir()) / "loop-seatbelt-provided"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)})
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_args.kwargs["env"]["TMPDIR"] == str(scratch)


def test_backend_guards_unbound_scratch_under_home(monkeypatch, tmp_path):
    """A direct writable scratch under home still denies private-name descendants."""
    home = tmp_path.with_name(tmp_path.name + "-home")
    home.chmod(0o700)
    temporary = home / "temporary"
    temporary.mkdir(mode=0o700)
    scratch = temporary / "loop-seatbelt-provided"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)})
    )

    assert result.outcome is SandboxOutcome.COMPLETED, result.detail
    patterns = regex_patterns(popen.call_args.args[0][2]).splitlines()
    scratch_prefix = f"^{re.escape(str(scratch))}/"
    assert any(
        re.search(pattern, str(scratch / ".ssh" / "key"))
        for pattern in patterns
        if pattern.startswith(scratch_prefix)
    )


def test_backend_rejects_unbound_scratch_under_private_temporary_ancestry(monkeypatch, tmp_path):
    """A caller-provided scratch path cannot claim the backend-owned scratch exception."""
    private = tmp_path / "Library" / "temporary"
    scratch = private / "loop-seatbelt-provided"
    scratch.mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)})
    )

    assert result.outcome is SandboxOutcome.INVALID
    assert "Protected directory cannot be a read root" in result.detail
    assert not result.possible_effects
    popen.assert_not_called()


def test_backend_audits_launch_with_the_approved_attempt_id(monkeypatch, tmp_path):
    """Native launch records correlate to the same approved request without shell text."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    approved = request(tmp_path, source="printf private-command")
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        assert MacOSSeatbeltBackend().run(approved).outcome is SandboxOutcome.COMPLETED
        assert telemetry.close(1)
    finally:
        set_telemetry(None)
    records = [item for item in adapter.records if item.event_name.startswith("sandbox.launch")]
    assert [item.event_name for item in records] == ["sandbox.launching", "sandbox.launched"]
    assert all(item.attributes["attempt_id"] == approved.attempt_id for item in records)
    assert all("private-command" not in str(item.attributes) for item in records)


def test_backend_does_not_launch_when_cancelled_during_launch_audit(monkeypatch, tmp_path):
    """Cancellation observed after launch auditing prevents a child and possible effects."""
    cancellation_event = Event()
    launch = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)

    def cancel_during_audit(event_name, **_attributes):
        """Cancel when the prelaunch audit records the approved attempt."""
        if event_name == "sandbox.launching":
            cancellation_event.set()

    monkeypatch.setattr("loop.execution.sandbox.macos.backend.telemetry_audit", cancel_during_audit)
    result = MacOSSeatbeltBackend().run(request(tmp_path), cancellation_event=cancellation_event)

    assert result.outcome is SandboxOutcome.CANCELLED
    assert not result.possible_effects
    launch.assert_not_called()


def test_backend_passes_opaque_command_source(monkeypatch, tmp_path):
    """The approved command source reaches the shell without textual rewriting."""
    source = "printf '%s' '$WORKSPACE_ROOT/quoted'"
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(tmp_path, source=source))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_args.args[0][-1] == source


def test_backend_granted_git_metadata_keeps_nested_git_protected(monkeypatch, tmp_path):
    """An approved root Git grant leaves nested repository metadata denied."""
    git = tmp_path / ".git"
    git.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, write_roots=(tmp_path, git)))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "/.+/\\.git" in regex_patterns(profile)


def test_backend_allows_approved_fresh_git_metadata_only_at_workspace_root(monkeypatch, tmp_path):
    """A missing top-level Git directory can be created without opening nested metadata."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, git_create=True))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "/.+/\\.git" in regex_patterns(profile)
    assert f'"^{tmp_path}/.*[.]git(/.*)?$"' not in profile


def test_backend_read_only_profile_has_no_workspace_write_grant(monkeypatch, tmp_path):
    """The OS profile denies workspace writes for an auto-approved read request."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path, write_roots=()))
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    write_rule = next(
        line for line in profile.splitlines() if line.startswith("(allow file-write* ")
    )
    assert str(tmp_path) not in write_rule


def test_backend_rejects_scratch_without_its_own_prefix(tmp_path):
    """Seatbelt alone enforces its scratch ownership convention before any launch."""
    with tempfile.TemporaryDirectory(prefix="loop-other-") as directory:
        bound = request(
            tmp_path,
            environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(Path(directory).resolve())},
        )
        result = MacOSSeatbeltBackend().run(bound)

    assert result.outcome is SandboxOutcome.INVALID
    assert "Unsafe command scratch directory" in result.detail


@pytest.mark.parametrize("name", ['quote"name', "slash\\name", "line\nname", "name[0]+;$()"])
def test_backend_encodes_complete_regex_strings(monkeypatch, tmp_path, name):
    """Filesystem and instruction names stay literal inside complete policy regex strings."""
    workspace = tmp_path / name
    workspace.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    instruction_name = name.replace("\\", "-")
    value = request(workspace, protected_instruction_names=(instruction_name,))
    assert MacOSSeatbeltBackend().run(value).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    patterns = regex_patterns(profile).splitlines() if "\n" not in name else None
    encoded = json.dumps(
        "^"
        + re.escape(str(workspace))
        + "/(.*/)?(AGENTS\\.md|SKILL\\.md|"
        + re.escape(instruction_name)
        + ")$",
        ensure_ascii=False,
    )
    assert "(regex " + encoded + ")" in profile
    assert '(regex #"' not in profile
    if patterns is not None:
        assert any(re.search(pattern, str(workspace / instruction_name)) for pattern in patterns)


@pytest.mark.parametrize("symlink", [False, True])
def test_backend_grants_only_bound_external_executable_files(monkeypatch, tmp_path, symlink):
    """Exact executable authority does not grant neighboring file or directory data."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir(mode=0o700)
    executable = tools / "target"
    executable.write_text("#!/bin/sh\nprintf exact")
    executable.chmod(0o700)
    spelling = tools / "tool" if symlink else executable
    if symlink:
        spelling.symlink_to(executable)
    details = executable.stat()
    value = request(
        workspace,
        environment={"PATH": str(tools)},
        executable_identities=(
            (spelling.name, spelling, executable, details.st_dev, details.st_ino),
        ),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    assert MacOSSeatbeltBackend().run(value).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    read = next(line for line in profile.splitlines() if line.startswith("(allow file-read* "))
    assert f'(literal "{executable}")' in read
    assert f'(literal "{spelling}")' in read
    assert f'(subpath "{tools}")' not in read


@pytest.mark.parametrize(
    "unsafe",
    [
        "root_world",
        "root_group",
        "root_owner",
        "mode",
        "owner",
        "hardlink",
        "private",
        "unmounted",
        "inode",
        "alias",
    ],
)
def test_backend_rejects_unsafe_exact_executable_authority(monkeypatch, tmp_path, unsafe):
    """Unsafe exact executable identity, root or file permissions refuse source launch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = tmp_path / (".ssh" if unsafe == "private" else "tools")
    tools.mkdir(mode=0o700)
    executable = tools / "tool"
    executable.write_text("executable")
    executable.chmod(0o700)
    details = executable.stat()
    value = request(
        workspace,
        environment={"PATH": str(tools)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )
    original_lstat, original_stat = Path.lstat, Path.stat

    def observed(path, original):
        """Expose changed metadata at the actual file or PATH-root boundary."""
        result = original(path)
        fields = {
            name: getattr(result, name)
            for name in ("st_mode", "st_uid", "st_gid", "st_dev", "st_ino", "st_nlink")
        }
        if path == tools:
            if unsafe == "root_world":
                fields["st_mode"] |= 0o002
            if unsafe == "root_group":
                fields.update(st_mode=fields["st_mode"] | 0o020, st_gid=os.getgid() + 1)
            if unsafe == "root_owner":
                fields["st_uid"] = os.getuid() + 1
        if path == executable:
            if unsafe == "mode":
                fields["st_mode"] |= 0o002
            if unsafe == "owner":
                fields["st_uid"] = os.getuid() + 1
            if unsafe == "hardlink":
                fields["st_nlink"] = 2
        return (
            SimpleNamespace(**fields)
            if (path == tools and unsafe.startswith("root_"))
            or (path == executable and unsafe in {"mode", "owner", "hardlink"})
            else result
        )

    monkeypatch.setattr(Path, "lstat", lambda path: observed(path, original_lstat))
    monkeypatch.setattr(Path, "stat", lambda path, **kwargs: observed(path, original_stat))
    if unsafe == "unmounted":
        monkeypatch.setattr(
            "loop.execution.sandbox.macos.backend._search_roots", lambda request: ()
        )
    if unsafe in {"inode", "alias"}:
        original = tools / "original"
        executable.rename(original)
        if unsafe == "alias":
            executable.symlink_to(original)
        else:
            executable.write_text("changed")
            executable.chmod(0o700)
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome in {SandboxOutcome.INVALID, SandboxOutcome.STALE}
    popen.assert_not_called()


def test_backend_reuses_workspace_executable_read_authority(monkeypatch, tmp_path):
    """An already readable workspace executable needs no separate exact-file grant."""
    executable = tmp_path / "tool"
    executable.write_text("#!/bin/sh\nprintf ok")
    executable.chmod(0o700)
    details = executable.stat()
    value = request(
        tmp_path,
        environment={"PATH": str(tmp_path)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    assert MacOSSeatbeltBackend().run(value).outcome is SandboxOutcome.COMPLETED
    assert f'(subpath "{tmp_path}")' in popen.call_args.args[0][2]


@pytest.mark.parametrize("changed", ["file_mode", "root_mode", "hardlink"])
def test_backend_revalidates_exact_executables_after_diagnostic_start(
    monkeypatch, tmp_path, changed
):
    """Metadata changes during diagnostic setup refuse a previously rendered exact grant."""
    workspace, tools = tmp_path / "workspace", tmp_path / "tools"
    workspace.mkdir()
    tools.mkdir(mode=0o700)
    executable = tools / "tool"
    executable.write_text("executable")
    executable.chmod(0o700)
    details = executable.stat()
    value = request(
        workspace,
        environment={"PATH": str(tools)},
        executable_identities=(("tool", executable, executable, details.st_dev, details.st_ino),),
    )
    monitor = MagicMock()

    def register(tag):
        """Change only the owned executable or its root after policy construction."""
        if changed == "file_mode":
            executable.chmod(0o777)
        elif changed == "root_mode":
            tools.chmod(0o777)
        else:
            os.link(executable, tools / "alias")

    monitor.register.side_effect = register
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda *args: monitor,
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    assert MacOSSeatbeltBackend().run(value).outcome is SandboxOutcome.INVALID
    popen.assert_not_called()
    monitor.unregister.assert_called_once()


def test_backend_protects_instruction_files_even_with_workspace_write_grant(monkeypatch, tmp_path):
    """Native write policy carves instructions and skills out of writable workspace roots."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    assert MacOSSeatbeltBackend().run(request(tmp_path)).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert "(AGENTS\\.md|SKILL\\.md)" in regex_patterns(profile)
    assert "/(.*/)?\\.agents/skills(/.*)?$" in regex_patterns(profile)
    assert "/(.*/)?\\.gitignore$" in regex_patterns(profile)
    assert "/(.*/)?\\.agentignore$" in regex_patterns(profile)
    assert profile.index("(deny file-write* ") > profile.index("(allow file-write* ")


def test_backend_protects_configured_instruction_name_at_every_depth(monkeypatch, tmp_path):
    """A custom instruction basename is anchored and escaped in the native deny policy."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    bound = request(tmp_path, protected_instruction_names=("POLICY+.md",))
    assert MacOSSeatbeltBackend().run(bound).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert r"AGENTS\.md|SKILL\.md|POLICY\+\.md" in regex_patterns(profile)
    assert f"{re.escape(str(tmp_path))}/(.*/)?(" in regex_patterns(profile)
    assert r"POLICY\+\.md)$" in regex_patterns(profile)


def test_backend_denies_private_reads_inside_every_granted_root(monkeypatch, tmp_path):
    """Every host-data grant retains private denials while fresh command scratch is exempt."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    writable = tmp_path / "writable"
    writable.mkdir()
    target = tmp_path / "alias-target"
    target.mkdir()
    mappings = tmp_path / "mappings"
    mappings.mkdir(mode=0o700)
    physical_alias = mappings / "r0"
    physical_alias.symlink_to(target)
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(
        request(
            workspace,
            read_roots=(extra, target),
            write_roots=(workspace, writable),
            read_aliases=(physical_alias,),
        )
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert f'(subpath "{workspace}")' in profile
    assert f'(subpath "{extra}")' in profile
    names = r"(\.[sS][sS][hH]|\.[aA][wW][sS]|\.[gG][nN][uU][pP][gG]|"
    names += r"\.[cC][oO][nN][fF][iI][gG]|\.[lL][oO][oO][pP]|[lL][iI][bB][rR][aA][rR][yY])"
    scratch = Path(popen.call_args.kwargs["env"]["TMPDIR"])
    for root in (workspace, extra, writable, target):
        assert f"{re.escape(str(root))}/(.*/)?{names}(/.*)?$" in regex_patterns(profile)
    assert f"{re.escape(str(scratch))}/(.*/)?{names}(/.*)?$" not in regex_patterns(profile)
    assert profile.index("(deny file-read-data") > profile.index("(allow file-read*")
    assert profile.index("(deny file-write* ") > profile.index("(allow file-write* ")


def test_backend_guards_protected_entries_and_their_ancestors_against_rename(monkeypatch, tmp_path):
    """Protected entries cannot be moved out of their pathname policy by renaming a parent."""
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("instructions")
    private_parent = tmp_path / "private-parent"
    private = private_parent / ".ssh"
    private.mkdir(parents=True)
    (private / "id_ed25519").write_text("canary")
    git_parent = tmp_path / "git-parent"
    git = git_parent / ".git"
    git.mkdir(parents=True)
    (git / "config").write_text("canary")
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    rename_rules = [
        line for line in profile.splitlines() if line.startswith("(deny file-write-unlink ")
    ]
    private_rules = [line for line in profile.splitlines() if line.startswith("(deny file-write* ")]
    assert any(r"(\.[sS][sS][hH]|\.[aA][wW][sS]" in regex_patterns(line) for line in private_rules)
    for protected in (nested, private_parent, private, git_parent, git):
        assert any(f'(literal "{protected}")' in line for line in rename_rules)
    assert all(f'(literal "{ordinary}")' not in line for line in rename_rules)


def test_backend_limits_baseline_reads_to_system_managed_roots(monkeypatch, tmp_path):
    """Writable installed-tool trees cannot receive recursive default reads."""
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())

    backend = MacOSSeatbeltBackend()
    assert backend.automatic_tool_read_roots(tmp_path, (), ()) == ()
    assert backend.policy_version == "macos-seatbelt-v2"
    assert SandboxCapability.LOCAL_NETWORK in backend.capabilities
    assert backend.scratch_prefix == "loop-seatbelt-"
    assert backend.trusted_file_owner_ids == tuple(dict.fromkeys((0, os.getuid())))
    result = backend.run(request(tmp_path))

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    names = r"(\.[sS][sS][hH]|\.[aA][wW][sS]|\.[gG][nN][uU][pP][gG]|"
    names += r"\.[cC][oO][nN][fF][iI][gG]|\.[lL][oO][oO][pP]|[lL][iI][bB][rR][aA][rR][yY])"
    for root in backend.system_read_roots:
        assert f"{re.escape(str(root))}/(.*/)?{names}(/.*)?$" in regex_patterns(profile)
    assert '(subpath "/System/Library")' in profile
    assert '(subpath "/System")' not in profile
    for root in ("/usr", "/usr/local", "/Library", "/opt/homebrew"):
        assert f'(subpath "{root}")' not in profile


def test_backend_auto_trusts_only_receipted_managed_package_roots(monkeypatch, tmp_path):
    """Homebrew auto-read eligibility belongs to the native platform adapter."""
    cellar = tmp_path / "Cellar"
    package = cellar / "sample" / "1.0"
    package.mkdir(parents=True)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend._MANAGED_TOOLCHAIN_PREFIXES", (cellar,)
    )

    assert not MacOSSeatbeltBackend.managed_tool_root(package)
    (package / "INSTALL_RECEIPT.json").write_text("{}", encoding="utf-8")
    assert MacOSSeatbeltBackend.managed_tool_root(package)
    assert not MacOSSeatbeltBackend.managed_tool_root(tmp_path)


def test_custom_package_receipt_is_candidate_only(tmp_path):
    """A nondefault prefix can suggest a package without becoming automatic authority."""
    package = tmp_path / "custom-prefix/Cellar/sample/1"
    package.mkdir(parents=True)
    (package / "INSTALL_RECEIPT.json").write_text("{}")
    assert MacOSSeatbeltBackend.installed_package_root(package)
    assert not MacOSSeatbeltBackend.managed_tool_root(package)


def test_installed_package_root_rejects_private_names(tmp_path):
    """Package receipts do not qualify roots containing a protected component."""
    assert not MacOSSeatbeltBackend.installed_package_root(tmp_path / ".ssh")


@pytest.mark.parametrize("target_root", [False, True])
def test_package_alias_must_contain_the_selected_executable(tmp_path, target_root):
    """Unrelated or root-wide aliases cannot identify an installed package."""
    opt = tmp_path / "prefix/opt"
    opt.mkdir(parents=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    alias = opt / "runtime"
    alias.symlink_to(Path("/") if target_root else unrelated, target_is_directory=True)
    selected = tmp_path / "selected/bin/engine"
    assert macos_backend._installed_package_layout(selected, alias / "bin/engine") is None


@pytest.mark.parametrize("ancestor", ["workspace", "home"])
def test_runtime_alias_never_inspects_a_workspace_or_home_ancestor(monkeypatch, tmp_path, ancestor):
    """A forged opt alias cannot turn a broad user directory into a package candidate."""
    broad = tmp_path / "store/parent"
    (broad / "bin").mkdir(parents=True)
    selected = broad / "bin/engine"
    selected.write_text("fixture")
    (broad / "INSTALL_RECEIPT.json").write_text("{}")
    workspace = broad / "project" if ancestor == "workspace" else tmp_path / "project"
    workspace.mkdir()
    home = broad if ancestor == "home" else tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: home)
    opt = tmp_path / "prefix/opt"
    opt.mkdir(parents=True)
    alias = opt / "runtime"
    alias.symlink_to(broad, target_is_directory=True)
    opener = MagicMock(side_effect=AssertionError("broad package metadata must not be read"))
    monkeypatch.setattr(macos_backend, "open_selected_file", opener)

    assert (
        MacOSSeatbeltBackend.installed_runtime_roots(
            selected, spelling=alias / "bin/engine", workspace=workspace
        )
        is None
    )
    opener.assert_not_called()


@pytest.mark.parametrize("ancestor", ["workspace", "home"])
def test_runtime_dependency_never_selects_a_workspace_or_home_ancestor(
    monkeypatch, tmp_path, ancestor
):
    """A dependency alias cannot suggest a broad directory containing user data."""
    store = tmp_path / "store"
    runtime = store / "runtime/1"
    dependency = store / "support/2"
    (runtime / "bin").mkdir(parents=True)
    dependency.mkdir(parents=True)
    selected = runtime / "bin/engine"
    selected.write_text("fixture")
    (runtime / "INSTALL_RECEIPT.json").write_text(
        '{"runtime_dependencies": [{"full_name": "support"}]}'
    )
    opt = tmp_path / "prefix/opt"
    opt.mkdir(parents=True)
    (opt / "runtime").symlink_to(runtime, target_is_directory=True)
    (opt / "support").symlink_to(dependency, target_is_directory=True)
    workspace = dependency / "project" if ancestor == "workspace" else tmp_path / "project"
    workspace.mkdir()
    home = dependency if ancestor == "home" else tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(macos_backend, "_trusted_home", lambda: home)

    assert (
        MacOSSeatbeltBackend.installed_runtime_roots(
            selected, spelling=opt / "runtime/bin/engine", workspace=workspace
        )
        is None
    )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_installed_package_receipt_must_be_a_single_link_regular_file(monkeypatch, tmp_path, kind):
    """Package discovery never reads or trusts a receipt linked to another file."""
    cellar = tmp_path / "Cellar"
    package = cellar / "sample/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/tool"
    executable.write_text("fixture")
    outside = tmp_path / "secret"
    outside.write_text('{"runtime_dependencies": []}')
    receipt = package / "INSTALL_RECEIPT.json"
    if kind == "symlink":
        receipt.symlink_to(outside)
    else:
        os.link(outside, receipt)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend._MANAGED_TOOLCHAIN_PREFIXES", (cellar,)
    )
    opener = MagicMock(side_effect=AssertionError("linked receipt must not be opened"))
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.open_selected_file", opener)

    assert not MacOSSeatbeltBackend.managed_tool_root(package)
    assert MacOSSeatbeltBackend.installed_runtime_roots(executable) is None
    opener.assert_not_called()


def test_installed_package_rechecks_receipt_link_count_after_open(monkeypatch, tmp_path):
    """A receipt hardlinked while it is being opened is never read."""
    package = tmp_path / "Cellar/sample/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/tool"
    executable.write_text("fixture")
    receipt = package / "INSTALL_RECEIPT.json"
    receipt.write_text('{"runtime_dependencies": []}')
    original_open = macos_backend.open_selected_file
    original_fstat = os.fstat
    opened = []

    def selected_open(root, path, identity):
        """Remember the descriptor after the initial identity check."""
        descriptor = original_open(root, path, identity)
        opened.append(descriptor)
        return descriptor

    def changed_fstat(descriptor):
        """Represent an added hard link after the descriptor was opened."""
        details = original_fstat(descriptor)
        if descriptor in opened:
            return SimpleNamespace(st_mode=details.st_mode, st_nlink=2)
        return details

    monkeypatch.setattr(macos_backend, "open_selected_file", selected_open)
    monkeypatch.setattr(macos_backend.os, "fstat", changed_fstat)
    assert MacOSSeatbeltBackend.installed_runtime_roots(executable) is None


def test_installed_package_rejects_receipt_changed_while_reading(monkeypatch, tmp_path):
    """A receipt changed during its bounded read cannot suggest dependencies."""
    package = tmp_path / "Cellar/sample/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/tool"
    executable.write_text("fixture")
    receipt = package / "INSTALL_RECEIPT.json"
    receipt.write_text('{"runtime_dependencies": []}')
    original = os.read

    def read_then_change(descriptor, size):
        """Change the owned fixture after reading its original bytes."""
        content = original(descriptor, size)
        receipt.write_text('{"runtime_dependencies": [], "changed": true}')
        return content

    monkeypatch.setattr(macos_backend.os, "read", read_then_change)
    assert MacOSSeatbeltBackend.installed_runtime_roots(executable) is None


def test_installed_package_never_inspects_a_protected_executable(tmp_path):
    """Private executables cannot trigger package discovery."""
    assert MacOSSeatbeltBackend.installed_runtime_roots(tmp_path / ".ssh/tool") is None


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"\xff",
        b"[]",
        b'{"runtime_dependencies": null}',
        b'{"runtime_dependencies": [null]}',
        b'{"runtime_dependencies": [{"full_name": "../private"}]}',
        b'{"runtime_dependencies": [{"full_name": "missing"}]}',
        json.dumps({"runtime_dependencies": [{}] * 33}).encode(),
        json.dumps({"runtime_dependencies": [{"full_name": "a" * 5000}]}).encode(),
        b"x" * (128 * 1024 + 1),
    ],
    ids=[
        "invalid-json",
        "invalid-utf8",
        "wrong-document",
        "wrong-dependencies",
        "wrong-dependency",
        "unsafe-name",
        "missing-alias",
        "too-many",
        "name-too-long",
        "too-large",
    ],
)
def test_installed_package_rejects_invalid_dependency_metadata(tmp_path, content):
    """Malformed, oversized, and unresolved metadata never grant dependency roots."""
    package = tmp_path / "Cellar/sample/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/tool"
    executable.write_text("fixture")
    (package / "INSTALL_RECEIPT.json").write_bytes(content)

    assert MacOSSeatbeltBackend.installed_runtime_roots(executable) is None


def test_installed_package_rejects_dependency_alias_outside_its_cellar(tmp_path):
    """Dependency metadata cannot authorize an alias to an unrelated directory."""
    package = tmp_path / "Cellar/sample/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/tool"
    executable.write_text("fixture")
    (package / "INSTALL_RECEIPT.json").write_text(
        '{"runtime_dependencies": [{"full_name": "other"}]}'
    )
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt/other").symlink_to(tmp_path, target_is_directory=True)
    assert MacOSSeatbeltBackend.installed_runtime_roots(executable) is None


@pytest.mark.parametrize("private", [False, True])
def test_trusted_argv_search_passes_only_selected_descriptors_under_read_only_policy(
    monkeypatch, tmp_path, private
):
    """A selected helper launches through Seatbelt without shell, network, or workspace writes."""
    if private:
        temporary = tmp_path / "Library" / "temporary"
        temporary.mkdir(mode=0o700, parents=True)
        safe = tmp_path / "safe"
        safe.mkdir(mode=0o700)
        monkeypatch.setattr(tempfile, "tempdir", str(temporary))
        monkeypatch.setattr("loop.execution.sandbox.macos.backend._PRIVATE_TEMP_ROOT", safe)
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    popen = MagicMock()
    capture = ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=0, stdout="matched")
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.supervise_process", lambda *_args, **_kwargs: capture
    )
    backend = MacOSSeatbeltBackend()

    result = backend.run_read_only_argv(
        executable, ("--json", "needle", "/dev/fd/3"), (3,), time.monotonic() + 5
    )

    assert result is capture
    argv = popen.call_args.args[0]
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[-3:] == ["--json", "needle", "/dev/fd/3"]
    assert popen.call_args.kwargs["pass_fds"] == (3,)
    assert "(deny default" in argv[2]
    assert "(allow network*)" not in argv[2]
    if private:
        assert Path(popen.call_args.kwargs["cwd"]).parent == safe


def test_trusted_argv_search_binds_verified_package_dependencies(monkeypatch, tmp_path):
    """Qualified helper runtime libraries remain readable without exposing selected source roots."""
    cellar = tmp_path / "Cellar"
    package = cellar / "ripgrep/1"
    dependency = cellar / "pcre2/2"
    (package / "bin").mkdir(parents=True)
    (dependency / "lib").mkdir(parents=True)
    (package / "INSTALL_RECEIPT.json").write_text(
        json.dumps({"runtime_dependencies": [{"full_name": "pcre2"}]})
    )
    (dependency / "INSTALL_RECEIPT.json").write_text("{}")
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt/pcre2").symlink_to(dependency)
    executable = package / "bin/rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend._MANAGED_TOOLCHAIN_PREFIXES", (cellar,)
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    capture = ProcessCapture(ProcessCaptureStatus.COMPLETED, exit_code=0, stdout="matched")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.supervise_process", lambda *args, **kwargs: capture
    )
    assert (
        MacOSSeatbeltBackend().run_read_only_argv(executable, (), (), time.monotonic() + 5)
        is capture
    )
    profile = popen.call_args.args[0][2]
    assert f'(subpath "{dependency}")' in profile
    assert f'(subpath "{package}")' in profile
    assert f'(subpath "{tmp_path}")' not in profile


def test_trusted_argv_search_refuses_invalid_package_metadata(monkeypatch, tmp_path):
    """Invalid package metadata cannot reach a native search launch."""
    package = tmp_path / "Cellar/search/1"
    (package / "bin").mkdir(parents=True)
    executable = package / "bin/rg"
    executable.write_text("fixture")
    (package / "INSTALL_RECEIPT.json").write_text("not json")
    launch = MagicMock(side_effect=AssertionError("invalid metadata must not launch"))
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    assert (
        MacOSSeatbeltBackend().run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    )
    launch.assert_not_called()


def test_trusted_argv_search_fails_closed_on_expiry_tamper_and_copy_error(monkeypatch, tmp_path):
    """An expired request or unsafe copy cannot reach a native child launch."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    backend = MacOSSeatbeltBackend()

    with monkeypatch.context() as scoped:
        scoped.setattr("loop.execution.sandbox.macos.backend.platform.system", lambda: "Linux")
        assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None

    assert (
        backend.run_read_only_argv(executable, (), (), time.monotonic() - 1).status
        is ProcessCaptureStatus.TIMED_OUT
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(SandboxRequest, "paths_are_current", lambda _self: False)
        assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    original_open = Path.open

    def denied_open(path, *args, **kwargs):
        """Fail only the copied helper destination before a native launch."""
        if path.name == "helper":
            raise OSError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    popen.assert_not_called()


def test_trusted_argv_search_stops_after_slow_helper_copy(monkeypatch, tmp_path):
    """A short fixed-argv deadline prevents launch after helper preparation runs long."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    clock = MagicMock(return_value=time.monotonic())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time.monotonic", clock)
    original_open = os.open

    def slow_open(path, *args, **kwargs):
        """Delay the copied helper input in a disposable fixture."""
        if path == executable:
            clock.return_value += 0.04
        return original_open(path, *args, **kwargs)

    launch = MagicMock()
    monkeypatch.setattr(os, "open", slow_open)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    started = clock.return_value
    result = MacOSSeatbeltBackend().run_read_only_argv(executable, (), (), started + 0.01)
    assert result.status is ProcessCaptureStatus.TIMED_OUT
    assert clock.return_value == pytest.approx(started + 0.04)
    launch.assert_not_called()


def test_trusted_argv_search_rejects_symlink_and_hardlink_helpers(monkeypatch, tmp_path):
    """Fixed-argv preparation never copies an aliased executable into private state."""
    executable = tmp_path / "rg"
    executable.write_bytes(b"verified helper")
    executable.chmod(0o755)
    symlink = tmp_path / "symlink-rg"
    symlink.symlink_to(executable)
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    backend = MacOSSeatbeltBackend()
    assert backend.run_read_only_argv(symlink, (), (), time.monotonic() + 5) is None
    assert backend.run_read_only_argv(tmp_path, (), (), time.monotonic() + 5) is None
    hardlink = tmp_path / "hardlink-rg"
    hardlink.hardlink_to(executable)
    assert backend.run_read_only_argv(executable, (), (), time.monotonic() + 5) is None
    launch.assert_not_called()


def test_backend_deadline_during_granted_tree_walk_prevents_launch(monkeypatch, tmp_path):
    """A slow granted-tree scan times out before any native command starts."""
    for index in range(10):
        (tmp_path / f"file-{index}").write_text("ordinary", encoding="utf-8")
    clock = MagicMock(return_value=time.monotonic())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.time.monotonic", clock)
    original_scandir = os.scandir

    def slow_scandir(path):
        """Delay entry enumeration only for the granted workspace."""
        if not isinstance(path, int) and Path(path) == tmp_path:
            clock.return_value += 0.04
        return original_scandir(path)

    launch = MagicMock()
    monkeypatch.setattr(os, "scandir", slow_scandir)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    started = clock.return_value
    result = MacOSSeatbeltBackend().run(request(tmp_path, deadline=started + 0.01))
    assert result.outcome is SandboxOutcome.TIMED_OUT
    assert clock.return_value == pytest.approx(started + 0.04)
    launch.assert_not_called()


def test_backend_pipe_failure_does_not_wait_past_deadline(monkeypatch, tmp_path):
    """A malformed sandbox child remains unavailable when reaping times out."""
    process = Process(stdout=None, stderr=None, wait_error=subprocess.TimeoutExpired("sh", 0))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is SandboxOutcome.UNAVAILABLE


@pytest.mark.parametrize("path_value", [":/usr/bin", "/usr/bin:", "relative:/usr/bin"])
def test_backend_binds_empty_and_relative_search_paths(tmp_path, path_value):
    """Empty and relative PATH entries bind to cwd instead of failing closed."""
    relative = tmp_path / "relative"
    relative.mkdir()

    bound = request(tmp_path, environment={"PATH": path_value, "LANG": "C"})

    if path_value.startswith(":"):
        assert bound.path_roots[0] == (tmp_path, tmp_path)
    if path_value.endswith(":"):
        assert bound.path_roots[-1] == (tmp_path, tmp_path)
    if path_value.startswith("relative:"):
        assert bound.path_roots[0] == (relative, relative)


@pytest.mark.parametrize("search_root", ["home", "temporary", "private", "darwin_tmp"])
def test_backend_omits_private_search_root(monkeypatch, tmp_path, search_root):
    """A private PATH entry does not block safe commands or gain lookup access."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    home = tmp_path.with_name(tmp_path.name + "-home")
    private = home / ".ssh"
    private.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend._trusted_home",
        lambda: home if search_root != "temporary" else temporary / "child",
    )
    selected = {
        "home": home,
        "temporary": temporary,
        "private": private,
        "darwin_tmp": Path("/private/tmp"),
    }[search_root]
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": f"{selected}:/usr/bin", "LANG": "C"})
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        (line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")),
        "",
    )
    if search_root == "darwin_tmp":
        data = next(line for line in profile.splitlines() if line.startswith("(allow file-read* "))
        assert '(literal "/private/tmp")' not in data
    else:
        assert f'(literal "{selected}")' not in metadata
    assert popen.call_args.kwargs["env"]["PATH"] == f"{selected}:/usr/bin"


def test_backend_omits_private_search_root_through_symlink(monkeypatch, tmp_path):
    """A PATH symlink into a private root gains no lookup access."""
    alias = tmp_path / "search"
    alias.symlink_to(tmp_path.with_name(tmp_path.name + "-home"), target_is_directory=True)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )

    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": str(alias), "LANG": "C"})
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        (line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")),
        "",
    )
    assert f'(literal "{alias}")' not in metadata


@pytest.mark.parametrize("macos_environment", ["real_home"], indirect=True)
def test_backend_fails_closed_when_account_home_is_unavailable(monkeypatch, tmp_path):
    """Missing trusted account home invalidates native preparation before launch."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.pwd.getpwuid",
        MagicMock(side_effect=KeyError("no account")),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.INVALID
    assert "home directory is unavailable" in result.detail
    popen.assert_not_called()


@pytest.mark.parametrize("macos_environment", ["real_home"], indirect=True)
@pytest.mark.parametrize("home_case", ["valid", "wrong_owner", "missing"])
def test_backend_validates_account_home_without_ambient_home(monkeypatch, tmp_path, home_case):
    """Trusted account lookup accepts a real home and rejects unsafe or absent homes."""

    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    home = tmp_path.with_name(tmp_path.name + "-account")
    if home_case != "missing":
        home.mkdir()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    if home_case == "wrong_owner":
        original_stat = Path.stat

        def changed_owner(path, *args, **kwargs):
            """Return a foreign UID only for the controlled account home."""
            details = original_stat(path, *args, **kwargs)
            if path == home:
                values = list(details)
                values[4] = os.getuid() + 1
                return os.stat_result(values)
            return details

        monkeypatch.setattr(Path, "stat", changed_owner)
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(tmp_path, environment={"PATH": str(home), "LANG": "C"})
    )
    if home_case == "valid":
        assert result.outcome is SandboxOutcome.COMPLETED
        popen.assert_called_once()
    else:
        assert result.outcome is SandboxOutcome.INVALID
        assert "home directory" in result.detail
        popen.assert_not_called()


def test_backend_external_path_grants_only_metadata_even_with_hardlink(monkeypatch, tmp_path):
    """An unused search directory exposes no file data or hardlink target."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    search = tmp_path / "search"
    search.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("fake-private")
    os.link(outside, search / "alias")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(workspace, environment={"PATH": str(search), "LANG": "C"})
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    metadata = next(
        line for line in profile.splitlines() if line.startswith("(allow file-read-metadata ")
    )
    data = next(line for line in profile.splitlines() if line.startswith("(allow file-read* "))
    assert f'(literal "{search}")' in metadata
    assert str(search) not in data


@pytest.mark.parametrize("config_kind", ["missing", "regular", "symlink", "hardlink"])
def test_backend_grants_only_safe_python_environment_config(monkeypatch, tmp_path, config_kind):
    """A Python search root permits absent metadata, grants exact files and rejects aliases."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    external = tmp_path / "venv"
    search = external / "bin"
    search.mkdir(parents=True)
    config = external / "pyvenv.cfg"
    if config_kind == "symlink":
        secret = external / "secret"
        secret.write_text("private")
        config.symlink_to(secret)
    elif config_kind != "missing":
        config.write_text("home = /usr/local/bin")
        if config_kind == "hardlink":
            os.link(config, external / "alias")
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(
        request(workspace, read_roots=(search,), environment={"PATH": str(search), "LANG": "C"})
    )
    if config_kind == "missing":
        assert result.outcome is SandboxOutcome.COMPLETED
        assert str(config) not in popen.call_args.args[0][2]
    elif config_kind == "regular":
        assert result.outcome is SandboxOutcome.COMPLETED
        assert f'(literal "{config}")' in popen.call_args.args[0][2]
        assert f'(subpath "{external}")' not in popen.call_args.args[0][2]
    else:
        assert result.outcome is SandboxOutcome.INVALID
        assert "Unsafe Python environment configuration" in result.detail
        popen.assert_not_called()


@pytest.mark.parametrize("config_kind", ["missing", "regular", "symlink"])
@pytest.mark.parametrize("include_search_root", [False, True])
def test_backend_skips_redundant_environment_metadata_inside_workspace(
    monkeypatch, tmp_path, config_kind, include_search_root
):
    """Existing workspace authority needs no extra metadata grant or unrelated validation."""
    workspace = tmp_path / "work"
    search = workspace / "bin"
    search.mkdir(parents=True)
    config = workspace / "pyvenv.cfg"
    if config_kind == "regular":
        config.write_text("unrelated metadata", encoding="utf-8")
    elif config_kind == "symlink":
        config.symlink_to(workspace / "unrelated")
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(
        request(
            workspace,
            read_roots=(search,) if include_search_root else (),
            environment={"PATH": str(search), "LANG": "C"},
        )
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    assert f'(subpath "{workspace}")' in profile
    assert f'(literal "{config}")' not in profile


def test_backend_records_tagged_kernel_denial_without_losing_shell_exit(monkeypatch, tmp_path):
    """A tagged child file denial remains distinct from the shell exit status."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")
    result, popen = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(records, running=True),
        output="/outside/secret: Permission denied",
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1
    assert result.stderr == "/outside/secret: Permission denied"
    assert "/outside/secret" in result.observed_denial
    assert result.diagnostic_status == "available"
    assert result.diagnostic_source == "macos-kernel-log"
    assert popen.call_args_list[1].args[0][0] == "/usr/bin/log"


def test_backend_preserves_nonstandard_exit_after_real_denial(monkeypatch, tmp_path):
    """A handled kernel denial and matching stderr cannot override exit code seven."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")
    result, _ = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(records),
        output="/outside/secret: Operation not permitted",
        returncode=7,
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 7
    assert "/outside/secret" in result.observed_denial


def test_backend_reports_live_denial_provenance(monkeypatch, tmp_path):
    """A live tagged record remains available when the bounded snapshot is empty."""
    monitor = MagicMock()
    monitor.denial.return_value = "Sandbox: cat(123) deny(1) file-read-data /outside/secret"
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda self, deadline: monitor,
    )
    snapshot = MagicMock(return_value=None)
    monkeypatch.setattr(macos_backend, "_read_denial_from_log", snapshot)

    result, popen = run_with_log(monkeypatch, tmp_path, None)

    assert result.observed_denial == monitor.denial.return_value
    assert result.diagnostic_status == "available"
    assert result.diagnostic_source == "macos-kernel-log"
    assert popen.call_count == 1
    snapshot.assert_called_once()


def test_backend_checks_buffered_denials_without_waiting_after_success(monkeypatch, tmp_path):
    """A successful command exposes buffered evidence without a snapshot or replay."""
    denial = "Sandbox: cat(1) deny(1) file-read-data /outside/secret"
    monitor = MagicMock()
    monitor.denial.side_effect = [None, denial]
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", lambda *_: monitor)
    snapshot = MagicMock(side_effect=AssertionError("success must not wait for a snapshot"))
    monkeypatch.setattr(macos_backend, "_read_denial_from_log", snapshot)
    monkeypatch.setattr(macos_backend, "_denial_from_log", snapshot)

    result, popen = run_with_log(monkeypatch, tmp_path, None, output="", returncode=0)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 0
    assert result.observed_denial == denial
    assert result.diagnostic_status == "available"
    assert monitor.denial.call_args.args[1] <= time.monotonic()
    snapshot.assert_not_called()
    assert popen.call_count == 1


def test_backend_success_with_no_buffered_denials_remains_unknown(monkeypatch, tmp_path):
    """A delayed or missing live record is not treated as proof of no denial."""
    monitor = MagicMock()
    monitor.denial.return_value = None
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", lambda *_: monitor)
    monkeypatch.setattr(
        macos_backend, "_denial_from_log", MagicMock(side_effect=AssertionError("unexpected wait"))
    )

    result, popen = run_with_log(monkeypatch, tmp_path, None, output="", returncode=0)

    assert result.exit_code == 0
    assert result.observed_denial == ""
    assert result.diagnostic_status == "unknown"
    assert popen.call_count == 1


def test_backend_merges_live_and_delayed_candidates(monkeypatch, tmp_path):
    """A delayed denied read remains visible beside an earlier same-priority observation."""
    startup = "Sandbox: true(1) deny(1) file-read-data /system/cache"
    live = "Sandbox: sh(2) deny(1) file-read-data /system/cache"
    later = "Sandbox: cat(3) deny(1) file-read-data /outside/secret"
    tag = "LOOP_SBX_" + "b" * 32
    monitor = MagicMock()
    monitor.denial.side_effect = [startup, live]
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", lambda *_: monitor)

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(kernel_record(f"{later}\n{tag}")))

    assert result.outcome is SandboxOutcome.COMPLETED
    candidates, _ = MacOSSeatbeltBackend().classify_denial_candidates(result.observed_denial)
    assert candidates == ((later, False), (live, False))


@pytest.mark.parametrize(
    "read",
    [
        "Sandbox: tool(1) deny(1) file-read-data /outside/ network-decoy",
        "Sandbox: deny(1) network-(1) deny(1) file-read-data /outside/input",
    ],
)
def test_denial_ranking_uses_the_operation_not_resource_text(monkeypatch, tmp_path, read):
    """A crafted filename or process name cannot inflate candidate priority."""
    write = "Sandbox: tool(2) deny(1) file-write-data /outside/output"
    monitor = MagicMock()
    monitor.denial.side_effect = [None, read]
    monkeypatch.setattr(macos_backend.SeatbeltDiagnosticService, "monitor", lambda *_: monitor)
    monkeypatch.setattr(macos_backend, "_read_denial_from_log", lambda *_: write)

    result, _ = run_with_log(monkeypatch, tmp_path, None)

    candidates, _ = MacOSSeatbeltBackend().classify_denial_candidates(result.observed_denial)
    assert candidates == ((write, False), (read, False))


def test_backend_bounds_denial_log_cleanup_without_masking_shell_result(monkeypatch, tmp_path):
    """A log child that ignores termination cannot extend command cleanup indefinitely."""
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._DENIAL_LOG_ATTEMPTS", 1)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._DENIAL_LOG_WAIT_SECONDS", 0.005)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend._DENIAL_STREAM_SHUTDOWN_SECONDS", 0.01
    )
    log_process = LogProcess(
        running=True,
        wait_error=subprocess.TimeoutExpired("/usr/bin/log", 0.01),
        stuck_after_kill=True,
    )
    try:
        result, _ = run_with_log(
            monkeypatch,
            tmp_path,
            log_process,
            output="Permission denied",
            returncode=23,
        )
    finally:
        log_process.finish()

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 23
    assert result.observed_denial == ""
    assert log_process.killed
    assert log_process.wait_timeouts == [0.01]
    assert log_process.stdout.closed


def test_backend_keeps_shell_result_when_log_poll_and_pipe_close_fail(monkeypatch, tmp_path):
    """Diagnostic cleanup errors cannot replace a completed shell result."""
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._DENIAL_LOG_ATTEMPTS", 1)
    process = PollErrorLogProcess()

    result, _ = run_with_log(
        monkeypatch,
        tmp_path,
        process,
        output="Permission denied",
        returncode=19,
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 19
    assert result.observed_denial == ""
    assert process.kill_attempts == 1
    assert process.stdout.closed
    assert process.wait_timeouts == [1.0]


def test_backend_attributes_denial_after_kernel_log_delay(monkeypatch, tmp_path):
    """A denial arriving after the first snapshot still produces a host offer outcome."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")

    result, popen = run_with_log(
        monkeypatch,
        tmp_path,
        LogProcess(),
        later_records=records,
        output="/outside/secret: Permission denied",
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert "/outside/secret" in result.observed_denial
    assert result.diagnostic_status == "available"
    assert result.diagnostic_source == "macos-kernel-log"
    assert popen.call_count == 3


def test_backend_prioritizes_network_denial_over_incidental_file_read(monkeypatch, tmp_path):
    """A later network denial supplies the actionable diagnostic for a child."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) file-read-data /venv/pyvenv.cfg\n{tag}")
    records += kernel_record(f"Sandbox: Python(123) deny(1) network-outbound remote:*:443\n{tag}")
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="socket.connect: Permission denied"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert "network-outbound" in result.observed_denial


def test_backend_does_not_attribute_unrelated_network_permission_text(monkeypatch, tmp_path):
    """A generic permission phrase does not tie a network denial to command failure."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) network-outbound remote:*:443\n{tag}")

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(records))

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "output", ["", "OSError: [Errno 9] Bad file descriptor", "Permission denied"]
)
def test_backend_preserves_nonzero_after_incidental_read_denial(monkeypatch, tmp_path, output):
    """A Python startup probe cannot turn an unrelated exit into a host offer."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: Python(123) deny(1) file-read-data /venv/pyvenv.cfg\n{tag}")

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(records), output=output)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_attributes_a_denied_symlink_target(monkeypatch, tmp_path):
    """A reported alias path can identify the kernel's denied canonical target."""
    target = tmp_path / "secret"
    target.write_text("private")
    (tmp_path / "alias").symlink_to(target)
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data {target}\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="alias: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert str(target) in result.observed_denial


def test_backend_rejects_relative_denial_path(monkeypatch, tmp_path):
    """A kernel record without an absolute denied path cannot justify host retry."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data relative\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="relative: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_ignores_unresolvable_stderr_path(monkeypatch, tmp_path):
    """A malformed child path cannot turn an ordinary exit into a host offer."""
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        """Simulate a candidate path that fails canonicalization."""
        if path.name == "loop":
            raise RuntimeError("symlink loop")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/secret\n{tag}")

    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="loop: Permission denied"
    )

    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_prefers_write_over_read_and_keeps_higher_priority(monkeypatch, tmp_path):
    """Later lower-priority log records do not replace an actionable write denial."""
    tag = "LOOP_SBX_" + "b" * 32
    records = kernel_record(f"Sandbox: bash(123) deny(1) file-write-data /outside/write\n{tag}")
    records += kernel_record(f"Sandbox: cat(123) deny(1) file-read-data /outside/write\n{tag}")
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="/outside/write: Permission denied"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert "/outside/write" in result.observed_denial


def test_backend_skips_log_inspection_after_deadline(monkeypatch, tmp_path):
    """A spent request deadline cannot start a fresh log query."""
    result, popen = run_with_log(monkeypatch, tmp_path, None, expire_after_wait=True)
    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_count == 1


def test_backend_bounds_log_bytes(monkeypatch, tmp_path):
    """Oversized log output cannot grow diagnostic memory without limit."""
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._LOG_LIMIT", 8)
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(b"long record without newline"))
    assert result.outcome is SandboxOutcome.COMPLETED


def test_backend_handles_log_query_without_readable_output(monkeypatch, tmp_path):
    """An idle query returns an ordinary nonzero result on its bounded timeout."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.select.select", lambda *args: ([], [], [])
    )
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess())
    assert result.outcome is SandboxOutcome.COMPLETED


def test_backend_preserves_result_on_log_read_error(monkeypatch, tmp_path):
    """A diagnostic read failure cannot turn an ordinary exit into a denial."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.select.select",
        MagicMock(side_effect=OSError("closed")),
    )
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess())
    assert result.outcome is SandboxOutcome.COMPLETED


@pytest.mark.parametrize(
    "records",
    [
        b"not json\n",
        b"42\n",
        b'{"processID":0,"processImagePath":"/kernel","senderImagePath":null}\n',
        (
            b'{"processID":0,"processImagePath":"/kernel",'
            b'"senderImagePath":"/Sandbox.kext/Sandbox","eventMessage":42}\n'
        ),
        kernel_record("Sandbox: cat(123) deny(1) file-read-data /outside\nWRONG"),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            process_id=123,
        ),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            image="/usr/bin/logger",
        ),
        kernel_record(
            "Sandbox: cat(123) deny(1) file-read-data /outside\nLOOP_SBX_" + "b" * 32,
            sender="/usr/bin/logger",
        ),
        kernel_record("Sandbox: cat(123) allow file-read-data /outside\nLOOP_SBX_" + "b" * 32),
        kernel_record("Sandbox: cat(123) deny(1) file-read-data /outside"),
    ],
)
def test_backend_does_not_infer_denial_from_untrusted_or_irrelevant_logs(
    monkeypatch, tmp_path, records
):
    """Only tagged kernel records for relevant OS operations classify a failure."""
    result, _ = run_with_log(
        monkeypatch, tmp_path, LogProcess(records), output="Operation not permitted"
    )
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1
    assert not result.observed_denial
    assert result.diagnostic_status == "unknown"
    assert result.diagnostic_source == "unknown"


@pytest.mark.parametrize(
    "operation, resource",
    [
        ("file-write-data", "/dev/tty"),
        ("file-read-data", "/Library/Preferences/Logging/example"),
        ("network-outbound", "/private/var/run/syslog"),
        ("sysctl-read", "kern.bootargs"),
        ("file-read-metadata", "/tmp"),
        ("process-exec", "/outside/tool"),
    ],
)
def test_backend_retains_tagged_denials_without_path_or_operation_allowlists(
    monkeypatch, tmp_path, operation, resource
):
    """Authenticated denials remain candidates regardless of host path or operation."""
    tag = "LOOP_SBX_" + "b" * 32
    denial = f"Sandbox: tool(123) deny(1) {operation} {resource}"
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(kernel_record(f"{denial}\n{tag}")))
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.observed_denial == denial
    assert result.diagnostic_source == "macos-kernel-log"


def test_backend_preserves_nonzero_when_log_query_fails(monkeypatch, tmp_path):
    """A broken log reader cannot invent a sandbox denial or host retry."""
    result, _ = run_with_log(monkeypatch, tmp_path, None, log_error=OSError("log unavailable"))
    assert result.outcome is SandboxOutcome.COMPLETED
    assert result.exit_code == 1


def test_backend_preserves_nonzero_when_log_pipe_is_missing(monkeypatch, tmp_path):
    """An unavailable log stream leaves an ordinary command failure intact."""
    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess(stdout=False))
    assert result.outcome is SandboxOutcome.COMPLETED


def test_backend_rejects_bad_profile_before_script(monkeypatch, tmp_path):
    """A launcher validation error is typed unavailable and starts no command."""
    probe = MagicMock(returncode=65, stderr="bad profile")
    popen = MagicMock()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert "exit 65" in result.detail
    popen.assert_not_called()


def test_backend_bounds_optional_probes_and_launches_every_profile(monkeypatch, tmp_path):
    """A missing observation gets bounded retries while every command still launches."""
    probe = MagicMock(returncode=0, stderr="")
    run = MagicMock(return_value=probe)
    launch = MagicMock(side_effect=[Process() for _ in range(4)])
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.run", run)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    approved = request(tmp_path)
    backend = MacOSSeatbeltBackend()

    for _ in range(4):
        assert backend.run(approved).outcome is SandboxOutcome.COMPLETED

    assert run.call_count == 3
    assert launch.call_count == 4


@pytest.mark.parametrize("failure", [PermissionError("denied"), subprocess.TimeoutExpired([], 1)])
def test_backend_classifies_preparation_and_probe_errors(monkeypatch, tmp_path, failure):
    """Preparation errors remain unavailable and probe expiry remains a timeout."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run", MagicMock(side_effect=failure)
    )
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is (
        SandboxOutcome.TIMED_OUT
        if isinstance(failure, subprocess.TimeoutExpired)
        else SandboxOutcome.UNAVAILABLE
    )


def test_backend_cancels_before_child_launch(monkeypatch, tmp_path):
    """An interrupted readiness check never starts a shell child."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(side_effect=KeyboardInterrupt),
    )
    launch = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", launch)

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.CANCELLED
    launch.assert_not_called()


def test_backend_rejects_hardlinked_writable_file(tmp_path):
    """The mandatory preflight rejects preexisting hardlinks before launch."""
    source = tmp_path / "one"
    source.write_text("safe")
    os.link(source, tmp_path / "two")
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "hardlinked file" in result.detail


@pytest.mark.parametrize("root_kind", ["workspace", "extra", "read_alias"])
def test_backend_rejects_hardlinked_read_root_even_when_read_only(tmp_path, root_kind):
    """Read-only roots and virtual targets cannot expose private inodes through hardlinks."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    granted = workspace if root_kind == "workspace" else extra
    private = granted / ".ssh"
    private.mkdir()
    source = private / "id_ed25519"
    source.write_text("fake-credential")
    os.link(source, granted / "public-alias")
    mappings = tmp_path / "mappings"
    mappings.mkdir(mode=0o700)
    physical_alias = mappings / "r0"
    physical_alias.symlink_to(extra)

    result = MacOSSeatbeltBackend().run(
        request(
            workspace,
            read_roots=(extra,) if root_kind in {"extra", "read_alias"} else (),
            write_roots=(),
            read_aliases=(physical_alias,) if root_kind == "read_alias" else (),
        )
    )

    assert result.outcome is SandboxOutcome.INVALID
    assert "hardlinked file" in result.detail


@pytest.mark.parametrize("root_kind", ["workspace", "extra", "read_alias", "write"])
@pytest.mark.parametrize("name", [".aws", ".AWS"])
def test_backend_rejects_protected_directory_as_read_root(tmp_path, root_kind, name):
    """A protected directory selected as a grant root never becomes readable."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = (workspace if root_kind == "workspace" else tmp_path) / name
    private.mkdir()
    mappings = tmp_path / "mappings"
    mappings.mkdir(mode=0o700)
    physical_alias = mappings / "r0"
    physical_alias.symlink_to(private)
    result = MacOSSeatbeltBackend().run(
        request(
            private if root_kind == "workspace" else workspace,
            read_roots=(private,) if root_kind in {"extra", "read_alias"} else (),
            write_roots=(private,) if root_kind == "write" else (),
            read_aliases=(physical_alias,) if root_kind == "read_alias" else (),
        )
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert "Protected directory cannot be a read root" in result.detail


def test_backend_rejects_read_alias_to_protected_descendant(tmp_path):
    """An alias cannot promote a protected descendant of an approved parent into a grant."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approved = tmp_path / "approved"
    approved.mkdir()
    private = approved / ".aws"
    private.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(private)
    result = MacOSSeatbeltBackend().run(
        request(workspace, read_roots=(approved,), read_aliases=(alias,))
    )
    assert result.outcome is SandboxOutcome.INVALID
    assert "Protected directory cannot be a read root" in result.detail


def test_backend_rejects_symlink_replacement_at_preflight(monkeypatch, tmp_path):
    """A writable root replaced by a symlink is rejected by the launch preflight."""
    value = request(tmp_path)
    real = tmp_path.with_name(tmp_path.name + "-real")
    tmp_path.rename(real)
    tmp_path.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: True)
    result = MacOSSeatbeltBackend().run(value)
    assert "not a real directory" in result.detail


def test_backend_walks_nested_directories(monkeypatch, tmp_path):
    """Preflight traverses nested writable directories before launch."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "regular.txt").write_text("safe")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=65, stderr="stop")),
    )
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "exit 65" in result.detail


def test_backend_detects_root_replacement_during_walk(monkeypatch, tmp_path):
    """A writable root replaced during its walk fails before launcher validation."""
    original_scandir = os.scandir

    class ReplacingScan:
        """Replace the workspace pathname when its directory scan closes."""

        def __init__(self, path):
            self.path = Path(path) if not isinstance(path, int) else None
            self.scan = original_scandir(path)

        def __enter__(self):
            return self.scan.__enter__()

        def __exit__(self, *args):
            result = self.scan.__exit__(*args)
            if self.path == tmp_path:
                old = tmp_path.with_name(tmp_path.name + "-old")
                tmp_path.rename(old)
                tmp_path.mkdir()
            return result

    monkeypatch.setattr("loop.execution.sandbox.macos.backend.os.scandir", ReplacingScan)
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert "changed during preflight" in result.detail


@pytest.mark.parametrize("states", [(True, False), (True, True, False), (True, True, True, False)])
def test_backend_revalidates_identity_around_launcher(monkeypatch, tmp_path, states):
    """Identity replacement through final monitor setup prevents shell execution."""
    value = request(tmp_path)
    current = iter(states)
    monkeypatch.setattr(SandboxRequest, "paths_are_current", lambda self: next(current))
    probe = MagicMock(returncode=0, stderr="")
    popen = MagicMock()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    result = MacOSSeatbeltBackend().run(value)
    assert result.outcome is SandboxOutcome.STALE
    assert "path identity changed" in result.detail
    popen.assert_not_called()


def test_backend_times_out_when_capture_reader_stays_alive(monkeypatch, tmp_path):
    """A reader that misses the request deadline yields a timed-out result."""

    class StuckThread:
        """Model a pipe reader that never completes."""

        def __init__(self, **kwargs):
            pass

        def start(self):
            """Leave the simulated reader pending."""

        def join(self, timeout=None):
            """Leave the simulated reader pending."""

        def is_alive(self):
            """Report that capture remains pending."""
            return True

    process = Process()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.threading.Thread", StuckThread)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is SandboxOutcome.TIMED_OUT


@pytest.mark.parametrize(
    ("process_factory", "outcome"),
    [
        (lambda: Process(stdout=None), SandboxOutcome.UNAVAILABLE),
        (
            lambda: Process(
                stdout="",
                stderr="sandbox-exec: sandbox_apply: Operation not permitted",
                returncode=71,
                started=False,
            ),
            SandboxOutcome.UNAVAILABLE,
        ),
        (
            lambda: Process(stdout="", stderr="sandbox-exec: forged failure", returncode=71),
            SandboxOutcome.COMPLETED,
        ),
        (
            lambda: Process(stdout="", stderr="sandbox-exec: forged failure", returncode=65),
            SandboxOutcome.COMPLETED,
        ),
        (
            lambda: Process(stdout="", stderr="ordinary failure", returncode=65),
            SandboxOutcome.COMPLETED,
        ),
        (
            lambda: Process(wait_error=subprocess.TimeoutExpired(["sandbox-exec"], 1)),
            SandboxOutcome.TIMED_OUT,
        ),
        (lambda: Process(wait_error=KeyboardInterrupt()), SandboxOutcome.CANCELLED),
    ],
)
def test_backend_classifies_child_capture_failures(monkeypatch, tmp_path, process_factory, outcome):
    """Missing pipes, nested launcher failure, and timeout each fail closed."""
    process = process_factory()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend._denial_from_log", lambda *args: None)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    from loop.utils import process as process_utils

    if isinstance(process.wait_error, subprocess.TimeoutExpired):
        deadline = request(tmp_path).deadline
        monkeypatch.setattr(process_utils.os, "waitid", lambda *_args: None, raising=False)
        monkeypatch.setattr(
            process_utils,
            "time",
            SimpleNamespace(monotonic=lambda: deadline + 1, sleep=lambda _seconds: None),
        )
    elif isinstance(process.wait_error, KeyboardInterrupt):
        monkeypatch.setattr(
            process_utils.os,
            "waitid",
            MagicMock(side_effect=KeyboardInterrupt()),
            raising=False,
        )
        process.wait_error = None
    result = MacOSSeatbeltBackend().run(request(tmp_path))
    assert result.outcome is outcome
    if outcome is SandboxOutcome.UNAVAILABLE:
        assert result.possible_effects
    if isinstance(process.wait_error, subprocess.TimeoutExpired):
        assert result.stdout == "ok"


@pytest.mark.parametrize("failed_stream", ["stdout", "stderr"])
def test_backend_preserves_partial_output_when_reader_fails(monkeypatch, tmp_path, failed_stream):
    """A reader failure remains unavailable with partial output and possible effects."""
    capture = ProcessCapture(
        ProcessCaptureStatus.UNAVAILABLE,
        stdout="LOOP_SHELL_STARTED\npartial-out",
        stderr="partial-err",
        stdout_discarded=None if failed_stream == "stdout" else 0,
        stderr_discarded=None if failed_stream == "stderr" else 0,
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.supervise_process", MagicMock(return_value=capture)
    )

    result, _ = run_with_log(monkeypatch, tmp_path, LogProcess())

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects is True
    assert result.detail == "Command output capture failed."
    assert result.stdout == "partial-out" and result.stderr == "partial-err"
    assert getattr(result, f"{failed_stream}_discarded") is None


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["LOOP_", "SHELL_STARTED\nhello"], [("stderr", "warning"), ("stdout", "hello")]),
        (["LOOP_SHELL_STARTED\n", "hello"], [("stderr", "warning"), ("stdout", "hello")]),
        (["different-startup-text", "hello"], []),
    ],
)
def test_backend_live_output_requires_verified_entry_marker(monkeypatch, tmp_path, parts, expected):
    """Both live streams remain hidden until native shell entry is verified."""
    seen = []
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen",
        MagicMock(return_value=Process()),
    )

    def supervise(*_args, output_observer, **_kwargs):
        """Feed split and invalid startup observations into native capture."""
        output_observer("stderr", "warning")
        for part in parts:
            output_observer("stdout", part)
        return ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "".join(parts), "warning")

    monkeypatch.setattr("loop.execution.sandbox.macos.backend.supervise_process", supervise)

    MacOSSeatbeltBackend._execute(
        request(tmp_path), "", {}, "", None, output_observer=lambda *item: seen.append(item)
    )

    assert seen == expected


@pytest.mark.parametrize("prepared_in_time", [False, True])
def test_native_command_runtime_starts_after_bounded_preparation(
    monkeypatch, tmp_path, prepared_in_time
):
    """The native launch obeys its prep ceiling and grants runtime only after spawn."""
    approved = request(tmp_path, deadline=100, runtime_timeout=30)
    monkeypatch.setattr(macos_backend.time, "monotonic", lambda: 10 if prepared_in_time else 70)
    launch = MagicMock(return_value=Process())
    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)
    capture = MagicMock(
        return_value=ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\nok", "")
    )
    monkeypatch.setattr(macos_backend, "supervise_process", capture)

    result = MacOSSeatbeltBackend._execute(approved, "", {}, "tag", None)

    if prepared_in_time:
        assert result.outcome is SandboxOutcome.COMPLETED
        assert capture.call_args.args[1] == 40
        launch.assert_called_once()
    else:
        assert result.outcome is SandboxOutcome.TIMED_OUT
        assert "preparation" in result.detail
        launch.assert_not_called()


@pytest.mark.parametrize("lease_live", [True, False])
def test_native_launch_binds_mediator_runtime_or_kills_expired_attempt(
    monkeypatch, tmp_path, lease_live
):
    """Mediator authority follows launch runtime and fails closed if setup expiry wins."""
    approved = request(tmp_path, deadline=100, runtime_timeout=30)
    monkeypatch.setattr(macos_backend.time, "monotonic", lambda: 10)
    launch = MagicMock(return_value=Process())
    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)
    capture = MagicMock(
        return_value=ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\nok", "")
    )
    monkeypatch.setattr(macos_backend, "supervise_process", capture)
    broker = MagicMock()
    broker.bind_runtime_deadline.return_value = lease_live

    result = MacOSSeatbeltBackend._execute(approved, "", {}, "tag", None, broker=broker)

    broker.bind_runtime_deadline.assert_called_once_with(40)
    assert result.outcome is (
        SandboxOutcome.COMPLETED if lease_live else SandboxOutcome.UNAVAILABLE
    )
    if not lease_live:
        assert result.possible_effects
        assert capture.call_args.args[1] == 10


def test_native_run_reserves_a_separate_preparation_deadline(monkeypatch, tmp_path):
    """The public backend passes a finite setup ceiling without altering caller identity."""
    approved = request(tmp_path, deadline=100, runtime_timeout=30)
    observed = []

    def refuse(prepared):
        """Observe only the request inspected before native preparation."""
        observed.append(prepared.deadline)
        return CommandProcessResult(SandboxOutcome.INVALID, detail="controlled refusal")

    monkeypatch.setattr(macos_backend, "_request_failure", refuse)

    result = MacOSSeatbeltBackend().run(approved)

    assert result.outcome is SandboxOutcome.INVALID
    assert observed == [70]
    assert approved.deadline == 100


def test_macos_refuses_owned_terminal_without_native_launch(monkeypatch, tmp_path):
    """Unavailable PTY enforcement cannot trigger a regular or privileged command run."""
    launch = MagicMock()
    monkeypatch.setattr(macos_backend.subprocess, "Popen", launch)
    approved = replace(request(tmp_path), terminal=True)
    result = MacOSSeatbeltBackend().run(approved, terminal_input=lambda: None)
    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert "not qualified" in result.detail
    launch.assert_not_called()


@pytest.mark.parametrize("failure_point", ["supervision", "monitor cleanup"])
def test_backend_post_launch_errors_preserve_possible_effects(monkeypatch, tmp_path, failure_point):
    """An error after child launch cannot establish that the command had no effects."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen", MagicMock(return_value=Process())
    )
    if failure_point == "supervision":
        monkeypatch.setattr(
            "loop.execution.sandbox.macos.backend.supervise_process",
            MagicMock(side_effect=OSError("supervision failed")),
        )
    else:
        monitor = MagicMock()
        monitor.denial.return_value = None
        monitor.unregister.side_effect = [None, OSError("monitor cleanup failed")]
        monkeypatch.setattr(
            "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
            lambda self, deadline: monitor,
        )

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects
    assert failure_point.split()[0] in result.detail


def test_backend_does_not_launch_when_diagnostic_stream_is_unavailable(monkeypatch, tmp_path):
    """A reported unavailable diagnostic stream still prevents unqualified command launch."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda _self, _deadline: None,
    )

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "Seatbelt denial stream is unavailable."
    assert not result.possible_effects
    popen.assert_not_called()


@pytest.mark.parametrize("boundary", ["start", "register", "command-register"])
def test_backend_rejects_diagnostic_setup_errors_without_host_retry(
    monkeypatch, tmp_path, boundary
):
    """A diagnostic setup exception cannot launch or offer unrestricted host recovery."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock()
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monitor = MagicMock()
    if boundary == "register":
        monitor.register.side_effect = PermissionError("diagnostic veto")
        get_monitor = MagicMock(return_value=monitor)
    elif boundary == "command-register":
        monitor.register.side_effect = [True, PermissionError("diagnostic veto")]
        get_monitor = MagicMock(return_value=monitor)
    else:
        get_monitor = MagicMock(side_effect=PermissionError("diagnostic veto"))
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor", get_monitor
    )

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.INVALID
    assert result.detail == "diagnostic veto"
    assert not result.possible_effects
    popen.assert_not_called()


def qualified_without_diagnostics(monkeypatch, tmp_path, capture, monitor):
    """Qualify the real backend against a controlled monitor-free selector probe."""
    native = MacOSSeatbeltBackend()
    monkeypatch.setattr(native, "_qualify_local_network", MagicMock())
    monkeypatch.setattr(native, "_qualify_mediated_egress", MagicMock())
    monkeypatch.setattr(native, "_qualify_terminal", MagicMock())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.create_backend", lambda: native)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.SeatbeltDiagnosticService.monitor",
        lambda _self, _deadline: monitor,
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    popen = MagicMock(side_effect=lambda *_args, **_kwargs: Process())
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)

    def supervise(_process, *_args, **_kwargs):
        """Produce the selector's expected control effect, then the requested child capture."""
        cwd = popen.call_args.kwargs["cwd"]
        if cwd != tmp_path:
            (cwd / "allowed").write_text("allowed", encoding="utf-8")
            return ProcessCapture(ProcessCaptureStatus.COMPLETED, 1, "LOOP_SHELL_STARTED\n")
        return capture

    monkeypatch.setattr("loop.execution.sandbox.macos.backend.supervise_process", supervise)
    selected = selection.select_sandbox_backend()
    assert selected is native
    return native, popen


@pytest.mark.parametrize("monitor_state", ["missing", "stopped"])
@pytest.mark.parametrize(
    ("capture", "outcome"),
    [
        (
            ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\nok"),
            SandboxOutcome.COMPLETED,
        ),
        (
            ProcessCapture(ProcessCaptureStatus.COMPLETED, 1, "LOOP_SHELL_STARTED\n"),
            SandboxOutcome.COMPLETED,
        ),
        (
            ProcessCapture(ProcessCaptureStatus.COMPLETED, 1, "unverified"),
            SandboxOutcome.UNAVAILABLE,
        ),
        (ProcessCapture(ProcessCaptureStatus.TIMED_OUT), SandboxOutcome.TIMED_OUT),
        (ProcessCapture(ProcessCaptureStatus.UNAVAILABLE), SandboxOutcome.UNAVAILABLE),
    ],
)
def test_qualified_backend_reports_unavailable_diagnostics(
    monkeypatch, tmp_path, monitor_state, capture, outcome
):
    """A qualified backend runs without a stopped stream and reports its absence honestly."""
    monitor = None
    if monitor_state == "stopped":
        monitor = MagicMock()
        monitor.register.return_value = False
    native, popen = qualified_without_diagnostics(monkeypatch, tmp_path, capture, monitor)
    try:
        result = native.run(request(tmp_path))
        assert result.outcome is outcome
        assert result.diagnostic_status == "unavailable"
        assert not result.observed_denial
        assert popen.call_count == 2
        if monitor is not None:
            monitor.unregister.assert_not_called()
    finally:
        native.close()


def test_monitorless_qualification_is_not_inherited_after_pid_change(monkeypatch, tmp_path):
    """A forked process cannot reuse its parent's monitor-free enforcement qualification."""
    capture = ProcessCapture(ProcessCaptureStatus.COMPLETED, 0, "LOOP_SHELL_STARTED\nok")
    native, popen = qualified_without_diagnostics(monkeypatch, tmp_path, capture, None)
    value = request(tmp_path)
    pid = os.getpid()
    native_os = SimpleNamespace(**{**vars(os), "getpid": lambda: pid + 1})
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.os", native_os)
    try:
        result = native.run(value)
        assert result.outcome is SandboxOutcome.INVALID
        assert result.detail == "Seatbelt denial stream is unavailable."
        assert popen.call_count == 1
    finally:
        native.close()


def test_backend_preserves_output_when_start_marker_is_missing(monkeypatch, tmp_path):
    """Missing captured entry evidence cannot prove that the child produced no effects."""
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen",
        MagicMock(return_value=Process(stdout="command output", started=False)),
    )

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects
    assert result.stdout == "command output"
    assert "entry could not be verified" in result.detail


def test_backend_cleans_up_child_if_launch_audit_fails(monkeypatch, tmp_path):
    """A post-spawn audit failure still terminates and reaps the owned child."""
    process = Process()
    process.wait = MagicMock(wraps=process.wait)
    terminate = MagicMock()
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.Popen", MagicMock(return_value=process)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", terminate)
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.telemetry_audit",
        MagicMock(side_effect=[None, OSError("audit failed")]),
    )

    result = MacOSSeatbeltBackend().run(request(tmp_path))

    assert result.outcome is SandboxOutcome.UNAVAILABLE
    assert result.possible_effects
    assert result.detail == "audit failed"
    terminate.assert_called_once_with(process)
    process.wait.assert_called_once()


@pytest.mark.parametrize("storage", ["app", "App é ;$' (data)"])
def test_backend_binds_durable_scratch_without_exposing_private_neighbors(
    monkeypatch, tmp_path, storage
):
    """A session root under Library retains private-name guards and exact directory authority."""
    scratch = tmp_path / "Library" / storage / "scratch" / "session"
    scratch.mkdir(parents=True, mode=0o700)
    aliases = tmp_path / "aliases"
    aliases.mkdir(mode=0o700)
    physical_alias = aliases / "temporary"
    physical_alias.symlink_to(scratch, target_is_directory=True)
    probe = MagicMock(returncode=0, stderr="")
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run", MagicMock(return_value=probe)
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.kill_process_group", MagicMock())
    value = request(
        tmp_path,
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        scratch_root=scratch,
    )

    result = MacOSSeatbeltBackend().run(value)

    assert result.outcome is SandboxOutcome.COMPLETED
    assert popen.call_args.kwargs["env"]["TMPDIR"] == str(scratch)
    profile = popen.call_args.args[0][2]
    assert f"(require-not (subpath {json.dumps(str(scratch), ensure_ascii=False)}))" in profile
    assert re.escape(str(scratch)) + "/(.*/)?" in regex_patterns(profile)
    assert scratch.is_dir()


def test_fresh_session_data_profile_preserves_outside_private_denials(monkeypatch, tmp_path):
    """Fresh data names are usable without dropping inherited or workspace private protection."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    scratch = tmp_path / "Library" / "files"
    scratch.mkdir(parents=True, mode=0o700)
    (scratch / ".loop").mkdir()
    (scratch / ".loop" / "nested").mkdir()
    popen = MagicMock(return_value=Process())
    monkeypatch.setattr(
        "loop.execution.sandbox.macos.backend.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stderr="")),
    )
    monkeypatch.setattr("loop.execution.sandbox.macos.backend.subprocess.Popen", popen)
    approved = request(
        workspace,
        environment={"PATH": "/usr/bin:/bin", "TMPDIR": str(scratch)},
        scratch_root=scratch,
        scratch_data=True,
    )
    assert MacOSSeatbeltBackend().run(approved).outcome is SandboxOutcome.COMPLETED
    profile = popen.call_args.args[0][2]
    patterns = regex_patterns(profile)
    assert f"^{re.escape(str(scratch))}/(.*/)?" not in patterns
    assert f"^{re.escape(str(workspace))}/(.*/)?" in patterns
    assert f'(require-not (subpath "{scratch}"))' in profile
