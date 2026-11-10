"""Verify optional Linux denial observation and exact attempt association."""

import hashlib
import io
import json
import os
import socket
import struct
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest

from loop.execution.sandbox.linux import diagnostics


class CapturedThread:
    """Capture observer workers so each case controls their order."""

    pending: ClassVar[list] = []

    def __init__(self, *, target, args=(), daemon):
        self.target = target
        self.args = args
        assert daemon

    def start(self):
        """Queue a worker without starting an external thread."""
        self.pending.append((self.target, self.args))


class ImmediateEvent:
    """Stop a failed discovery loop without waiting for time."""

    def __init__(self):
        self.value = False

    def set(self):
        """Set the cooperative stop signal."""
        self.value = True

    def is_set(self):
        """Return the cooperative stop signal."""
        return self.value

    def wait(self, _timeout):
        """Stop the next retry immediately."""
        self.value = True


@pytest.fixture
def observation(monkeypatch, tmp_path):
    """Provide an owned fake journal, socket peer, and protected service membership."""
    CapturedThread.pending = []
    monkeypatch.setattr(
        diagnostics,
        "threading",
        SimpleNamespace(Thread=CapturedThread, Event=ImmediateEvent, Lock=threading.Lock),
    )
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: 10)
    monkeypatch.setattr(
        diagnostics.shutil, "which", lambda *_args, **_kwargs: "/usr/bin/journalctl"
    )
    monkeypatch.setattr(
        diagnostics,
        "bind_host_file",
        Mock(return_value=SimpleNamespace(open_verified=lambda: nullcontext(9))),
    )
    monkeypatch.setattr(diagnostics, "CGROUP_ROOT", tmp_path)
    group_name = "loopgen_demo@123-456.service"
    group = tmp_path / "loopgen_demo.slice" / group_name
    group.mkdir(parents=True)
    (group / "cgroup.procs").write_text("123\n", encoding="ascii")
    original_read = Path.read_text
    original_stat = Path.stat

    def read(path, *args, **kwargs):
        """Provide stable kernel-owned synthetic process metadata."""
        value = str(path)
        if value == "/proc/sys/kernel/random/boot_id":
            return "a" * 32
        if value == "/proc/123/stat":
            return "123 (fixture) S " + "0 " * 18 + "42"
        if value == "/proc/123/cgroup":
            return f"0::/loopgen_demo.slice/{group_name}\n"
        return original_read(path, *args, **kwargs)

    def stat(path, *args, **kwargs):
        """Treat the synthetic service directory as root protected."""
        if path == group:
            return SimpleNamespace(st_uid=0, st_mode=0o40755)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "stat", stat)
    fake_os = SimpleNamespace(
        fstat=lambda _fd: SimpleNamespace(st_ino=111),
        pread=lambda _fd, _size, _offset: b"\x7fELF",
        readlink=lambda _path: "socket:[222]",
        scandir=os.scandir,
        defpath=os.defpath,
        pathsep=os.pathsep,
        environ={},
    )
    monkeypatch.setattr(diagnostics, "os", fake_os)
    message = struct.pack("=BBBBIII", socket.AF_UNIX, 1, 1, 0, 111, 0, 0)
    attribute = struct.pack("=HHI", 8, 2, 222)
    header = struct.pack("=IHHII", 16 + len(message) + len(attribute), 20, 0, 1, 0)
    query = Mock()
    query.__enter__ = Mock(return_value=query)
    query.__exit__ = Mock(return_value=False)
    query.recvfrom.return_value = (header + message + attribute, (0, 0))
    fake_socket = SimpleNamespace(
        AF_UNIX=socket.AF_UNIX,
        AF_NETLINK=16,
        SOCK_RAW=socket.SOCK_RAW,
        socket=Mock(return_value=query),
    )
    monkeypatch.setattr(diagnostics, "socket", fake_socket)
    process = Mock()
    process.stdout = io.BytesIO()
    process.poll.return_value = None
    fake_subprocess = SimpleNamespace(Popen=Mock(return_value=process), PIPE=-1, DEVNULL=-3)
    monkeypatch.setattr(diagnostics, "subprocess", fake_subprocess)
    suffix = hashlib.sha256(group_name.encode()).hexdigest()[:48]
    fields = (
        'audit: type=1400 audit(123.45:6): apparmor="DENIED" operation="open" '
        f'namespace="root//demo" profile="payload_{"b" * 64}_{suffix}" '
        'name="/outside/file" pid=123'
    )
    event = {
        "_TRANSPORT": "kernel",
        "_BOOT_ID": "a" * 32,
        "__MONOTONIC_TIMESTAMP": "10000001",
        "_SOURCE_MONOTONIC_TIMESTAMP": "10000001",
        "MESSAGE": fields,
    }
    connection = Mock()
    connection.fileno.return_value = 3
    return SimpleNamespace(
        event=event,
        connection=connection,
        query=query,
        process=process,
        group=group,
        group_name=group_name,
        socket=fake_socket,
        os=fake_os,
        subprocess=fake_subprocess,
    )


def run_observer(observation, lines):
    """Deliver synthetic events after associating the live receiver."""
    observation.process.stdout = io.BytesIO(lines)
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, identify = CapturedThread.pending
    identify[0](*identify[1])
    reader[0](*reader[1])
    return observer.finish()


def test_reports_only_the_matching_kernel_observation(observation):
    """A live matching peer and kernel event produce a bounded unverified candidate."""
    result = run_observer(observation, json.dumps(observation.event).encode() + b"\n")

    assert result == ("apparmor-open /outside/file", "available")
    observation.process.kill.assert_called_once()
    observation.process.wait.assert_called_once()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_TRANSPORT", "stdout"),
        ("_BOOT_ID", "b" * 32),
        ("__MONOTONIC_TIMESTAMP", "1"),
        ("_SOURCE_MONOTONIC_TIMESTAMP", "1"),
        ("MESSAGE", "ordinary output"),
        ("MESSAGE", None),
    ],
)
def test_rejects_untrusted_journal_metadata(observation, field, value):
    """Wrong provenance, boot, time, or message cannot become a candidate."""
    observation.event[field] = value
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "",
        "unknown",
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('apparmor="DENIED"', 'apparmor="ALLOWED"'),
        ('namespace="root//demo"', 'namespace="root//other"'),
        ('operation="open"', 'operation="bad operation"'),
        ('profile="payload_', 'profile="other_'),
        ('name="/outside/file"', 'name="relative"'),
        ("pid=123", "pid=123 pid=456"),
        ("audit(123.45:6):", "audit(invalid):"),
    ],
)
def test_rejects_or_sanitizes_ambiguous_kernel_fields(observation, before, after):
    """Only structured same-generation records can disclose a safe candidate."""
    observation.event["MESSAGE"] = observation.event["MESSAGE"].replace(before, after)
    candidate, status = run_observer(observation, json.dumps(observation.event).encode() + b"\n")
    if after == 'name="relative"':
        assert (candidate, status) == ("apparmor-open <resource unavailable>", "available")
    else:
        assert (candidate, status) == ("", "unknown")


@pytest.mark.parametrize(
    "line", [b"bad json\n", b"[]\n", b'{"_TRANSPORT":"kernel","_TRANSPORT":"kernel"}\n']
)
def test_rejects_invalid_journal_envelopes(observation, line):
    """Malformed or duplicate-field input cannot spoof a kernel observation."""
    assert run_observer(observation, line) == ("", "unknown")


def test_unrelated_and_oversized_events_do_not_cross_attempts(observation):
    """Wrong service hashes and oversized records cannot contaminate a result."""
    event = dict(observation.event)
    event["MESSAGE"] = event["MESSAGE"].replace(observation.group_name, "other")
    event["MESSAGE"] = event["MESSAGE"].replace(
        hashlib.sha256(observation.group_name.encode()).hexdigest()[:48], "c" * 48
    )
    lines = b"x" * 65537 + b"\n" + json.dumps(event).encode() + b"\n"
    assert run_observer(observation, lines) == ("", "unknown")


def test_failed_socket_association_never_surfaces_other_events(observation):
    """An absent peer prevents attributing a valid journal event to this request."""
    observation.query.recvfrom.return_value = (b"short", (0, 0))
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "",
        "unknown",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "sender",
        "length",
        "kind",
        "sequence",
        "family",
        "inode",
        "short-attribute",
        "long-attribute",
        "short-peer",
        "duplicate-peer",
        "other-attribute",
        "zero-peer",
        "tail",
    ],
)
def test_rejects_invalid_or_ambiguous_socket_identity(observation, mutation):
    """Malformed, foreign, or ambiguous kernel peer replies cannot bind an attempt."""
    data, sender = observation.query.recvfrom.return_value
    data = bytearray(data)
    if mutation == "sender":
        sender = (1, 0)
    elif mutation == "length":
        struct.pack_into("=I", data, 0, len(data) - 1)
    elif mutation == "kind":
        struct.pack_into("=H", data, 4, 2)
    elif mutation == "sequence":
        struct.pack_into("=I", data, 8, 2)
    elif mutation == "family":
        data[16] = 0
    elif mutation == "inode":
        struct.pack_into("=I", data, 20, 999)
    elif mutation == "short-attribute":
        struct.pack_into("=H", data, 32, 3)
    elif mutation == "long-attribute":
        struct.pack_into("=H", data, 32, 100)
    elif mutation == "short-peer":
        struct.pack_into("=H", data, 32, 4)
    elif mutation == "duplicate-peer":
        data.extend(data[32:40])
        struct.pack_into("=I", data, 0, len(data))
    elif mutation == "other-attribute":
        struct.pack_into("=H", data, 34, 3)
    elif mutation == "zero-peer":
        struct.pack_into("=I", data, 36, 0)
    else:
        data.append(0)
        struct.pack_into("=I", data, 0, len(data))
    observation.query.recvfrom.return_value = (bytes(data), sender)
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "",
        "unknown",
    )


@pytest.mark.parametrize(
    "kind",
    [
        "limit",
        "unsafe",
        "oversized",
        "many-pids",
        "bad-pid",
        "missing-proc",
        "wrong-peer",
        "duplicate",
        "changed-socket",
    ],
)
def test_untrusted_or_ambiguous_service_membership_is_unknown(observation, monkeypatch, kind):
    """Unreadable, excessive, changed, or ambiguous membership cannot identify a request."""
    if kind == "limit":
        monkeypatch.setattr(diagnostics, "_MAX_SERVICES", 0)
    elif kind == "unsafe":
        previous = Path.stat
        monkeypatch.setattr(
            Path,
            "stat",
            lambda path, *args, **kwargs: (
                SimpleNamespace(st_uid=10, st_mode=0o40755)
                if path == observation.group
                else previous(path, *args, **kwargs)
            ),
        )
    elif kind == "oversized":
        (observation.group / "cgroup.procs").write_text("1" * 4097)
    elif kind == "many-pids":
        (observation.group / "cgroup.procs").write_text("123\n" * 129)
    elif kind == "bad-pid":
        (observation.group / "cgroup.procs").write_text("bad\n")
    elif kind == "missing-proc":
        (observation.group / "cgroup.procs").write_text("999999999\n")
    elif kind == "wrong-peer":
        observation.os.readlink = lambda _path: "socket:[999]"
    elif kind == "duplicate":
        (observation.group / "cgroup.procs").write_text("123\n123\n")
    else:
        good = observation.query.recvfrom.return_value
        observation.query.recvfrom.side_effect = [good, (b"short", (0, 0))]
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "",
        "unknown",
    )


def test_ignores_unrelated_services_and_corrupt_process_metadata(observation):
    """Other service names and a process that disappears do not prevent a valid match."""
    (observation.group.parent / "other.service").mkdir()
    (observation.group / "cgroup.procs").write_text("999999999\n123\n")
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "apparmor-open /outside/file",
        "available",
    )


def test_socket_query_failure_does_not_affect_the_command(observation):
    """Unavailable kernel socket diagnostics leave attribution unknown."""
    observation.query.recvfrom.side_effect = OSError("unavailable")
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "",
        "unknown",
    )


def test_journal_discovery_uses_protected_nonstandard_host_paths(observation, monkeypatch):
    """An unsafe earlier executable cannot hide a protected custom host installation."""
    observation.os.defpath = "/usr/bin:/usr/bin:relative"
    observation.os.environ = {"PATH": "/home/user/bin:/opt/system/bin"}
    which = Mock(
        side_effect=lambda _name, path: (
            f"{path}/journalctl" if path in {"/home/user/bin", "/opt/system/bin"} else None
        )
    )
    monkeypatch.setattr(diagnostics.shutil, "which", which)
    protected = SimpleNamespace(open_verified=lambda: nullcontext(9))

    def bind(path):
        """Reject the user-writable candidate and accept the protected host tool."""
        if str(path).startswith("/home/"):
            raise PermissionError("unprotected")
        return protected

    monkeypatch.setattr(diagnostics, "bind_host_file", bind)
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "apparmor-open /outside/file",
        "available",
    )
    assert which.call_count == 3
    assert which.call_args.kwargs["path"] == "/opt/system/bin"


def test_journal_discovery_bounds_host_path_search(observation, monkeypatch):
    """An unusually long host search path cannot create unbounded diagnostic work."""
    monkeypatch.setattr(diagnostics, "_MAX_SEARCH_DIRECTORIES", 1)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda *_args, **_kwargs: None)
    assert run_observer(observation, b"") == ("", "unknown")


def test_script_journal_reader_is_skipped_for_protected_elf(observation, monkeypatch):
    """A protected script cannot delegate host execution to a writable interpreter."""
    observation.os.defpath = "/usr/bin:/opt/system/bin"
    observation.os.environ = {}
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _name, path: f"{path}/journalctl")
    monkeypatch.setattr(
        diagnostics,
        "bind_host_file",
        lambda path: SimpleNamespace(
            open_verified=lambda: nullcontext(11 if str(path).startswith("/usr/") else 9)
        ),
    )
    observation.os.pread = lambda fd, _size, _offset: b"#!/u" if fd == 11 else b"\x7fELF"
    assert run_observer(observation, json.dumps(observation.event).encode() + b"\n") == (
        "apparmor-open /outside/file",
        "available",
    )


def test_pinned_journal_executable_is_rechecked_before_execution(observation):
    """A changed executable type cannot be launched even after successful discovery."""
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, _identify = CapturedThread.pending
    observation.os.pread = Mock(side_effect=[b"\x7fELF", b"#!/u"])
    reader[0](*reader[1])
    assert observer.finish() == ("", "unknown")
    observation.subprocess.Popen.assert_not_called()


@pytest.mark.parametrize("failure", ["missing-program", "bad-boot", "bind", "spawn", "no-stdout"])
def test_unavailable_journal_never_affects_the_command(observation, monkeypatch, failure):
    """Unavailable journal setup produces unknown diagnostics without launching commands again."""
    if failure == "missing-program":
        monkeypatch.setattr(diagnostics.shutil, "which", lambda *_args, **_kwargs: None)
    elif failure == "bad-boot":
        previous = Path.read_text
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda path, *args, **kwargs: (
                "invalid"
                if str(path) == "/proc/sys/kernel/random/boot_id"
                else previous(path, *args, **kwargs)
            ),
        )
    elif failure == "bind":
        monkeypatch.setattr(diagnostics, "bind_host_file", Mock(side_effect=PermissionError()))
    elif failure == "spawn":
        observation.subprocess.Popen.side_effect = OSError("cannot start")
    else:
        observation.process.stdout = None
        observer = diagnostics.LinuxDenialObserver()
        observer.start(observation.connection, "demo")
        reader, identify = CapturedThread.pending
        identify[0](*identify[1])
        reader[0](*reader[1])
        assert observer.finish() == ("", "unknown")
        return
    assert run_observer(observation, b"") == ("", "unknown")


def test_discarding_oversized_multiline_input_preserves_later_records(observation):
    """Oversized journal data is discarded through the next complete record boundary."""
    lines = b"x" * 150000 + b"\n" + json.dumps(observation.event).encode() + b"\n"
    assert run_observer(observation, lines) == ("apparmor-open /outside/file", "available")


def test_deduplicates_and_bounds_multiple_candidates(observation):
    """Distinct current-attempt denials remain separate within the bounded result."""
    lines = []
    for index in range(6):
        event = dict(observation.event)
        event["MESSAGE"] = event["MESSAGE"].replace(
            "/outside/file", f"/outside/{index}" + "x" * 490
        )
        lines.append(json.dumps(event).encode() + b"\n")
    lines.append(lines[-1])
    candidate, status = run_observer(observation, b"".join(lines))
    assert status == "available"
    assert len(candidate) <= 2000
    records = json.loads(candidate.removeprefix(diagnostics.DENIAL_RECORDS_PREFIX))
    assert len(records) <= 4
    assert len(records) == len(set(records))
    assert records[-1].startswith("apparmor-open /outside/5")
    assert records[-1].endswith("…")


def test_thread_start_failure_leaves_diagnostics_unknown(observation, monkeypatch):
    """A thread resource failure never changes command execution."""
    monkeypatch.setattr(diagnostics.threading, "Thread", Mock(side_effect=RuntimeError()))
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    assert observer.finish() == ("", "unknown")


def test_workers_do_not_start_io_after_attempt_completion(observation):
    """Already-completed attempts do not perform optional observation work."""
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, identify = CapturedThread.pending
    assert observer.finish() == ("", "unknown")
    identify[0](*identify[1])
    reader[0](*reader[1])
    observation.subprocess.Popen.assert_not_called()
    observation.socket.socket.assert_not_called()


def test_attempt_can_finish_during_journal_discovery(observation, monkeypatch):
    """A delayed executable verification cannot launch after the attempt finishes."""
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, _identify = CapturedThread.pending

    def bind(_path):
        """Complete the attempt while the optional journal executable is being checked."""
        assert observer.finish() == ("", "unknown")
        return SimpleNamespace(open_verified=lambda: nullcontext(9))

    monkeypatch.setattr(diagnostics, "bind_host_file", bind)
    reader[0](*reader[1])
    observation.subprocess.Popen.assert_not_called()


def test_discovery_has_a_fixed_retry_budget(observation, monkeypatch):
    """A receiver that never appears does not cause unbounded background polling."""
    observation.os.readlink = lambda _path: "socket:[999]"
    monkeypatch.setattr(ImmediateEvent, "wait", lambda *_args: None)
    monkeypatch.setattr(diagnostics, "_IDENTITY_ATTEMPTS", 1)
    assert run_observer(observation, b"") == ("", "unknown")


@pytest.mark.parametrize("kill_fails", [False, True])
def test_stopping_during_journal_launch_cleans_up_without_waiting(observation, kill_fails):
    """A delayed journal launch cannot outlive the completed attempt."""
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, _identify = CapturedThread.pending

    def launch(*_args, **_kwargs):
        """Complete the command while the diagnostic process is starting."""
        assert observer.finish() == ("", "unknown")
        return observation.process

    observation.subprocess.Popen.side_effect = launch
    if kill_fails:
        observation.process.kill.side_effect = OSError("already stopped")
    reader[0](*reader[1])
    observation.process.kill.assert_called_once()
    observation.process.wait.assert_called_once()


def test_journal_termination_failure_is_only_diagnostic(observation):
    """A vanished or unkillable observer cannot replace the command result."""
    observation.process.stdout = io.BytesIO()
    observer = diagnostics.LinuxDenialObserver()
    observer.start(observation.connection, "demo")
    reader, identify = CapturedThread.pending
    identify[0](*identify[1])
    reader[0](*reader[1])
    observation.process.kill.side_effect = OSError("already stopped")
    assert observer.finish() == ("", "unknown")
