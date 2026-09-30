"""Verify the live macOS denial reader without depending on the host log daemon."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loop.execution.sandbox import macos


class StreamProcess:
    """Feed an in-memory stream to a synchronously scheduled denial reader."""

    stdout: io.BytesIO
    running: bool
    killed: bool
    consume: object

    def __init__(self, *, header: bool = True, running: bool = True) -> None:
        self.stdout = io.BytesIO()
        self.running = running
        self.killed = False
        self.consume = None
        if header:
            self.send(b"[]\n")

    def send(self, line: bytes) -> None:
        """Deliver simulated log content synchronously to its registered reader."""
        position = self.stdout.tell()
        self.stdout.seek(0, io.SEEK_END)
        self.stdout.write(line)
        self.stdout.seek(position)
        if self.consume is not None:
            self.consume()

    def poll(self) -> int | None:
        """Report whether the simulated log stream is active."""
        return None if self.running and not self.killed else 0

    def kill(self) -> None:
        """Stop the simulated process without allocating native resources."""
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        """Return the simulated process status without waiting."""
        return 0

    def finish(self) -> None:
        """End the stream without a process kill."""
        self.running = False


@pytest.fixture(autouse=True)
def deterministic_reader(monkeypatch):
    """Schedule reader deliveries immediately and replace timed events with state doubles."""

    def event_factory():
        """Represent readiness without waiting for real time or another thread."""
        event = MagicMock()
        event.wait.return_value = False
        event.set.side_effect = lambda: setattr(event.wait, "return_value", True)
        return event

    def thread_factory(*, target, daemon):
        """Connect in-memory log deliveries directly to the real record consumer."""
        process = macos.subprocess.Popen.return_value
        thread = MagicMock()
        process.consume = target
        thread.start.side_effect = (
            target
            if not isinstance(process.stdout, io.BytesIO) or process.stdout.getvalue()
            else None
        )
        thread.is_alive.side_effect = lambda: (
            process.poll() is None and isinstance(process.stdout, io.BytesIO)
        )
        return thread

    monkeypatch.setattr(macos, "time", SimpleNamespace(**(vars(time) | {"monotonic": lambda: 100})))
    monkeypatch.setattr(
        macos,
        "threading",
        SimpleNamespace(**(vars(threading) | {"Thread": thread_factory, "Event": event_factory})),
    )


def record(tag: str, operation: str = "network-outbound", **changes: object) -> bytes:
    """Encode one tagged kernel record with optional field changes."""
    event: dict[str, object] = {
        "processID": 0,
        "processImagePath": "/kernel",
        "senderImagePath": "/Sandbox.kext/Sandbox",
        "eventMessage": f"Sandbox: sh deny(1) {operation} example\n{tag}",
    }
    event.update(changes)
    return json.dumps(event).encode() + b"\n"


def assert_denial(monitor: macos.DenialMonitor, tag: str, operation: str) -> None:
    """Require the synchronously delivered trusted denial without a polling deadline."""
    assert operation in (monitor.denial(tag, 101) or "")


def test_live_monitor_accepts_only_exact_trusted_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore invalid, untrusted and other-attempt lines while ranking real denials."""
    process = StreamProcess()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(macos.subprocess, "Popen", popen)
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "a" * 32
    other = "LOOP_SBX_" + "b" * 32
    try:
        assert monitor.healthy()
        assert popen.call_args.args[0][:3] == ["/usr/bin/log", "stream", "--style"]
        assert popen.call_args.kwargs["close_fds"] is True
        monitor.register(tag)
        process.send(b"not-json\n" + b"[]\n")
        process.send(record(other))
        process.send(record(tag, processID=123))
        process.send(record(tag, senderImagePath="/untrusted"))
        process.send(record(tag, eventMessage=42))
        process.send(record(tag, operation="file-read-data"))
        assert_denial(monitor, tag, "file-read-data")
        process.send(record(tag, operation="network-outbound"))
        assert_denial(monitor, tag, "network-outbound")
        process.send(record(tag, operation="file-write-data"))
        assert "network-outbound" in (monitor.denial(tag, 101) or "")
        monitor.unregister(tag)
        assert monitor.denial(tag, 101) is None
    finally:
        monitor.close()
    assert process.killed
    assert not monitor.healthy()


def test_live_monitor_ranks_tagged_denials_after_incidental_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A higher-priority tagged denial remains visible after a later lower-priority one."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "d" * 32
    target = tmp_path / "secret"
    try:
        monitor.register(tag)
        process.send(
            record(
                tag,
                eventMessage=f"Sandbox: cat deny(1) file-read-data {target}\n{tag}",
            )
        )
        process.send(
            record(
                tag,
                eventMessage=f"Sandbox: sh deny(1) file-read-data /Library/Preferences/Logging/missing\n{tag}",
            )
        )
        assert_denial(monitor, tag, "file-read-data")
        assert str(target) in (monitor.denial(tag, 101) or "")
    finally:
        monitor.close()


def test_live_monitor_bounds_lines_and_handles_no_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large or unsolicited records cannot accumulate in the shared reader."""
    process = StreamProcess()
    monkeypatch.setattr(macos, "_LOG_LIMIT", 512)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "c" * 32
    try:
        assert monitor.denial(tag, 101) is None
        monitor.register(tag)
        process.send(b" " * (macos._LOG_LIMIT + 1) + b"\n")
        process.send(record(tag, operation="file-read-data", processImagePath="/other"))
        process.send(record(tag, operation="file-read-data", eventMessage="missing tag"))
        process.send(record(tag, operation="process-fork"))
        assert monitor.denial(tag, 100.2) is None
    finally:
        monitor.close()


def test_live_monitor_fails_closed_when_stream_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unready or exited live stream cannot authorize a command launch."""
    unready = StreamProcess(header=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=unready))
    with pytest.raises(OSError, match="did not become ready"):
        macos.DenialMonitor(100)
    assert unready.killed

    exited = StreamProcess(running=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=exited))
    with pytest.raises(OSError, match="exited before command launch"):
        macos.DenialMonitor(101)


def test_live_monitor_reader_failure_stops_accepting_denials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An I/O failure in the log reader ends its health and cannot authorize a command."""
    process = StreamProcess()
    real_stdout = process.stdout

    class FailingStream:
        """Supply one ready line before the log stream fails."""

        def __init__(self) -> None:
            self.reads = 0

        def readline(self, _limit: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return b"[]\n"
            raise OSError("log read failed")

        def close(self) -> None:
            real_stdout.close()

    process.stdout = FailingStream()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    try:
        assert not monitor.healthy()
        with pytest.raises(OSError, match="stopped"):
            monitor.register("LOOP_SBX_" + "d" * 32)
    finally:
        monitor.close()


def test_live_monitor_rejects_stopped_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped log stream is detected before an attempt is registered."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    process.finish()
    assert not monitor.healthy()
    with pytest.raises(OSError, match="stopped"):
        monitor.register("LOOP_SBX_" + "d" * 32)
    monitor.close()


def test_shared_monitor_reuses_and_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    """One parent reuses a healthy stream and replaces a stopped stream."""
    first = MagicMock()
    second = MagicMock()
    first.healthy.side_effect = [True, False]
    second.healthy.return_value = True
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(macos, "DenialMonitor", factory)
    service = macos.SeatbeltDiagnosticService()
    assert service.monitor(101) is first
    assert service.monitor(101) is first
    assert service.monitor(101) is second
    first.close.assert_called_once()
    service.close()
    second.close.assert_called_once()
    service.close()
    second.close.assert_called_once()


def test_shared_monitor_discards_inherited_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forked child does not retain its parent's live stream object."""
    parent = MagicMock()
    replacement = MagicMock()
    service = macos.SeatbeltDiagnosticService()
    service._monitor = parent
    service._pid = -1
    monkeypatch.setattr(macos, "DenialMonitor", MagicMock(return_value=replacement))
    assert service.monitor(101) is replacement
    parent.close.assert_not_called()


def test_application_diagnostics_share_and_recreate_one_stream(monkeypatch):
    """Two backends share a monitor until the application closes its diagnostic owner."""
    first = MagicMock()
    first.healthy.return_value = True
    second = MagicMock()
    second.healthy.return_value = True
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(macos, "DenialMonitor", factory)
    service = macos.SeatbeltDiagnosticService()
    left = macos.MacOSSeatbeltBackend(service)
    right = macos.MacOSSeatbeltBackend(service)
    assert right is not left
    assert service.monitor(101) is first
    assert service.monitor(101) is first
    left.close()
    first.close.assert_not_called()
    service.close()
    first.close.assert_called_once()
    assert service.monitor(101) is second
    service.close()
    second.close.assert_called_once()
    owned = macos.MacOSSeatbeltBackend()
    owned.close()


def test_diagnostic_owner_discards_forked_readiness(monkeypatch):
    """A PID change forces a fresh monitor and launch probe without closing the parent stream."""
    parent = MagicMock()
    parent.healthy.return_value = True
    child = MagicMock()
    child.healthy.return_value = True
    monkeypatch.setattr(macos, "DenialMonitor", MagicMock(side_effect=[parent, child]))
    service = macos.SeatbeltDiagnosticService()
    assert service.monitor(101) is parent
    service.mark_parent_verified()
    assert service.parent_verified()
    real_pid = os.getpid()
    monkeypatch.setattr(macos.os, "getpid", lambda: real_pid + 1)
    assert not service.parent_verified()
    assert service.monitor(101) is child
    parent.close.assert_not_called()
    service.close()
    child.close.assert_called_once()
