"""Verify the live macOS denial reader without depending on the host log daemon."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock

import pytest

from loop.execution.sandbox import macos


class StreamProcess:
    """Feed a controllable, line-oriented pipe to the live denial reader."""

    def __init__(self, *, header: bool = True, running: bool = True) -> None:
        read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb")
        self.running = running
        self.killed = False
        if header:
            self.send(b"[]\n")
        if not running:
            os.close(self._write_fd)

    def send(self, line: bytes) -> None:
        """Write one simulated log line into the stream."""
        os.write(self._write_fd, line)

    def poll(self) -> int | None:
        """Report whether the simulated log stream is active."""
        return None if self.running and not self.killed else 0

    def kill(self) -> None:
        """Stop the simulated process and release the reader."""
        self.killed = True
        os.close(self._write_fd)

    def wait(self, timeout: float | None = None) -> int:
        """Return the simulated process status."""
        return 0

    def finish(self) -> None:
        """End the stream without a process kill."""
        self.running = False
        os.close(self._write_fd)


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


def wait_for_denial(monitor: macos.DenialMonitor, tag: str, operation: str) -> None:
    """Wait briefly for a reader-thread update without assuming its schedule."""
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if operation in (monitor.denial(tag, deadline) or ""):
            return
        time.sleep(0.001)
    pytest.fail(f"No {operation} denial arrived")


def test_live_monitor_accepts_only_exact_trusted_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore invalid, untrusted and other-attempt lines while ranking real denials."""
    process = StreamProcess()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(macos.subprocess, "Popen", popen)
    monitor = macos.DenialMonitor(time.monotonic() + 1)
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
        wait_for_denial(monitor, tag, "file-read-data")
        process.send(record(tag, operation="network-outbound"))
        wait_for_denial(monitor, tag, "network-outbound")
        process.send(record(tag, operation="file-write-data"))
        assert "network-outbound" in (monitor.denial(tag, time.monotonic() + 1) or "")
        monitor.unregister(tag)
        assert monitor.denial(tag, time.monotonic() + 1) is None
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
    monitor = macos.DenialMonitor(time.monotonic() + 1)
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
        wait_for_denial(monitor, tag, "file-read-data")
        assert str(target) in (monitor.denial(tag, time.monotonic() + 1) or "")
    finally:
        monitor.close()


def test_live_monitor_bounds_lines_and_handles_no_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large or unsolicited records cannot accumulate in the shared reader."""
    process = StreamProcess()
    monkeypatch.setattr(macos, "_LOG_LIMIT", 512)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(time.monotonic() + 1)
    tag = "LOOP_SBX_" + "c" * 32
    try:
        assert monitor.denial(tag, time.monotonic() + 1) is None
        monitor.register(tag)
        process.send(b" " * (macos._LOG_LIMIT + 1) + b"\n")
        process.send(record(tag, operation="file-read-data", processImagePath="/other"))
        process.send(record(tag, operation="file-read-data", eventMessage="missing tag"))
        process.send(record(tag, operation="process-fork"))
        assert monitor.denial(tag, time.monotonic() + 0.2) is None
    finally:
        monitor.close()


def test_live_monitor_fails_closed_when_stream_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unready or exited live stream cannot authorize a command launch."""
    unready = StreamProcess(header=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=unready))
    with pytest.raises(OSError, match="did not become ready"):
        macos.DenialMonitor(time.monotonic())
    assert unready.killed

    exited = StreamProcess(running=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=exited))
    with pytest.raises(OSError, match="exited before command launch"):
        macos.DenialMonitor(time.monotonic() + 1)


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
    monitor = macos.DenialMonitor(time.monotonic() + 1)
    try:
        monitor._reader.join(timeout=1)
        assert not monitor.healthy()
        with pytest.raises(OSError, match="stopped"):
            monitor.register("LOOP_SBX_" + "d" * 32)
    finally:
        monitor.close()


def test_live_monitor_rejects_stopped_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped log stream is detected before an attempt is registered."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(time.monotonic() + 1)
    process.finish()
    monitor._reader.join(timeout=1)
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
    assert service.monitor(time.monotonic() + 1) is first
    assert service.monitor(time.monotonic() + 1) is first
    assert service.monitor(time.monotonic() + 1) is second
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
    assert service.monitor(time.monotonic() + 1) is replacement
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
    assert service.monitor(time.monotonic() + 1) is first
    assert service.monitor(time.monotonic() + 1) is first
    left.close()
    first.close.assert_not_called()
    service.close()
    first.close.assert_called_once()
    assert service.monitor(time.monotonic() + 1) is second
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
    assert service.monitor(time.monotonic() + 1) is parent
    service.mark_parent_verified()
    assert service.parent_verified()
    real_pid = os.getpid()
    monkeypatch.setattr(macos.os, "getpid", lambda: real_pid + 1)
    assert not service.parent_verified()
    assert service.monitor(time.monotonic() + 1) is child
    parent.close.assert_not_called()
    service.close()
    child.close.assert_called_once()
