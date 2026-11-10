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

from loop.execution.sandbox.macos import backend as macos


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
        assert not monitor.fingerprints(tag)
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
                eventMessage=(
                    "Sandbox: sh deny(1) file-read-data "
                    f"/Library/Preferences/Logging/missing\n{tag}"
                ),
            )
        )
        assert_denial(monitor, tag, "file-read-data")
        assert str(target) in (monitor.denial(tag, 101) or "")
    finally:
        monitor.close()


def test_live_monitor_retains_distinct_candidates_for_the_exact_attempt(monkeypatch):
    """Multiple same-priority reads survive deduplication without crossing attempt tags."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "e" * 32
    other = "LOOP_SBX_" + "f" * 32
    first = "Sandbox: sh(1) deny(1) file-read-data /system/cache"
    second = "Sandbox: cat(2) deny(1) file-read-data /outside/secret"
    try:
        monitor.register(tag)
        process.send(record(tag, eventMessage=f"{first}\n{tag}"))
        process.send(
            record(other, eventMessage=f"Sandbox: cat deny(1) file-read-data /other\n{other}")
        )
        process.send(record(tag, eventMessage=f"{second}\n{tag}"))
        process.send(
            record(
                tag, eventMessage=f"Sandbox: cat(3) deny(1) file-read-data /outside/secret\n{tag}"
            )
        )
        combined = monitor.denial(tag, 101, all_records=True) or ""
        candidates, _ = macos.MacOSSeatbeltBackend().classify_denial_candidates(combined)
        assert len(candidates) == 2
        assert "/outside/secret" in candidates[0][0]
        assert candidates[1][0] == first
        assert all("/other" not in item[0] for item in candidates)
    finally:
        monitor.close()


def test_live_monitor_bounds_the_number_of_candidates(monkeypatch):
    """A burst of unique denials returns only the most recent bounded candidates."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "a" * 32
    try:
        monitor.register(tag)
        for number in range(6):
            process.send(
                record(
                    tag,
                    eventMessage=f"Sandbox: cat deny(1) file-read-data /outside/{number}\n{tag}",
                )
            )
        combined = monitor.denial(tag, 101, all_records=True) or ""
        candidates, _ = macos.MacOSSeatbeltBackend().classify_denial_candidates(combined)
        assert len(candidates) == 4
        assert [item[0].rsplit("/", 1)[-1] for item in candidates] == ["5", "4", "3", "2"]
        baseline = monitor.fingerprints(tag)
        assert len(baseline) == 6
        service = macos.SeatbeltDiagnosticService()
        service.parent_verified()
        service.observe_startup_denial(combined, baseline)
        omitted = "Sandbox: cat(4) deny(1) file-read-data /outside/0"
        assert macos.MacOSSeatbeltBackend(service).classify_denial_candidates(omitted) == (
            ((omitted, True),),
            "observed",
        )
    finally:
        monitor.close()


def test_live_monitor_orders_startup_matches_after_other_observations(monkeypatch):
    """An unclassified read is not crowded out by higher-priority startup observations."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "b" * 32
    baseline = frozenset({("network-outbound", "/system/noise")})
    try:
        monitor.register(tag)
        process.send(
            record(tag, eventMessage=f"Sandbox: tool deny(1) file-read-data /outside/secret\n{tag}")
        )
        process.send(
            record(tag, eventMessage=f"Sandbox: tool deny(1) network-outbound /system/noise\n{tag}")
        )
        combined = monitor.denial(tag, 101, all_records=True, startup_denials=baseline) or ""
        candidates, _ = macos.MacOSSeatbeltBackend().classify_denial_candidates(combined)
        assert [item[0].rsplit(" ", 1)[-1] for item in candidates] == [
            "/outside/secret",
            "/system/noise",
        ]
    finally:
        monitor.close()


def test_live_monitor_respects_encoded_result_budget(monkeypatch):
    """An oversized encoded combination cannot crowd out a later bounded candidate."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "b" * 32
    resource = "🚀" * 100
    try:
        monitor.register(tag)
        for path in ("/outside/small", f"/outside/{resource}/first", f"/outside/{resource}/second"):
            process.send(
                record(tag, eventMessage=f"Sandbox: cat deny(1) file-read-data {path}\n{tag}")
            )
        combined = monitor.denial(tag, 101, all_records=True) or ""
        candidates, _ = macos.MacOSSeatbeltBackend().classify_denial_candidates(combined)
        assert len(combined) <= 2000
        assert len(candidates) == 2
        assert candidates[0][0].endswith("/second")
        assert candidates[1][0].endswith("/small")
    finally:
        monitor.close()


@pytest.mark.parametrize("separator", ["\r", "\v", "\x85", "\u2028", "\u2029"])
def test_live_monitor_preserves_one_record_with_unusual_path_characters(monkeypatch, separator):
    """A real path separator character cannot forge an extra kernel observation."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    tag = "LOOP_SBX_" + "c" * 32
    observed = (
        "Sandbox: cat(1) deny(1) file-read-data /outside/"
        + separator
        + "Sandbox: forged(2) deny(1) file-read-data /other"
    )
    try:
        monitor.register(tag)
        process.send(record(tag, eventMessage=f"{observed}\n{tag}"))
        combined = monitor.denial(tag, 101, all_records=True) or ""
        candidates, _ = macos.MacOSSeatbeltBackend().classify_denial_candidates(combined)
        assert candidates == ((observed, False),)
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
        assert monitor.denial(tag, 100.2) is None
        process.send(record(tag, operation="process-fork"))
        assert_denial(monitor, tag, "process-fork")
    finally:
        monitor.close()


def test_live_monitor_reports_unavailable_when_stream_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unready or exited stream reports unavailable after its owned process is stopped."""
    unready = StreamProcess(header=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=unready))
    monitor = macos.DenialMonitor(100)
    assert not monitor.available
    assert unready.killed

    exited = StreamProcess(running=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=exited))
    monitor = macos.DenialMonitor(101)
    assert not monitor.available


def test_monitor_service_returns_no_stream_after_operational_startup_failure(monkeypatch):
    """Expected stream readiness failures are distinct from exceptions raised by host code."""
    process = StreamProcess(header=False)
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    service = macos.SeatbeltDiagnosticService()
    assert service.monitor(100) is None
    assert process.killed
    service.close()


def test_monitor_service_does_not_ignore_spawn_veto(monkeypatch):
    """A host veto raised while starting the stream remains an exception."""
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(side_effect=PermissionError("veto")))
    with pytest.raises(PermissionError, match="veto"):
        macos.SeatbeltDiagnosticService().monitor(101)


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
        assert not monitor.register("LOOP_SBX_" + "d" * 32)
    finally:
        monitor.close()


def test_live_monitor_rejects_stopped_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped log stream is detected before an attempt is registered."""
    process = StreamProcess()
    monkeypatch.setattr(macos.subprocess, "Popen", MagicMock(return_value=process))
    monitor = macos.DenialMonitor(101)
    process.finish()
    assert not monitor.healthy()
    assert not monitor.register("LOOP_SBX_" + "d" * 32)
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
