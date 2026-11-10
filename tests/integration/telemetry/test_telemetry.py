"""Verify native telemetry writer scheduling and SQLite lifecycle."""

import logging
import queue
import sqlite3
import threading

import pytest

from loop.telemetry import (
    MemoryTelemetryAdapter,
    SQLiteTelemetryAdapter,
    Telemetry,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def untimed_queue(monkeypatch):
    """Block on queue events instead of advancing a real telemetry interval."""
    get = queue.Queue.get

    def receive(target, block=True, timeout=None):
        """Receive an event without an elapsed-time wait."""
        return get(target, block=block)

    monkeypatch.setattr(queue.Queue, "get", receive)


class BlockingAdapter(MemoryTelemetryAdapter):
    """Block the first background write until a test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def write_batch(self, records):
        """Wait before retaining the first offered batch."""
        self.started.set()
        self.release.wait()
        super().write_batch(records)


class SignalingAdapter(MemoryTelemetryAdapter):
    """Signal when the background writer persists a batch."""

    def __init__(self) -> None:
        super().__init__()
        self.written = threading.Event()

    def write_batch(self, records):
        """Retain an offered batch and signal its completion."""
        super().write_batch(records)
        self.written.set()


def test_activity_overload_drops_only_activity_and_preserves_priority_records():
    """A saturated activity queue drops activity while retaining priority records."""
    adapter = BlockingAdapter()
    telemetry = Telemetry(adapter, queue_capacity=1, batch_size=1, flush_seconds=10)
    telemetry.activity("writing")
    adapter.started.wait()
    telemetry.activity("queued")
    telemetry.activity("dropped")
    telemetry.audit("preserved", decision="allow")
    adapter.release.set()
    assert telemetry.dropped_activity == 1
    assert telemetry.close(None)
    event_names = [record.event_name for record in adapter.records]
    assert event_names[0] == "writing"
    assert set(event_names[1:]) == {"preserved", "queued"}


def test_sqlite_lifecycle_runs_in_writer_thread_and_closes_cleanly(tmp_path, caplog):
    """SQLite telemetry owns writes, flushes, and closure in its writer thread."""
    path = tmp_path / "telemetry.db"
    adapter = SQLiteTelemetryAdapter(path, workspace_id="workspace")
    telemetry = Telemetry(adapter, workspace_id="workspace", batch_size=1, flush_seconds=1)
    with caplog.at_level(logging.ERROR, logger="loop.telemetry.telemetry"):
        telemetry.activity("persisted")
        assert telemetry.close(None)
    assert "Telemetry operation failed" not in caplog.text

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT event_name FROM telemetry_records").fetchall() == [
            ("persisted",)
        ]
    finally:
        connection.close()


def test_shutdown_barrier_rejects_late_activity_and_coordinates_lifecycle_calls():
    """Closing rejects later activity while concurrent flush and close calls wait safely."""
    adapter = BlockingAdapter()
    telemetry = Telemetry(adapter, batch_size=1, flush_seconds=10)
    telemetry.activity("accepted")
    adapter.started.wait()

    closing = threading.Thread(target=telemetry.close, args=(None,))
    closing.start()
    assert telemetry.close(0) is False
    assert telemetry.flush(0) is False
    telemetry.activity("ignored")
    telemetry.audit("also.ignored", decision="allow")
    telemetry.error("ignored.error", error_type="test.ignored")
    telemetry.error("ignored.error", error_type="test.ignored", exception=RuntimeError())

    adapter.release.set()
    closing.join()
    assert not closing.is_alive()
    assert telemetry.close(None)
    assert [record.event_name for record in adapter.records] == ["accepted"]


def test_flush_timeout_close_retry_and_closed_emission_are_safe():
    """Blocked persistence times out, later closes cleanly, and ignores post-close records."""
    adapter = BlockingAdapter()
    telemetry = Telemetry(adapter, queue_capacity=2, batch_size=1, flush_seconds=10)
    telemetry.activity("writing")
    adapter.started.wait()
    telemetry.activity("queued")
    assert telemetry.flush(0) is False
    assert telemetry.close(0) is False
    adapter.release.set()
    assert telemetry.close(None)
    count = len(adapter.records)
    telemetry.activity("ignored")
    assert len(adapter.records) == count


def test_partial_batches_flush_on_interval_without_an_explicit_barrier(monkeypatch):
    """The background interval persists low-volume batches before shutdown."""
    get = queue.Queue.get
    received = False
    expired = False

    def receive(target, *args, **kwargs):
        """Deliver one event, then simulate its interval expiring immediately."""
        nonlocal received, expired
        if received and not expired:
            expired = True
            raise queue.Empty
        item = get(target, *args, **kwargs)
        received = True
        return item

    monkeypatch.setattr(queue.Queue, "get", receive)
    adapter = SignalingAdapter()
    telemetry = Telemetry(adapter, batch_size=10, flush_seconds=10)
    telemetry.activity("interval")
    adapter.written.wait()
    assert len(adapter.records) == 1
    assert telemetry.close(None)


def test_idle_writer_and_zero_timeout_shutdown_are_recoverable(monkeypatch):
    """Idle polling stays healthy and a zero-timeout stop can be completed by a later close."""
    idle_poll = threading.Event()
    original_get = queue.Queue.get
    first_poll = True

    def return_immediate_empty_once(target, *args, **kwargs):
        """Advance the writer through one idle interval without waiting for real time."""
        nonlocal first_poll
        if first_poll:
            first_poll = False
            idle_poll.set()
            raise queue.Empty
        return original_get(target, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "get", return_immediate_empty_once)
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=10)
    idle_poll.wait()
    stopped = telemetry.close(0)
    if not stopped:
        assert telemetry.close(None)
