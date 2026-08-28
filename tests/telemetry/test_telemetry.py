"""Tests for the correlated telemetry facade and background writer."""

import logging
import queue
import threading
from unittest.mock import Mock

import pytest

from loop.telemetry import (
    MemoryTelemetryAdapter,
    Telemetry,
    TelemetryContext,
    get_telemetry,
    set_telemetry,
    telemetry_activity,
    telemetry_audit,
    telemetry_error,
    telemetry_span,
    telemetry_trace_event,
)
from loop.utils import payload_digest


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


def test_telemetry_records_all_signals_with_nested_w3c_correlation():
    """Facade records minimized signals and complete traces with nested parent identifiers."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    with telemetry.context(session_id="session", message_sequence=2) as root:
        telemetry.activity("activity", prompt="excluded", count=1)
        with telemetry.context() as child:
            telemetry.error(
                "failure",
                error_type="test.failed",
                exception=RuntimeError("private"),
            )
            telemetry.audit("decision", decision="allow")
            payload = {"input": "complete"}
            telemetry.trace_event(
                "request", payload=payload, payload_sha256=payload_digest(payload)
            )
    assert telemetry.flush(1)
    assert telemetry.close(1)
    assert telemetry.close(1)
    assert telemetry.flush(1)

    records = adapter.records
    assert [record.signal for record in records] == ["activity", "error", "audit", "trace"]
    assert "prompt" not in records[0].attributes
    assert records[0].trace_id == root.trace_id
    assert records[1].span_id == child.span_id
    assert records[1].parent_span_id == root.span_id
    assert len(root.trace_id) == 32
    assert len(root.span_id) == 16
    assert records[-1].payload_sha256 == payload_digest({"input": "complete"})


def test_telemetry_generates_public_w3c_identifiers():
    """Public helpers generate correctly sized lowercase hexadecimal identifiers."""
    trace_id = Telemetry.trace_id()
    span_id = Telemetry.span_id()

    assert len(trace_id) == 32
    assert len(span_id) == 16
    assert int(trace_id, 16) >= 0
    assert int(span_id, 16) >= 0


def test_explicit_context_binding_and_process_facade_lifecycle():
    """Explicit bindings restore prior state and the process facade can be replaced safely."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    context = TelemetryContext(session_id="session", trace_id="a" * 32, span_id="b" * 16)
    token = telemetry.bind(context)
    telemetry.activity("bound")
    telemetry.reset(token)
    set_telemetry(telemetry)
    assert get_telemetry() is telemetry
    set_telemetry(None)
    assert get_telemetry() is None
    assert telemetry.close(1)
    assert adapter.records[0].session_id == "session"


def test_process_helpers_are_no_ops_without_a_facade_and_delegate_when_active():
    """Process helpers centralize optional dispatch and correlation span behavior."""
    with telemetry_span(session_id="ignored") as context:
        assert context is None
        telemetry_activity("ignored")
        telemetry_error("ignored", error_type="ignored")
        telemetry_audit("ignored")
        telemetry_trace_event("ignored")

    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        with telemetry_span(session_id="session", message_sequence=3) as context:
            assert context is not None
            telemetry_activity("activity", count=1)
            telemetry_error("error", error_type="test.failed", exception=RuntimeError())
            telemetry_audit("audit", decision="allow")
            telemetry_trace_event(
                "trace", payload_factory=lambda: {"safe": True}, digest_payload=True
            )
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert [record.signal for record in adapter.records] == [
        "activity",
        "error",
        "audit",
        "trace",
    ]
    assert all(record.session_id == "session" for record in adapter.records)
    assert all(record.message_sequence == 3 for record in adapter.records)
    assert adapter.records[-1].payload_sha256 == payload_digest({"safe": True})


def test_process_helpers_keep_low_level_activity_and_isolate_deferred_trace_failures(caplog):
    """Low-level activity survives without telemetry and deferred payload failures stay isolated."""
    with caplog.at_level(logging.INFO, logger="loop.operational"):
        telemetry_activity("startup.safe")
    assert "Operational activity" in caplog.text

    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        with caplog.at_level(logging.ERROR):
            telemetry_trace_event(
                "trace.failed",
                payload_factory=lambda: (_ for _ in ()).throw(RuntimeError("private")),
                digest_payload=True,
            )
        assert "Telemetry operation failed" in caplog.text
        assert "private" not in caplog.text
        assert telemetry.close(1)
    finally:
        set_telemetry(None)


def test_facade_rejects_invalid_configuration_and_unsupported_trace_values(caplog):
    """Invalid setup fails early while malformed records are isolated and diagnosed safely."""
    with pytest.raises(ValueError, match="must be positive"):
        Telemetry(queue_capacity=0)
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    with caplog.at_level(logging.ERROR):
        telemetry.trace_event("invalid", payload=object())
    assert "Telemetry operation failed" in caplog.text
    assert telemetry.close(1)


def test_activity_overload_drops_only_activity_and_priority_falls_back_synchronously(
    monkeypatch,
):
    """A saturated queue drops activity but synchronously offers priority records."""
    adapter = BlockingAdapter()
    telemetry = Telemetry(adapter, queue_capacity=1, batch_size=1, flush_seconds=10)
    telemetry.activity("writing")
    assert adapter.started.wait(1)
    telemetry.activity("queued")
    telemetry.activity("dropped")
    fallback_started = threading.Event()
    original_put = queue.Queue.put

    def reject_priority_record(target, item, *args, **kwargs):
        """Force the priority record through the synchronous overload path."""
        if getattr(item, "event_name", None) == "preserved":
            fallback_started.set()
            raise queue.Full
        return original_put(target, item, *args, **kwargs)

    monkeypatch.setattr(queue.Queue, "put", reject_priority_record)
    audit = threading.Thread(
        target=telemetry.audit,
        args=("preserved",),
        kwargs={"decision": "allow"},
    )
    audit.start()
    assert fallback_started.wait(1)
    adapter.release.set()
    audit.join()
    assert telemetry.dropped_activity == 1
    assert telemetry.close(1)
    event_names = [record.event_name for record in adapter.records]
    assert event_names[0] == "writing"
    assert set(event_names[1:]) == {"preserved", "queued"}


def test_adapter_failures_and_flush_timeouts_do_not_escape(caplog):
    """Adapter and blocked-queue failures are diagnosed without reaching instrumented code."""
    adapter = Mock()
    adapter.write_batch.side_effect = OSError("private disk detail")
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    with caplog.at_level(logging.ERROR):
        telemetry.audit("record", decision="allow")
        assert telemetry.flush(1) is False
    assert "Telemetry operation failed" in caplog.text
    assert "private disk detail" not in caplog.text
    assert telemetry.close(1) is False


def test_flush_timeout_close_retry_and_closed_emission_are_safe():
    """Blocked persistence times out, later closes cleanly, and ignores post-close records."""
    adapter = BlockingAdapter()
    telemetry = Telemetry(adapter, queue_capacity=2, batch_size=1, flush_seconds=10)
    telemetry.activity("writing")
    assert adapter.started.wait(1)
    telemetry.activity("queued")
    assert telemetry.flush(0) is False
    assert telemetry.close(0) is False
    adapter.release.set()
    assert telemetry.close(1)
    count = len(adapter.records)
    telemetry.activity("ignored")
    assert len(adapter.records) == count


def test_partial_batches_flush_on_interval_without_an_explicit_barrier():
    """The background interval persists low-volume batches before shutdown."""
    adapter = SignalingAdapter()
    telemetry = Telemetry(adapter, batch_size=10, flush_seconds=0.01)
    telemetry.activity("interval")
    assert adapter.written.wait(1)
    assert len(adapter.records) == 1
    assert telemetry.close(1)


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
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    assert idle_poll.wait(1)
    stopped = telemetry.close(0)
    if not stopped:
        assert telemetry.close(1)


def test_adapter_flush_and_close_failures_are_isolated(caplog):
    """Adapter lifecycle exceptions return failure and use independent safe diagnostics."""
    flush_adapter = Mock()
    flush_adapter.flush.side_effect = OSError("private flush")
    flushing = Telemetry(flush_adapter, flush_seconds=0.01)
    with caplog.at_level(logging.ERROR):
        assert flushing.flush(1) is False
    flush_adapter.flush.side_effect = None
    assert flushing.close(1)

    close_adapter = Mock()
    close_adapter.close.side_effect = OSError("private close")
    closing = Telemetry(close_adapter, flush_seconds=0.01)
    with caplog.at_level(logging.ERROR):
        assert closing.close(1) is False
    assert closing.close(1)
    assert "private flush" not in caplog.text
    assert "private close" not in caplog.text
