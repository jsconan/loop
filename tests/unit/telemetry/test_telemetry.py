"""Tests for the correlated telemetry facade and background writer."""

import logging
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
    assert records[0].severity == "debug"
    assert "prompt" not in records[0].attributes
    assert records[0].trace_id == root.trace_id
    assert records[1].span_id == child.span_id
    assert records[1].parent_span_id == root.span_id
    assert len(root.trace_id) == 32
    assert len(root.span_id) == 16
    assert records[-1].payload_sha256 == payload_digest({"input": "complete"})


def test_telemetry_attributes_every_signal_to_its_workspace():
    """Workspace-aware telemetry attaches identity without disclosing its path."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01, workspace_id="workspace")

    telemetry.activity("activity")
    telemetry.trace_event("trace")
    assert telemetry.close(1)

    assert [record.workspace_id for record in adapter.records] == ["workspace", "workspace"]
    assert all("workspace.root" not in record.attributes for record in adapter.records)
    assert all(record.schema_version == 2 for record in adapter.records)


def test_trace_metadata_is_inherited_by_nested_traces_and_remains_scoped():
    """Trace metadata reaches descendant traces, supports local overrides, and is reset."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    with telemetry.context(metadata={"conversation.id": "root", "feature.variant": "v1"}):
        telemetry.trace_event("root")
        with telemetry.context(metadata={"feature.variant": "v2"}):
            telemetry.trace_event("child", **{"conversation.id": "event"})
        telemetry.activity("activity")
    telemetry.trace_event("outside")
    assert telemetry.close(1)

    root, child, activity, outside = adapter.records
    assert dict(root.attributes) == {"conversation.id": "root", "feature.variant": "v1"}
    assert dict(child.attributes) == {"conversation.id": "event", "feature.variant": "v2"}
    assert not dict(activity.attributes)
    assert not dict(outside.attributes)


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


def test_process_helpers_preserve_activity_severity_and_isolate_trace_failures(caplog):
    """Activity preserves its level in both outputs and trace preparation failures stay isolated."""
    with caplog.at_level(logging.DEBUG, logger="loop.operational"):
        telemetry_activity("startup.safe")
        telemetry_activity("startup.visible", severity="info")
    assert "Operational activity" in caplog.text
    assert [record.levelno for record in caplog.records[-2:]] == [logging.DEBUG, logging.INFO]

    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        telemetry_activity("persisted", severity="warning")
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
    assert adapter.records[0].severity == "warning"


def test_facade_rejects_invalid_configuration_and_unsupported_trace_values(caplog):
    """Invalid setup fails early while malformed records are isolated and diagnosed safely."""
    with pytest.raises(ValueError, match="must be positive"):
        Telemetry(queue_capacity=0)
    telemetry = Telemetry(MemoryTelemetryAdapter(), flush_seconds=0.01)
    with caplog.at_level(logging.ERROR):
        telemetry.trace_event("invalid", payload=object())
    assert "Telemetry operation failed" in caplog.text
    assert telemetry.close(1)


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


def test_activity_overload_preserves_priority_records():
    """Queue saturation drops activity immediately while retaining audit evidence."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, queue_capacity=1, batch_size=1)
    telemetry.activity("accepted")
    telemetry.activity("dropped")
    telemetry.audit("preserved", decision="allow")
    assert telemetry.dropped_activity == 1
    assert telemetry.close()
    assert [record.event_name for record in adapter.records] == ["accepted", "preserved"]


def test_shutdown_rejects_new_records_before_writer_settlement(telemetry_writer):
    """An admitted close rejects late records and exposes unfinished lifecycle results."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter)
    telemetry.activity("accepted")
    assert telemetry.flush(0) is False
    assert telemetry.close(0) is False
    assert telemetry.close(0) is False
    assert telemetry.flush(0) is False
    telemetry.activity("ignored")
    telemetry.audit("ignored", decision="deny")
    telemetry.error("ignored", error_type="test.ignored")
    telemetry.error("ignored", error_type="test.ignored", exception=RuntimeError())
    assert telemetry.close()
    assert [record.event_name for record in adapter.records] == ["accepted"]
    telemetry.activity("ignored.after.close")
    assert len(adapter.records) == 1


def test_writer_flushes_partial_batches_on_an_empty_queue_response(telemetry_writer):
    """Explicit empty-queue responses flush pending records and tolerate an idle writer."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, batch_size=10)
    telemetry.activity("partial")
    telemetry_writer.empty_reads.update({0, 2})
    assert telemetry.close()
    assert [record.event_name for record in adapter.records] == ["partial"]
