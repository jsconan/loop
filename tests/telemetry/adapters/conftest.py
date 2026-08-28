"""Provide representative telemetry records to adapter suites."""

from types import MappingProxyType

import pytest

from loop.telemetry import TelemetryRecord


@pytest.fixture
def telemetry_record():
    """Build one representative immutable telemetry record."""
    return TelemetryRecord(
        record_id="tel_test",
        timestamp_ns=1,
        observed_timestamp_ns=2,
        signal="trace",
        event_name="test.event",
        event_sequence=3,
        session_id="session",
        message_sequence=4,
        trace_id="a" * 32,
        span_id="b" * 16,
        parent_span_id="c" * 16,
        severity="info",
        attributes=MappingProxyType({"model": "test"}),
        payload=MappingProxyType({"value": "complete"}),
        payload_sha256="digest",
    )
