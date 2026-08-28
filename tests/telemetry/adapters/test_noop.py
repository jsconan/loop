"""Tests for the no-op telemetry persistence adapter."""

from loop.telemetry import NoOpTelemetryAdapter


def test_noop_adapter_follows_the_common_lifecycle(telemetry_record):
    """No-op persistence accepts and discards records through its full lifecycle."""
    adapter = NoOpTelemetryAdapter()
    adapter.write_batch((telemetry_record,))
    adapter.flush()
    adapter.close()
