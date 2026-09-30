"""Tests for the in-memory telemetry persistence adapter."""

from loop.telemetry import MemoryTelemetryAdapter


def test_memory_adapter_retains_stable_ordered_snapshots(telemetry_record):
    """Memory persistence retains accepted records in write order."""
    adapter = MemoryTelemetryAdapter()
    adapter.write_batch((telemetry_record,))
    adapter.flush()
    adapter.close()

    assert adapter.records == (telemetry_record,)
