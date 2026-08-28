"""Retain telemetry records in process memory."""

from collections.abc import Sequence
from threading import Lock

from ..models import TelemetryRecord


class MemoryTelemetryAdapter:
    """Retain records in memory for embedding and deterministic tests."""

    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []
        self._lock = Lock()

    @property
    def records(self) -> tuple[TelemetryRecord, ...]:
        """Return a stable snapshot of persisted records.

        Returns:
            tuple[TelemetryRecord, ...]: Records in adapter write order.
        """
        with self._lock:
            return tuple(self._records)

    def write_batch(self, records: Sequence[TelemetryRecord]) -> None:
        """Append one record batch atomically.

        Args:
            records (Sequence[TelemetryRecord]): Immutable records to retain.
        """
        with self._lock:
            self._records.extend(records)

    def flush(self) -> None:
        """Complete immediately because writes are synchronous."""

    def close(self) -> None:
        """Complete immediately because no external resources exist."""
