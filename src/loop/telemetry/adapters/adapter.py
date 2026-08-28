"""Define the telemetry persistence adapter contract."""

from collections.abc import Sequence
from typing import Protocol

from ..models import TelemetryRecord


class TelemetryAdapter(Protocol):
    """Accept normalized records without knowing instrumentation call sites."""

    def write_batch(self, records: Sequence[TelemetryRecord]) -> None:
        """Persist one ordered record batch.

        Args:
            records (Sequence[TelemetryRecord]): Immutable records to persist atomically.
        """

    def flush(self) -> None:
        """Flush adapter-owned buffers."""

    def close(self) -> None:
        """Release adapter-owned resources."""
