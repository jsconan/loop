"""Discard telemetry records through a no-op adapter."""

from collections.abc import Sequence

from ..models import TelemetryRecord


class NoOpTelemetryAdapter:
    """Discard every telemetry record."""

    def write_batch(self, records: Sequence[TelemetryRecord]) -> None:
        """Discard an ordered record batch.

        Args:
            records (Sequence[TelemetryRecord]): Records intentionally ignored.
        """

    def flush(self) -> None:
        """Complete immediately because no buffers exist."""

    def close(self) -> None:
        """Complete immediately because no resources exist."""
