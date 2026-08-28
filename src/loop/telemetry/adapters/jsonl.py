"""Persist telemetry records as canonical JSON Lines."""

import json
from collections.abc import Sequence
from pathlib import Path

from ... import constants
from ..models import TelemetryRecord
from ..policy import thaw


class JSONLTelemetryAdapter:
    """Append canonical telemetry objects to a local JSON Lines file.

    Args:
        path (Path | str): Destination JSONL path created lazily on the first write.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def write_batch(self, records: Sequence[TelemetryRecord]) -> None:
        """Append one canonical line per record.

        Args:
            records (Sequence[TelemetryRecord]): Immutable records to append.
        """
        if not records:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        with self._path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(_record_json(record) + "\n")
        self._path.chmod(constants.PRIVATE_FILE_MODE)

    def flush(self) -> None:
        """Complete immediately because each batch closes its file."""

    def close(self) -> None:
        """Complete immediately because each batch closes its file."""


def _record_json(record: TelemetryRecord) -> str:
    value = {
        "record_id": record.record_id,
        "timestamp_ns": record.timestamp_ns,
        "observed_timestamp_ns": record.observed_timestamp_ns,
        "signal": record.signal,
        "event_name": record.event_name,
        "severity": record.severity,
        "session_id": record.session_id,
        "message_sequence": record.message_sequence,
        "event_sequence": record.event_sequence,
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "parent_span_id": record.parent_span_id,
        "attributes": thaw(record.attributes),
        "payload": thaw(record.payload),
        "payload_sha256": record.payload_sha256,
        "schema_version": record.schema_version,
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
