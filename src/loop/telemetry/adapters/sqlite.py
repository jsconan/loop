"""Persist telemetry records and payloads in SQLite."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from ... import constants
from ..models import TelemetryRecord
from ..policy import thaw

_SCHEMA_VERSION = 1


class SQLiteTelemetryAdapter:
    """Persist metadata and potentially large payloads in a local SQLite database.

    Args:
        path (Path | str): SQLite telemetry database path.
        workspace_id (str): Workspace identity linked from every stored record.
        busy_timeout_ms (int): Maximum milliseconds to wait for a database lock.
    """

    _path: Path
    _busy_timeout_ms: int
    _connection: sqlite3.Connection | None
    _workspace_id: str

    def __init__(
        self,
        path: Path | str,
        *,
        workspace_id: str,
        busy_timeout_ms: int = constants.DEFAULT_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS,
    ) -> None:
        self._path = Path(path)
        if not workspace_id:
            raise ValueError("Workspace identifier must not be empty.")
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        self._workspace_id = workspace_id
        self._busy_timeout_ms = busy_timeout_ms
        self._connection: sqlite3.Connection | None = None
        self._initialize()

    @property
    def path(self) -> Path:
        """Return the configured database path.

        Returns:
            Path: Telemetry database path.
        """
        return self._path

    def write_batch(self, records: Sequence[TelemetryRecord]) -> None:
        """Persist one record batch in a single transaction.

        Args:
            records (Sequence[TelemetryRecord]): Immutable records to persist.
        """
        if not records:
            return
        connection = self._connect()
        with connection:
            for record in records:
                if record.workspace_id != self._workspace_id:
                    raise ValueError(
                        f"Telemetry record workspace '{record.workspace_id}' does not match "
                        f"storage workspace '{self._workspace_id}'."
                    )
                payload_id = None
                if record.payload is not None:
                    payload = json.dumps(
                        thaw(record.payload), ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    cursor = connection.execute(
                        "INSERT INTO telemetry_payloads(encoding, size_bytes, payload) "
                        "VALUES (?, ?, ?)",
                        ("json", len(payload), payload),
                    )
                    payload_id = cursor.lastrowid
                connection.execute(
                    """
                    INSERT INTO telemetry_records(
                        record_id, timestamp_ns, observed_ns, signal, event_name, severity,
                        workspace_id, session_id, message_sequence, event_sequence, trace_id,
                        span_id, parent_span_id, attributes, payload_id, payload_sha256,
                        schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.timestamp_ns,
                        record.observed_timestamp_ns,
                        record.signal,
                        record.event_name,
                        record.severity,
                        record.workspace_id,
                        record.session_id,
                        record.message_sequence,
                        record.event_sequence,
                        record.trace_id,
                        record.span_id,
                        record.parent_span_id,
                        json.dumps(
                            thaw(record.attributes),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        payload_id,
                        record.payload_sha256,
                        record.schema_version,
                    ),
                )

    def flush(self) -> None:
        """Commit pending adapter work."""
        if self._connection is not None:
            self._connection.commit()

    def close(self) -> None:
        """Checkpoint and close the writer connection."""
        if self._connection is not None:
            self._connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self._connection.close()
            self._connection = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        connection = sqlite3.connect(self._path)
        self._path.chmod(constants.PRIVATE_FILE_MODE)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS telemetry_payloads (
                payload_id INTEGER PRIMARY KEY,
                encoding TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS telemetry_records (
                record_id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                observed_ns INTEGER NOT NULL,
                signal TEXT NOT NULL,
                event_name TEXT NOT NULL,
                severity TEXT,
                workspace_id TEXT,
                session_id TEXT,
                message_sequence INTEGER,
                event_sequence INTEGER NOT NULL,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT,
                attributes TEXT NOT NULL,
                payload_id INTEGER,
                payload_sha256 TEXT,
                schema_version INTEGER NOT NULL,
                FOREIGN KEY(payload_id) REFERENCES telemetry_payloads(payload_id)
            );
            CREATE INDEX IF NOT EXISTS ix_telemetry_session_sequence
                ON telemetry_records(session_id, event_sequence);
            CREATE INDEX IF NOT EXISTS ix_telemetry_trace_time
                ON telemetry_records(trace_id, timestamp_ns);
            CREATE INDEX IF NOT EXISTS ix_telemetry_span ON telemetry_records(span_id);
            CREATE INDEX IF NOT EXISTS ix_telemetry_parent ON telemetry_records(parent_span_id);
            CREATE INDEX IF NOT EXISTS ix_telemetry_event_time
                ON telemetry_records(event_name, timestamp_ns);
            """
        )
        self._connection = connection
        return connection

    def _initialize(self) -> None:
        """Initialize telemetry storage for its linked workspace."""
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if schema_version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"Telemetry schema version {schema_version} is newer than supported "
                    f"version {_SCHEMA_VERSION}."
                )
            if schema_version < _SCHEMA_VERSION:
                self._migrate_records(connection, self._workspace_id)
                connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._connection = None

    @staticmethod
    def _migrate_records(connection: sqlite3.Connection, workspace_id: str) -> None:
        """Add workspace identity and remove historical workspace paths from records."""
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(telemetry_records)").fetchall()
        }
        if "workspace_id" not in columns:
            connection.execute("ALTER TABLE telemetry_records ADD COLUMN workspace_id TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_telemetry_workspace_time "
            "ON telemetry_records(workspace_id, timestamp_ns) WHERE workspace_id IS NOT NULL"
        )
        rows = connection.execute(
            "SELECT record_id, attributes FROM telemetry_records "
            "WHERE workspace_id IS NULL OR attributes LIKE '%workspace.root%'"
        ).fetchall()
        for record_id, encoded_attributes in rows:
            sanitized_attributes = encoded_attributes
            try:
                attributes = json.loads(encoded_attributes)
            except (TypeError, json.JSONDecodeError):
                attributes = None
            if isinstance(attributes, dict) and "workspace.root" in attributes:
                del attributes["workspace.root"]
                sanitized_attributes = json.dumps(
                    attributes, ensure_ascii=False, separators=(",", ":")
                )
            connection.execute(
                "UPDATE telemetry_records SET workspace_id = COALESCE(workspace_id, ?), "
                "attributes = ? WHERE record_id = ?",
                (workspace_id, sanitized_attributes, record_id),
            )
