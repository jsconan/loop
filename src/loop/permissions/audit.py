"""Persist and export centralized permission audit records."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .. import constants


class SQLitePermissionAudit:
    """Store permission audit records for every workspace in one SQLite database.

    Args:
        path (Path | str): Central audit database path.
        busy_timeout_ms (int): Maximum milliseconds to wait for a database lock.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        busy_timeout_ms: int = constants.DEFAULT_TELEMETRY_SQLITE_BUSY_TIMEOUT_MS,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive.")
        self._path = Path(path).resolve()
        self._busy_timeout_ms = busy_timeout_ms
        self._initialize()

    @property
    def path(self) -> Path:
        """Return the audit database path.

        Returns:
            Path: Central audit database path.
        """
        return self._path

    def append(self, workspace_id: str, event_name: str, payload: dict[str, object]) -> None:
        """Append one minimized workspace-correlated audit record.

        Args:
            workspace_id (str): Stable workspace identifier.
            event_name (str): Audit event name.
            payload (dict[str, object]): Sanitized audit attributes.

        Raises:
            ValueError: If the workspace identifier or event name is empty.
        """
        if not workspace_id or not event_name:
            raise ValueError("Audit workspace and event name must be non-empty.")
        with closing(self._connect()) as connection, connection:  # pylint: disable=confusing-with-statement
            connection.execute(
                "INSERT INTO permission_audit_records("
                "record_id, timestamp_ns, workspace_id, process_id, event_name, payload_json, "
                "schema_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    time.time_ns(),
                    workspace_id,
                    os.getpid(),
                    event_name,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    1,
                ),
            )

    def export_jsonl(self, destination: Path | str, *, workspace_id: str | None = None) -> Path:
        """Export a stable, ordered JSONL audit snapshot.

        Args:
            destination (Path | str): New export file path.
            workspace_id (str | None): Optional workspace filter.

        Returns:
            Path: Created export file path.

        Raises:
            FileExistsError: If the destination already exists.
        """
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(f"Audit export already exists: {target}")
        query = (
            "SELECT record_id, timestamp_ns, workspace_id, process_id, event_name, payload_json, "
            "schema_version FROM permission_audit_records"
        )
        parameters: tuple[str, ...] = ()
        if workspace_id is not None:
            query += " WHERE workspace_id = ?"
            parameters = (workspace_id,)
        query += " ORDER BY timestamp_ns, record_id"
        target.parent.mkdir(parents=True, exist_ok=True)
        with (
            closing(self._connect()) as connection,
            target.open("x", encoding="utf-8") as output,
        ):
            for row in connection.execute(query, parameters):
                record = {
                    "record_id": row[0],
                    "timestamp_ns": row[1],
                    "workspace_id": row[2],
                    "process_id": row[3],
                    "event_name": row[4],
                    "payload": json.loads(row[5]),
                    "schema_version": row[6],
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
        return target

    def _connect(self) -> sqlite3.Connection:
        """Open a configured audit connection."""
        connection = sqlite3.connect(self.path)
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return connection

    def _initialize(self) -> None:
        """Create the private audit database and schema when absent."""
        self.path.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        self.path.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        with closing(self._connect()) as connection:  # noqa: SIM117
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS permission_audit_records (
                        record_id TEXT PRIMARY KEY,
                        timestamp_ns INTEGER NOT NULL,
                        workspace_id TEXT NOT NULL,
                        process_id INTEGER NOT NULL,
                        event_name TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        schema_version INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS permission_audit_workspace_time "
                    "ON permission_audit_records(workspace_id, timestamp_ns)"
                )
        self.path.chmod(constants.PRIVATE_FILE_MODE)
