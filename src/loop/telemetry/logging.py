"""Configure independent, minimized operational logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from .. import constants
from ..utils import PrivateRotatingTextFile, sha256_digest


def import_legacy_operational_log(source: Path | str, destination: Path | str) -> int:
    """Idempotently append legacy operational records to the central log.

    Args:
        source (Path | str): Legacy operational log path.
        destination (Path | str): Central operational log path.

    Returns:
        int: Number of newly imported records.
    """
    legacy = Path(source).resolve()
    target = Path(destination).resolve()
    if not legacy.is_file():
        return 0
    target.parent.mkdir(mode=constants.PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    imported = 0
    lock = FileLock(f"{target}.lock", mode=constants.PRIVATE_FILE_MODE)
    with lock:
        known = set()
        for candidate in target.parent.glob(f"{target.name}*"):
            if not candidate.is_file() or candidate.name.endswith(".lock"):
                continue
            with candidate.open(encoding="utf-8", errors="replace") as existing:
                for line in existing:
                    try:
                        migration_id = json.loads(line).get("migration_id")
                    except (AttributeError, json.JSONDecodeError):
                        continue
                    if migration_id is not None:
                        known.add(migration_id)
        with (
            legacy.open(encoding="utf-8", errors="replace") as input_file,
            target.open("a", encoding="utf-8") as output,
        ):
            for line_number, line in enumerate(input_file, 1):
                migration_id = sha256_digest(f"{legacy}:{line_number}:{line}")
                if migration_id in known:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = {"message": line.rstrip("\r\n"), "level": "UNKNOWN"}
                if not isinstance(value, dict):
                    value = {"message": value, "level": "UNKNOWN"}
                value["migration_id"] = migration_id
                output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
                imported += 1
        target.chmod(constants.PRIVATE_FILE_MODE)
    return imported


class SafeRotatingFileHandler(logging.Handler):
    """Write rotating private log records with a content-free failure fallback.

    Args:
        path (Path | str): Operational log destination.
        max_bytes (int): Maximum active-file size before the next record rotates it.
        backup_count (int): Number of numbered archives to retain.
    """

    _output: PrivateRotatingTextFile

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int = constants.DEFAULT_OPERATIONAL_LOG_BYTES,
        backup_count: int = constants.DEFAULT_OPERATIONAL_LOG_BACKUPS,
    ) -> None:
        super().__init__()
        self._output = PrivateRotatingTextFile(
            path,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
        self._output.prepare()

    def emit(self, record: logging.LogRecord) -> None:
        """Write one formatted record without exposing handler failures."""
        try:
            self._output.append(self.format(record) + "\n")
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Report a failed log write without exposing the record or exception."""
        del record
        try:
            sys.stderr.write("Operational logging handler failed\n")
        except Exception:  # noqa: BLE001,S110  # pylint: disable=broad-exception-caught
            pass


class SafeOperationalFormatter(logging.Formatter):
    """Format fixed log messages and allowlisted structured diagnostic fields."""

    _FIELDS = (
        "event.name",
        "error.type",
        "error.instance",
        "error.operation",
        "exception.type",
        "telemetry.component",
        "telemetry.failure",
        "workspace_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Return one minimized JSON log line.

        Args:
            record (logging.LogRecord): Standard-library record to format.

        Returns:
            str: Single-line JSON without exception contents or tracebacks.
        """
        value = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "timestamp_ns": int(record.created * 1_000_000_000),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage().replace("\r", "\\r").replace("\n", "\\n"),
        }
        for field in self._FIELDS:
            if field in record.__dict__ and record.__dict__[field] is not None:
                item = record.__dict__[field]
                value[field] = (
                    item if isinstance(item, (str, int, float, bool)) else type(item).__name__
                )
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class WorkspaceFilter(logging.Filter):
    """Stamp one sanitized workspace identity on operational records."""

    def __init__(self, workspace_id: str) -> None:
        super().__init__()
        self._workspace_id = workspace_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach workspace context and accept the record."""
        record.workspace_id = self._workspace_id
        return True


def configure_operational_logging(
    path: Path | str,
    *,
    level: str = constants.DEFAULT_OPERATIONAL_LOG_LEVEL,
    max_bytes: int | None = None,
    backup_count: int | None = None,
    workspace_id: str | None = None,
) -> logging.Handler | None:
    """Install an owner-local rotating handler without disabling stderr fallback.

    Args:
        path (Path | str): Operational log destination.
        level (str): Minimum logging severity.
        max_bytes (int | None): Maximum active log size before rotation.
        backup_count (int | None): Number of rotated archives to retain.
        workspace_id (str | None): Active workspace stamped on emitted records.

    Returns:
        logging.Handler | None: Installed handler, or ``None`` when setup failed.
    """
    destination = Path(path)
    resolved_max_bytes = (
        max_bytes if max_bytes is not None else constants.DEFAULT_OPERATIONAL_LOG_BYTES
    )
    resolved_backup_count = (
        backup_count if backup_count is not None else constants.DEFAULT_OPERATIONAL_LOG_BACKUPS
    )
    try:
        handler = SafeRotatingFileHandler(
            destination,
            max_bytes=resolved_max_bytes,
            backup_count=resolved_backup_count,
        )
        handler.addFilter(logging.Filter("loop"))
        if workspace_id is not None:
            handler.addFilter(WorkspaceFilter(workspace_id))
        handler.setFormatter(SafeOperationalFormatter())
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(level)
        return handler
    except (OSError, ValueError):
        logging.getLogger(__name__).critical(
            "Operational logging initialization failed",
            extra={"error.type": "logging.initialization_failed"},
        )
        return None
