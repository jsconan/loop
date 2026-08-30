"""Configure independent, minimized operational logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .. import constants
from ..utils import PrivateRotatingTextFile


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


def configure_operational_logging(
    path: Path | str,
    *,
    level: str = constants.DEFAULT_OPERATIONAL_LOG_LEVEL,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Handler | None:
    """Install an owner-local rotating handler without disabling stderr fallback.

    Args:
        path (Path | str): Operational log destination.

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
