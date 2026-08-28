"""Configure independent, minimized operational logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .. import constants


class SafeRotatingFileHandler(RotatingFileHandler):
    """Report handler failures through a content-free stderr fallback."""

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


def configure_operational_logging(path: Path | str) -> logging.Handler | None:
    """Install an owner-local rotating handler without disabling stderr fallback.

    Args:
        path (Path | str): Operational log destination.

    Returns:
        logging.Handler | None: Installed handler, or ``None`` when setup failed.
    """
    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.chmod(constants.PRIVATE_DIRECTORY_MODE)
        handler = SafeRotatingFileHandler(
            destination,
            maxBytes=constants.DEFAULT_OPERATIONAL_LOG_BYTES,
            backupCount=constants.DEFAULT_OPERATIONAL_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.addFilter(logging.Filter("loop"))
        handler.setFormatter(SafeOperationalFormatter())
        destination.chmod(constants.PRIVATE_FILE_MODE)
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(constants.DEFAULT_OPERATIONAL_LOG_LEVEL)
        return handler
    except (OSError, ValueError):
        logging.getLogger(__name__).critical(
            "Operational logging initialization failed",
            extra={"error.type": "logging.initialization_failed"},
        )
        return None
