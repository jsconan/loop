"""Tests for independent minimized operational logging."""

import json
import logging
from unittest.mock import Mock

import loop.telemetry.logging as logging_module
from loop.telemetry.logging import SafeOperationalFormatter, configure_operational_logging


def test_safe_formatter_excludes_exception_contents_and_normalizes_fields():
    """Operational formatting emits one safe JSON line without traceback material."""
    record = logging.LogRecord("loop.test", logging.ERROR, __file__, 1, "Failed\nline", (), None)
    record.__dict__["error.type"] = "test.failed"
    record.__dict__["exception.type"] = "builtins.RuntimeError"
    record.__dict__["ignored"] = "private"

    value = json.loads(SafeOperationalFormatter().format(record))

    assert value["message"] == "Failed\\nline"
    assert value["exception.type"] == "builtins.RuntimeError"
    assert "ignored" not in value


def test_configure_operational_logging_writes_rotating_local_file(tmp_path):
    """Bootstrap configuration installs a working owner-local file handler."""
    handler = configure_operational_logging(tmp_path / ".loop" / "loop.log")
    logger = logging.getLogger("loop.operational.test")
    logger.error("Safe failure", extra={"error.type": "test.failed"})
    handler.flush()
    logging.getLogger().removeHandler(handler)
    handler.close()

    value = json.loads((tmp_path / ".loop" / "loop.log").read_text(encoding="utf-8"))
    assert value["message"] == "Safe failure"
    assert value["error.type"] == "test.failed"
    assert (tmp_path / ".loop").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / ".loop" / "loop.log").stat().st_mode & 0o777 == 0o600


def test_configure_operational_logging_falls_back_when_file_setup_fails(monkeypatch, caplog):
    """Bootstrap file failures are reported safely through normal logging fallback."""
    monkeypatch.setattr(
        logging_module,
        "SafeRotatingFileHandler",
        Mock(side_effect=OSError("private")),
    )

    with caplog.at_level(logging.CRITICAL):
        handler = configure_operational_logging("/unavailable/loop.log")

    assert handler is None
    assert "Operational logging initialization failed" in caplog.text
    assert "private" not in caplog.text


def test_handler_failures_use_content_free_stderr_fallback(tmp_path, capsys):
    """A broken operational file handler reports failure without exposing its log record."""
    handler = logging_module.SafeRotatingFileHandler(tmp_path / "loop.log")
    record = logging.LogRecord(
        "loop.test",
        logging.ERROR,
        __file__,
        1,
        "private payload",
        (),
        None,
    )

    handler.handleError(record)
    handler.close()

    output = capsys.readouterr().err
    assert output == "Operational logging handler failed\n"
    assert "private payload" not in output


def test_handler_failure_fallback_never_raises(tmp_path, monkeypatch):
    """A broken stderr fallback cannot propagate into the instrumented application."""
    handler = logging_module.SafeRotatingFileHandler(tmp_path / "loop.log")
    monkeypatch.setattr(
        logging_module.sys.stderr,
        "write",
        Mock(side_effect=OSError("stderr unavailable")),
    )

    handler.handleError(
        logging.LogRecord("loop.test", logging.ERROR, __file__, 1, "failure", (), None)
    )
    handler.close()
