"""Tests for independent minimized operational logging."""

import json
import logging
from unittest.mock import Mock

import loop.telemetry.logging as logging_module
from loop.telemetry.logging import (
    SafeOperationalFormatter,
    configure_operational_logging,
    import_legacy_operational_log,
)


def test_legacy_log_import_is_streaming_idempotent_and_normalizes_records(tmp_path):
    """Legacy JSON, scalar, and malformed lines append once using durable record identities."""
    source = tmp_path / "legacy.log"
    destination = tmp_path / "state" / "loop.log"
    assert import_legacy_operational_log(tmp_path / "missing", destination) == 0
    source.write_text('{"event.name":"one"}\n"scalar"\nmalformed\n', encoding="utf-8")
    destination.parent.mkdir()
    destination.write_text("not-json\n{}\n", encoding="utf-8")
    (destination.parent / "loop.log.1").write_text('{"migration_id":"known"}\n', encoding="utf-8")

    assert import_legacy_operational_log(source, destination, workspace_id="workspace") == 3
    moved = tmp_path / "moved.log"
    source.rename(moved)
    assert import_legacy_operational_log(moved, destination, workspace_id="workspace") == 0
    values = [json.loads(line) for line in destination.read_text().splitlines()[2:]]
    assert values[0]["event.name"] == "one"
    assert values[0]["workspace_id"] == "workspace"
    assert values[1]["message"] == "scalar"
    assert values[2]["message"] == "malformed"

    fallback_source = tmp_path / "unowned.log"
    fallback_destination = tmp_path / "fallback" / "loop.log"
    fallback_source.write_text("legacy\n", encoding="utf-8")
    assert import_legacy_operational_log(fallback_source, fallback_destination) == 1
    assert import_legacy_operational_log(fallback_source, fallback_destination) == 0


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
    handler = configure_operational_logging(
        tmp_path / ".loop" / "loop.log", workspace_id="workspace"
    )
    logger = logging.getLogger("loop.operational.test")
    logger.error("Safe failure", extra={"error.type": "test.failed"})
    handler.flush()
    logging.getLogger().removeHandler(handler)
    handler.close()

    value = json.loads((tmp_path / ".loop" / "loop.log").read_text(encoding="utf-8"))
    assert value["message"] == "Safe failure"
    assert value["error.type"] == "test.failed"
    assert value["workspace_id"] == "workspace"
    assert isinstance(value["timestamp_ns"], int)
    assert (tmp_path / ".loop").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / ".loop" / "loop.log").stat().st_mode & 0o777 == 0o600


def test_configure_operational_logging_rotates_private_archives(tmp_path, monkeypatch):
    """Rollover retains bounded operational JSONL archives with private permissions."""
    monkeypatch.setattr("loop.constants.DEFAULT_OPERATIONAL_LOG_BYTES", 1)
    monkeypatch.setattr("loop.constants.DEFAULT_OPERATIONAL_LOG_BACKUPS", 1)
    path = tmp_path / ".loop" / "loop.log"
    handler = configure_operational_logging(path)
    logger = logging.getLogger("loop.operational.rotation")

    logger.error("first")
    logger.error("second")
    handler.flush()
    logging.getLogger().removeHandler(handler)
    handler.close()

    archive = path.with_name("loop.log.1")
    assert json.loads(path.read_text(encoding="utf-8"))["message"] == "second"
    assert json.loads(archive.read_text(encoding="utf-8"))["message"] == "first"
    assert path.stat().st_mode & 0o777 == 0o600
    assert archive.stat().st_mode & 0o777 == 0o600


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


def test_handler_emit_failures_use_content_free_stderr_fallback(tmp_path, monkeypatch, capsys):
    """A write failure during logging cannot expose the formatted record or propagate."""
    monkeypatch.setattr(
        logging_module.PrivateRotatingTextFile,
        "append",
        Mock(side_effect=OSError("unavailable")),
    )
    handler = logging_module.SafeRotatingFileHandler(tmp_path / "loop.log")
    logger = logging.getLogger("loop.test.emit_failure")
    previous_propagation = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False

    try:
        logger.error("private payload")
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagation
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
