"""Tests for the application-wide problem contract."""

import logging

import pytest

from loop import Problem, ProblemException, log_problem
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry


def _raise_private_error():
    """Raise an exception with a traceback frame suitable for safe location logging."""
    raise ValueError("private diagnostic")


def test_problem_from_exception_uses_defaults_and_explicit_safe_context():
    """Exception conversion keeps safe fields while generating an opaque occurrence ID."""
    problem = Problem.from_exception(
        ValueError("invalid value"),
        code="example.invalid",
        title="Invalid example",
        operation="validate_example",
    )

    assert problem.detail == "invalid value"
    assert problem.instance.startswith("err_")
    assert problem.metadata == {}


def test_problem_exception_carries_the_structured_failure():
    """Problem exceptions retain the original problem and expose its detail as exception text."""
    problem = Problem(code="example.failed", title="Example failed", detail="Try again.")

    error = ProblemException(problem)

    assert error.problem is problem
    assert str(error) == "Try again."


@pytest.mark.parametrize(
    ("severity", "level"),
    [("warning", logging.WARNING), ("error", logging.ERROR), ("fatal", logging.CRITICAL)],
)
def test_log_problem_records_stable_attributes_and_severity(caplog, severity, level):
    """Problem logging maps severity and exposes identifiers as structured attributes."""
    problem = Problem(
        code="example.failed",
        title="Example failed",
        detail="Try again.",
        severity=severity,
        operation="example",
        instance="err_test",
    )

    with caplog.at_level(level):
        log_problem(logging.getLogger("tests.problem"), problem)

    record = caplog.records[-1]
    assert record.levelno == level
    assert record.__dict__["error.type"] == "example.failed"
    assert record.__dict__["error.instance"] == "err_test"
    assert record.__dict__["error.operation"] == "example"
    assert "exception.location" not in record.__dict__


def test_log_problem_records_only_exception_type(caplog):
    """Exception-backed logs retain type without arbitrary message or traceback contents."""
    error = ValueError("private diagnostic")
    problem = Problem(code="example.failed", title="Example failed", detail="Safe detail.")

    with caplog.at_level(logging.ERROR):
        log_problem(logging.getLogger("tests.problem.exception"), problem, error)

    record = caplog.records[-1]
    assert record.__dict__["exception.type"] == "builtins.ValueError"
    assert "exception.location" not in record.__dict__
    assert "exception.message" not in record.__dict__
    assert record.exc_info is None
    assert "private diagnostic" not in record.getMessage()


@pytest.mark.parametrize(
    ("severity", "level"),
    [("warning", logging.WARNING), ("error", logging.ERROR), ("fatal", logging.CRITICAL)],
)
def test_log_problem_records_exception_location_for_each_severity(caplog, severity, level):
    """Every exception-backed severity records only the final traceback frame location."""
    with pytest.raises(ValueError) as caught:
        _raise_private_error()
    error = caught.value
    problem = Problem(
        code="example.failed",
        title="Example failed",
        detail="Safe detail.",
        severity=severity,
    )

    with caplog.at_level(level):
        log_problem(logging.getLogger("tests.problem.location"), problem, error)

    traceback = error.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    expected_location = (
        f"{frame.f_globals['__name__']}.{frame.f_code.co_name}:{traceback.tb_lineno}"
    )
    record = caplog.records[-1]
    assert record.levelno == level
    assert record.__dict__["exception.location"] == expected_location
    assert record.exc_info is None
    assert "private diagnostic" not in record.getMessage()


def test_log_problem_omits_location_when_frame_has_no_module_name(monkeypatch, caplog):
    """A traceback without a string module name remains safe to report."""
    with pytest.raises(ValueError) as caught:
        _raise_private_error()
    monkeypatch.setitem(globals(), "__name__", None)
    problem = Problem(code="example.failed", title="Example failed", detail="Safe detail.")

    with caplog.at_level(logging.ERROR):
        log_problem(logging.getLogger("tests.problem.no_module"), problem, caught.value)

    assert "exception.location" not in caplog.records[-1].__dict__


def test_log_problem_uses_safe_last_resort_without_configured_logging(capsys):
    """Unconfigured logging reports a safe summary through Python's stderr fallback."""
    logger = logging.getLogger("tests.problem.unconfigured")
    logger.handlers.clear()
    logger.propagate = False
    error = ValueError("private diagnostic")
    problem = Problem(code="example.failed", title="Example failed", detail="Safe detail.")

    log_problem(logger, problem, error)

    output = capsys.readouterr().err
    assert "Example failed" in output
    assert "private diagnostic" not in output


@pytest.mark.parametrize("with_traceback", [False, True])
def test_log_problem_also_records_structured_telemetry_when_available(with_traceback):
    """Telemetry preserves safe exception locations only when a traceback is available."""
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)
    try:
        problem = Problem(
            code="example.failed",
            title="Example failed",
            detail="Safe detail.",
            operation="example",
        )
        error = ValueError("private")
        if with_traceback:
            try:
                raise error
            except ValueError as raised:
                error = raised
        log_problem(logging.getLogger("tests.problem.telemetry"), problem, error)
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    record = adapter.records[0]
    assert record.event_name == "problem.reported"
    assert record.attributes["error.type"] == "example.failed"
    assert record.attributes["exception.type"] == "builtins.ValueError"
    if with_traceback:
        assert record.attributes["exception.location"].startswith(f"{__name__}.")
    else:
        assert "exception.location" not in record.attributes
    assert "private" not in repr(record)
