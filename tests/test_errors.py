"""Tests for the application-wide problem contract."""

import logging

import pytest

from loop import Problem, ProblemException, log_problem


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


def test_log_problem_records_exception_type_message_and_traceback(caplog):
    """Exception-backed problem logs retain diagnostic context outside the user payload."""
    error = ValueError("private diagnostic")
    problem = Problem(code="example.failed", title="Example failed", detail="Safe detail.")

    with caplog.at_level(logging.ERROR):
        log_problem(logging.getLogger("tests.problem.exception"), problem, error)

    record = caplog.records[-1]
    assert record.__dict__["exception.type"] == "builtins.ValueError"
    assert record.__dict__["exception.message"] == "private diagnostic"
    assert record.exc_info[:2] == (ValueError, error)


def test_log_problem_does_not_expose_tracebacks_without_configured_logging(capsys):
    """Unconfigured logging does not send diagnostic tracebacks to the user terminal."""
    logger = logging.getLogger("tests.problem.unconfigured")
    logger.handlers.clear()
    logger.propagate = False
    error = ValueError("private diagnostic")
    problem = Problem(code="example.failed", title="Example failed", detail="Safe detail.")

    log_problem(logger, problem, error)

    assert capsys.readouterr().err == ""
