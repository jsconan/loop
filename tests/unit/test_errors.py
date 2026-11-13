"""Tests for the application-wide problem contract."""

import logging

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAMLError

from loop import Problem, ProblemException, log_problem
from loop.errors import safe_error_detail
from loop.permissions import PermissionConfiguration
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import load_yaml


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


@pytest.mark.parametrize("key", ["synthetic-secret", 123456789])
def test_permission_diagnostic_redacts_dynamic_mapping_keys(key):
    """Invalid dynamic policy keys cannot appear in user-facing diagnostics."""
    with pytest.raises(ValidationError) as raised:
        PermissionConfiguration.model_validate({"defaults": {key: "allow"}}, extra="forbid")
    detail = safe_error_detail(raised.value, PermissionConfiguration, trusted_types=(YAMLError,))
    assert "defaults" in detail
    assert str(key) not in detail


def test_permission_diagnostic_describes_excessive_nesting():
    """A nesting limit reports an actionable cause without displaying source text."""
    with pytest.raises(YAMLError) as raised:
        load_yaml("[" * 2000 + "0" + "]" * 2000)
    assert "nesting exceeds" in safe_error_detail(
        raised.value, PermissionConfiguration, trusted_types=(YAMLError,)
    )


def test_permission_diagnostic_identifies_invalid_encoding():
    """Encoding failures report the byte position instead of source text."""
    error = UnicodeDecodeError("utf-8", b"bad\xff", 3, 4, "invalid")
    assert "byte 3" in safe_error_detail(error, PermissionConfiguration)


def test_safe_error_detail_does_not_expose_exception_messages():
    """Only explicitly trusted messages and safe replacements are displayed."""
    error = ValueError("private value")
    assert safe_error_detail(error, PermissionConfiguration) == (
        "ValueError while reading or validating data."
    )
    assert (
        safe_error_detail(
            error, PermissionConfiguration, trusted_messages=frozenset({"private value"})
        )
        == "private value"
    )
    assert (
        safe_error_detail(error, PermissionConfiguration, redacted_prefixes={"private": "safe"})
        == "safe"
    )


def test_safe_error_detail_reports_safe_exception_locations():
    """File, encoding and JSON failures retain their useful safe locations."""
    import json

    assert "line 1, column 2" in safe_error_detail(
        json.JSONDecodeError("private", "{", 1), PermissionConfiguration
    )
    assert "errno 2" in safe_error_detail(
        FileNotFoundError(2, "private", "config.toml"), PermissionConfiguration
    )
    assert "byte 0" in safe_error_detail(
        UnicodeEncodeError("ascii", "é", 0, 1, "private"), PermissionConfiguration
    )


@pytest.mark.parametrize(
    ("settings", "cause"),
    [
        (
            {"repetition_min_pattern_size": 5, "repetition_max_pattern_size": 4},
            "Minimum repetition pattern size cannot exceed the maximum.",
        ),
        (
            {
                "repetition_min_pattern_size": 0,
                "repetition_max_pattern_size": 1,
                "repetition_min_count": 0,
            },
            "Repetition minimum count must be at least two when enabled.",
        ),
    ],
)
def test_configuration_repetition_diagnostic_explains_the_cause(settings, cause):
    """Safe repetition constraint failures retain the specific explanation."""
    from loop.configuration import ApplicationSettings

    with pytest.raises(ValidationError) as error:
        ApplicationSettings.model_validate({"backend": settings})
    assert cause in safe_error_detail(error.value, ApplicationSettings)


def test_managed_rule_diagnostic_explains_the_cause():
    """Managed policy diagnostics explain invalid authority without displaying rule data."""
    from loop.permissions.managed import ManagedPolicy

    with pytest.raises(ValidationError) as error:
        ManagedPolicy.model_validate(
            {
                "version": 1,
                "identity": "a" * 32,
                "revision": 1,
                "rules": [{"id": "secret-id", "decision": "allow"}],
            }
        )
    detail = safe_error_detail(error.value, ManagedPolicy)
    assert "safe distinct IDs and deny or ask decisions" in detail
    assert "secret-id" not in detail


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"config_version": 999}, "config_version: must be less than or equal to 1"),
        ({"config_version": True}, "config_version: must be an integer"),
        ({"backend": {"request_timeout_seconds": 0}}, "must be greater than 0"),
        ({"app": {"temperature": -1}}, "must be greater than or equal to 0"),
    ],
)
def test_schema_diagnostics_explain_safe_constraints(values, expected):
    """Validation reports trusted numeric bounds and types without reporting input values."""
    from loop.configuration import ApplicationSettings

    with pytest.raises(ValidationError) as raised:
        ApplicationSettings.model_validate(values)
    assert expected in safe_error_detail(raised.value, ApplicationSettings)


def test_schema_diagnostics_redact_untrusted_discriminator_values():
    """An invalid union discriminator cannot enter a user-facing validation message."""
    from typing import Annotated, Literal

    from pydantic import BaseModel, Field

    class First(BaseModel):
        """Represent one permitted discriminator."""

        kind: Literal["first"]

    class Second(BaseModel):
        """Represent another permitted discriminator."""

        kind: Literal["second"]

    class Wrapper(BaseModel):
        """Select a model using a discriminator."""

        item: Annotated[First | Second, Field(discriminator="kind")]

    with pytest.raises(ValidationError) as raised:
        Wrapper.model_validate({"item": {"kind": "synthetic-secret"}})
    detail = safe_error_detail(raised.value, Wrapper)
    assert "synthetic-secret" not in detail
    assert "item" in detail


@pytest.mark.parametrize("bound", ["synthetic-secret", True, float("inf")])
def test_schema_diagnostics_ignore_untrusted_constraint_context(bound):
    """Malformed comparison context cannot place source data in a diagnostic."""
    from loop.configuration import ApplicationSettings

    error = ValidationError.from_exception_data(
        "configuration",
        [{"type": "greater_than", "loc": ("config_version",), "input": 0, "ctx": {"gt": bound}}],
    )
    detail = safe_error_detail(error, ApplicationSettings)
    assert detail == "config_version: must be greater than"
    assert "synthetic-secret" not in detail


def test_schema_diagnostics_report_finite_fractional_bounds():
    """A finite schema threshold is explained without exposing the invalid value."""
    from loop.configuration import ApplicationSettings

    error = ValidationError.from_exception_data(
        "configuration",
        [{"type": "less_than", "loc": ("config_version",), "input": 99, "ctx": {"lt": 0.5}}],
    )
    assert safe_error_detail(error, ApplicationSettings) == "config_version: must be less than 0.5"


def test_file_diagnostic_explains_operating_system_cause_without_exception_text():
    """I/O diagnostics name the OS cause while omitting arbitrary exception messages."""
    import errno

    error = OSError(errno.ELOOP, "synthetic-secret", "config.toml")
    detail = safe_error_detail(error, PermissionConfiguration)
    assert "config.toml" in detail
    assert "symbolic" in detail.lower()
    assert "synthetic-secret" not in detail


def test_file_diagnostic_handles_unavailable_errno_without_exception_text():
    """An I/O failure without an OS error number still redacts exception contents."""
    detail = safe_error_detail(OSError("synthetic-secret"), PermissionConfiguration)
    assert detail == "Cannot read file (errno unavailable)."


def test_schema_diagnostics_explain_literal_and_pattern_without_input():
    """Diagnostics show field-owned choices and patterns without exposing rejected values."""
    from loop.configuration import ApplicationSettings
    from loop.permissions.managed import ManagedPolicy

    with pytest.raises(ValidationError) as choice_error:
        ApplicationSettings.model_validate({"app": {"reasoning_effort": "synthetic-secret"}})
    choice = safe_error_detail(choice_error.value, ApplicationSettings)
    assert '"none"' in choice and '"high"' in choice and '"max"' in choice
    assert "synthetic-secret" not in choice
    with pytest.raises(ValidationError) as pattern_error:
        ManagedPolicy.model_validate({"version": 1, "identity": "synthetic-secret", "revision": 1})
    pattern = safe_error_detail(pattern_error.value, ManagedPolicy)
    assert "^[0-9a-f]{32}$" in pattern
    assert "synthetic-secret" not in pattern


@pytest.mark.parametrize("kind", ["literal_error", "string_pattern_mismatch"])
def test_schema_diagnostics_never_trust_forged_context(kind):
    """Error context cannot substitute a secret for model-owned schema constraints."""
    from loop.configuration import ApplicationSettings
    from loop.permissions.managed import ManagedPolicy

    model = ApplicationSettings if kind == "literal_error" else ManagedPolicy
    location = ("app", "reasoning_effort") if kind == "literal_error" else ("identity",)
    context = (
        {"expected": "synthetic-secret"}
        if kind == "literal_error"
        else {"pattern": "synthetic-secret"}
    )
    error = ValidationError.from_exception_data(
        "configuration",
        [{"type": kind, "loc": location, "input": "synthetic-secret", "ctx": context}],
    )
    detail = safe_error_detail(error, model)
    assert "synthetic-secret" not in detail
    assert ('"high"' if kind == "literal_error" else "^[0-9a-f]{32}$") in detail


def test_schema_diagnostics_resolve_mapping_keys_values_and_list_items():
    """Dynamic policy paths use their own schema constraints while redacting unknown keys."""
    with pytest.raises(ValidationError) as raised:
        PermissionConfiguration.model_validate(
            {
                "defaults": {"synthetic-secret": "allow", "filesystem.read": "invalid"},
                "rules": [{"id": "id", "decision": "invalid"}],
            }
        )
    detail = safe_error_detail(raised.value, PermissionConfiguration)
    assert "synthetic-secret" not in detail
    assert '"filesystem.read"' in detail
    assert '"allow", "ask", "deny"' in detail
    assert "<index>" in detail


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            {"kind": "file", "path": "/x", "expected_kind": "synthetic-secret"},
            '"file", "directory", "symlink"',
        ),
        (
            {"kind": "process", "argv": ["cmd"], "cwd": "/tmp", "boundary": "synthetic-secret"},
            '"host", "sandboxed"',
        ),
        (
            {"kind": "synthetic-secret"},
            'field "kind" must select one of "file", "network", "process", "session"',
        ),
        ({"path": "/x"}, 'field "kind" must select one of "file", "network", "process", "session"'),
    ],
)
def test_schema_diagnostics_explain_discriminated_targets(target, expected):
    """Tagged policy targets report their field choices without exposing rejected input."""
    with pytest.raises(ValidationError) as raised:
        PermissionConfiguration.model_validate(
            {"rules": [{"id": "id", "decision": "deny", "target": target}]}
        )
    detail = safe_error_detail(raised.value, PermissionConfiguration)
    assert expected in detail
    assert "synthetic-secret" not in detail


def test_schema_diagnostics_preserve_nullable_choice_semantics():
    """Suggestions do not imply nullable alternatives are invalid."""
    from loop.configuration import ApplicationSettings

    assert ApplicationSettings.model_validate({"app": {"reasoning_effort": None}})
    with pytest.raises(ValidationError) as raised:
        ApplicationSettings.model_validate({"app": {"reasoning_effort": "wrong"}})
    assert "supported values include" in safe_error_detail(raised.value, ApplicationSettings)


@pytest.mark.parametrize("kind", ["union_tag_invalid", "union_tag_not_found"])
def test_schema_diagnostics_redact_forged_union_context(kind):
    """Discriminator details come from the schema rather than exception context."""
    context = {"discriminator": "synthetic-secret"}
    if kind == "union_tag_invalid":
        context.update({"tag": "synthetic-secret", "expected_tags": "synthetic-secret"})
    error = ValidationError.from_exception_data(
        "policy", [{"type": kind, "loc": ("rules", 0, "target"), "input": {}, "ctx": context}]
    )
    detail = safe_error_detail(error, PermissionConfiguration)
    assert 'field "kind"' in detail
    assert '"file", "network", "process", "session"' in detail
    assert "synthetic-secret" not in detail


def test_schema_diagnostics_fall_back_for_unknown_locations():
    """Unrecognized paths do not borrow constraints from an unrelated field."""
    error = ValidationError.from_exception_data(
        "policy",
        [
            {
                "type": "literal_error",
                "loc": ("synthetic-secret",),
                "input": "private",
                "ctx": {"expected": "private"},
            }
        ],
    )
    detail = safe_error_detail(error, PermissionConfiguration)
    assert detail == "<unknown>: must be a supported value"
    assert "private" not in detail


def test_schema_diagnostics_explain_single_literal():
    """A single literal constraint is reported from its schema."""
    from typing import Literal

    from pydantic import BaseModel

    class Only(BaseModel):
        """Accept one declared value."""

        value: Literal["only"]

    with pytest.raises(ValidationError) as raised:
        Only.model_validate({"value": "synthetic-secret"})
    detail = safe_error_detail(raised.value, Only)
    assert detail == 'value: supported values include "only"'


def test_schema_diagnostics_handle_repeated_schema_references(monkeypatch):
    """Repeated references terminate and report a single trusted constraint."""
    from typing import Literal

    from pydantic import BaseModel

    class Choice(BaseModel):
        """Accept one value with an equivalent schema reference."""

        value: Literal["only"]

    schema = {
        "properties": {
            "value": {"anyOf": [{"$ref": "#/$defs/Choice"}, {"$ref": "#/$defs/Choice"}]}
        },
        "$defs": {"Choice": {"const": "only"}},
    }
    monkeypatch.setattr(Choice, "model_json_schema", classmethod(lambda cls: schema))
    with pytest.raises(ValidationError) as raised:
        Choice.model_validate({"value": "synthetic-secret"})
    assert safe_error_detail(raised.value, Choice) == 'value: supported values include "only"'


@pytest.mark.parametrize("case", ["reference", "branch", "tag", "mapping_key"])
def test_schema_diagnostics_fail_safely_for_unresolvable_schema_paths(monkeypatch, case):
    """Unresolvable schema references and mappings cannot disclose error context."""
    from pydantic import BaseModel

    class Policy(BaseModel):
        """Provide a trusted schema for a synthetic validation failure."""

        value: str

    missing = {"$ref": "#/$defs/missing"}
    if case == "reference":
        field = missing
        location = ("value",)
        kind = "literal_error"
        context = {"expected": "synthetic-secret"}
    elif case == "branch":
        field = {"discriminator": {"propertyName": "kind", "mapping": {"known": "#/$defs/missing"}}}
        location = ("value", "known", "child")
        kind = "literal_error"
        context = {"expected": "synthetic-secret"}
    elif case == "tag":
        field = {"discriminator": {"mapping": {"known": "#/$defs/missing"}}}
        location = ("value",)
        kind = "union_tag_not_found"
        context = {"discriminator": "synthetic-secret"}
    else:
        field = {"additionalProperties": {"enum": ["only"]}}
        location = ("value", "synthetic-secret", "[key]")
        kind = "literal_error"
        context = {"expected": "synthetic-secret"}
    schema = {"properties": {"value": field}, "$defs": {}}
    monkeypatch.setattr(Policy, "model_json_schema", classmethod(lambda cls: schema))
    error = ValidationError.from_exception_data(
        "policy", [{"type": kind, "loc": location, "input": "synthetic-secret", "ctx": context}]
    )
    detail = safe_error_detail(error, Policy)
    assert "synthetic-secret" not in detail
    assert "supported" in detail


def test_schema_diagnostics_do_not_mix_ambiguous_choice_sets(monkeypatch):
    """Conflicting schema alternatives fall back rather than presenting misleading choices."""
    from typing import Literal

    from pydantic import BaseModel

    class Choice(BaseModel):
        """Provide one invalid value against an ambiguous schema."""

        value: Literal["first"]

    schema = {"properties": {"value": {"anyOf": [{"enum": ["first"]}, {"enum": ["second"]}]}}}
    monkeypatch.setattr(Choice, "model_json_schema", classmethod(lambda cls: schema))
    with pytest.raises(ValidationError) as raised:
        Choice.model_validate({"value": "synthetic-secret"})
    assert safe_error_detail(raised.value, Choice) == "value: must be a supported value"
