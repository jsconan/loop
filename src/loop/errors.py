"""Define the application-wide problem reporting contract."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .telemetry import get_telemetry

_VALIDATION_MESSAGES = {
    "bool_parsing": "must be a boolean",
    "bool_type": "must be a boolean",
    "dict_type": "must be a mapping",
    "enum": "must be a supported value",
    "extra_forbidden": "unrecognized field",
    "finite_number": "must be a finite number",
    "float_parsing": "must be a number",
    "float_type": "must be a number",
    "int_parsing": "must be an integer",
    "int_type": "must be an integer",
    "list_type": "must be a list",
    "literal_error": "must be a supported value",
    "missing": "required field is missing",
    "string_pattern_mismatch": "must match the required string format",
    "string_type": "must be a string",
    "union_tag_invalid": "requires a supported discriminator",
    "union_tag_not_found": "requires a supported discriminator",
}
_VALIDATION_BOUNDS = {
    "greater_than": ("gt", "must be greater than"),
    "greater_than_equal": ("ge", "must be greater than or equal to"),
    "less_than": ("lt", "must be less than"),
    "less_than_equal": ("le", "must be less than or equal to"),
}


def _schema_constraints(
    schema: dict[str, Any], location: tuple[object, ...], *, kind: str
) -> tuple[str, ...] | None:
    """Find unambiguous constraints for one validation path in the trusted model schema."""
    definitions = schema.get("$defs", {})
    pending = [(schema, 0)]
    visited = set()
    constraints = set()
    while pending:
        node, position = pending.pop()
        marker = (id(node), position)
        if marker in visited:
            continue
        visited.add(marker)
        discriminator = node.get("discriminator", {})
        mapping = discriminator.get("mapping", {})
        if position < len(location) and isinstance(location[position], str):
            branch = mapping.get(location[position])
            if isinstance(branch, str) and branch.startswith("#/$defs/"):
                target = definitions.get(branch.removeprefix("#/$defs/"))
                if isinstance(target, dict):
                    pending.append((target, position + 1))
                continue
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = definitions.get(reference.removeprefix("#/$defs/"))
            if isinstance(target, dict):
                pending.append((target, position))
        for key in ("anyOf", "oneOf", "allOf"):
            pending.extend(
                (child, position) for child in node.get(key, ()) if isinstance(child, dict)
            )
        if position == len(location):
            if kind == "string_pattern_mismatch" and isinstance(node.get("pattern"), str):
                constraints.add((json.dumps(node["pattern"]),))
            elif kind in {"enum", "literal_error"} and isinstance(node.get("enum"), list):
                constraints.add(tuple(json.dumps(value) for value in node["enum"]))
            elif kind in {"enum", "literal_error"} and "const" in node:
                constraints.add((json.dumps(node["const"]),))
            elif kind in {"union_tag_invalid", "union_tag_not_found"} and mapping:
                name = discriminator.get("propertyName")
                if isinstance(name, str):
                    constraints.add((json.dumps(name), *(json.dumps(value) for value in mapping)))
            continue
        part = location[position]
        properties = node.get("properties", {})
        if isinstance(part, str) and part in properties:
            pending.append((properties[part], position + 1))
        elif isinstance(part, int) and isinstance(node.get("items"), dict):
            pending.append((node["items"], position + 1))
        elif isinstance(node.get("additionalProperties"), dict):
            if location[position + 1 : position + 2] == ("[key]",):
                key_schema = node.get("propertyNames")
                if isinstance(key_schema, dict):
                    pending.append((key_schema, position + 2))
            else:
                pending.append((node["additionalProperties"], position + 1))
    if len(constraints) == 1:
        return next(iter(constraints))
    return None


def safe_error_detail(
    error: BaseException,
    model: type[BaseModel],
    *,
    trusted_types: tuple[type[BaseException], ...] = (),
    trusted_messages: frozenset[str] = frozenset(),
    redacted_prefixes: Mapping[str, str] | None = None,
) -> str:
    """Report a validation or loading failure without exposing untrusted values.

    Args:
        error (BaseException): Loading or validation failure.
        model (type[BaseModel]): Model defining trusted validation path segments.
        trusted_types (tuple[type[BaseException], ...]): Owner-defined safe exception types.
        trusted_messages (frozenset[str]): Exact owner-defined messages safe to display.
        redacted_prefixes (Mapping[str, str] | None): Unsafe prefixes and safe replacements.

    Returns:
        str: Sanitized error detail with trusted field names and positions where available.
    """
    if isinstance(error, ValidationError):
        names = {"[key]"}
        schema = model.model_json_schema()
        pending = [schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                properties = item.get("properties")
                if isinstance(properties, dict):
                    names.update(properties)
                if isinstance(item.get("const"), str):
                    names.add(item["const"])
                if isinstance(item.get("enum"), list):
                    names.update(value for value in item["enum"] if isinstance(value, str))
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        details = []
        for issue in error.errors(include_input=False, include_context=True, include_url=False):
            location = tuple(
                "<index>" if isinstance(part, int) else part if part in names else "<unknown>"
                for part in issue["loc"]
            )
            if issue["type"] == "extra_forbidden" and location:
                location = (*location[:-1], "<unknown>")
            kind = issue["type"]
            if kind.startswith("loop_"):
                message = issue["msg"]
            elif kind in {
                "enum",
                "literal_error",
                "string_pattern_mismatch",
                "union_tag_invalid",
                "union_tag_not_found",
            }:
                constraints = _schema_constraints(schema, issue["loc"], kind=kind)
                if constraints:
                    if kind == "string_pattern_mismatch":
                        message = f"string values must match pattern {constraints[0]}"
                    elif kind in {"union_tag_invalid", "union_tag_not_found"}:
                        message = (
                            f"field {constraints[0]} must select one of "
                            f"{', '.join(constraints[1:])}"
                        )
                    else:
                        message = f"supported values include {', '.join(constraints)}"
                else:
                    message = _VALIDATION_MESSAGES.get(kind, kind)
            elif kind in _VALIDATION_BOUNDS:
                key, description = _VALIDATION_BOUNDS[kind]
                bound = issue.get("ctx", {}).get(key)
                if (
                    isinstance(bound, (int, float))
                    and not isinstance(bound, bool)
                    and (isinstance(bound, int) or math.isfinite(bound))
                ):
                    message = f"{description} {bound}"
                else:
                    message = description
            else:
                message = _VALIDATION_MESSAGES.get(kind, kind)
            details.append(f"{'.'.join(str(part) for part in location)}: {message}")
        prefix = f"{error.title}: " if error.title.startswith("Configuration in ") else ""
        return prefix + "; ".join(details)
    message = str(error)
    if isinstance(error, trusted_types) or message in trusted_messages:
        return message
    if redacted_prefixes is not None:
        for prefix, replacement in redacted_prefixes.items():
            if message.startswith(prefix):
                return replacement
    if isinstance(error, (UnicodeDecodeError, UnicodeEncodeError)):
        return f"Invalid text encoding at byte {error.start}."
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON at line {error.lineno}, column {error.colno}."
    if isinstance(error, PermissionError):
        return f"Permission denied while reading {error.filename or 'file'}."
    if isinstance(error, OSError):
        if error.errno is not None:
            return (
                f"Cannot read {error.filename or 'file'}: "
                f"{os.strerror(error.errno)} (errno {error.errno})."
            )
        return f"Cannot read {error.filename or 'file'} (errno unavailable)."
    return f"{type(error).__name__} while reading or validating data."


class Problem(BaseModel):
    """Describe one safe, structured application failure.

    Args:
        code (str): Stable machine-readable failure code.
        title (str): Stable human-readable summary.
        detail (str): Safe, actionable description of this occurrence.
        severity (Literal["warning", "error", "fatal"]): User-visible impact level.
        retryable (bool): Whether repeating the operation may succeed.
        operation (str | None): Operation that failed, when known.
        instance (str): Opaque identifier for this occurrence.
        metadata (dict[str, Any]): Safe structured context for diagnostics and clients.
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1)
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    severity: Literal["warning", "error", "fatal"] = "error"
    retryable: bool = False
    operation: str | None = None
    instance: str = Field(default_factory=lambda: f"err_{uuid4().hex}")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        code: str,
        title: str,
        detail: str | None = None,
        severity: Literal["warning", "error", "fatal"] = "error",
        retryable: bool = False,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Create a safe problem from an exception at a reporting boundary.

        Args:
            error (BaseException): Original exception supplying the default detail.
            code (str): Stable machine-readable failure code.
            title (str): Stable human-readable summary.
            detail (str | None): Safe replacement detail, or the exception text when omitted.
            severity (Literal["warning", "error", "fatal"]): User-visible impact level.
            retryable (bool): Whether repeating the operation may succeed.
            operation (str | None): Operation that failed, when known.
            metadata (dict[str, Any] | None): Safe structured context for clients.

        Returns:
            Problem: Structured failure containing no traceback or implicit exception metadata.
        """
        return cls(
            code=code,
            title=title,
            detail=detail if detail is not None else str(error),
            severity=severity,
            retryable=retryable,
            operation=operation,
            metadata=metadata or {},
        )


class ProblemException(Exception):
    """Carry a structured problem through an exception-based control-flow boundary.

    Args:
        problem (Problem): Safe application problem represented by this exception.
    """

    problem: Problem

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        super().__init__(problem.detail)


def log_problem(
    logger: logging.Logger,
    problem: Problem,
    error: BaseException | None = None,
) -> None:
    """Record a problem through configured application logging.

    Args:
        logger (logging.Logger): Logger receiving the problem record.
        problem (Problem): Structured failure shared with the user or protocol consumer.
        error (BaseException | None): Original exception supplying its qualified type and, when
            available, the raised frame's module, function, and line. Exception text and traceback
            contents are never recorded. Defaults to ``None`` when the problem did not originate
            from an exception.
    """
    level = {
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.CRITICAL,
    }[problem.severity]
    attributes = {
        "error.type": problem.code,
        "error.instance": problem.instance,
        "error.operation": problem.operation,
    }
    if error is not None:
        attributes.update(
            {
                "exception.type": f"{type(error).__module__}.{type(error).__qualname__}",
            }
        )
        location = _exception_location(error)
        if location is not None:
            attributes["exception.location"] = location
    logger.log(
        level,
        "%s [%s]",
        problem.title,
        problem.instance,
        extra=attributes,
    )
    telemetry = get_telemetry()
    if telemetry is not None:
        telemetry_attributes = {
            "operation": problem.operation,
            "problem_instance": problem.instance,
        }
        if "exception.location" in attributes:
            telemetry_attributes["exception.location"] = attributes["exception.location"]
        telemetry.error(
            "problem.reported",
            error_type=problem.code,
            severity=problem.severity,
            exception=error,
            **telemetry_attributes,
        )


def _exception_location(error: BaseException) -> str | None:
    """Return only the final traceback frame's module, function, and line number."""
    traceback = error.__traceback__
    if traceback is None:
        return None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    module = frame.f_globals.get("__name__")
    function = frame.f_code.co_name
    if not isinstance(module, str):
        return None
    return f"{module}.{function}:{traceback.tb_lineno}"
