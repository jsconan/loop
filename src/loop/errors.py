"""Define the application-wide problem reporting contract."""

from __future__ import annotations

import logging
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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
        error (BaseException | None): Original exception whose type and traceback should be
            recorded, or ``None`` when the problem did not originate from an exception.
    """
    if not logger.hasHandlers():
        return

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
    exc_info = None
    if error is not None:
        attributes.update(
            {
                "exception.type": f"{type(error).__module__}.{type(error).__qualname__}",
                "exception.message": str(error),
            }
        )
        exc_info = (type(error), error, error.__traceback__)
    logger.log(
        level,
        "%s: %s [%s]",
        problem.title,
        problem.detail,
        problem.instance,
        extra=attributes,
        exc_info=exc_info,
    )
