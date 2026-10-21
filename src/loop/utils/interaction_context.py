"""Carry session, run, and operation identity through interaction adapters."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .identifiers import new_uuid


@dataclass(frozen=True, slots=True)
class InteractionCorrelation:
    """Identify the execution scope that owns one interaction request.

    Args:
        session_id (str | None): Session owning the request, when available.
        run_id (str | None): Agent run owning the request, when available.
        operation_id (str | None): Tool operation owning the request, when available.
    """

    session_id: str | None = None
    run_id: str | None = None
    operation_id: str | None = None


_CURRENT_INTERACTION: ContextVar[InteractionCorrelation | None] = ContextVar(
    "loop_interaction_correlation", default=None
)


def current_interaction_correlation() -> InteractionCorrelation:
    """Return the identity active in this execution context, or an empty identity.

    Returns:
        InteractionCorrelation: Active identity, or an empty identity outside a run.
    """
    return _CURRENT_INTERACTION.get() or InteractionCorrelation()


@contextmanager
def interaction_run(session_id: str) -> Generator[None]:
    """Bind interactions to a session and run, reusing a run when already nested.

    Args:
        session_id (str): Identifier of the session that owns the run.

    Yields:
        None: Control while the run identity is active.
    """
    parent = current_interaction_correlation()
    run_id = parent.run_id if parent.session_id == session_id else str(new_uuid())
    token = _CURRENT_INTERACTION.set(InteractionCorrelation(session_id, run_id))
    try:
        yield
    finally:
        _CURRENT_INTERACTION.reset(token)


@contextmanager
def interaction_operation(operation_id: str) -> Generator[None]:
    """Bind interactions to one operation while preserving session and run identity.

    Args:
        operation_id (str): Stable model tool-call identity for the operation.

    Yields:
        None: Control while the operation identity is active.
    """
    parent = current_interaction_correlation()
    token = _CURRENT_INTERACTION.set(
        InteractionCorrelation(parent.session_id, parent.run_id, operation_id)
    )
    try:
        yield
    finally:
        _CURRENT_INTERACTION.reset(token)
