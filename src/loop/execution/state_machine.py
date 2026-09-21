"""Enforce fail-closed sandbox attempt lifecycle transitions."""

from __future__ import annotations

from enum import StrEnum

from .results import ExecutionResult


class AttemptState(StrEnum):
    """Identify the lifecycle state of one sandbox attempt."""

    NEW = "new"
    AUTHORIZED = "authorized"
    ATTESTED = "attested"
    RUNNING = "running"
    TERMINAL = "terminal"


_ALLOWED_TRANSITIONS = {
    AttemptState.NEW: frozenset({AttemptState.AUTHORIZED, AttemptState.TERMINAL}),
    AttemptState.AUTHORIZED: frozenset({AttemptState.ATTESTED, AttemptState.TERMINAL}),
    AttemptState.ATTESTED: frozenset({AttemptState.RUNNING, AttemptState.TERMINAL}),
    AttemptState.RUNNING: frozenset({AttemptState.TERMINAL}),
    AttemptState.TERMINAL: frozenset(),
}


class AttemptStateMachine:
    """Advance one sandbox attempt through its closed lifecycle.

    The machine deliberately has no host state or transition: any sandbox failure reaches the
    sole terminal state and must be returned to the caller as a sandbox result.
    """

    state: AttemptState
    result: ExecutionResult | None

    def __init__(self) -> None:
        """Initialize a new, unstarted sandbox attempt."""
        self.state = AttemptState.NEW
        self.result = None

    def transition(self, target: AttemptState) -> None:
        """Advance to one permitted nonterminal state.

        Args:
            target (AttemptState): Next lifecycle state.

        Raises:
            ValueError: If the transition is invalid or terminal.
        """
        if target is AttemptState.TERMINAL or target not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(f"Cannot transition from {self.state.value} to {target.value}.")
        self.state = target

    def finish(self, result: ExecutionResult) -> None:
        """Record one terminal sandbox result.

        Args:
            result (ExecutionResult): Closed result produced by the sandbox lifecycle.

        Raises:
            ValueError: If this attempt is already terminal.
        """
        if AttemptState.TERMINAL not in _ALLOWED_TRANSITIONS[self.state]:
            raise ValueError("A terminal attempt cannot produce another result.")
        self.state = AttemptState.TERMINAL
        self.result = result
