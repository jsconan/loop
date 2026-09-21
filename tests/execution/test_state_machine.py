"""Tests for closed sandbox lifecycle transitions."""

import pytest

from loop.execution.results import Completed, InfrastructureFailure
from loop.execution.state_machine import AttemptState, AttemptStateMachine


def test_state_machine_accepts_every_valid_lifecycle_transition():
    """An ordinary attempt reaches a single terminal sandbox result."""
    attempt = AttemptStateMachine()

    attempt.transition(AttemptState.AUTHORIZED)
    attempt.transition(AttemptState.ATTESTED)
    attempt.transition(AttemptState.RUNNING)
    attempt.finish(Completed(request_id="request", exit_code=0))

    assert attempt.state is AttemptState.TERMINAL
    assert attempt.result == Completed(request_id="request", exit_code=0)


@pytest.mark.parametrize(
    ("initial", "target"),
    [
        (AttemptState.NEW, AttemptState.ATTESTED),
        (AttemptState.AUTHORIZED, AttemptState.RUNNING),
        (AttemptState.ATTESTED, AttemptState.AUTHORIZED),
        (AttemptState.RUNNING, AttemptState.AUTHORIZED),
        (AttemptState.TERMINAL, AttemptState.RUNNING),
        (AttemptState.NEW, AttemptState.TERMINAL),
    ],
)
def test_state_machine_rejects_every_invalid_or_terminal_transition(
    initial: AttemptState, target: AttemptState
):
    """Invalid transitions cannot manufacture another execution boundary."""
    attempt = AttemptStateMachine()
    if initial is AttemptState.AUTHORIZED:
        attempt.transition(initial)
    elif initial is AttemptState.ATTESTED:
        attempt.transition(AttemptState.AUTHORIZED)
        attempt.transition(initial)
    elif initial is AttemptState.RUNNING:
        attempt.transition(AttemptState.AUTHORIZED)
        attempt.transition(AttemptState.ATTESTED)
        attempt.transition(initial)
    elif initial is AttemptState.TERMINAL:
        attempt.finish(InfrastructureFailure(request_id="request"))

    with pytest.raises(ValueError):
        attempt.transition(target)


def test_state_machine_allows_any_sandbox_failure_to_finish_before_launch():
    """Infrastructure failure terminates from authorization without a host transition."""
    attempt = AttemptStateMachine()
    attempt.transition(AttemptState.AUTHORIZED)
    failure = InfrastructureFailure(request_id="request", diagnostic_id="diagnostic")

    attempt.finish(failure)

    assert attempt.state is AttemptState.TERMINAL
    assert attempt.result is failure
    with pytest.raises(ValueError):
        attempt.finish(failure)
