"""Verify the responsive conversation command provider."""

from unittest.mock import Mock

import pytest

from loop import CommandCompletion, CommandContext, Interaction
from loop.conversation.commands import ConversationCommands
from loop.conversation.models import ConversationStatus, SteeringOutcome


@pytest.fixture
def command_context():
    """Provide a fresh interaction and invocation context."""
    interaction = Mock(spec=Interaction)
    return CommandContext("conversation", interaction), interaction


def test_provider_declares_responsive_controls_with_freeform_steering():
    """The provider registers all controls and leaves steering text free-form."""
    provider = ConversationCommands(Mock())
    registrations = provider.get_commands()

    assert [registration.function.__name__ for registration in registrations] == [
        "interrupt",
        "respond",
        "status",
        "steer",
    ]
    assert all(registration.responsive for registration in registrations)
    assert registrations[-1].completion == CommandCompletion()


@pytest.mark.parametrize(
    ("accepted", "method", "message"),
    [
        (True, "info", "Interrupt requested; waiting for the active operation to settle..."),
        (False, "warning", "No agent run is active."),
    ],
)
def test_interrupt_reports_the_control_outcome(command_context, accepted, method, message):
    """Interrupt feedback reflects whether the active run accepted the request."""
    context, interaction = command_context
    control = Mock()
    control.request_interrupt.return_value = accepted

    ConversationCommands(control).interrupt(context)

    getattr(interaction, method).assert_called_once_with(message)


@pytest.mark.parametrize("pending", [False, True])
def test_respond_reports_only_when_no_interaction_is_pending(command_context, pending):
    """A missing interaction produces a warning while a pending one is requested."""
    context, interaction = command_context
    control = Mock()
    control.request_response.return_value = pending

    ConversationCommands(control).respond(context)

    if pending:
        interaction.warning.assert_not_called()
    else:
        interaction.warning.assert_called_once_with("No input request is pending.")


def test_status_formats_the_controller_snapshot(command_context):
    """Status displays the current activity and queued ordinary input count."""
    context, interaction = command_context
    control = Mock()
    control.status_snapshot.return_value = ConversationStatus("Working", 3)

    ConversationCommands(control).status(context)

    interaction.info.assert_called_once_with("Working; 3 queued inputs.")


@pytest.mark.parametrize(
    ("message", "outcome", "method", "feedback"),
    [
        ((), None, "warning", "Usage: /steer <message>"),
        (
            ("fix",),
            SteeringOutcome.RUN_INACTIVE,
            "warning",
            "No agent run is active; send a normal message instead.",
        ),
        (
            ("fix",),
            SteeringOutcome.INTERRUPT_REJECTED,
            "warning",
            "No agent run is active.",
        ),
        (
            ("fix",),
            SteeringOutcome.INTERRUPTED,
            "info",
            "Interrupt requested; waiting for the active operation to settle...",
        ),
        (("fix",), SteeringOutcome.QUEUE_FULL, None, None),
    ],
)
def test_steer_reports_validation_and_control_outcomes(
    command_context, message, outcome, method, feedback
):
    """Steering reports usage and lifecycle outcomes without duplicating queue warnings."""
    context, interaction = command_context
    control = Mock()
    control.steer.return_value = outcome

    ConversationCommands(control).steer(context, message)

    if outcome is None:
        control.steer.assert_not_called()
    else:
        control.steer.assert_called_once_with(" ".join(message), interaction)
    if method is None:
        interaction.info.assert_not_called()
        interaction.warning.assert_not_called()
    else:
        getattr(interaction, method).assert_called_once_with(feedback)
