"""Verify the responsive conversation command provider."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from loop import (
    CommandCapability,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CompletionManager,
    CompletionValue,
    Interaction,
)
from loop.conversation.commands import ConversationCommands
from loop.conversation.models import ConversationStatus, ResponseOutcome, SteeringOutcome


@pytest.fixture
def command_context():
    """Provide a fresh interaction and invocation context."""
    interaction = Mock(spec=Interaction)
    return CommandContext("conversation", interaction), interaction


def test_provider_declares_state_capabilities_and_response_completion():
    """The provider gates each control by conversation state and leaves steering free-form."""
    control = Mock()
    provider = ConversationCommands(control)
    registrations = provider.get_commands()

    assert [registration.function.__name__ for registration in registrations] == [
        "interrupt",
        "respond",
        "status",
        "steer",
    ]
    assert registrations[0].capabilities == frozenset({CommandCapability.RUN_ACTIVE})
    assert registrations[1].capabilities == frozenset({CommandCapability.PENDING_RESPONSE})
    assert registrations[2].capabilities == frozenset(
        {CommandCapability.CONVERSATION_ACTIVE, CommandCapability.CONTEXT_READ}
    )
    assert registrations[3].capabilities == frozenset({CommandCapability.RUN_ACTIVE})
    assert registrations[1].completion.provider == control.response_completions
    assert registrations[-1].completion == CommandCompletion()


def test_respond_argument_completes_the_pending_response_values():
    """The respond command suggests values provided by its active interaction request."""
    control = Mock()
    control.response_completions.return_value = (
        CompletionValue("Yes", "Approve the request."),
        CompletionValue("No", "Decline the request."),
    )
    manager = CommandManager(providers=(ConversationCommands(control),))
    completion = CompletionManager((manager.completion_adapter,))

    values = list(completion.get_completions(Document("/respond y"), CompleteEvent()))

    assert [(value.text, value.display_meta_text) for value in values] == [
        ("Yes", "Approve the request.")
    ]


def test_conversation_commands_are_absent_without_conversation_provider():
    """A manager outside asynchronous conversation has no conversation-only commands."""
    manager = CommandManager()

    assert "respond" not in {item.name for item in manager.commands}
    assert "interrupt" not in {item.name for item in manager.active_commands}


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


@pytest.mark.parametrize(
    ("outcome", "answer", "warning"),
    [
        (ResponseOutcome.NO_REQUEST, (), "No input request is pending."),
        (ResponseOutcome.PROMPTED, (), None),
        (ResponseOutcome.ANSWERED, ("Yes",), None),
        (
            ResponseOutcome.INVALID,
            ("unexpected",),
            "Choose one of the listed responses, or use /respond without an answer.",
        ),
        (
            ResponseOutcome.PROMPT_REQUIRED,
            ("Yes",),
            "Use /respond without an answer for this request.",
        ),
        (
            ResponseOutcome.MASKED_INPUT_REQUIRED,
            ("secret",),
            "This request requires masked input; use /respond without an answer.",
        ),
    ],
)
def test_respond_reports_request_outcomes(command_context, outcome, answer, warning):
    """A response is prompted, submitted, or rejected with outcome-specific feedback."""
    context, interaction = command_context
    control = Mock()
    control.request_response.return_value = outcome

    ConversationCommands(control).respond(context, answer)

    control.request_response.assert_called_once_with(" ".join(answer) if answer else None)
    if warning is None:
        interaction.warning.assert_not_called()
    else:
        interaction.warning.assert_called_once_with(warning)


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
