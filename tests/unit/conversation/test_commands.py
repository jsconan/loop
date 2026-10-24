"""Verify the responsive conversation command provider."""

import asyncio
from inspect import iscoroutinefunction
from unittest.mock import AsyncMock, Mock

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
from loop.conversation.commands import (
    ConversationCommands,
    ConversationControl,
    ConversationRecoveryCommands,
)
from loop.conversation.models import (
    ConversationStatus,
    PendingInputPriorityOutcome,
    PendingInputRecallOutcome,
    ResponseOutcome,
    SteeringOutcome,
)


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
        "pause",
        "stop",
        "respond",
        "status",
        "steer_async",
        "queue_async",
        "edit_async",
        "edit_async",
    ]
    assert registrations[1].capabilities == frozenset({CommandCapability.RUN_ACTIVE})
    assert registrations[2].capabilities == frozenset({CommandCapability.RUN_ACTIVE})
    assert registrations[5].name == "steer"
    assert registrations[6].name == "queue"
    assert registrations[0].capabilities == frozenset({CommandCapability.RUN_ACTIVE})
    assert registrations[3].capabilities == frozenset({CommandCapability.PENDING_RESPONSE})
    assert registrations[4].capabilities == frozenset(
        {CommandCapability.CONVERSATION_ACTIVE, CommandCapability.CONTEXT_READ}
    )
    assert registrations[5].capabilities == frozenset({CommandCapability.CONVERSATION_ACTIVE})
    assert registrations[6].capabilities == frozenset({CommandCapability.CONVERSATION_ACTIVE})
    assert registrations[7].name == "edit"
    assert registrations[8].name == "recall"
    assert registrations[7].capabilities == frozenset({CommandCapability.CONVERSATION_ACTIVE})
    assert registrations[8].capabilities == frozenset({CommandCapability.CONVERSATION_ACTIVE})
    assert registrations[3].completion.provider == control.response_completions
    assert registrations[6].completion == CommandCompletion()


def test_control_protocol_declares_sync_and_async_control_methods():
    """The control contract preserves native sync methods beside async counterparts."""
    assert callable(ConversationControl.steer)
    assert callable(ConversationControl.queue)
    assert callable(ConversationControl.request_pause)
    assert callable(ConversationControl.request_stop_run)
    assert iscoroutinefunction(ConversationControl.steer_async)
    assert iscoroutinefunction(ConversationControl.queue_async)
    assert iscoroutinefunction(ConversationControl.reprioritize_latest_async)
    assert iscoroutinefunction(ConversationControl.recall_latest_async)


def test_recovery_provider_declares_state_gated_async_commands():
    """Continue requires a resumable run while retry requires a recorded failure."""
    registrations = ConversationRecoveryCommands(Mock()).get_commands()

    assert [registration.name for registration in registrations] == ["continue", "retry"]
    assert [registration.function.__name__ for registration in registrations] == [
        "continue_async",
        "retry_async",
    ]
    assert [registration.capabilities for registration in registrations] == [
        frozenset({CommandCapability.RUN_CONTINUABLE}),
        frozenset({CommandCapability.RUN_RETRYABLE}),
    ]


@pytest.mark.parametrize(
    ("continuable", "pending", "warning", "info"),
    [
        (False, False, "There is no run to continue.", None),
        (True, True, "Recovery remains pending; use /retry to try again.", None),
        (True, False, None, "Run continued."),
    ],
)
def test_continue_reports_recovery_results(command_context, continuable, pending, warning, info):
    """Continue requires a resumable run and reports whether recovery completed."""
    context, interaction = command_context
    control = Mock(continuable=continuable)
    control.recover_pending_async = AsyncMock(return_value=pending)

    asyncio.run(ConversationRecoveryCommands(control).continue_async(context))

    assert control.recover_pending_async.await_count == int(continuable)
    if continuable:
        control.recover_pending_async.assert_awaited_once_with()
    if warning is not None:
        interaction.warning.assert_called_once_with(warning)
    else:
        interaction.warning.assert_not_called()
    if info is not None:
        interaction.info.assert_called_once_with(info)
    else:
        interaction.info.assert_not_called()


@pytest.mark.parametrize(
    ("retry_pending", "pending", "warning", "info"),
    [
        (False, False, "There is nothing to retry.", None),
        (True, True, "Recovery remains pending; use /retry to try again.", None),
        (True, False, None, "Recovery completed."),
    ],
)
def test_retry_reports_recovery_results(command_context, retry_pending, pending, warning, info):
    """Retry is available only for a recorded failure and reports its final state."""
    context, interaction = command_context
    control = Mock(retry_pending=retry_pending)
    control.recover_pending_async = AsyncMock(return_value=pending)

    asyncio.run(ConversationRecoveryCommands(control).retry_async(context))

    assert control.recover_pending_async.await_count == int(retry_pending)
    if retry_pending:
        control.recover_pending_async.assert_awaited_once_with(retry=True)
    if warning is not None:
        interaction.warning.assert_called_once_with(warning)
    else:
        interaction.warning.assert_not_called()
    if info is not None:
        interaction.info.assert_called_once_with(info)
    else:
        interaction.info.assert_not_called()


@pytest.mark.parametrize(
    ("accepted", "method", "message"),
    [
        (True, "info", "Run paused at its next safe checkpoint."),
        (False, "warning", "No agent run is active."),
    ],
)
def test_pause_reports_the_run_control_outcome(command_context, accepted, method, message):
    """Pause feedback distinguishes an accepted request from an inactive run."""
    context, interaction = command_context
    control = Mock()
    control.request_pause.return_value = accepted

    ConversationCommands(control).pause(context)

    control.request_pause.assert_called_once_with()
    getattr(interaction, method).assert_called_once_with(message)


@pytest.mark.parametrize(
    ("accepted", "method", "message"),
    [
        (True, "info", "Run stop requested; use /continue to resume it later."),
        (False, "warning", "No agent run is active."),
    ],
)
def test_stop_reports_the_conversation_outcome(command_context, accepted, method, message):
    """Stop feedback distinguishes an accepted run stop from an unavailable run."""
    context, interaction = command_context
    control = Mock()
    control.request_stop_run.return_value = accepted

    ConversationCommands(control).stop(context)

    control.request_stop_run.assert_called_once_with()
    getattr(interaction, method).assert_called_once_with(message)


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
            "Choose one of the listed responses.",
        ),
        (
            ResponseOutcome.PROMPT_REQUIRED,
            ("Yes",),
            "Answer this request in its active prompt.",
        ),
        (
            ResponseOutcome.MASKED_INPUT_REQUIRED,
            ("secret",),
            "Answer this request in its masked prompt.",
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
    ("message", "accepted", "priority_outcome", "expected_warning"),
    [
        ((), None, PendingInputPriorityOutcome.NO_PENDING_INPUT, "No pending message to queue"),
        (("after", "this"), True, None, None),
        (("after",), False, None, "No agent run is active; send a normal message instead."),
    ],
)
def test_queue_reports_admission_outcomes(
    command_context, message, accepted, priority_outcome, expected_warning
):
    """Queued messages preserve their text and report empty, accepted, and rejected input."""
    context, interaction = command_context
    control = Mock()
    control.queue_async = AsyncMock(return_value=accepted)
    control.reprioritize_latest_async = AsyncMock(return_value=priority_outcome)

    asyncio.run(ConversationCommands(control).queue_async(context, message))

    if accepted is None:
        control.queue_async.assert_not_awaited()
        control.reprioritize_latest_async.assert_awaited_once_with(steer=False)
    else:
        control.queue_async.assert_awaited_once_with(" ".join(message), interaction)
    if expected_warning is None:
        interaction.warning.assert_not_called()
    else:
        interaction.warning.assert_called_once_with(expected_warning)


def test_queue_without_arguments_reprioritizes_the_latest_waiting_message(command_context):
    """Queue without text demotes the latest mailbox message to ordinary priority."""
    context, interaction = command_context
    control = Mock()
    control.reprioritize_latest_async = AsyncMock(return_value=PendingInputPriorityOutcome.UPDATED)

    asyncio.run(ConversationCommands(control).queue_async(context))

    control.reprioritize_latest_async.assert_awaited_once_with(steer=False)
    interaction.info.assert_called_once_with("Newest pending message set to queue priority.")
    interaction.warning.assert_not_called()


def test_queue_reports_idle_admission_for_the_next_run(command_context):
    """Explicit queue input accepted between runs is described as a next-run message."""
    context, interaction = command_context
    control = Mock()
    control.queue_async = AsyncMock(return_value=True)
    control.status_snapshot.return_value = ConversationStatus("Idle", 1)

    asyncio.run(ConversationCommands(control).queue_async(context, ("after", "this")))

    control.queue_async.assert_awaited_once_with("after this", interaction)
    interaction.info.assert_called_once_with("Queued input for the next run.")
    interaction.warning.assert_not_called()


@pytest.mark.parametrize(
    ("message", "outcome", "priority_outcome", "method", "feedback"),
    [
        (
            (),
            None,
            PendingInputPriorityOutcome.NO_PENDING_INPUT,
            "warning",
            "No pending message to steer",
        ),
        (
            ("fix",),
            SteeringOutcome.RUN_INACTIVE,
            None,
            "warning",
            "No agent run is active; send a normal message instead.",
        ),
        (
            ("fix",),
            SteeringOutcome.INTERRUPT_REJECTED,
            None,
            "warning",
            "No agent run is active.",
        ),
        (
            ("fix",),
            SteeringOutcome.INTERRUPTED,
            None,
            "info",
            "Interrupt requested; waiting for the active operation to settle...",
        ),
        (
            ("fix",),
            SteeringOutcome.QUEUED,
            None,
            "info",
            "Steering input accepted for the next run.",
        ),
        (("fix",), SteeringOutcome.QUEUE_FULL, None, None, None),
    ],
)
def test_steer_reports_validation_and_control_outcomes(
    command_context, message, outcome, priority_outcome, method, feedback
):
    """Steering reports usage and lifecycle outcomes without duplicating queue warnings."""
    context, interaction = command_context
    control = Mock()
    control.steer_async = AsyncMock(return_value=outcome)
    control.reprioritize_latest_async = AsyncMock(return_value=priority_outcome)

    asyncio.run(ConversationCommands(control).steer_async(context, message))

    if outcome is None:
        control.steer_async.assert_not_awaited()
        control.reprioritize_latest_async.assert_awaited_once_with(steer=True)
    else:
        control.steer_async.assert_awaited_once_with(" ".join(message), interaction)
    if method is None:
        interaction.info.assert_not_called()
        interaction.warning.assert_not_called()
    else:
        getattr(interaction, method).assert_called_once_with(feedback)


@pytest.mark.parametrize(
    ("priority_outcome", "feedback"),
    [
        (
            PendingInputPriorityOutcome.UPDATED,
            "Newest pending message set to steering priority.",
        ),
        (
            PendingInputPriorityOutcome.INTERRUPTED,
            "Interrupt requested; waiting for the active operation to settle...",
        ),
    ],
)
def test_steer_without_arguments_reprioritizes_and_interrupts(
    command_context, priority_outcome, feedback
):
    """Steer without text prioritizes the latest mailbox message and reports interruption."""
    context, interaction = command_context
    control = Mock()
    control.reprioritize_latest_async = AsyncMock(return_value=priority_outcome)

    asyncio.run(ConversationCommands(control).steer_async(context))

    control.reprioritize_latest_async.assert_awaited_once_with(steer=True)
    interaction.info.assert_called_once_with(feedback)
    interaction.warning.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "method", "feedback"),
    [
        (
            PendingInputRecallOutcome.NO_PENDING_INPUT,
            "warning",
            "No pending message to edit.",
        ),
        (
            PendingInputRecallOutcome.RECALLED,
            "info",
            "Pending message moved to the editor.",
        ),
    ],
)
def test_edit_reports_recall_outcomes(command_context, outcome, method, feedback):
    """The edit and recall aliases report empty and successful mailbox recalls."""
    context, interaction = command_context
    control = Mock()
    control.recall_latest_async = AsyncMock(return_value=outcome)

    asyncio.run(ConversationCommands(control).edit_async(context))

    control.recall_latest_async.assert_awaited_once_with()
    getattr(interaction, method).assert_called_once_with(feedback)
