"""Verify optional interaction scopes for existing synchronous adapters."""

from unittest.mock import Mock

import pytest

from loop import Interaction


def test_unspecified_interaction_does_not_claim_fresh_host_approval():
    """Existing input adapters fail closed until they opt into fresh host decisions."""
    assert Interaction.supports_fresh_host_approval.fget(Mock()) is False


def test_default_terminal_context_preserves_synchronous_execution_and_errors():
    """Legacy adapters require no terminal handoff and do not suppress operation failures."""
    operation = Mock()
    with Interaction.terminal_context(Mock()):
        operation()
    operation.assert_called_once_with()
    with pytest.raises(ValueError, match="operation failed"), Interaction.terminal_context(Mock()):
        raise ValueError("operation failed")


@pytest.mark.parametrize("choice", [None, "exit", "unknown", []])
def test_recovery_prompt_requires_explicit_supported_choice(choice):
    """Exit and invalid input halt rather than implicitly selecting a recovery action."""
    import asyncio
    from unittest.mock import AsyncMock

    from loop.interaction import ConsoleInteraction, prompt_recovery_async
    from loop.utils import ShutdownRequested

    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value=choice)
    with pytest.raises(ShutdownRequested):
        asyncio.run(
            prompt_recovery_async(
                interaction,
                "Recover?",
                operation="recovery",
            )
        )


def test_recovery_prompt_returns_selected_action():
    """A listed recovery action is returned without implying any side effect."""
    import asyncio
    from unittest.mock import AsyncMock

    from loop.interaction import ConsoleInteraction, prompt_recovery_async

    interaction = Mock(spec=ConsoleInteraction)
    interaction.prompt_async = AsyncMock(return_value="retry")
    assert (
        asyncio.run(
            prompt_recovery_async(
                interaction,
                "Recover?",
                operation="recovery",
            )
        )
        == "retry"
    )


@pytest.mark.parametrize("choice", ["retry", "exit", "unknown", None, []])
def test_synchronous_recovery_requires_an_explicit_supported_choice(choice):
    """Synchronous recovery accepts only a selected non-exit action."""
    from loop.interaction import prompt_recovery
    from loop.utils import ShutdownRequested

    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = choice
    if choice == "retry":
        assert (
            prompt_recovery(
                interaction,
                "Recover?",
            )
            == "retry"
        )
    else:
        with pytest.raises(ShutdownRequested):
            prompt_recovery(
                interaction,
                "Recover?",
            )


def test_recovery_prompt_builds_only_requested_actions():
    """The common choices and shortcuts include only explicitly available actions."""
    from loop.interaction import prompt_recovery

    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "reset"
    assert (
        prompt_recovery(
            interaction, "Recover?", continue_label="Safe bypass", reset_label="Archive"
        )
        == "reset"
    )
    interaction.prompt.assert_called_once_with(
        "Recover?",
        exit_commands=(),
        choices={
            "retry": "Reload after fixing the problem",
            "continue": "Safe bypass",
            "reset": "Archive",
            "exit": "Halt",
        },
        index={"retry": "R", "continue": "C", "reset": "T", "exit": "Q"},
    )


def test_recovery_prompt_rejects_unavailable_action():
    """An input adapter cannot select an action the owner did not offer."""
    from loop.interaction import prompt_recovery
    from loop.utils import ShutdownRequested

    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = "reset"
    with pytest.raises(ShutdownRequested):
        prompt_recovery(interaction, "Recover?")
