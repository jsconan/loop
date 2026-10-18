"""Expose responsive controls for an active conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol

from ..commands import CommandContext, CommandRegistration, CommandRemainder
from ..completion import CommandCompletion
from .models import ConversationStatus, SteeringOutcome

if TYPE_CHECKING:
    from ..interaction import Interaction


class ConversationControl(Protocol):
    """Describe conversation actions available to the command provider."""

    def request_interrupt(self) -> bool:
        """Request interruption of the active run.

        Returns:
            bool: Whether an active run accepted the interruption request.
        """

    def request_response(self) -> bool:
        """Request a response to the currently pending interaction.

        Returns:
            bool: Whether an interaction is waiting for a response.
        """

    def status_snapshot(self) -> ConversationStatus:
        """Return the current activity and queued input count.

        Returns:
            ConversationStatus: Snapshot suitable for displaying to the user.
        """

    def steer(self, message: str, interaction: Interaction) -> SteeringOutcome:
        """Queue a correction and request interruption of the active run.

        Args:
            message (str): Correction text to prioritize ahead of ordinary input.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            SteeringOutcome: Result distinguishing interruption, inactivity, and backpressure.
        """


class ConversationCommands:
    """Register user-facing controls for a responsive conversation.

    Args:
        control (ConversationControl): Conversation lifecycle actions used by each command.
    """

    _control: ConversationControl

    def __init__(self, control: ConversationControl) -> None:
        self._control = control

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return responsive interruption, status, steering, and response commands.

        Returns:
            tuple[CommandRegistration, ...]: Commands exposed by this provider.
        """
        return (
            CommandRegistration(self.interrupt, responsive=True),
            CommandRegistration(self.respond, responsive=True),
            CommandRegistration(self.status, responsive=True),
            CommandRegistration(
                self.steer,
                completion=CommandCompletion(),
                responsive=True,
            ),
        )

    def interrupt(self, context: CommandContext) -> None:
        """Interrupt the active run while leaving the conversation open.

        Args:
            context (CommandContext): Interaction used to report the outcome.
        """
        if self._control.request_interrupt():
            context.interaction.info(
                "Interrupt requested; waiting for the active operation to settle..."
            )
        else:
            context.interaction.warning("No agent run is active.")

    def respond(self, context: CommandContext) -> None:
        """Answer the pending interaction request.

        Args:
            context (CommandContext): Interaction used to report a missing request.
        """
        if not self._control.request_response():
            context.interaction.warning("No input request is pending.")

    def status(self, context: CommandContext) -> None:
        """Show conversation activity and the number of queued inputs.

        Args:
            context (CommandContext): Interaction used to display conversation state.
        """
        status = self._control.status_snapshot()
        context.interaction.info(f"{status.state}; {status.queued_inputs} queued inputs.")

    def steer(
        self,
        context: CommandContext,
        message: Annotated[tuple[str, ...], CommandRemainder()] = (),
    ) -> None:
        """Prioritize a correction message and interrupt the active run.

        Args:
            context (CommandContext): Interaction used to report the steering outcome.
            message (tuple[str, ...]): Correction text split into shell tokens. Defaults to an
                empty tuple, which reports the usage requirement.
        """
        text = " ".join(message)
        if not text:
            context.interaction.warning("Usage: /steer <message>")
            return

        outcome = self._control.steer(text, context.interaction)
        if outcome is SteeringOutcome.RUN_INACTIVE:
            context.interaction.warning("No agent run is active; send a normal message instead.")
        elif outcome is SteeringOutcome.INTERRUPT_REJECTED:
            context.interaction.warning("No agent run is active.")
        elif outcome is SteeringOutcome.INTERRUPTED:
            context.interaction.info(
                "Interrupt requested; waiting for the active operation to settle..."
            )
