"""Expose responsive controls for an active conversation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol

from ..commands import CommandContext, CommandRegistration, CommandRemainder
from ..completion import CommandCompletion, CompletionValue
from ..models import CommandCapability
from .models import ConversationStatus, ResponseOutcome, SteeringOutcome

if TYPE_CHECKING:
    from ..interaction import Interaction


class ConversationControl(Protocol):
    """Describe conversation actions available to the command provider."""

    def request_interrupt(self) -> bool:
        """Request interruption of the active run.

        Returns:
            bool: Whether an active run accepted the interruption request.
        """

    def request_response(self, answer: str | None = None) -> ResponseOutcome:
        """Request an interactive response or submit one to the pending interaction.

        Args:
            answer (str | None): Answer text to submit directly, or ``None`` to open the separate
                response prompt.

        Returns:
            ResponseOutcome: Whether the pending request was answered, prompted, or rejected.
        """

    def response_completions(self) -> tuple[CompletionValue, ...]:
        """Return the choices accepted by the pending response request.

        Returns:
            tuple[CompletionValue, ...]: Current response values, or an empty tuple when the
                request accepts unrestricted text or requires masked input.
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
    """Register user-facing controls for an asynchronous conversation.

    Args:
        control (ConversationControl): Conversation lifecycle actions used by each command.
    """

    _control: ConversationControl

    def __init__(self, control: ConversationControl) -> None:
        self._control = control

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return controls annotated with the conversation states that support them.

        Returns:
            tuple[CommandRegistration, ...]: Commands exposed by this provider.
        """
        return (
            CommandRegistration(
                self.interrupt,
                capabilities=frozenset({CommandCapability.RUN_ACTIVE}),
            ),
            CommandRegistration(
                self.respond,
                completion=CommandCompletion(provider=self._control.response_completions),
                capabilities=frozenset({CommandCapability.PENDING_RESPONSE}),
            ),
            CommandRegistration(
                self.status,
                capabilities=frozenset(
                    {CommandCapability.CONVERSATION_ACTIVE, CommandCapability.CONTEXT_READ}
                ),
            ),
            CommandRegistration(
                self.steer,
                completion=CommandCompletion(),
                capabilities=frozenset({CommandCapability.RUN_ACTIVE}),
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

    def respond(
        self,
        context: CommandContext,
        answer: Annotated[tuple[str, ...], CommandRemainder()] = (),
    ) -> None:
        """Answer the pending interaction request or open its dedicated prompt.

        Args:
            context (CommandContext): Interaction used to report a missing request.
            answer (tuple[str, ...]): Optional response text, completed from the pending request's
                selectable choices when it has any. Defaults to an empty tuple.
        """
        text = " ".join(answer)
        outcome = self._control.request_response(text if text.strip() else None)
        if outcome is ResponseOutcome.NO_REQUEST:
            context.interaction.warning("No input request is pending.")
        elif outcome is ResponseOutcome.INVALID:
            context.interaction.warning(
                "Choose one of the listed responses, or use /respond without an answer."
            )
        elif outcome is ResponseOutcome.PROMPT_REQUIRED:
            context.interaction.warning("Use /respond without an answer for this request.")
        elif outcome is ResponseOutcome.MASKED_INPUT_REQUIRED:
            context.interaction.warning(
                "This request requires masked input; use /respond without an answer."
            )

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
