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
        """Submit a direct response when the native prompt does not own input.

        Args:
            answer (str | None): Answer text to submit directly, or ``None`` to keep the request
                in its native prompt.

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

    async def steer_async(self, message: str, interaction: Interaction) -> SteeringOutcome:
        """Queue a correction and request interruption of the active run.

        Args:
            message (str): Correction text to prioritize ahead of ordinary input.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            SteeringOutcome: Result distinguishing interruption, inactivity, and backpressure.

        Raises:
            asyncio.CancelledError: If durable admission is cancelled after its write settles.
        """

    def queue(self, message: str, interaction: Interaction) -> bool:
        """Queue a message to run after the active agent run completes.

        Args:
            message (str): Text to add to the conversation queue.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            bool: Whether the message was admitted.
        """

    async def queue_async(self, message: str, interaction: Interaction) -> bool:
        """Queue a message to run after the active agent run completes.

        Args:
            message (str): Text to add to the conversation queue.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            bool: Whether the message was admitted.

        Raises:
            asyncio.CancelledError: If durable admission is cancelled after its write settles.
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
                self.steer_async,
                name="steer",
                completion=CommandCompletion(),
                capabilities=frozenset({CommandCapability.RUN_ACTIVE}),
            ),
            CommandRegistration(
                self.queue_async,
                name="queue",
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
        """Submit an answer to a pending question when direct responses are available.

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
            context.interaction.warning("Choose one of the listed responses.")
        elif outcome is ResponseOutcome.PROMPT_REQUIRED:
            context.interaction.warning("Answer this request in its active prompt.")
        elif outcome is ResponseOutcome.MASKED_INPUT_REQUIRED:
            context.interaction.warning("Answer this request in its masked prompt.")

    def status(self, context: CommandContext) -> None:
        """Show conversation activity and the number of queued inputs.

        Args:
            context (CommandContext): Interaction used to display conversation state.
        """
        status = self._control.status_snapshot()
        context.interaction.info(f"{status.state}; {status.queued_inputs} queued inputs.")

    async def steer_async(
        self,
        context: CommandContext,
        message: Annotated[tuple[str, ...], CommandRemainder()] = (),
    ) -> None:
        """Prioritize a correction message and interrupt the active run.

        Args:
            context (CommandContext): Interaction used to report the steering outcome.
            message (tuple[str, ...]): Correction text split into shell tokens. Defaults to an
                empty tuple, which reports the usage requirement.

        Returns:
            None: After reporting the control outcome.

        Raises:
            asyncio.CancelledError: If durable admission is cancelled after its write settles.
        """
        text = " ".join(message)
        if not text:
            context.interaction.warning("Usage: /steer <message>")
            return

        outcome = await self._control.steer_async(text, context.interaction)
        if outcome is SteeringOutcome.RUN_INACTIVE:
            context.interaction.warning("No agent run is active; send a normal message instead.")
        elif outcome is SteeringOutcome.INTERRUPT_REJECTED:
            context.interaction.warning("No agent run is active.")
        elif outcome is SteeringOutcome.INTERRUPTED:
            context.interaction.info(
                "Interrupt requested; waiting for the active operation to settle..."
            )

    async def queue_async(
        self,
        context: CommandContext,
        message: Annotated[tuple[str, ...], CommandRemainder()] = (),
    ) -> None:
        """Queue a message to run after the current agent run completes.

        Args:
            context (CommandContext): Interaction used to report the queue outcome.
            message (tuple[str, ...]): Message text split into shell tokens. Defaults to an
                empty tuple, which reports the usage requirement.

        Returns:
            None: After reporting the queue outcome.

        Raises:
            asyncio.CancelledError: If durable admission is cancelled after its write settles.
        """
        text = " ".join(message)
        if not text:
            context.interaction.warning("Usage: /queue <message>")
        elif not await self._control.queue_async(text, context.interaction):
            context.interaction.warning("No agent run is active; send a normal message instead.")
