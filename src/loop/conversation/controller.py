"""Keep terminal interaction responsive around serialized conversation execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, suppress
from contextvars import copy_context
from dataclasses import replace
from itertools import count
from queue import Empty, Full
from threading import Lock, RLock, get_ident
from typing import TYPE_CHECKING, cast

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
)
from prompt_toolkit.document import Document
from prompt_toolkit.patch_stdout import patch_stdout

from ..commands import CommandManager
from ..completion import CompletionManager, CompletionValue
from ..control import RunControl, RunInterrupted
from ..interaction import AsyncInteractionInput, Interaction
from ..models import CommandCapability, ConversationInputMode, SteeringMessage
from ..session.models import InputDeliveredEvent, InputQueuedEvent, InputRecalledEvent
from ..telemetry import telemetry_activity
from ..utils import ChoiceItem, choice_items
from .commands import ConversationCommands
from .interaction import MailboxInteraction
from .mailbox import InputMailbox
from .models import (
    EVENT_CONFIRM,
    EVENT_OUTPUT,
    EVENT_PROMPT,
    EVENT_TERMINAL,
    EVENT_TERMINAL_EXIT,
    AdmittedConversationInput,
    ConversationCompletion,
    ConversationRunState,
    ConversationStatus,
    InputAdmission,
    InputAdmissionKind,
    InteractionCall,
    PendingInputPriorityOutcome,
    PendingInputRecallOutcome,
    ResponseOutcome,
    RunStateChanged,
    SteeringOutcome,
)

if TYPE_CHECKING:
    from ..session import SessionManager

_QUEUE_CAPACITY = 16
_STEERING_PRIORITY = 0
_INPUT_PRIORITY = 1
_STATUS_COMPONENT = "session_controller"


class GuardedCompleter(Completer):
    """Read live completion sources only while their execution owner is idle.

    Args:
        source (Completer): Full completion source for an idle conversation.
        gate (RLock): Lock held while the execution owner accesses live conversation state.
        fallback (Completer): Safe completion source to use while execution is busy.
    """

    _source: Completer
    _gate: RLock
    _fallback: Completer

    def __init__(self, source: Completer, gate: RLock, fallback: Completer) -> None:
        self._source = source
        self._gate = gate
        self._fallback = fallback

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Generator[Completion]:
        """Yield live completions when idle and safe command completions while busy.

        Args:
            document (Document): Current input document.
            complete_event (CompleteEvent): Completion trigger metadata.

        Yields:
            Completion: Suggestions from live providers while idle or safe commands while busy.
        """
        if not self._gate.acquire(blocking=False):
            yield from self._fallback.get_completions(document, complete_event)
            return

        try:
            yield from self._source.get_completions(document, complete_event)
        finally:
            self._gate.release()


class SessionController:
    """Own one responsive UI and one serialized conversation execution lifecycle.

    Construct conversation services with ``interaction`` and ``control`` from this controller,
    then supply the conversation operation to ``run``. ``ApplicationRuntime`` registers a
    ``ConversationCommands`` provider and binds the loop's command manager; direct compositions do
    the same. An unbound controller builds a standalone registry from that provider when it starts.
    All domain operations stay on one execution thread. The UI owns prompts, presentation, and a
    bounded input mailbox. Bound sessions retain accepted inputs until consumption or rejection.
    Plain text during an active run follows the configured queue or steering mode. Explicit
    ``/steer`` interrupts immediately; ``/queue`` queues input after the run. ``/pause`` parks at a
    safe checkpoint, while ``/stop`` settles the run and leaves this conversation owner available
    for ``/continue``. Only application exit shuts down the runtime. Each controller runs once.

    Args:
        interaction (Interaction): UI implementing ``AsyncInteractionInput`` as well as output.
        input_mode (ConversationInputMode): Active-run routing for plain text.
            Defaults to ``queue``.

    Raises:
        TypeError: If the UI does not implement asynchronous input.
        ValueError: If the input mode is neither ``queue`` nor ``steer``.
    """

    _source: Interaction
    _input: AsyncInteractionInput
    _interaction: MailboxInteraction
    _command_manager: CommandManager | None
    _command_completer: CompletionManager
    _control: RunControl
    _inbox: InputMailbox
    _sequence: count
    _gate: RLock
    _mailbox_lock: Lock
    _mailbox_available: asyncio.Event | None
    _input_mode: ConversationInputMode
    _loop: asyncio.AbstractEventLoop | None
    _events: asyncio.Queue[InteractionCall | RunStateChanged] | None
    _pending: InteractionCall | None
    _completer: Completer | None
    _ready: bool
    _idle: bool
    _used: bool
    _ended: bool
    _ui_thread: int | None
    _terminal_owned: bool
    _stopping: bool
    _answer_input: asyncio.Task | None
    _message_input: asyncio.Task | None
    _command_task: asyncio.Future | None
    _discarded: int
    _idle_command_pending: bool
    _draft: str | None
    _draft_bypass_commands: bool
    _input_signal: asyncio.Event
    _async_owner: bool
    _session_manager: SessionManager | None
    _queued_input_ids: set[str]

    def __init__(
        self,
        interaction: Interaction,
        *,
        input_mode: ConversationInputMode = "queue",
    ) -> None:
        if not isinstance(interaction, AsyncInteractionInput):
            raise TypeError("Session controller requires asynchronous input.")
        if input_mode not in ("queue", "steer"):
            raise ValueError("Conversation input mode must be 'queue' or 'steer'.")
        self._source = interaction
        self._input = cast(AsyncInteractionInput, interaction)
        self._interaction = MailboxInteraction(interaction)
        self._command_manager = None
        self._command_completer = CompletionManager()
        self._control = RunControl()
        self._inbox = InputMailbox(_QUEUE_CAPACITY)
        self._sequence = count()
        self._gate = RLock()
        self._mailbox_lock = Lock()
        self._mailbox_available = None
        self._input_mode = input_mode
        self._loop = None
        self._events = None
        self._pending = None
        self._completer = None
        self._ready = False
        self._idle = True
        self._used = False
        self._ended = False
        self._ui_thread = None
        self._terminal_owned = False
        self._stopping = False
        self._answer_input = None
        self._message_input = None
        self._command_task = None
        self._discarded = 0
        self._idle_command_pending = False
        self._draft = None
        self._draft_bypass_commands = False
        self._input_signal = asyncio.Event()
        self._async_owner = False
        self._session_manager = None
        self._queued_input_ids = set()

    @property
    def input_mode(self) -> ConversationInputMode:
        """Return the active-run route for plain text.

        Returns:
            ConversationInputMode: Queue or steering priority for active-run text.
        """
        return self._input_mode

    @input_mode.setter
    def input_mode(self, mode: ConversationInputMode) -> None:
        """Set the active-run route for plain text.

        Args:
            mode (ConversationInputMode): Queue or steering priority for active-run text.

        Returns:
            None: After the active-run input mode is set.

        Raises:
            ValueError: If the mode is neither ``queue`` nor ``steer``.
        """
        if mode not in ("queue", "steer"):
            raise ValueError("Conversation input mode must be 'queue' or 'steer'.")
        self._input_mode = mode

    def bind_command_manager(self, command_manager: CommandManager) -> None:
        """Bind the loop-owned command registry for contextual dispatch and completion.

        Args:
            command_manager (CommandManager): Manager containing this controller's registrations.

        Raises:
            RuntimeError: If command dispatch is bound after the controller starts.
        """
        if self._used:
            raise RuntimeError("Bind the command manager before running the session controller.")
        self._command_manager = command_manager
        command_manager.set_capability_provider(self.command_capabilities)
        self._command_completer = CompletionManager((command_manager.completion_adapter,))

    def bind_session_manager(self, session_manager: SessionManager) -> None:
        """Bind durable session lifecycle records before starting the controller.

        Args:
            session_manager (SessionManager): Session persistence owner for this conversation.

        Raises:
            RuntimeError: If persistence is bound after the controller starts.
        """
        if self._used:
            raise RuntimeError("Bind session persistence before running the session controller.")
        self._session_manager = session_manager
        sequence = (
            max(
                (
                    event.sequence
                    for event in session_manager.session.events
                    if isinstance(event, InputQueuedEvent)
                ),
                default=-1,
            )
            + 1
        )
        self._sequence = count(sequence)

    def command_capabilities(self) -> frozenset[CommandCapability]:
        """Return commands supported by the current controller lifecycle state.

        Returns:
            frozenset[CommandCapability]: Active controller capabilities. Before startup
                and after settlement, only session-idle and context inspection/update
                capabilities are returned, so conversation-only controls stay hidden.
        """
        if not self.running:
            return frozenset(
                {
                    CommandCapability.SESSION_IDLE,
                    CommandCapability.CONTEXT_READ,
                    CommandCapability.CONTEXT_UPDATE,
                }
            )
        if self._ended or self._stopping or self._control.stopped:
            return frozenset({CommandCapability.CONTEXT_READ})

        capabilities = {
            CommandCapability.CONVERSATION_ACTIVE,
            CommandCapability.CONVERSATION_CONTROL,
            CommandCapability.CONTEXT_READ,
            CommandCapability.CONTEXT_UPDATE,
        }
        if self._control.active:
            capabilities.add(CommandCapability.RUN_ACTIVE)
        if self._pending is not None:
            capabilities.add(CommandCapability.PENDING_RESPONSE)
        if self._control.paused or (
            not self._control.active
            and self._session_manager is not None
            and self._session_manager.continuable
        ):
            capabilities.add(CommandCapability.RUN_CONTINUABLE)
        if (
            not self._control.active
            and self._session_manager is not None
            and self._session_manager.retry_pending
        ):
            capabilities.add(CommandCapability.RUN_RETRYABLE)
        if not self._control.active and self._pending is None:
            capabilities.add(CommandCapability.SESSION_IDLE)
        return frozenset(capabilities)

    def request_interrupt(self) -> bool:
        """Interrupt the active run and revoke its separate pending question.

        Returns:
            bool: Whether an active run accepted the interruption request.
        """
        if not self._control.interrupt():
            return False
        telemetry_activity("conversation.run.interrupt_requested", component=_STATUS_COMPONENT)
        request = self._pending
        if request is not None:
            if not request.result.done():
                request.result.set_exception(RunInterrupted())
            if self._pending is request:
                self._pending = None
        if self._answer_input is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._answer_input.cancel)
        return True

    def request_pause(self) -> bool:
        """Park the current run without cancelling its owner or active operation.

        Returns:
            bool: Whether an active run accepted the pause request.
        """
        if not self._control.pause():
            return False
        telemetry_activity("conversation.run.pause_requested", component=_STATUS_COMPONENT)
        return True

    def request_stop_run(self) -> bool:
        """Settle the active run without stopping the conversation owner.

        Returns:
            bool: Whether the stop request was accepted.
        """
        if not self._control.request_stop_run():
            return False
        telemetry_activity("conversation.stop_requested", component=_STATUS_COMPONENT)
        return True

    def request_response(self, answer: str | None = None) -> ResponseOutcome:
        """Submit a direct answer when the native prompt does not own input.

        Args:
            answer (str | None): Text to submit, or ``None`` to keep the active native prompt.

        Returns:
            ResponseOutcome: Whether the request was prompted, answered, or rejected.
        """
        request = self._pending
        if request is None:
            return ResponseOutcome.NO_REQUEST
        if request.result.done():
            if self._pending is request:
                self._pending = None
            return ResponseOutcome.NO_REQUEST
        if answer is None:
            return ResponseOutcome.PROMPTED
        if request.kind == EVENT_TERMINAL:
            return ResponseOutcome.PROMPT_REQUIRED
        if request.kind == EVENT_PROMPT and request.kwargs.get("secret", False):
            return ResponseOutcome.MASKED_INPUT_REQUIRED
        if self._answer_input is not None:
            return ResponseOutcome.PROMPT_REQUIRED
        if not answer.strip() and request.kind == EVENT_CONFIRM:
            request.result.set_result(request.kwargs.get("default", False))
            self._pending = None
            return ResponseOutcome.ANSWERED
        if not answer.strip() and request.kind == EVENT_PROMPT:
            default = request.kwargs.get("default")
            if default is None:
                return ResponseOutcome.INVALID
            request.result.set_result(default)
            self._pending = None
            return ResponseOutcome.ANSWERED

        if request.kind == EVENT_CONFIRM:
            choices = choice_items({True: "Yes", False: "No"}, index=("y", "n"))
        elif request.kind == EVENT_PROMPT:
            if request.kwargs.get("choices") is None:
                request.result.set_result(answer)
                self._pending = None
                return ResponseOutcome.ANSWERED
            try:
                choices = self._pending_response_choices(request)
            except ValueError:
                return ResponseOutcome.INVALID
        else:
            return ResponseOutcome.PROMPT_REQUIRED

        selected = answer.strip().casefold()
        choice = next(
            (item for item in choices if selected in {item.index.casefold(), item.name.casefold()}),
            None,
        )
        if choice is None:
            return ResponseOutcome.INVALID
        request.result.set_result(choice.value)
        self._pending = None
        return ResponseOutcome.ANSWERED

    def response_completions(self) -> tuple[CompletionValue, ...]:
        """Return accepted choices for the current response request.

        Returns:
            tuple[CompletionValue, ...]: Choices accepted by the pending prompt. Empty when the
                prompt is absent, unrestricted, secret, or requires an exclusive terminal handoff.
        """
        request = self._pending
        if request is None or request.kind == EVENT_TERMINAL:
            return ()
        if request.kind == EVENT_CONFIRM:
            choices = choice_items({True: "Yes", False: "No"}, index=("y", "n"))
        elif request.kind == EVENT_PROMPT:
            if request.kwargs.get("secret", False):
                return ()
            if request.kwargs.get("choices") is None:
                return ()
            try:
                choices = self._pending_response_choices(request)
            except ValueError:
                return ()
        else:
            return ()
        return tuple(
            CompletionValue(item.name, item.description or f"Select {item.index}.")
            for item in choices
        )

    @staticmethod
    def _pending_response_choices(request: InteractionCall) -> tuple[ChoiceItem, ...]:
        """Normalize pending prompt choices once for completion and direct answer handling."""
        choices = request.kwargs["choices"]
        if not isinstance(choices, Mapping):
            choices = tuple(choices)
            request.kwargs["choices"] = choices
        index = request.kwargs.get("index")
        if index is not None and not isinstance(index, Mapping):
            index = tuple(index)
            request.kwargs["index"] = index
        return choice_items(choices, index=index)

    def status_snapshot(self) -> ConversationStatus:
        """Return a stable view of conversation activity and queued inputs.

        Returns:
            ConversationStatus: Current activity and pending ordinary input count.
        """
        state = (
            "Paused"
            if self._control.paused
            else "Stopping"
            if self._control.stop_requested
            else "Retry pending"
            if not self._control.active
            and self._session_manager is not None
            and self._session_manager.retry_pending
            else "Continue pending"
            if not self._control.active
            and self._session_manager is not None
            and self._session_manager.continuable
            else "Recovery pending"
            if not self._control.active
            and self._session_manager is not None
            and self._session_manager.recovery_pending
            else "Interrupting"
            if self._control.interrupted
            else "Cancelling"
            if self._control.cancelling
            else "Cancelled"
            if self._control.cancelled
            else "Steering"
            if self._control.steering_pending
            else "Waiting for your response"
            if self._pending is not None
            else ("Idle" if self._idle else "Working")
        )
        return ConversationStatus(state=state, queued_inputs=self._inbox.size())

    def steer(self, message: str, interaction: Interaction) -> SteeringOutcome:
        """Queue a correction and interrupt the active run.

        Args:
            message (str): Correction text to prioritize ahead of ordinary input.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            SteeringOutcome: Result distinguishing interruption, inactivity, and backpressure.
        """
        if not self._control.active:
            return SteeringOutcome.RUN_INACTIVE
        if self._session_manager is not None:
            event = self._session_manager.queue_input(
                message,
                priority=_STEERING_PRIORITY,
                bypass_commands=True,
            )
            admission = self._admission_from_event(event)
            if not self._enqueue_steering(message, admission=admission, report_full=False):
                self._reject_admission(admission)
                interaction.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
                return SteeringOutcome.QUEUE_FULL
            if not self.request_interrupt():
                return SteeringOutcome.INTERRUPT_REJECTED
            return SteeringOutcome.INTERRUPTED
        if not self._enqueue(
            SteeringMessage(message),
            priority=_STEERING_PRIORITY,
            interaction=interaction,
        ):
            return SteeringOutcome.QUEUE_FULL
        if not self.request_interrupt():
            return SteeringOutcome.INTERRUPT_REJECTED
        return SteeringOutcome.INTERRUPTED

    async def steer_async(self, message: str, interaction: Interaction) -> SteeringOutcome:
        """Persist a steering input and interrupt the active run when one is running.

        Args:
            message (str): Correction text to prioritize ahead of ordinary queued input.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            SteeringOutcome: Result distinguishing interruption, queued input, inactivity, and
                backpressure.

        Raises:
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        if not self._control.active and not self.running:
            return SteeringOutcome.RUN_INACTIVE
        admission = await self._durable_admission_async(
            message,
            priority=_STEERING_PRIORITY,
            bypass_commands=True,
        )
        if self._control.active:
            if not await self._enqueue_steering_async(
                message,
                admission=admission,
                report_full=False,
            ):
                await self._reject_admission_async(admission)
                interaction.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
                return SteeringOutcome.QUEUE_FULL
            if self.request_interrupt():
                return SteeringOutcome.INTERRUPTED
            return (
                SteeringOutcome.INTERRUPT_REJECTED
                if self._control.active
                else SteeringOutcome.QUEUED
            )
        if not await self._enqueue_async(message, admission=admission, report_full=False):
            await self._reject_admission_async(admission)
            interaction.warning(
                "Input queue is full; this input was not accepted. Try again later."
            )
            return SteeringOutcome.QUEUE_FULL
        return SteeringOutcome.QUEUED

    def queue(self, message: str, interaction: Interaction) -> bool:
        """Admit an explicit message for execution after the active run completes.

        Args:
            message (str): Text to add at ordinary queue priority.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            bool: Whether the active run admitted the message.
        """
        if not self._control.active:
            return False
        if self._session_manager is not None:
            event = self._session_manager.queue_input(
                message,
                priority=_INPUT_PRIORITY,
                bypass_commands=True,
            )
            admission = self._admission_from_event(event)
            if self._enqueue(
                message,
                admission=admission,
                interaction=interaction,
                report_full=False,
            ):
                return True
            self._reject_admission(admission)
            interaction.warning(
                "Input queue is full; this input was not accepted. Try again later."
            )
            return False
        return self._enqueue(SteeringMessage(message), interaction=interaction)

    async def queue_async(self, message: str, interaction: Interaction) -> bool:
        """Persist a follow-up before admitting it at ordinary queue priority.

        Args:
            message (str): Text to add after the active run.
            interaction (Interaction): UI used to report queue admission feedback.

        Returns:
            bool: Whether the active conversation admitted the message.

        Raises:
            asyncio.CancelledError: If persistence settles after cancellation.
        """
        if not self._control.active and not self.running:
            return False
        admission = await self._durable_admission_async(
            message,
            priority=_INPUT_PRIORITY,
            bypass_commands=True,
        )
        if await self._enqueue_async(
            message,
            admission=admission,
            interaction=interaction,
            report_full=False,
        ):
            return True
        await self._reject_admission_async(admission)
        interaction.warning("Input queue is full; this input was not accepted. Try again later.")
        return False

    async def reprioritize_latest_async(
        self,
        *,
        steer: bool,
    ) -> PendingInputPriorityOutcome:
        """Change the newest waiting mailbox input and request interruption when steering.

        Args:
            steer (bool): Whether to move the input to steering priority.

        Returns:
            PendingInputPriorityOutcome: Whether an input was updated and a run interrupted.

        Raises:
            asyncio.CancelledError: If durable persistence settles after cancellation.
        """
        await self._acquire_mailbox_lock_async()
        try:
            session_manager = self._session_manager
            session_id = session_manager.session.id if session_manager is not None else None
            delivered_ids = (
                {
                    event.input_id
                    for event in session_manager.session.events
                    if isinstance(event, InputDeliveredEvent)
                }
                if session_manager is not None
                else set()
            )
            eligible = [
                item
                for item in self._inbox.snapshot()
                if item.kind in {InputAdmissionKind.MESSAGE, InputAdmissionKind.RESUME}
                and (item.input_id is None or item.session_id == session_id)
                and item.input_id not in delivered_ids
            ]
            if not eligible:
                return PendingInputPriorityOutcome.NO_PENDING_INPUT
            target = max(eligible, key=lambda item: item.sequence)
            priority = _STEERING_PRIORITY if steer else _INPUT_PRIORITY
            if target.priority != priority and target.input_id is not None:
                if session_manager is None:
                    return PendingInputPriorityOutcome.NO_PENDING_INPUT
                try:
                    await session_manager.change_input_priority_async(target.input_id, priority)
                except asyncio.CancelledError:
                    persisted = next(
                        (
                            item
                            for item in session_manager.session.pending_inputs
                            if item.input_id == target.input_id
                        ),
                        None,
                    )
                    if persisted is not None and persisted.priority == priority:
                        self._inbox.reprioritize(target, priority)
                        self._withdraw_steering(target)
                    raise
            if target.priority != priority:
                if not self._inbox.reprioritize(target, priority):
                    return PendingInputPriorityOutcome.NO_PENDING_INPUT
                self._withdraw_steering(target)
        finally:
            self._release_mailbox_lock()
        if steer and self._control.active and self.request_interrupt():
            return PendingInputPriorityOutcome.INTERRUPTED
        return PendingInputPriorityOutcome.UPDATED

    async def recall_latest_async(self) -> PendingInputRecallOutcome:
        """Return the newest waiting mailbox message to the active editor.

        Returns:
            PendingInputRecallOutcome: Whether a waiting message was recalled.

        Raises:
            asyncio.CancelledError: If durable persistence settles after cancellation.
        """
        await self._acquire_mailbox_lock_async()
        try:
            session_manager = self._session_manager
            session_id = session_manager.session.id if session_manager is not None else None
            delivered_ids = (
                {
                    event.input_id
                    for event in session_manager.session.events
                    if isinstance(event, InputDeliveredEvent)
                }
                if session_manager is not None
                else set()
            )
            pending_ids = (
                {event.input_id for event in session_manager.session.pending_inputs}
                if session_manager is not None
                else set()
            )
            eligible = [
                item
                for item in self._inbox.snapshot()
                if item.kind in {InputAdmissionKind.MESSAGE, InputAdmissionKind.RESUME}
                and isinstance(item.message, str)
                and (
                    item.input_id is None
                    or (
                        session_manager is not None
                        and item.session_id == session_id
                        and item.input_id in pending_ids
                        and item.input_id not in delivered_ids
                    )
                )
            ]
            if not eligible:
                return PendingInputRecallOutcome.NO_PENDING_INPUT
            target = max(eligible, key=lambda item: item.sequence)
            message = cast(str, target.message)
            if target.input_id is not None and session_manager is not None:
                try:
                    await session_manager.recall_input_async(target.input_id)
                except asyncio.CancelledError:
                    recalled = any(
                        isinstance(event, InputRecalledEvent) and event.input_id == target.input_id
                        for event in session_manager.session.events
                    )
                    if recalled:
                        self._remove_recalled_input(target)
                        self._draft = message
                        self._draft_bypass_commands = target.bypass_commands
                    raise
            self._remove_recalled_input(target)
            self._draft = message
            self._draft_bypass_commands = target.bypass_commands
        finally:
            self._release_mailbox_lock()
        return PendingInputRecallOutcome.RECALLED

    def _remove_recalled_input(self, admission: InputAdmission) -> None:
        """Remove a recalled admission and release its durable mailbox identity."""
        self._inbox.remove(admission)
        self._withdraw_steering(admission)
        if admission.input_id is not None:
            self._queued_input_ids.discard(admission.input_id)

    def _withdraw_steering(self, admission: InputAdmission) -> None:
        """Withdraw the steering request associated with this exact admission."""
        if admission.priority == _STEERING_PRIORITY:
            self._control.withdraw_steering(id(admission))

    @property
    def interaction(self) -> Interaction:
        """Return the interaction to inject into conversation-owned services.

        During execution, its synchronous methods must be called outside the UI thread. UI
        code uses the original surface instead.

        Returns:
            Interaction: UI-thread dispatch during execution and direct interaction during setup.
        """
        return self._interaction

    @property
    def control(self) -> RunControl:
        """Return shutdown, interruption, and pause control for the conversation's agent runner.

        Returns:
            RunControl: Cancellation signals shared with the serialized execution owner.
        """
        return self._control

    @property
    def running(self) -> bool:
        """Return whether execution still owns its conversation services.

        Returns:
            bool: Whether the controller has started and not yet settled.
        """
        return self._loop is not None

    def stop(self) -> None:
        """Request safe shutdown from the UI or another thread.

        The controller drains pending interaction and waits for active synchronous operations to
        settle. Calling this before startup prevents execution from accepting its first input.

        Raises:
            RuntimeError: If the active UI loop rejects shutdown scheduling.
        """
        loop = self._loop
        if loop is None:
            self._control.stop()
        else:
            try:
                loop.call_soon_threadsafe(self._stop)
            except RuntimeError:
                if self._loop is not None:
                    raise
                self._control.stop()

    def run(
        self,
        conversation: Callable[[Callable[[Completer], object]], None],
        *,
        route_idle_commands: bool = False,
    ) -> None:
        """Run a synchronous conversation while accepting input and dispatching UI requests.

        The operation must use its supplied input callback for ordinary conversation messages.
        Nested prompts and all output use this controller's ``interaction``. Shutdown stops
        admission and waits for the execution owner to settle before returning or raising. This
        method owns an event loop for mailbox delivery and rejects calls from an existing loop.

        Args:
            conversation (Callable[[Callable[[Completer], object]], None]): Synchronous operation
                accepting a message reader. The reader takes a completer and returns text or
                ``False`` on shutdown.
            route_idle_commands (bool): Pass commands entered at an idle boundary through the
                conversation input reader. This lets the execution owner dispatch commands that
                may prompt; their prompts open immediately on the async UI. Defaults to ``False``
                for standalone controllers.

        Returns:
            None: After the synchronous callback and all owned operations settle.

        Raises:
            RuntimeError: If called from an event loop or this controller has already run.
            BaseException: If the conversation or UI fails.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Synchronous SessionController.run() cannot run inside an event loop; "
                "await SessionController.run_async() instead."
            )
        with asyncio.Runner() as runner:
            runner.run(
                self._run_sync_callback_async(
                    conversation,
                    route_idle_commands=route_idle_commands,
                )
            )

    async def _run_sync_callback_async(
        self,
        conversation: Callable[[Callable[[Completer], object]], None],
        *,
        route_idle_commands: bool = False,
    ) -> None:
        """Host a synchronous callback on a worker while this loop owns UI delivery."""
        if self._command_manager is None:
            self.bind_command_manager(
                CommandManager(
                    providers=(ConversationCommands(self),),
                    interaction=self._source,
                )
            )
        if self._used:
            raise RuntimeError("Session controller can only run once.")
        self._used = True
        self._loop = asyncio.get_running_loop()
        self._ui_thread = get_ident()
        self._events = asyncio.Queue(maxsize=1)
        self._interaction.bind(self._send, self._send_async)
        await self._restore_pending_inputs_async()
        telemetry_activity("conversation.started", component=_STATUS_COMPONENT)
        primary = None
        try:
            with (
                patch_stdout(raw=True),
                ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="loop-conversation",
                ) as executor,
            ):
                worker = self._loop.run_in_executor(
                    executor,
                    copy_context().run,
                    self._execute,
                    conversation,
                )
                try:
                    await self._drive_async(worker, route_idle_commands=route_idle_commands)
                except BaseException as error:
                    primary = error
                    self._stop()
                    cleanup = asyncio.create_task(self._settle_async(worker))
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            continue
                    command_failure = cleanup.result()
                    if command_failure is not None and command_failure is not error:
                        error.add_note(
                            f"Command also stopped with {type(command_failure).__name__}."
                        )
                    failure = worker.exception()
                    if failure is not None and failure is not error:
                        error.add_note(f"Conversation also stopped with {type(failure).__name__}.")
                    raise
        finally:
            self._interaction.bind(None)
            self._loop = None
            self._ui_thread = None
            self._events = None
            telemetry_activity("conversation.stopped", component=_STATUS_COMPONENT)
            if self._discarded:
                try:
                    self._source.warning(
                        f"Stopped; {self._discarded} queued inputs were not executed."
                    )
                except Exception as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        f"Queue cancellation reporting also failed with {type(error).__name__}."
                    )
        if self._control.stopped and not self._ended:
            self._source.conversation_ended()

    async def run_async(
        self,
        conversation: Callable[[Callable[[Completer], Awaitable[object]]], Awaitable[None]],
        *,
        route_idle_commands: bool = False,
    ) -> None:
        """Run async conversation ownership on the UI event loop while accepting input.

        The conversation must await its supplied message reader. Async interaction requests are
        routed through the same UI mailbox as the synchronous controller path.

        Args:
            conversation (Callable[[Callable[[Completer], Awaitable[object]]], Awaitable[None]]):
                Async operation receiving an awaitable ordinary-message reader.
            route_idle_commands (bool): Pass idle-boundary commands to the conversation owner.
                Defaults to ``False``.

        Raises:
            RuntimeError: If this controller has already run.
            asyncio.CancelledError: After owned execution settles when this task is cancelled.
            BaseException: If the conversation or UI fails after owned execution settles.

        Returns:
            None: After the conversation and its owned operations have settled.
        """
        if self._command_manager is None:
            self.bind_command_manager(
                CommandManager(
                    providers=(ConversationCommands(self),),
                    interaction=self._source,
                )
            )
        if self._used:
            raise RuntimeError("Session controller can only run once.")
        self._used = True
        self._loop = asyncio.get_running_loop()
        self._ui_thread = get_ident()
        self._events = asyncio.Queue(maxsize=1)
        self._input_signal = asyncio.Event()
        self._async_owner = True
        self._interaction.bind(self._send, self._send_async)
        await self._restore_pending_inputs_async()
        telemetry_activity("conversation.started", component=_STATUS_COMPONENT)
        primary = None
        try:
            with patch_stdout(raw=True):
                worker = asyncio.create_task(self._execute_async(conversation))
                try:
                    await self._drive_async(worker, route_idle_commands=route_idle_commands)
                except BaseException as error:
                    primary = error
                    self._stop()
                    cleanup = asyncio.create_task(self._settle_async(worker))
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            continue
                    command_failure = cleanup.result()
                    if command_failure is not None and command_failure is not error:
                        error.add_note(
                            f"Command also stopped with {type(command_failure).__name__}."
                        )
                    failure = worker.exception()
                    if failure is not None and failure is not error:
                        error.add_note(f"Conversation also stopped with {type(failure).__name__}.")
                    raise
        finally:
            self._interaction.bind(None)
            self._async_owner = False
            self._loop = None
            self._ui_thread = None
            self._events = None
            telemetry_activity("conversation.stopped", component=_STATUS_COMPONENT)
            if self._discarded:
                try:
                    self._source.warning(
                        f"Stopped; {self._discarded} queued inputs were not executed."
                    )
                except Exception as error:
                    if primary is None:
                        raise
                    primary.add_note(
                        f"Queue cancellation reporting also failed with {type(error).__name__}."
                    )
        if self._control.stopped and not self._ended:
            self._source.conversation_ended()

    def _execute(self, conversation: Callable) -> None:
        """Keep all conversation state on one owner and normalize requested interruption."""
        try:
            with self._gate:
                conversation(self._receive)
        except KeyboardInterrupt:
            if not self._control.stopped:
                raise

    async def _execute_async(self, conversation: Callable) -> None:
        """Run async conversation code and normalize requested interruption."""
        try:
            await conversation(self._receive_async)
        except KeyboardInterrupt:
            if not self._control.stopped:
                raise

    def _send(self, event: InteractionCall | RunStateChanged) -> object:
        """Send one acknowledged UI request from the execution owner."""
        if get_ident() == self._ui_thread:
            if not self._async_owner:
                raise RuntimeError("Mailbox interaction must run outside the UI thread.")
            if not isinstance(event, InteractionCall) or event.kind not in {
                EVENT_OUTPUT,
                EVENT_TERMINAL_EXIT,
            }:
                raise RuntimeError("Synchronous mailbox input cannot run on the UI thread.")
            try:
                result = event.function(*event.args, **event.kwargs)
            except BaseException as error:
                event.result.set_exception(error)
                raise
            else:
                event.result.set_result(result)
                if event.function == self._source.conversation_ended:
                    self._ended = True
                return result
            finally:
                if event.kind == EVENT_TERMINAL_EXIT:
                    self._terminal_owned = False
        loop = cast(asyncio.AbstractEventLoop, self._loop)
        events = cast(asyncio.Queue, self._events)
        asyncio.run_coroutine_threadsafe(events.put(event), loop).result()
        return event.result.result()

    async def _send_async(self, event: InteractionCall | RunStateChanged) -> object:
        """Send and await one correlated UI request from async conversation ownership."""
        events = cast(asyncio.Queue, self._events)
        await events.put(event)
        return await asyncio.wrap_future(event.result)

    def _receive(self, completer: Completer) -> object:
        """Release domain ownership while waiting for the next admitted input."""
        if self._control.stopped:
            return False
        self._gate.release()
        try:
            self._send(RunStateChanged(completer))
            commands_only = (
                self._session_manager is not None and self._session_manager.recovery_pending
            )
            while True:
                with self._mailbox_access():
                    try:
                        admission = self._inbox.get_nowait(commands_only=commands_only)
                        break
                    except Empty:
                        pass
                if not self._inbox.wait_until_not_empty(commands_only=commands_only):
                    return False
        finally:
            # Restore the owner's existing scope after lending the gate to completion readers.
            self._gate.acquire()
        if admission.kind is InputAdmissionKind.STOP:
            return False
        if self._control.stopped:
            return False
        if admission.input_id is not None:
            if (
                self._session_manager is None
                or admission.session_id != self._session_manager.session.id
            ):
                self._queued_input_ids.discard(admission.input_id)
                self._source.warning("Queued input belongs to a different session and was skipped.")
                return self._receive(completer)
            self._session_manager.mark_input_delivered(admission.input_id)
        self._send(RunStateChanged(completer, ConversationRunState.WORKING))
        if admission.input_id is not None and admission.session_id is not None:
            return AdmittedConversationInput(
                input_id=admission.input_id,
                session_id=admission.session_id,
                message=str(admission.message),
                bypass_commands=admission.bypass_commands,
            )
        return admission.message

    async def _receive_async(self, completer: Completer) -> object:
        """Await the next admitted input without blocking the UI event loop."""
        if self._control.stopped:
            return False
        await self._send_async(RunStateChanged(completer))
        admission = await self._get_input_async()
        if admission.kind is InputAdmissionKind.STOP:
            return False
        if self._control.stopped:
            return False
        if admission.input_id is not None:
            if (
                self._session_manager is None
                or admission.session_id != self._session_manager.session.id
            ):
                self._queued_input_ids.discard(admission.input_id)
                self._source.warning("Queued input belongs to a different session and was skipped.")
                return await self._receive_async(completer)
            await self._session_manager.mark_input_delivered_async(admission.input_id)
        await self._send_async(RunStateChanged(completer, state=ConversationRunState.WORKING))
        if admission.input_id is not None and admission.session_id is not None:
            return AdmittedConversationInput(
                input_id=admission.input_id,
                session_id=admission.session_id,
                message=str(admission.message),
                bypass_commands=admission.bypass_commands,
            )
        return admission.message

    async def _drive_async(self, worker: asyncio.Future, *, route_idle_commands: bool) -> None:
        """Multiplex execution requests and user input without sharing domain state."""
        events = cast(asyncio.Queue, self._events)
        incoming = asyncio.create_task(events.get())
        typing = None
        active_waiter = None
        typing_bypass_commands = False
        worker_done = False
        try:
            while True:
                if (
                    not worker_done
                    and not self._control.stopped
                    and active_waiter is None
                    and self._idle_command_pending
                ):
                    active_waiter = asyncio.create_task(self._control.wait_until_active_async())
                if (
                    not worker_done
                    and self._ready
                    and typing is None
                    and not self._control.stopped
                    and not self._terminal_owned
                    and not self._idle_command_pending
                    and self._command_task is None
                ):
                    if self._draft is not None:
                        typing_bypass_commands = self._draft_bypass_commands
                        typing = asyncio.create_task(
                            self._input.prompt_async(
                                completer=self._completer,
                                default="",
                                initial=self._draft,
                            )
                        )
                        self._draft = None
                        self._draft_bypass_commands = False
                    else:
                        typing_bypass_commands = False
                        typing = asyncio.create_task(
                            self._input.prompt_async(completer=self._completer, default="")
                        )
                    self._message_input = typing
                watched = {incoming}
                if not worker_done:
                    watched.add(worker)
                if typing is not None:
                    watched.add(typing)
                command_task = self._command_task
                if command_task is not None:
                    watched.add(command_task)
                if active_waiter is not None:
                    watched.add(active_waiter)
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if incoming in done:
                    event = incoming.result()
                    incoming = asyncio.create_task(events.get())
                    if (
                        isinstance(event, RunStateChanged)
                        and event.state is ConversationRunState.IDLE
                        and active_waiter is not None
                    ):
                        active_waiter.cancel()
                        active_waiter = None
                    if (
                        isinstance(event, InteractionCall)
                        and event.kind not in {EVENT_OUTPUT, EVENT_TERMINAL_EXIT}
                        and typing is not None
                    ):
                        if typing.done():
                            submitted = typing
                            typing = None
                            self._message_input = None
                            if not submitted.cancelled():
                                await self._accept_async(
                                    submitted,
                                    route_idle_commands=route_idle_commands,
                                    bypass_commands=typing_bypass_commands,
                                )
                        else:
                            draft = self._input.current_input()
                            if isinstance(draft, str):
                                self._draft = draft
                                self._draft_bypass_commands = typing_bypass_commands
                            self._input.set_current_input("")
                            typing.cancel()
                            await asyncio.gather(typing, return_exceptions=True)
                            typing = None
                            self._message_input = None
                    await self._deliver_async(event)
                if typing is not None and typing in done:
                    submitted = typing
                    typing = None
                    self._message_input = None
                    if not submitted.cancelled():
                        await self._accept_async(
                            submitted,
                            route_idle_commands=route_idle_commands,
                            bypass_commands=typing_bypass_commands,
                        )
                if worker in done:
                    completion = ConversationCompletion(
                        asyncio.CancelledError() if worker.cancelled() else worker.exception()
                    )
                    if completion.error is not None:
                        raise completion.error
                    worker_done = True
                if active_waiter is not None and active_waiter in done:
                    active_waiter.result()
                    self._idle_command_pending = False
                    active_waiter = None
                if command_task is not None and command_task in done:
                    self._command_task = None
                    command_task.result()
                    if cast(CommandManager, self._command_manager).exit_requested:
                        self._stop()
                        self._source.info("Stopping after the active operation settles...")
                if worker_done and self._command_task is None:
                    return
        except BaseException:
            self._stop()
            raise
        finally:
            incoming.cancel()
            if not incoming.cancelled() and incoming.done():
                await self._deliver_async(incoming.result())
            if typing is not None:
                typing.cancel()
            if active_waiter is not None:
                active_waiter.cancel()
            await asyncio.gather(
                incoming,
                *([typing] if typing is not None else []),
                *([active_waiter] if active_waiter is not None else []),
                return_exceptions=True,
            )
            self._message_input = None

    async def _deliver_async(self, event: InteractionCall | RunStateChanged) -> None:
        """Apply one UI event or await its blocking request through the native input surface."""
        if isinstance(event, RunStateChanged):
            self._ready = True
            self._idle = event.state is ConversationRunState.IDLE
            if event.state is ConversationRunState.IDLE:
                self._idle_command_pending = False
                await self._restore_pending_inputs_async()
            self._completer = GuardedCompleter(
                event.completer,
                self._gate,
                self._command_completer,
            )
            event.result.set_result(None)
            return
        if event.result.done():
            return
        if event.kind not in {EVENT_OUTPUT, EVENT_TERMINAL_EXIT}:
            if self._interaction_revoked(event):
                return
            with ExitStack() as resources:
                if self._session_manager is not None:
                    try:
                        resources.enter_context(
                            self._session_manager.interaction_execution(event.session_id)
                        )
                        await self._session_manager.record_interaction_requested_async(
                            event.request_id,
                            kind=event.kind,
                            session_id=event.session_id,
                            run_id=event.run_id,
                            operation_id=event.operation_id,
                        )
                    except BaseException as error:
                        if not event.result.done():
                            event.result.set_exception(error)
                        if isinstance(error, asyncio.CancelledError):
                            raise
                        return
                self._pending = event
                answered, result = await self._answer_async()
            if answered and not event.result.done():
                event.result.set_result(result)
            return
        try:
            result = event.function(*event.args, **event.kwargs)
        except BaseException as error:  # noqa: BLE001 - return failures to the blocked owner
            event.result.set_exception(error)
        else:
            if event.function == self._source.conversation_ended:
                self._ended = True
            event.result.set_result(result)
        finally:
            if event.kind == EVENT_TERMINAL_EXIT:
                self._terminal_owned = False

    async def _accept_async(
        self,
        submitted: asyncio.Task,
        *,
        route_idle_commands: bool,
        bypass_commands: bool = False,
    ) -> None:
        """Resolve input without hiding errors, then apply controls or admit ordinary text."""
        if self._stopping:
            with suppress(asyncio.CancelledError):
                submitted.result()
            return
        value = submitted.result()
        if value is False:
            self._stop()
            self._source.info("Stopping after the active operation settles...")
            return
        command_manager = cast(CommandManager, self._command_manager)
        if (
            not bypass_commands
            and route_idle_commands
            and (
                self._idle
                or (
                    self._session_manager is not None
                    and self._session_manager.recovery_pending
                    and not self._control.active
                    and self._pending is None
                )
            )
            and command_manager.is_user_command(value)
        ):
            self._idle_command_pending = await self._enqueue_async(value, command=True)
            return
        if not bypass_commands and command_manager.is_user_command(value):
            self._command_task = asyncio.create_task(
                self._run_command_async(command_manager, value)
            )
            return
        if value == "":
            self._source.warning("Please enter a message!")
        elif self._session_manager is not None:
            resume = (
                self._control.paused
                or self._control.stop_requested
                or (self._session_manager.recovery_pending and not self._control.active)
            )
            steer = self._control.active and (resume or self._input_mode == "steer")
            priority = _STEERING_PRIORITY if steer or resume else _INPUT_PRIORITY
            admission = await self._durable_admission_async(
                str(value),
                priority=priority,
                bypass_commands=True,
            )
            if resume:
                admission = replace(admission, kind=InputAdmissionKind.RESUME)
            admitted = (
                await self._enqueue_steering_async(
                    str(value), admission=admission, report_full=False
                )
                if steer
                else await self._enqueue_async(
                    admission.message,
                    admission=admission,
                    report_full=False,
                )
            )
            if admitted and resume:
                self._control.continue_run()
            if not admitted:
                await self._reject_admission_async(admission)
                self._source.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
        elif self._control.active and (self._control.paused or self._input_mode == "steer"):
            if await self._enqueue_steering_async(value):
                self._control.continue_run()
        else:
            await self._enqueue_async(value)

    def _run_command(self, command_manager: CommandManager, value: str) -> bool:
        """Dispatch one user command through the controller-owned interaction mailbox."""
        try:
            return command_manager.handle_user_command(value, interaction=self._interaction)
        except (RunInterrupted, KeyboardInterrupt) as error:
            if not isinstance(error, RunInterrupted) and not self._control.stopped:
                raise
            return True

    async def _run_command_async(self, command_manager: CommandManager, value: str) -> bool:
        """Dispatch one user command through the async owner and interaction mailbox."""
        owns_async_dispatch = not self._async_owner
        if owns_async_dispatch:
            self._async_owner = True
        try:
            return await command_manager.handle_user_command_async(
                value,
                interaction=self._interaction,
            )
        except (RunInterrupted, KeyboardInterrupt) as error:
            if not isinstance(error, RunInterrupted) and not self._control.stopped:
                raise
            return True
        finally:
            if owns_async_dispatch:
                self._async_owner = False

    @contextmanager
    def _mailbox_access(self) -> Generator[None]:
        """Release synchronous mailbox access with the same waiter notification as async access."""
        self._mailbox_lock.acquire()
        try:
            yield
        finally:
            self._release_mailbox_lock()

    async def _acquire_mailbox_lock_async(self) -> None:
        """Wait cooperatively until the mailbox is available to the async owner."""
        available = self._mailbox_available
        if available is None:
            available = asyncio.Event()
            self._mailbox_available = available
        while not self._mailbox_lock.acquire(blocking=False):
            available.clear()
            if self._mailbox_lock.acquire(blocking=False):
                return
            await available.wait()

    def _release_mailbox_lock(self) -> None:
        """Release the mailbox lock and wake any async owner waiting for it."""
        self._mailbox_lock.release()
        available = self._mailbox_available
        if available is None:
            return
        loop = self._loop
        if loop is None or get_ident() == self._ui_thread:
            available.set()
        else:
            loop.call_soon_threadsafe(available.set)

    async def _get_input_async(self) -> InputAdmission:
        """Dequeue one mailbox item without racing a durable priority transition."""
        while True:
            await self._acquire_mailbox_lock_async()
            try:
                self._input_signal.clear()
                try:
                    return self._inbox.get_nowait(
                        commands_only=(
                            self._session_manager is not None
                            and self._session_manager.recovery_pending
                        )
                    )
                except Empty:
                    pass
            finally:
                self._release_mailbox_lock()
            await self._input_signal.wait()

    def _put_mailbox_item_locked(self, admission: InputAdmission) -> bool:
        """Add one input while the mailbox lock is held, unless shutdown has started."""
        if self._stopping:
            return False
        return self._inbox.try_put(admission)

    def _report_enqueued(self, interaction: Interaction) -> None:
        """Wake the input owner and report one accepted mailbox admission."""
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        pending = self._inbox.size()
        telemetry_activity(
            "conversation.input.queued",
            component=_STATUS_COMPONENT,
            pending=pending,
        )
        if not self._idle:
            interaction.info(f"Queued input ({pending} pending).")

    def _enqueue(
        self,
        value: object,
        *,
        priority: int = _INPUT_PRIORITY,
        interaction: Interaction | None = None,
        admission: InputAdmission | None = None,
        report_full: bool = True,
    ) -> bool:
        """Admit bounded input without losing existing work when the mailbox is full."""
        active_interaction = interaction if interaction is not None else self._source
        try:
            queued = admission or InputAdmission(
                priority,
                next(self._sequence),
                InputAdmissionKind.MESSAGE,
                value,
            )
            with self._mailbox_access():
                if not self._put_mailbox_item_locked(queued):
                    return False
        except Full:
            telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
            if report_full:
                active_interaction.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
            return False
        self._report_enqueued(active_interaction)
        return True

    async def _enqueue_async(
        self,
        value: object,
        *,
        priority: int = _INPUT_PRIORITY,
        interaction: Interaction | None = None,
        admission: InputAdmission | None = None,
        report_full: bool = True,
        command: bool = False,
    ) -> bool:
        """Admit bounded input without blocking the UI loop on cross-thread mailbox access."""
        active_interaction = interaction if interaction is not None else self._source
        queued = admission or InputAdmission(
            priority,
            next(self._sequence),
            InputAdmissionKind.COMMAND if command else InputAdmissionKind.MESSAGE,
            value,
        )
        await self._acquire_mailbox_lock_async()
        try:
            try:
                if not self._put_mailbox_item_locked(queued):
                    return False
            except Full:
                telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
                if report_full:
                    active_interaction.warning(
                        "Input queue is full; this input was not accepted. Try again later."
                    )
                return False
        finally:
            self._release_mailbox_lock()
        self._report_enqueued(active_interaction)
        return True

    async def _durable_admission_async(
        self,
        content: str,
        *,
        priority: int,
        bypass_commands: bool,
    ) -> InputAdmission:
        """Persist an accepted message and build its mailbox record."""
        if self._session_manager is None:
            message = SteeringMessage(content) if bypass_commands else content
            return InputAdmission(
                priority,
                next(self._sequence),
                InputAdmissionKind.MESSAGE,
                message,
                bypass_commands=bypass_commands,
            )
        event = await self._session_manager.queue_input_async(
            content,
            priority=priority,
            bypass_commands=bypass_commands,
        )
        return self._admission_from_event(event)

    def _admission_from_event(self, event: InputQueuedEvent) -> InputAdmission:
        """Convert a persisted accepted input into its ordered mailbox record."""
        admission = InputAdmission(
            event.priority,
            event.sequence,
            InputAdmissionKind.MESSAGE,
            event.content,
            input_id=event.input_id,
            session_id=event.session_id,
            bypass_commands=event.bypass_commands,
        )
        self._queued_input_ids.add(event.input_id)
        return admission

    def _reject_admission(self, admission: InputAdmission) -> None:
        """Persist input rejection through the synchronous writer."""
        if admission.input_id is None or self._session_manager is None:
            return
        self._session_manager.reject_input(admission.input_id)
        self._queued_input_ids.discard(admission.input_id)

    async def _reject_admission_async(self, admission: InputAdmission) -> None:
        """Persist rejection before returning a durable admission failure to the UI."""
        if admission.input_id is None or self._session_manager is None:
            return
        await self._session_manager.reject_input_async(admission.input_id)
        self._queued_input_ids.discard(admission.input_id)

    async def _restore_pending_inputs_async(self) -> None:
        """Restore durable accepted inputs into the bounded mailbox in admission order."""
        if (
            self._session_manager is None
            or self._stopping
            or self._session_manager.recovery_pending
        ):
            return
        await self._acquire_mailbox_lock_async()
        try:
            pending = self._session_manager.session.pending_inputs
            pending_ids = {event.input_id for event in pending}
            self._queued_input_ids.intersection_update(pending_ids)
            available = self._inbox.capacity - self._inbox.size()
            for event in pending:
                if available <= 0:
                    break
                if event.input_id in self._queued_input_ids:
                    continue
                admission = InputAdmission(
                    event.priority,
                    event.sequence,
                    InputAdmissionKind.MESSAGE,
                    event.content,
                    input_id=event.input_id,
                    session_id=event.session_id,
                    bypass_commands=event.bypass_commands,
                )
                try:
                    if not self._put_mailbox_item_locked(admission):
                        break
                except Full:
                    break
                self._queued_input_ids.add(event.input_id)
                available -= 1
        finally:
            self._release_mailbox_lock()
        if self._inbox and self._loop is not None:
            self._input_signal.set()

    def _enqueue_steering(
        self,
        value: str,
        *,
        admission: InputAdmission | None = None,
        report_full: bool = True,
    ) -> bool:
        """Atomically queue active ordinary input as steering for the next safe boundary."""
        message = SteeringMessage(value)
        durable_admission = admission or InputAdmission(
            _STEERING_PRIORITY,
            next(self._sequence),
            InputAdmissionKind.MESSAGE,
            message,
        )

        try:
            with self._mailbox_access():
                accepted = self._control.request_steering(
                    lambda: self._put_mailbox_item_locked(durable_admission),
                    key=id(durable_admission),
                )
        except Full:
            accepted = False
        if not accepted:
            if self._stopping:
                return False
            if self._control.active:
                telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
                if report_full:
                    self._source.warning(
                        "Input queue is full; this input was not accepted. Try again later."
                    )
                return False
            return self._enqueue(
                value,
                admission=admission,
                report_full=report_full,
            )
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        telemetry_activity(
            "conversation.input.queued",
            component=_STATUS_COMPONENT,
            pending=self._inbox.size(),
        )
        self._source.info("Steering accepted; finishing the current operation first.")
        return True

    async def _enqueue_steering_async(
        self,
        value: str,
        *,
        admission: InputAdmission | None = None,
        report_full: bool = True,
    ) -> bool:
        """Admit steering without blocking the UI loop on cross-thread mailbox access."""
        durable_admission = admission or InputAdmission(
            _STEERING_PRIORITY,
            next(self._sequence),
            InputAdmissionKind.MESSAGE,
            SteeringMessage(value),
        )
        await self._acquire_mailbox_lock_async()
        try:
            try:
                accepted = self._control.request_steering(
                    lambda: self._put_mailbox_item_locked(durable_admission),
                    key=id(durable_admission),
                )
            except Full:
                accepted = False
        finally:
            self._release_mailbox_lock()
        if accepted:
            self._report_enqueued(self._source)
            self._source.info("Steering accepted; finishing the current operation first.")
            return True
        if self._stopping:
            return False
        if self._control.active:
            telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
            if report_full:
                self._source.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
            return False
        return await self._enqueue_async(
            value,
            admission=admission,
            report_full=report_full,
        )

    def _interaction_revoked(self, request: InteractionCall) -> bool:
        """Refuse completed requests and revoke input after shutdown or run interruption."""
        if request.result.done():
            return True
        if self._control.stopped:
            request.result.set_exception(KeyboardInterrupt())
            return True
        if self._control.interrupted:
            request.result.set_exception(RunInterrupted())
            return True
        return False

    async def _answer_async(self) -> tuple[bool, object]:
        """Settle the pending request and return its answer before the caller releases ownership."""
        request = cast(InteractionCall, self._pending)
        answer_input = None
        outcome = "answered"
        primary = None
        resolved_result = None
        answered = False
        try:
            if self._interaction_revoked(request):
                return False, None
            if request.kind == EVENT_TERMINAL:
                result = request.function(*request.args, **request.kwargs)
                self._terminal_owned = True
            else:
                function = (
                    self._input.confirm_async
                    if request.kind == EVENT_CONFIRM
                    else self._input.prompt_async
                )
                answer_input = asyncio.create_task(function(*request.args, **request.kwargs))
                self._answer_input = answer_input
                loop = asyncio.get_running_loop()

                def cancel_input(future: Future) -> None:
                    """Cancel only this request's input when its waiting owner abandons it."""
                    if future.cancelled() and not answer_input.done():
                        loop.call_soon_threadsafe(answer_input.cancel)

                request.result.add_done_callback(cancel_input)
                result = await answer_input
            resolved_result = result
            answered = True
        except asyncio.CancelledError as error:
            if self._pending is request and not request.result.done():
                request.result.set_exception(KeyboardInterrupt())
            interrupted = (
                not request.result.cancelled()
                and request.result.done()
                and isinstance(request.result.exception(), RunInterrupted)
            )
            revoked = self._stopping or interrupted or request.result.cancelled()
            outcome = "revoked" if revoked else "failed"
            if self._pending is request and not revoked:
                primary = error
        except BaseException as error:  # noqa: BLE001 - return all UI failures to the owner
            outcome = "failed"
            if self._pending is request and not request.result.done():
                request.result.set_exception(error)
                primary = error
        finally:
            if self._pending is request:
                self._pending = None
            if self._answer_input is answer_input:
                self._answer_input = None
            if self._session_manager is not None:
                request_error = None
                if request.result.done() and not request.result.cancelled():
                    request_error = request.result.exception()
                if request.result.cancelled() or isinstance(
                    request_error, (KeyboardInterrupt, RunInterrupted)
                ):
                    outcome = "revoked"
                elif request_error is not None:
                    outcome = "failed"
                try:
                    await self._session_manager.record_interaction_settled_async(
                        request.request_id,
                        outcome,
                    )
                except BaseException as error:
                    if not request.result.done():
                        request.result.set_exception(error)
                    if primary is None:
                        raise
                    primary.add_note(
                        f"Interaction settlement also failed with {type(error).__name__}."
                    )
        if primary is not None:
            raise primary
        return answered, resolved_result

    def _stop(self) -> int:
        """Stop admission, revoke pending input, and wake an idle execution owner."""
        if self._stopping:
            return 0
        self._stopping = True
        self._control.stop()
        if self._answer_input is not None:
            self._answer_input.cancel()
        if self._message_input is not None:
            self._message_input.cancel()
        telemetry_activity("conversation.stopping", component=_STATUS_COMPONENT)
        request = self._pending
        if request is not None:
            if not request.result.done():
                request.result.set_exception(KeyboardInterrupt())
            if self._pending is request:
                self._pending = None
        discarded = self._inbox.stop_and_replace(
            InputAdmission(_INPUT_PRIORITY, next(self._sequence), InputAdmissionKind.STOP)
        )
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        self._discarded += discarded
        return discarded

    async def _settle_async(self, worker: asyncio.Future) -> BaseException | None:
        """Drain acknowledged UI work until execution and command owners release resources."""
        events = cast(asyncio.Queue, self._events)
        command_failure = None
        while not worker.done() or self._command_task is not None:
            incoming = asyncio.create_task(events.get())
            try:
                watched = {incoming}
                if not worker.done():
                    watched.add(worker)
                command_task = self._command_task
                if command_task is not None:
                    watched.add(command_task)
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if incoming in done:
                    await self._deliver_async(incoming.result())
                if command_task is not None and command_task in done:
                    self._command_task = None
                    try:
                        command_task.result()
                    except BaseException as error:  # noqa: BLE001 - preserve the primary failure
                        command_failure = error
            finally:
                incoming.cancel()
                await asyncio.gather(incoming, return_exceptions=True)
        return command_failure
