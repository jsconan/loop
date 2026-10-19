"""Keep terminal interaction responsive around serialized conversation execution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from itertools import count
from queue import Empty, Full, PriorityQueue
from threading import RLock, get_ident
from typing import cast

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
from ..models import CommandCapability, SteeringMessage
from ..telemetry import telemetry_activity
from ..utils import ChoiceItem, choice_items
from .commands import ConversationCommands
from .interaction import MailboxInteraction
from .models import (
    EVENT_CONFIRM,
    EVENT_OUTPUT,
    EVENT_PROMPT,
    EVENT_TERMINAL,
    EVENT_TERMINAL_EXIT,
    ConversationStatus,
    InputReady,
    InteractionCall,
    ResponseOutcome,
    SteeringOutcome,
)

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
    bounded input mailbox. Pending inputs are transient until the conversation consumes them.
    Ordinary input during an active run is admitted as steering and applied after the current
    model response or tool settles. Explicit ``/steer`` interrupts immediately; ``/queue`` queues
    input after the run. Each controller runs once.

    Args:
        interaction (Interaction): UI implementing ``AsyncInteractionInput`` as well as output.

    Raises:
        TypeError: If the UI does not implement asynchronous input.
    """

    _source: Interaction
    _input: AsyncInteractionInput
    _interaction: MailboxInteraction
    _command_manager: CommandManager | None
    _command_completer: CompletionManager
    _control: RunControl
    _inbox: PriorityQueue
    _sequence: count
    _gate: RLock
    _loop: asyncio.AbstractEventLoop | None
    _events: asyncio.Queue | None
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
    _input_signal: asyncio.Event
    _async_owner: bool

    def __init__(self, interaction: Interaction) -> None:
        if not isinstance(interaction, AsyncInteractionInput):
            raise TypeError("Session controller requires asynchronous input.")
        self._source = interaction
        self._input = cast(AsyncInteractionInput, interaction)
        self._interaction = MailboxInteraction(interaction)
        self._command_manager = None
        self._command_completer = CompletionManager()
        self._control = RunControl()
        self._inbox = PriorityQueue(maxsize=_QUEUE_CAPACITY)
        self._sequence = count()
        self._gate = RLock()
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
        self._input_signal = asyncio.Event()
        self._async_owner = False

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
            request.result.set_exception(RunInterrupted())
            self._pending = None
        if self._answer_input is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._answer_input.cancel)
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
            "Interrupting"
            if self._control.interrupted
            else "Steering"
            if self._control.steering_pending
            else "Waiting for your response"
            if self._pending is not None
            else ("Idle" if self._idle else "Working")
        )
        return ConversationStatus(state=state, queued_inputs=self._inbox.qsize())

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
        if not self._enqueue(
            SteeringMessage(message),
            priority=_STEERING_PRIORITY,
            interaction=interaction,
        ):
            return SteeringOutcome.QUEUE_FULL
        if not self.request_interrupt():
            return SteeringOutcome.INTERRUPT_REJECTED
        return SteeringOutcome.INTERRUPTED

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
        return self._enqueue(SteeringMessage(message), interaction=interaction)

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
        """Return shutdown and scoped interruption control for the conversation's agent runner.

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

    async def run(
        self,
        conversation: Callable[[Callable[[Completer], object]], None],
        *,
        route_idle_commands: bool = False,
    ) -> None:
        """Run a conversation while accepting input and dispatching UI requests.

        The operation must use its supplied input callback for ordinary conversation messages.
        Nested prompts and all output use this controller's ``interaction``. Cancellation stops
        admission and waits for the execution owner to settle before returning or raising.

        Args:
            conversation (Callable[[Callable[[Completer], object]], None]): Synchronous operation
                accepting a message reader. The reader takes a completer and returns text or
                ``False`` on shutdown.
            route_idle_commands (bool): Pass commands entered at an idle boundary through the
                conversation input reader. This lets the execution owner dispatch commands that
                may prompt; their prompts open immediately on the async UI. Defaults to ``False``
                for standalone controllers.

        Raises:
            RuntimeError: If this controller has already run.
            asyncio.CancelledError: After owned execution settles when this task is cancelled.
            BaseException: If the conversation or UI fails.
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
        self._interaction.bind(self._send)
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
                    await self._drive(worker, route_idle_commands=route_idle_commands)
                except BaseException as error:
                    primary = error
                    self._stop()
                    cleanup = asyncio.create_task(self._settle(worker))
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
        telemetry_activity("conversation.started", component=_STATUS_COMPONENT)
        primary = None
        try:
            with patch_stdout(raw=True):
                worker = asyncio.create_task(self._execute_async(conversation))
                try:
                    await self._drive(worker, route_idle_commands=route_idle_commands)
                except BaseException as error:
                    primary = error
                    self._stop()
                    cleanup = asyncio.create_task(self._settle(worker))
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

    def _send(self, event: InteractionCall | InputReady) -> object:
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

    async def _send_async(self, event: InteractionCall) -> object:
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
            self._send(InputReady(completer))
            _, _, message = self._inbox.get()
        finally:
            # Restore the owner's existing scope after lending the gate to completion readers.
            self._gate.acquire()
        if self._control.stopped:
            return False
        self._send(InputReady(completer, busy=True))
        return message

    async def _receive_async(self, completer: Completer) -> object:
        """Await the next admitted input without blocking the UI event loop."""
        if self._control.stopped:
            return False
        await self._send_async(InputReady(completer))
        while True:
            try:
                _, _, message = self._inbox.get_nowait()
                break
            except Empty:
                self._input_signal.clear()
                if not self._inbox.empty():
                    continue
                await self._input_signal.wait()
        if self._control.stopped:
            return False
        await self._send_async(InputReady(completer, busy=True))
        return message

    async def _drive(self, worker: asyncio.Future, *, route_idle_commands: bool) -> None:
        """Multiplex execution requests and user input without sharing domain state."""
        events = cast(asyncio.Queue, self._events)
        incoming = asyncio.create_task(events.get())
        typing = None
        worker_done = False
        try:
            while True:
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
                        typing = asyncio.create_task(
                            self._input.prompt_async(
                                completer=self._completer,
                                default="",
                                initial=self._draft,
                            )
                        )
                        self._draft = None
                    else:
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
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if incoming in done:
                    event = incoming.result()
                    incoming = asyncio.create_task(events.get())
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
                                await self._accept(
                                    submitted,
                                    route_idle_commands=route_idle_commands,
                                )
                        else:
                            draft = self._input.current_input()
                            if isinstance(draft, str):
                                self._draft = draft
                            self._input.set_current_input("")
                            typing.cancel()
                            await asyncio.gather(typing, return_exceptions=True)
                            typing = None
                            self._message_input = None
                    await self._deliver(event)
                if typing is not None and typing in done:
                    submitted = typing
                    typing = None
                    self._message_input = None
                    if not submitted.cancelled():
                        await self._accept(submitted, route_idle_commands=route_idle_commands)
                if worker in done:
                    worker.result()
                    worker_done = True
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
                await self._deliver(incoming.result())
            if typing is not None:
                typing.cancel()
            await asyncio.gather(
                incoming, *([typing] if typing is not None else []), return_exceptions=True
            )
            self._message_input = None

    async def _deliver(self, event: InteractionCall | InputReady) -> None:
        """Apply one UI event or await its blocking request through the native input surface."""
        if isinstance(event, InputReady):
            self._ready = True
            self._idle = not event.busy
            if not event.busy:
                self._idle_command_pending = False
            self._completer = GuardedCompleter(
                event.completer,
                self._gate,
                self._command_completer,
            )
            event.result.set_result(None)
            return
        if event.kind not in {EVENT_OUTPUT, EVENT_TERMINAL_EXIT}:
            if self._control.stopped:
                event.result.set_exception(KeyboardInterrupt())
                return
            if self._control.interrupted:
                event.result.set_exception(RunInterrupted())
                return
            self._pending = event
            await self._answer()
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

    async def _accept(self, submitted: asyncio.Task, *, route_idle_commands: bool) -> None:
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
        if route_idle_commands and self._idle and command_manager.is_user_command(value):
            self._idle_command_pending = self._enqueue(value)
            return
        if command_manager.is_user_command(value):
            self._command_task = cast(asyncio.AbstractEventLoop, self._loop).run_in_executor(
                None,
                copy_context().run,
                self._run_command,
                command_manager,
                value,
            )
            return
        if value == "":
            self._source.warning("Please enter a message!")
        elif self._control.active:
            self._enqueue_steering(value)
        else:
            self._enqueue(value)

    def _run_command(self, command_manager: CommandManager, value: str) -> bool:
        """Dispatch one user command through the controller-owned interaction mailbox."""
        try:
            return command_manager.handle_user_command(value, interaction=self._interaction)
        except (RunInterrupted, KeyboardInterrupt) as error:
            if not isinstance(error, RunInterrupted) and not self._control.stopped:
                raise
            return True

    def _enqueue(
        self,
        value: str,
        *,
        priority: int = _INPUT_PRIORITY,
        interaction: Interaction | None = None,
    ) -> bool:
        """Admit bounded input without losing existing work when the mailbox is full."""
        active_interaction = interaction if interaction is not None else self._source
        try:
            self._inbox.put_nowait((priority, next(self._sequence), value))
        except Full:
            telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
            active_interaction.warning(
                "Input queue is full; this input was not accepted. Try again later."
            )
            return False
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        telemetry_activity(
            "conversation.input.queued",
            component=_STATUS_COMPONENT,
            pending=self._inbox.qsize(),
        )
        if not self._idle:
            active_interaction.info(f"Queued input ({self._inbox.qsize()} pending).")
        return True

    def _enqueue_steering(self, value: str) -> bool:
        """Atomically queue active ordinary input as steering for the next safe boundary."""
        message = SteeringMessage(value)
        sequence = next(self._sequence)

        def admit() -> bool:
            """Insert without UI work while run-control admission is locked."""
            try:
                self._inbox.put_nowait((_STEERING_PRIORITY, sequence, message))
            except Full:
                return False
            return True

        if not self._control.request_steering(admit):
            if self._control.active:
                telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
                self._source.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
                return False
            return self._enqueue(value)
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        telemetry_activity(
            "conversation.input.queued",
            component=_STATUS_COMPONENT,
            pending=self._inbox.qsize(),
        )
        self._source.info("Steering accepted; finishing the current operation first.")
        return True

    async def _answer(self) -> None:
        """Resolve only the explicit pending request through the UI's async input surface."""
        request = cast(InteractionCall, self._pending)
        try:
            if request.kind == EVENT_TERMINAL:
                result = request.function(*request.args, **request.kwargs)
                self._terminal_owned = True
            else:
                function = (
                    self._input.confirm_async
                    if request.kind == EVENT_CONFIRM
                    else self._input.prompt_async
                )
                self._answer_input = asyncio.create_task(function(*request.args, **request.kwargs))
                result = await self._answer_input
        except asyncio.CancelledError:
            if not request.result.done():
                request.result.set_exception(KeyboardInterrupt())
            interrupted = isinstance(request.result.exception(), RunInterrupted)
            if not self._stopping and not interrupted:
                raise
        except BaseException as error:
            request.result.set_exception(error)
            raise
        else:
            request.result.set_result(result)
        finally:
            self._pending = None
            self._answer_input = None

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
        if self._pending is not None:
            self._pending.result.set_exception(KeyboardInterrupt())
            self._pending = None
        discarded = 0
        while True:
            try:
                self._inbox.get_nowait()
            except Empty:
                break
            discarded += 1
        self._inbox.put_nowait((_INPUT_PRIORITY, next(self._sequence), False))
        loop = self._loop
        if loop is not None:
            if get_ident() == self._ui_thread:
                self._input_signal.set()
            else:
                loop.call_soon_threadsafe(self._input_signal.set)
        self._discarded += discarded
        return discarded

    async def _settle(self, worker: asyncio.Future) -> BaseException | None:
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
                    await self._deliver(incoming.result())
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
