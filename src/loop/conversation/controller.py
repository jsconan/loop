"""Keep terminal interaction responsive around serialized conversation execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from queue import Empty, Full, Queue
from threading import RLock, get_ident
from typing import cast

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.patch_stdout import patch_stdout

from ..control import RunControl
from ..interaction import AsyncInteractionInput, Interaction
from ..telemetry import telemetry_activity
from .interaction import MailboxInteraction
from .models import (
    EVENT_CONFIRM,
    EVENT_OUTPUT,
    EVENT_TERMINAL,
    EVENT_TERMINAL_EXIT,
    InputReady,
    InteractionCall,
)

_QUEUE_CAPACITY = 16
_STATUS_COMPONENT = "session_controller"


class GuardedCompleter(Completer):
    """Read live completion sources only while their execution owner is idle."""

    _source: Completer
    _gate: RLock

    def __init__(self, source: Completer, gate: RLock) -> None:
        self._source = source
        self._gate = gate

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Generator[Completion]:
        """Yield current completions when execution is idle.

        Args:
            document (Document): Current input document.
            complete_event (CompleteEvent): Completion trigger metadata.

        Yields:
            Completion: Suggestions from the idle conversation's live providers.
        """
        if not self._gate.acquire(blocking=False):
            return

        try:
            yield from self._source.get_completions(document, complete_event)
        finally:
            self._gate.release()


class SessionController:
    """Own one responsive UI and one serialized conversation execution lifecycle.

    Construct conversation services with ``interaction`` and ``control`` from this controller,
    then supply their conversation operation to ``run``. All domain operations stay on one
    execution thread. The UI owns prompts, presentation, and a bounded input mailbox. Pending
    inputs are transient until the conversation consumes them. Each controller runs once.

    Args:
        interaction (Interaction): UI implementing ``AsyncInteractionInput`` as well as output.

    Raises:
        TypeError: If the UI does not implement asynchronous input.
    """

    _source: Interaction
    _input: AsyncInteractionInput
    _interaction: MailboxInteraction
    _control: RunControl
    _inbox: Queue
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
    _discarded: int

    def __init__(self, interaction: Interaction) -> None:
        if not isinstance(interaction, AsyncInteractionInput):
            raise TypeError("Session controller requires asynchronous input.")
        self._source = interaction
        self._input = cast(AsyncInteractionInput, interaction)
        self._interaction = MailboxInteraction(interaction)
        self._control = RunControl()
        self._inbox = Queue(maxsize=_QUEUE_CAPACITY)
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
        self._discarded = 0

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
        """Return cooperative cancellation for the conversation's agent runner.

        Returns:
            RunControl: Stop signal shared with the serialized execution owner.
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
    ) -> None:
        """Run a conversation while accepting input and dispatching UI requests.

        The operation must use its supplied input callback for ordinary conversation messages.
        Nested prompts and all output use this controller's ``interaction``. Cancellation stops
        admission and waits for the execution owner to settle before returning or raising.

        Args:
            conversation (Callable[[Callable[[Completer], object]], None]): Synchronous operation
                accepting a message reader. The reader takes a completer and returns text or
                ``False`` on shutdown.

        Raises:
            RuntimeError: If this controller has already run.
            asyncio.CancelledError: After owned execution settles when this task is cancelled.
            BaseException: If the conversation or UI fails.
        """
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
                    await self._drive(worker)
                except BaseException as error:
                    primary = error
                    self._stop()
                    cleanup = asyncio.create_task(self._settle(worker))
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            continue
                    cleanup.result()
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

    def _execute(self, conversation: Callable) -> None:
        """Keep all conversation state on one owner and normalize requested interruption."""
        try:
            with self._gate:
                conversation(self._receive)
        except KeyboardInterrupt:
            if not self._control.stopped:
                raise

    def _send(self, event: InteractionCall | InputReady) -> object:
        """Send one acknowledged UI request from the execution owner."""
        loop = cast(asyncio.AbstractEventLoop, self._loop)
        events = cast(asyncio.Queue, self._events)
        if get_ident() == self._ui_thread:
            raise RuntimeError("Mailbox interaction must run outside the UI thread.")
        asyncio.run_coroutine_threadsafe(events.put(event), loop).result()
        return event.result.result()

    def _receive(self, completer: Completer) -> object:
        """Release domain ownership while waiting for the next admitted input."""
        if self._control.stopped:
            return False
        self._gate.release()
        try:
            self._send(InputReady(completer))
            message = self._inbox.get()
        finally:
            # Restore the owner's existing scope after lending the gate to completion readers.
            self._gate.acquire()
        if self._control.stopped:
            return False
        self._send(InputReady(completer, busy=True))
        return message

    async def _drive(self, worker: asyncio.Future) -> None:
        """Multiplex execution requests and user input without sharing domain state."""
        events = cast(asyncio.Queue, self._events)
        incoming = asyncio.create_task(events.get())
        typing = None
        try:
            while True:
                if (
                    self._ready
                    and typing is None
                    and not self._control.stopped
                    and not self._terminal_owned
                ):
                    typing = asyncio.create_task(
                        self._input.prompt_async(completer=self._completer, default="")
                    )
                    self._message_input = typing
                watched = {worker, incoming}
                if typing is not None:
                    watched.add(typing)
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if incoming in done:
                    event = incoming.result()
                    incoming = asyncio.create_task(events.get())
                    await self._deliver(event)
                if typing is not None and typing in done:
                    submitted = typing
                    typing = None
                    self._message_input = None
                    await self._accept(submitted)
                if worker in done:
                    worker.result()
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
        """Apply one event on the UI thread or retain its separate input request."""
        if isinstance(event, InputReady):
            self._ready = True
            self._idle = not event.busy
            self._completer = GuardedCompleter(event.completer, self._gate)
            event.result.set_result(None)
            return
        if event.kind not in {EVENT_OUTPUT, EVENT_TERMINAL_EXIT}:
            if self._control.stopped:
                event.result.set_exception(KeyboardInterrupt())
                return
            self._pending = event
            if not self._ready or self._terminal_owned:
                await self._answer()
            else:
                detail = (
                    "Exclusive terminal access requested."
                    if event.kind == EVENT_TERMINAL
                    else f"Input requested: {event.args[0] or 'Choose a value.'}"
                )
                self._source.info(
                    detail
                    + "\nPress Enter or use /respond to answer; other messages remain queued."
                )
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

    async def _accept(self, submitted: asyncio.Task) -> None:
        """Resolve input without hiding errors, then apply controls or admit ordinary text."""
        if self._stopping:
            if not submitted.cancelled():
                submitted.result()
            return
        value = submitted.result()
        if value is False or value in ("/exit", "/quit"):
            self._stop()
            self._source.info("Stopping after the active operation settles...")
        elif value == "/status":
            state = (
                "Waiting for your response"
                if self._pending is not None
                else ("Idle" if self._idle else "Working")
            )
            self._source.info(f"{state}; {self._inbox.qsize()} queued inputs.")
        elif value == "/respond" or (value == "" and self._pending is not None):
            if self._pending is None:
                self._source.warning("No input request is pending.")
            else:
                await self._answer()
        elif value == "":
            self._source.warning("Please enter a message!")
        else:
            try:
                self._inbox.put_nowait(value)
            except Full:
                telemetry_activity("conversation.queue.full", component=_STATUS_COMPONENT)
                self._source.warning(
                    "Input queue is full; this input was not accepted. Try again later."
                )
            else:
                telemetry_activity(
                    "conversation.input.queued",
                    component=_STATUS_COMPONENT,
                    pending=self._inbox.qsize(),
                )
                if not self._idle:
                    self._source.info(f"Queued input ({self._inbox.qsize()} pending).")

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
            if not self._stopping:
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
        self._inbox.put_nowait(False)
        self._discarded += discarded
        return discarded

    async def _settle(self, worker: asyncio.Future) -> None:
        """Drain acknowledged UI work until the execution owner releases all resources."""
        events = cast(asyncio.Queue, self._events)
        while not worker.done():
            incoming = asyncio.create_task(events.get())
            try:
                done, _ = await asyncio.wait(
                    {worker, incoming}, return_when=asyncio.FIRST_COMPLETED
                )
                if incoming in done:
                    await self._deliver(incoming.result())
            finally:
                incoming.cancel()
                await asyncio.gather(incoming, return_exceptions=True)
