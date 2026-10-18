"""Verify responsive mailbox admission and serialized conversation ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event, Thread, get_ident
from unittest.mock import MagicMock, Mock

import pytest
from prompt_toolkit.completion import CompleteEvent, DummyCompleter, WordCompleter
from prompt_toolkit.document import Document

from loop import (
    CommandCapability,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CommandRegistration,
    CompletionManager,
    CompletionValue,
    ConsoleInteraction,
    Interaction,
    RunInterrupted,
    SessionController,
)
from loop.conversation import ConversationCommands
from loop.conversation import controller as controller_module
from loop.conversation.models import (
    EVENT_CONFIRM,
    EVENT_PROMPT,
    EVENT_TERMINAL,
    InteractionCall,
    ResponseOutcome,
    SteeringOutcome,
)
from loop.models import SteeringMessage


@pytest.fixture
def ui(monkeypatch):
    """Provide async input and output doubles without global terminal patching."""
    monkeypatch.setattr(controller_module, "patch_stdout", lambda **_kwargs: nullcontext())
    source = MagicMock(spec=ConsoleInteraction)
    source.response_context.side_effect = nullcontext
    source.terminal_context.side_effect = nullcontext
    return source


def read_inputs(ui, inputs):
    """Read normal UI messages from a deterministic async mailbox."""

    async def read(*_args, **_kwargs):
        """Wait for an explicitly supplied user input."""
        return await inputs.get()

    ui.prompt_async.side_effect = read


def test_controller_requires_async_input():
    """Legacy synchronous adapters remain outside the responsive controller."""
    with pytest.raises(TypeError, match="asynchronous input"):
        SessionController(MagicMock(spec=Interaction))


def test_controller_exposes_setup_interaction_and_a_single_execution_lifecycle(ui):
    """Setup uses the original UI and a settled controller cannot be started twice."""
    controller = SessionController(ui)
    assert controller.running is False
    assert controller.control.stopped is False
    controller.interaction.info("setup")
    ui.info.assert_called_once_with("setup")

    asyncio.run(controller.run(lambda _receive: None))

    assert controller.running is False
    controller.interaction.info("settled")
    with pytest.raises(RuntimeError, match="Bind the command manager before running"):
        controller.bind_command_manager(CommandManager())
    with pytest.raises(RuntimeError, match="only run once"):
        asyncio.run(controller.run(lambda _receive: None))


def test_controller_commands_appear_in_help_and_complete_values_while_busy(ui):
    """Registered responsive controls and later argument values remain discoverable."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
    controller.bind_command_manager(manager)

    def choose(context: CommandContext, color: str) -> None:
        """Choose a configured color."""

    def queued() -> None:
        """Run after the active operation settles."""

    def unavailable() -> None:
        """Remain unavailable while the active operation is running."""

    manager.register(
        CommandRegistration(
            choose,
            capabilities=frozenset({CommandCapability.RUN_ACTIVE}),
            completion=CommandCompletion(values=(CompletionValue("blue"),)),
        )
    )
    manager.register(queued)
    manager.register(unavailable)
    manager.call("help")
    setup_names = {item.name for item in ui.table.call_args.args[0]}
    assert {"interrupt", "respond", "status", "steer", "choose"}.isdisjoint(setup_names)
    assert "queued" in setup_names

    async def scenario():
        """Read a completion request while the conversation owns active work."""
        inputs = asyncio.Queue()
        working, captured = asyncio.Event(), asyncio.Event()
        release = Event()
        busy_completions = []
        busy_command_completions = []
        busy_help_names = []

        async def read(*_args, **kwargs):
            """Capture command completion before waiting for the next mailbox input."""
            completer = kwargs["completer"]
            completions = list(completer.get_completions(Document("/choose b"), CompleteEvent()))
            if working.is_set():
                busy_completions[:] = completions
                busy_command_completions[:] = list(
                    completer.get_completions(Document("/"), CompleteEvent())
                )
                captured.set()
            return await inputs.get()

        ui.prompt_async.side_effect = read

        def conversation(receive):
            """Hold execution ownership after accepting the initial message."""
            receive(DummyCompleter())
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
            receive(DummyCompleter())

        ui.info.side_effect = lambda message="": working.set() if message == "working" else None
        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            manager.handle_user_command("/help", interaction=ui)
            busy_help_names[:] = [item.name for item in ui.table.call_args.args[0]]
            inputs.put_nowait("/status")
            await captured.wait()
            assert [item.text for item in busy_completions] == ["blue"]
            commands = {item.text: item for item in busy_command_completions}
            assert "/choose" in commands
            assert "/respond" not in commands
            assert "/queued" not in commands
            assert "/unavailable" not in commands
            assert set(busy_help_names) == {
                "help",
                "exit",
                "quit",
                "interrupt",
                "status",
                "steer",
                "choose",
            }
            release.set()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert "status" not in {item.name for item in manager.active_commands}


def test_controller_rejects_inactive_and_unknown_slash_input_without_queueing(ui):
    """Inactive commands and unknown slash input are explained instead of queued as messages."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)

    def serialized() -> None:
        """Represent a command owned by the conversation loop."""

    manager.register(serialized)
    controller.bind_command_manager(manager)
    received = []

    async def scenario():
        """Reject slash input during execution, then admit an ordinary follow-up."""
        inputs = asyncio.Queue()
        working = asyncio.Event()
        inactive_rejected = asyncio.Event()
        unknown_rejected = asyncio.Event()
        release = Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": working.set() if message == "working" else None
        ui.warning.side_effect = lambda _message: inactive_rejected.set()
        ui.report.side_effect = lambda _problem: unknown_rejected.set()

        def conversation(receive):
            """Consume a normal input before and after an active run scope."""
            received.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
            received.append(receive(DummyCompleter()))

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await working.wait()
            inputs.put_nowait("/serialized")
            await inactive_rejected.wait()
            inputs.put_nowait("/missing value")
            await unknown_rejected.wait()
            assert received == ["start"]
            release.set()
            inputs.put_nowait("follow-up")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert received == ["start", "follow-up"]


def test_response_command_completions_follow_the_pending_request(ui):
    """Selectable response prompts expose their accepted values through the command catalog."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Choose one:",),
        kwargs={"choices": {"red": "Red", "blue": "Blue"}},
        kind=EVENT_PROMPT,
    )
    controller._pending = request

    assert [item.value for item in controller.response_completions()] == ["Red", "Blue"]
    assert controller.request_response("bLuE") is ResponseOutcome.ANSWERED
    assert request.result.result() == "blue"
    assert controller._pending is None


def test_command_capabilities_track_active_run_pending_response_and_settlement(ui):
    """Conversation controls appear only while the corresponding lifecycle state exists."""
    controller = SessionController(ui)
    manager = CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
    controller.bind_command_manager(manager)

    def names():
        return {command.name for command in manager.active_commands}

    assert {"interrupt", "respond", "status", "steer"}.isdisjoint(names())
    controller._loop = Mock()
    with controller.control.execution():
        active_names = names()
        assert {"interrupt", "status", "steer"} <= active_names
        assert "respond" not in active_names
        controller._pending = InteractionCall(
            function=ui.prompt,
            args=("Choose:",),
            kwargs={"choices": {"yes": "Yes", "no": "No"}},
            kind=EVENT_PROMPT,
        )
        assert "respond" in names()
        completer = CompletionManager((manager.completion_adapter,))
        completions = list(completer.get_completions(Document("/respond y"), CompleteEvent()))
        assert [item.text for item in completions] == ["Yes"]
    controller._pending = None
    controller._loop = None
    assert {"interrupt", "respond", "status", "steer"}.isdisjoint(names())


def test_command_capabilities_during_shutdown_allow_only_inspection(ui):
    """Shutdown keeps inspection available while refusing conversation controls and edits."""
    controller = SessionController(ui)
    controller._loop = Mock()
    controller._stopping = True

    assert controller.command_capabilities() == frozenset({CommandCapability.CONTEXT_READ})


def test_response_completion_caches_one_shot_prompt_choices(ui):
    """Completion does not consume one-shot choices before direct response validation."""
    controller = SessionController(ui)
    request = InteractionCall(
        ui.prompt,
        ("Choose one:",),
        {"choices": (value for value in ("red", "blue"))},
        kind=EVENT_PROMPT,
    )
    controller._pending = request

    assert [item.value for item in controller.response_completions()] == ["red", "blue"]
    assert controller.request_response("blue") is ResponseOutcome.ANSWERED
    assert request.result.result() == "blue"


@pytest.mark.parametrize(
    ("kind", "kwargs", "answer", "outcome"),
    [
        (EVENT_PROMPT, {}, "A free form answer", ResponseOutcome.ANSWERED),
        (EVENT_PROMPT, {"secret": True}, "secret", ResponseOutcome.MASKED_INPUT_REQUIRED),
        (EVENT_TERMINAL, {}, "answer", ResponseOutcome.PROMPT_REQUIRED),
        (EVENT_PROMPT, {"choices": ("first", "second")}, "3", ResponseOutcome.INVALID),
        (EVENT_PROMPT, {"choices": ("first", "second")}, "2", ResponseOutcome.ANSWERED),
        (
            EVENT_PROMPT,
            {"choices": ("first", "second"), "index": ("same", "same")},
            "first",
            ResponseOutcome.INVALID,
        ),
        (EVENT_CONFIRM, {}, "y", ResponseOutcome.ANSWERED),
        (EVENT_CONFIRM, {}, "maybe", ResponseOutcome.INVALID),
        ("unknown", {}, "answer", ResponseOutcome.PROMPT_REQUIRED),
    ],
)
def test_direct_response_resolves_supported_prompt_kinds(ui, kind, kwargs, answer, outcome):
    """Direct responses preserve free text, typed choices, and confirmation values."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    assert controller.request_response(answer) is outcome
    if outcome is ResponseOutcome.ANSWERED:
        assert request.result.done()
    else:
        assert not request.result.done()


def test_request_response_without_value_opens_the_pending_prompt(ui):
    """An empty response command delegates to the original masked or selectable UI prompt."""
    controller = SessionController(ui)
    assert controller.request_response() is ResponseOutcome.NO_REQUEST
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs={}, kind=EVENT_PROMPT)
    controller._pending = request

    assert controller.request_response() is ResponseOutcome.PROMPTED
    assert controller._respond_requested is True


def test_response_completions_handle_confirm_secret_free_text_and_invalid_choices(ui):
    """Only valid, non-secret choices are completed for pending responses."""
    controller = SessionController(ui)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Confirm?",), {}, kind=EVENT_CONFIRM)
    assert [item.value for item in controller.response_completions()] == ["Yes", "No"]

    controller._pending = InteractionCall(
        ui.prompt, ("Secret?",), {"secret": True, "choices": ("a", "b")}, kind=EVENT_PROMPT
    )
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Exclusive?",), {}, kind=EVENT_TERMINAL)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Text?",), {}, kind=EVENT_PROMPT)
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(
        ui.prompt,
        ("Invalid choices?",),
        {"choices": ("a", "b"), "index": ("same", "same")},
        kind=EVENT_PROMPT,
    )
    assert controller.response_completions() == ()

    controller._pending = InteractionCall(ui.prompt, ("Unknown?",), {}, kind="unknown")
    assert controller.response_completions() == ()


def test_steering_reports_when_active_control_already_rejects_interruption(ui):
    """A stopped control refuses interruption even while its execution scope is unwinding."""
    controller = SessionController(ui)

    with controller.control.execution():
        controller.control.stop()
        outcome = controller.steer("correction", ui)

    assert outcome is SteeringOutcome.INTERRUPT_REJECTED


def test_steering_rejects_when_no_run_is_active(ui):
    """Steering cannot admit corrections while the run control is idle."""
    controller = SessionController(ui)

    assert controller.steer("correction", ui) is SteeringOutcome.RUN_INACTIVE
    assert controller.status_snapshot().queued_inputs == 0


def test_controller_accepts_messages_and_status_during_serialized_work(ui):
    """UI input remains live while execution retains one owner and processes queued inputs FIFO."""
    controller = SessionController(ui)
    release = Event()
    received, execution_threads, output_threads = [], [], []

    async def scenario():
        """Queue follow-ups behind a controlled execution boundary."""
        inputs = asyncio.Queue()
        started, queued, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe presentation on the UI thread and signal execution boundaries."""
            output_threads.append(get_ident())
            if message == "working":
                started.set()
            elif message.startswith("Queued input"):
                queued.set()
            elif message == "processed":
                finished.set()

        ui.info.side_effect = output

        def conversation(receive):
            """Own mutable conversation state while blocking work is in progress."""
            while (message := receive(DummyCompleter())) is not False:
                received.append(message)
                execution_threads.append(get_ident())
                if message == "first":
                    controller.interaction.info("working")
                    release.wait()
                else:
                    controller.interaction.info("processed")
            controller.interaction.conversation_ended()

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await started.wait()
            inputs.put_nowait("/status")
            inputs.put_nowait("second")
            await queued.wait()
            assert received == ["first"]
            assert controller.running is True
            assert "Working; 0 queued inputs." in [item.args[0] for item in ui.info.call_args_list]
            release.set()
            await finished.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)
        assert set(output_threads) == {get_ident()}

    asyncio.run(scenario())
    assert received == ["first", "second"]
    assert len(set(execution_threads)) == 1
    assert execution_threads[0] != get_ident()
    ui.conversation_ended.assert_called_once_with()


@pytest.mark.parametrize("reply", ["", "/respond"])
def test_controller_never_uses_queued_chat_as_an_approval(ui, reply):
    """Only an explicit reply path reaches confirmation; queued messages remain ordinary input."""
    controller = SessionController(ui)
    decisions, messages = [], []
    ui.confirm_async.return_value = False

    async def scenario():
        """Queue text that resembles approval before opening the actual decision prompt."""
        inputs = asyncio.Queue()
        pending, queued, processed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe safe progress through the separate input and approval channels."""
            if message.startswith("Input requested"):
                pending.set()
            elif message.startswith("Queued input"):
                queued.set()
            elif message == "processed":
                processed.set()

        ui.info.side_effect = output

        def conversation(receive):
            """Ask for approval without draining the conversation input queue."""
            while (message := receive(DummyCompleter())) is not False:
                messages.append(message)
                if message == "start":
                    decisions.append(controller.interaction.confirm("Write the file?"))
                else:
                    controller.interaction.info("processed")

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await pending.wait()
            inputs.put_nowait("yes")
            await queued.wait()
            ui.confirm_async.assert_not_awaited()
            inputs.put_nowait("/status")
            inputs.put_nowait(reply)
            await processed.wait()
            inputs.put_nowait(False)
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == [False]
    assert messages == ["start", "yes"]
    ui.confirm_async.assert_awaited_once_with("Write the file?", default=False)
    ui.confirm.assert_not_called()
    assert any(
        item.args[0].startswith("Waiting for your response") for item in ui.info.call_args_list
    )


def test_controller_preserves_startup_prompts_and_masked_input(ui):
    """Prompts before the first ordinary input boundary retain their exact typed results."""
    controller = SessionController(ui)
    ui.prompt_async.return_value = "private-value"
    values = []

    def conversation(_receive):
        """Read a startup secret through the UI without ordinary mailbox input."""
        values.append(controller.interaction.prompt("Credential:", secret=True, exit_commands=()))

    asyncio.run(controller.run(conversation))

    assert values == ["private-value"]
    ui.prompt_async.assert_awaited_once_with(
        "Credential:",
        completer=None,
        choices=None,
        index=None,
        default=None,
        secret=True,
        exit_commands=(),
    )
    ui.prompt.assert_not_called()


def test_controller_stops_before_startup_without_opening_input(ui):
    """A pre-start stop request prevents admission and leaves interaction usable afterward."""
    controller = SessionController(ui)
    controller.stop()
    values = []
    asyncio.run(controller.run(lambda receive: values.append(receive(DummyCompleter()))))

    assert values == [False]
    assert controller.control.stopped is True
    assert controller.running is False
    ui.prompt_async.assert_not_awaited()
    ui.conversation_ended.assert_called_once_with()


def test_controller_bounds_queued_inputs_and_reports_unexecuted_inputs_on_exit(ui):
    """Full admission explicitly rejects input and shutdown does not execute queued work."""
    controller = SessionController(ui)
    release = Event()
    messages = []

    async def scenario():
        """Fill the bounded queue while execution waits at an owned operation."""
        inputs = asyncio.Queue()
        started, rejected, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": (
            started.set()
            if message == "working"
            else stopping.set()
            if message.startswith("Stopping")
            else None
        )
        ui.warning.side_effect = lambda message: (
            rejected.set() if "queue is full" in message else None
        )

        def conversation(receive):
            """Keep the execution owner busy until the test releases it."""
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("working")
            release.wait()
            messages.append(receive(DummyCompleter()))

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await started.wait()
            for index in range(17):
                inputs.put_nowait(f"queued {index}")
            await rejected.wait()
            inputs.put_nowait("/exit")
            await stopping.wait()
            assert not task.done()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["first", False]
    assert any(
        "16 queued inputs were not executed" in item.args[0] for item in ui.warning.call_args_list
    )


def test_controller_revokes_pending_approval_on_shutdown(ui):
    """Shutdown unblocks a waiting execution owner without granting pending approval."""
    controller = SessionController(ui)
    decisions = []

    async def scenario():
        """Stop after the UI has announced the request but before any reply."""
        inputs = asyncio.Queue()
        pending = asyncio.Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda text="": (
            pending.set() if text.startswith("Input requested") else None
        )

        def conversation(receive):
            """Request a decision from the independent approval channel."""
            receive(DummyCompleter())
            decisions.append(controller.interaction.confirm("Approve?", default=True))

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await pending.wait()
            inputs.put_nowait("/quit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    ui.confirm_async.assert_not_awaited()


def test_controller_keeps_live_completion_reads_out_of_active_execution(ui):
    """Live completion works when idle and does not inspect services during active mutation."""
    controller = SessionController(ui)
    release = Event()
    completions = []

    async def scenario():
        """Inspect the supplied completion capability at idle and busy input boundaries."""
        inputs = asyncio.Queue()
        started, busy_read = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": started.set() if text == "working" else None

        async def read(*_args, **kwargs):
            """Use the UI's injected completer without accessing controller state."""
            if started.is_set():
                completions.append(
                    list(kwargs["completer"].get_completions(Document("he"), CompleteEvent()))
                )
                busy_read.set()
            else:
                completions.append(
                    list(kwargs["completer"].get_completions(Document("he"), CompleteEvent()))
                )
            return await inputs.get()

        ui.prompt_async.side_effect = read

        def conversation(receive):
            """Hold the owner gate around a synchronous operation."""
            receive(WordCompleter(["hello"]))
            controller.interaction.info("working")
            release.wait()

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await started.wait()
            # An already-open editor can submit status to start a fresh completion request.
            inputs.put_nowait("/status")
            await busy_read.wait()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert [item.text for item in completions[0]] == ["hello"]
    assert completions[-1] == []


def test_controller_propagates_worker_errors_and_restores_direct_interaction(ui):
    """Execution errors settle their owner before the controller restores setup interaction."""
    controller = SessionController(ui)

    def conversation(_receive):
        """Fail at the execution boundary without leaving background operations."""
        raise ValueError("execution failed")

    with pytest.raises(ValueError, match="execution failed"):
        asyncio.run(controller.run(conversation))

    assert controller.running is False
    controller.interaction.info("after failure")
    ui.info.assert_called_with("after failure")


def test_controller_cancellation_waits_for_execution_cleanup(ui):
    """Task cancellation drains UI cleanup and retains ownership until the worker settles."""
    controller = SessionController(ui)
    release = Event()
    cleanup_threads = []

    async def scenario():
        """Cancel during active work and then release its deterministic cleanup boundary."""
        started, cleaned = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": (
            started.set() if text == "working" else cleaned.set() if text == "cleaned" else None
        )

        def conversation(_receive):
            """Finalize presentation after an in-progress synchronous operation settles."""
            try:
                controller.interaction.info("working")
                release.wait()
            finally:
                cleanup_threads.append(get_ident())
                controller.interaction.info("cleaned")

        task = asyncio.create_task(controller.run(conversation))
        try:
            await started.wait()
            task.cancel()
            # Scheduling an ordinary UI callback remains possible while shutdown is pending.
            await asyncio.get_running_loop().run_in_executor(None, release.set)
            await cleaned.wait()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert len(cleanup_threads) == 1
    assert controller.running is False


def test_controller_hands_off_exclusive_terminal_without_starting_another_editor(ui):
    """Direct authentication owns stdin until its scope and nested approvals have settled."""
    controller = SessionController(ui)
    release = Event()
    decisions = []

    async def scenario():
        """Explicitly hand off the terminal before simulating direct authentication."""
        inputs = asyncio.Queue()
        pending, held, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe terminal ownership without sharing domain objects."""
            if message.startswith("Exclusive terminal"):
                pending.set()
            elif message == "held":
                held.set()
            elif message == "released":
                finished.set()

        ui.info.side_effect = output
        ui.confirm_async.return_value = False

        def conversation(receive):
            """Reserve stdin around an operation that uses direct authentication."""
            receive(DummyCompleter())
            with controller.interaction.terminal_context():
                decisions.append(controller.interaction.confirm("Install?"))
                controller.interaction.info("held")
                release.wait()
            controller.interaction.info("released")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("manage")
            await pending.wait()
            inputs.put_nowait("/respond")
            await held.wait()
            count = ui.prompt_async.await_count
            assert count == 2
            ui.confirm_async.assert_awaited_once_with("Install?", default=False)
            release.set()
            await finished.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == [False]


def test_controller_rejects_blocking_mailbox_calls_from_its_ui_thread(ui):
    """Accidental synchronous UI calls fail explicitly instead of deadlocking the mailbox."""
    controller = SessionController(ui)

    async def scenario():
        """Exercise the public injected interaction while the controller is active."""
        ready = asyncio.Event()

        def show(message=""):
            """Check the UI-thread ownership guard from inside real dispatch."""
            if message == "ready":
                with pytest.raises(RuntimeError, match="outside the UI thread"):
                    controller.interaction.info("invalid nested call")
                ready.set()

        ui.info.side_effect = show
        await controller.run(lambda _receive: controller.interaction.info("ready"))
        assert ready.is_set()

    asyncio.run(scenario())


def test_controller_preserves_output_failures_at_the_execution_boundary(ui):
    """Presentation failures return to their blocked owner and do not leave a running worker."""
    controller = SessionController(ui)
    ui.answer.side_effect = ValueError("render failed")

    with pytest.raises(ValueError, match="render failed"):
        asyncio.run(controller.run(lambda _receive: controller.interaction.answer("answer")))

    assert controller.running is False


def test_controller_applies_backpressure_to_concurrent_tool_output(ui):
    """Auxiliary tool threads cannot overflow UI dispatch or render outside its owner."""
    controller = SessionController(ui)
    threads = []
    ui.info.side_effect = lambda _text: threads.append(get_ident())

    def conversation(_receive):
        """Join all tool-owned output producers before releasing conversation ownership."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(controller.interaction.info, (str(index) for index in range(30))))

    asyncio.run(controller.run(conversation))

    assert sorted(int(item.args[0]) for item in ui.info.call_args_list) == list(range(30))
    assert set(threads) == {get_ident()}


def test_controller_reports_blank_input_and_missing_response_requests(ui):
    """UI control commands never become model input when their request is absent."""
    controller = SessionController(ui)

    async def scenario():
        """Submit idle UI controls before closing the ordinary input channel."""
        inputs = asyncio.Queue()
        read_inputs(ui, inputs)
        for value in (
            "",
            "/respond",
            "/interrupt",
            "/steer",
            "/steer correction",
            "/status",
            False,
        ):
            inputs.put_nowait(value)
        await controller.run(lambda receive: receive(DummyCompleter()))

    asyncio.run(scenario())
    assert [item.args[0] for item in ui.warning.call_args_list] == [
        "Please enter a message!",
        (
            "Command '/respond' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
        (
            "Command '/interrupt' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
        (
            "Command '/steer' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
        (
            "Command '/steer' is unavailable in the current conversation state. "
            "Type /help to see commands that can run now."
        ),
    ]
    ui.info.assert_any_call("Idle; 0 queued inputs.")


@pytest.mark.parametrize("cancelled", [False, True])
def test_controller_propagates_errors_and_cancellation_from_active_request_input(ui, cancelled):
    """Input failures revoke their own request and leave no blocked execution owner."""
    controller = SessionController(ui)

    async def answer(*_args, **_kwargs):
        """Fail while a startup confirmation owns the input surface."""
        if cancelled:
            raise asyncio.CancelledError
        raise ValueError("input failed")

    ui.confirm_async.side_effect = answer
    expected = asyncio.CancelledError if cancelled else ValueError
    with pytest.raises(expected):
        asyncio.run(controller.run(lambda _receive: controller.interaction.confirm("Approve?")))

    assert controller.running is False


def test_controller_external_stop_cancels_an_active_decision_prompt(ui):
    """A stop request revokes an opened approval without waiting for a stale user reply."""
    controller = SessionController(ui)

    async def scenario():
        """Stop while a confirmation is actively waiting for terminal input."""
        opened = asyncio.Event()

        async def answer(*_args, **_kwargs):
            """Keep the approval surface open until the controller cancels it."""
            opened.set()
            await asyncio.Event().wait()

        ui.confirm_async.side_effect = answer
        task = asyncio.create_task(
            controller.run(
                lambda _receive: controller.interaction.confirm("Approve?", default=True)
            )
        )
        try:
            await opened.wait()
            controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert controller.running is False
    ui.conversation_ended.assert_called_once_with()


def test_controller_rejects_a_new_approval_requested_after_stop(ui):
    """An operation finishing during shutdown cannot open or obtain a new approval."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Stop while work is active and let it attempt a late input request."""
        started = asyncio.Event()
        ui.info.side_effect = lambda text="": started.set() if text == "working" else None

        def conversation(_receive):
            """Attempt confirmation after the active operation has settled."""
            controller.interaction.info("working")
            release.wait()
            controller.interaction.confirm("Too late", default=True)

        task = asyncio.create_task(controller.run(conversation))
        try:
            await started.wait()
            controller.stop()
            release.set()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.confirm_async.assert_not_awaited()


def test_controller_retains_cleanup_failure_evidence_during_repeated_cancellation(ui):
    """Repeated cancellation cannot detach the owner or hide a cleanup failure."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Cancel again after the worker has observed shutdown and before releasing it."""
        started, cleaning = asyncio.Event(), asyncio.Event()
        ui.info.side_effect = lambda text="": (
            started.set() if text == "working" else cleaning.set() if text == "cleaning" else None
        )

        def conversation(_receive):
            """Wait for shutdown, report cleanup, then fail after explicit release."""
            controller.interaction.info("working")
            # Waiting uses the public stop signal, with no elapsed-time assertion.
            try:
                controller.control.wait(60)
            except KeyboardInterrupt:
                controller.interaction.info("cleaning")
                release.wait()
            raise ValueError("cleanup failed")

        task = asyncio.create_task(controller.run(conversation))
        try:
            await started.wait()
            task.cancel()
            await cleaning.wait()
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert any("ValueError" in note for note in raised.value.__notes__)
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("returned_input", [False, True])
def test_controller_external_stop_releases_an_idle_reader_without_admitting_late_input(
    ui, returned_input
):
    """Stopping an idle owner cancels its editor and rejects input returned during shutdown."""
    controller = SessionController(ui)
    messages = []

    async def scenario():
        """Stop with either an open editor or a completed input waiting for dispatch."""
        opened = asyncio.Event()

        async def read(*_args, **_kwargs):
            """Return text concurrently with stop or remain suspended until cancelled."""
            opened.set()
            if returned_input:
                controller.stop()
                return "late input"
            await asyncio.Event().wait()

        ui.prompt_async.side_effect = read
        task = asyncio.create_task(
            controller.run(lambda receive: messages.append(receive(DummyCompleter())))
        )
        try:
            await opened.wait()
            if not returned_input:
                controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == [False]
    assert controller.running is False


def test_controller_preserves_unrequested_worker_interruption(ui):
    """An unexpected worker interrupt is propagated after execution has settled."""
    controller = SessionController(ui)

    def conversation(_receive):
        """Interrupt without receiving a cooperative stop request."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(controller.run(conversation))
    assert controller.running is False


def test_controller_stop_tolerates_the_event_loop_settling_in_another_thread(ui, monkeypatch):
    """A concurrent stop remains harmless when execution settles before its callback is sent."""
    controller = SessionController(ui)
    release_worker, release_stopper = Event(), Event()
    errors = []
    stopper_id = []

    async def scenario():
        """Hold a public scheduler call across the execution lifecycle's final boundary."""
        started, stopping = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        schedule = loop.call_soon_threadsafe
        ui.info.side_effect = lambda _text: started.set()

        def call_soon(callback, *args, **kwargs):
            """Model a loop closing after a stopper has read its active reference."""
            if stopper_id and get_ident() == stopper_id[0]:
                schedule(stopping.set)
                release_stopper.wait()
                raise RuntimeError("event loop closed")
            return schedule(callback, *args, **kwargs)

        def stop():
            """Request shutdown from an external caller and retain any unexpected error."""
            stopper_id.append(get_ident())
            try:
                controller.stop()
            except RuntimeError as error:
                errors.append(error)

        def conversation(_receive):
            """Keep execution active until the stop scheduler is paused."""
            controller.interaction.info("working")
            release_worker.wait()

        monkeypatch.setattr(loop, "call_soon_threadsafe", call_soon)
        task = asyncio.create_task(controller.run(conversation))
        thread = Thread(target=stop)
        try:
            await started.wait()
            thread.start()
            await stopping.wait()
            release_worker.set()
            await task
        finally:
            release_worker.set()
            release_stopper.set()
            monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
            await loop.run_in_executor(None, thread.join)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert errors == []
    assert controller.control.stopped is True


def test_controller_stop_preserves_real_scheduler_failures(ui, monkeypatch):
    """An active loop's scheduling failure is not mistaken for an already-settled lifecycle."""
    controller = SessionController(ui)

    async def scenario():
        """Fail a stop submission while the UI still owns the active lifecycle."""
        loop = asyncio.get_running_loop()

        def output(_text):
            """Exercise stop on a blocked owner without affecting later scheduling."""

            def fail(*_args, **_kwargs):
                """Report an actual scheduler failure before execution has settled."""
                raise RuntimeError("scheduler failed")

            with monkeypatch.context() as patch:
                patch.setattr(loop, "call_soon_threadsafe", fail)
                with pytest.raises(RuntimeError, match="scheduler failed"):
                    controller.stop()

        ui.info.side_effect = output
        await controller.run(lambda _receive: controller.interaction.info("working"))

    asyncio.run(scenario())


def test_controller_cancellation_does_not_lose_dequeued_output(ui, monkeypatch):
    """An interaction already removed from the mailbox is acknowledged during cancellation."""
    real_queue = asyncio.Queue
    controller = SessionController(ui)

    async def scenario():
        """Cancel exactly when the public queue receive returns its first interaction."""
        cancelled = False

        class CancellationQueue(real_queue):
            """Expose cancellation at a deterministic mailbox receive boundary."""

            async def get(self):
                """Cancel the owning controller after dequeuing its first item."""
                nonlocal cancelled
                item = await super().get()
                if not cancelled:
                    cancelled = True
                    task.cancel()
                return item

        monkeypatch.setattr(controller_module.asyncio, "Queue", CancellationQueue)
        task = asyncio.create_task(
            controller.run(lambda _receive: controller.interaction.info("buffered output"))
        )
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    ui.info.assert_called_once_with("buffered output")
    assert controller.running is False


@pytest.mark.parametrize("worker_failure", [False, True])
def test_controller_reports_discarded_inputs_without_hiding_primary_failures(ui, worker_failure):
    """Queue cancellation is reported after settlement, with primary failure evidence retained."""
    controller = SessionController(ui)
    release = Event()
    ui.warning.side_effect = ValueError("queue display failed")

    async def scenario():
        """Keep one queued input pending while shutdown or execution failure ends the owner."""
        inputs = asyncio.Queue()
        working, queued, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda text="": (
            working.set()
            if text == "working"
            else queued.set()
            if text.startswith("Queued")
            else stopping.set()
            if text.startswith("Stopping")
            else None
        )

        def conversation(receive):
            """Fail or settle only after a second input has been admitted to the queue."""
            receive(DummyCompleter())
            controller.interaction.info("working")
            release.wait()
            if worker_failure:
                raise LookupError("worker failed")

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("unexecuted")
            await queued.wait()
            if not worker_failure:
                inputs.put_nowait(False)
                await stopping.wait()
            release.set()
            expected = LookupError if worker_failure else ValueError
            with pytest.raises(expected) as raised:
                await task
            if worker_failure:
                assert "queue cancellation reporting" in raised.value.__notes__[0].casefold()
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.warning.assert_called_once_with("Stopped; 1 queued inputs were not executed.")


def test_steering_precedes_followups_and_keeps_corrections_in_submission_order(ui):
    """Corrections interrupt the current scope, then retain FIFO ahead of ordinary queued text."""

    controller = SessionController(ui)
    release = Event()
    messages = []

    async def scenario():
        """Keep the execution owner held until both corrections and status are acknowledged."""
        inputs = asyncio.Queue()
        working, interrupted, status, completed = (asyncio.Event() for _ in range(4))
        read_inputs(ui, inputs)
        requests = []

        def info(text=""):
            """Observe UI transitions without inspecting controller state."""
            if text == "working":
                working.set()
            elif text.startswith("Interrupt requested"):
                requests.append(text)
                if len(requests) == 2:
                    interrupted.set()
            elif text == "Interrupting; 4 queued inputs.":
                status.set()
            elif text == "done":
                completed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Simulate an active tool boundary and then consume four admitted inputs."""
            messages.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                with pytest.raises(RunInterrupted):
                    controller.control.checkpoint()
            for _ in range(4):
                messages.append(receive(DummyCompleter()))
            controller.interaction.info("done")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            for value in ("followup one", "followup two", "/steer correction", "/steer /new"):
                inputs.put_nowait(value)
            await interrupted.wait()
            inputs.put_nowait("/status")
            await status.wait()
            release.set()
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["first", "correction", "/new", "followup one", "followup two"]
    assert all(isinstance(value, SteeringMessage) for value in messages[1:3])


@pytest.mark.parametrize("command", ["/interrupt", "/steer correction"])
def test_run_interruption_revokes_pending_approval_without_ending_the_chat(ui, command):
    """Active approvals are revoked with a scoped interrupt and are never answered by steering."""

    controller = SessionController(ui)
    decisions, messages = [], []

    async def scenario():
        """Interrupt a pending request, then verify subsequent ordinary input still works."""
        inputs = asyncio.Queue()
        pending, finished, processed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        def info(text=""):
            """Observe approval and execution transitions through visible output."""
            if text.startswith("Input requested"):
                pending.set()
            elif text == "interrupted":
                finished.set()
            elif text == "processed":
                processed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Catch only a run interrupt and leave the reader open for another message."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                decisions.append(controller.interaction.confirm("Approve?", default=True))
            controller.interaction.info("interrupted")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("processed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await pending.wait()
            inputs.put_nowait(command)
            await finished.wait()
            assert controller.control.stopped is False
            if command == "/interrupt":
                inputs.put_nowait("later")
            await processed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    assert messages == ["later" if command == "/interrupt" else "correction"]
    ui.confirm_async.assert_not_awaited()


def test_rejected_steering_does_not_interrupt_or_displace_a_full_mailbox(ui):
    """Capacity refusal leaves the active run and all sixteen accepted follow-ups intact."""
    controller = SessionController(ui)
    release = Event()
    interrupted = []
    received = []

    async def scenario():
        """Fill the queue behind an active scope, reject steering, then end normally."""
        inputs = asyncio.Queue()
        working, rejected, checked = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        def info(text=""):
            """Expose controlled execution boundaries through UI output."""
            if text == "working":
                working.set()
            elif text == "checked":
                checked.set()

        ui.info.side_effect = info
        ui.warning.side_effect = lambda text: rejected.set() if "queue is full" in text else None

        def conversation(receive):
            """Hold active ownership until rejected admission is observable."""
            receive(DummyCompleter())
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                interrupted.append(controller.control.interrupted)
                controller.control.checkpoint()
            for _ in range(16):
                received.append(receive(DummyCompleter()))
            controller.interaction.info("checked")

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            for index in range(16):
                inputs.put_nowait(f"followup {index}")
            inputs.put_nowait("/steer rejected")
            await rejected.wait()
            release.set()
            await checked.wait()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert interrupted == [False]
    assert received == [f"followup {index}" for index in range(16)]
    assert sum("queue is full" in call.args[0] for call in ui.warning.call_args_list) == 1


@pytest.mark.parametrize("operation", ["prompt", "confirm", "terminal_context"])
def test_interrupted_runs_cannot_open_late_questions_or_terminal_handoffs(ui, operation):
    """A question arriving after the control is accepted is revoked before any UI is opened."""
    controller = SessionController(ui)
    release = Event()

    async def scenario():
        """Accept interruption while the execution owner is held before requesting UI."""
        inputs = asyncio.Queue()
        working, interrupted, completed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        def info(text=""):
            """Expose the request and interrupt ordering through UI acknowledgements."""
            if text == "working":
                working.set()
            elif text.startswith("Interrupt requested"):
                interrupted.set()
            elif text == "completed":
                completed.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Try to open a stale approval or terminal scope after interruption."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                controller.interaction.info("working")
                release.wait()
                if operation == "terminal_context":
                    with controller.interaction.terminal_context():
                        pytest.fail("Interrupted work must not obtain terminal ownership")
                else:
                    getattr(controller.interaction, operation)("Approve?")
            controller.interaction.info("completed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("/interrupt")
            await interrupted.wait()
            release.set()
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    ui.confirm_async.assert_not_awaited()
    ui.terminal_context.assert_not_called()
    assert all(not call.args for call in ui.prompt_async.call_args_list)
