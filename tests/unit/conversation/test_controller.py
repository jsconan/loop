"""Verify responsive mailbox admission and serialized conversation ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Empty, PriorityQueue
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
    assert {"interrupt", "respond", "status", "steer", "queue", "choose"}.isdisjoint(setup_names)
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
                "queue",
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


def test_controller_routes_idle_commands_through_the_conversation_owner(ui):
    """Idle commands can prompt through the async UI without sync prompt-toolkit calls."""
    controller = SessionController(ui)
    selected = []

    def choose_scope(context: CommandContext) -> None:
        """Choose a scope through the command interaction."""
        selected.append(
            context.interaction.prompt(
                "Choose configuration scope:",
                choices={"user": "Save for current user", "session": "This session only"},
                index={"user": "u", "session": "s"},
            )
        )

    manager = CommandManager(
        commands=(choose_scope,),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    prompts = []
    received = []
    inputs = asyncio.Queue()

    async def prompt(*args, **kwargs):
        """Return deterministic editor input and a selected native prompt response."""
        if args == ("Choose configuration scope:",):
            prompts.append((args, kwargs))
            return "session"
        return await inputs.get()

    async def scenario():
        """Dispatch an idle command and open its choice prompt without another Enter."""
        ui.prompt_async.side_effect = prompt
        ui.prompt.side_effect = AssertionError("Synchronous UI input used during async runtime.")
        inputs.put_nowait("/choose_scope")
        inputs.put_nowait(False)

        def conversation(receive):
            """Dispatch commands on the conversation owner between idle reads."""
            while (value := receive(DummyCompleter())) is not False:
                received.append(value)
                manager.handle_user_command(value)

        task = asyncio.create_task(controller.run(conversation, route_idle_commands=True))
        try:
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert received == ["/choose_scope"]
    assert selected == ["session"]
    assert len(prompts) == 1
    assert prompts[0][0] == ("Choose configuration scope:",)
    assert prompts[0][1]["choices"] == {
        "user": "Save for current user",
        "session": "This session only",
    }
    assert prompts[0][1]["index"] == {"user": "u", "session": "s"}
    ui.prompt.assert_not_called()


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("prompt", "session"), ("confirm", True)],
)
def test_active_commands_use_native_async_blocking_interactions(ui, kind, expected):
    """Run-active commands resolve prompts and confirmations through the native async UI."""
    controller = SessionController(ui)
    release = Event()
    active, question_open, completed = (asyncio.Event() for _ in range(3))
    answers = []

    def choose(context: CommandContext) -> None:
        """Request the configured blocking interaction from an eligible command."""
        if kind == "prompt":
            answer = context.interaction.prompt(
                "Choose a scope:", choices={"session": "This session only"}
            )
        else:
            answer = context.interaction.confirm("Continue?", default=False)
        answers.append(answer)
        context.interaction.info("command completed")

    manager = CommandManager(
        commands=(
            CommandRegistration(choose, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()

    async def prompt(*args, **_kwargs):
        """Answer the command question and wait for ordinary editor input otherwise."""
        if kind == "prompt" and args == ("Choose a scope:",):
            question_open.set()
            return expected
        return await inputs.get()

    async def confirm(*_args, **_kwargs):
        """Answer the command confirmation through the async confirmation API."""
        question_open.set()
        return expected

    def info(message=""):
        """Expose worker activity and command completion to the test event loop."""
        if message == "working":
            active.set()
        elif message == "command completed":
            completed.set()

    ui.prompt_async.side_effect = prompt
    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = info

    def conversation(receive):
        """Hold an active execution while the command asks its blocking question."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()
        receive(DummyCompleter())

    async def scenario():
        """Submit an active command and wait for its UI question to resolve."""
        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/choose")
            await question_open.wait()
            await completed.wait()
            release.set()
            controller.stop()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == [expected]
    if kind == "prompt":
        ui.prompt_async.assert_any_await(
            "Choose a scope:",
            completer=None,
            choices={"session": "This session only"},
            index=None,
            default=None,
            secret=False,
        )
        ui.confirm_async.assert_not_awaited()
    else:
        ui.confirm_async.assert_awaited_once_with("Continue?", default=False)


def test_active_command_finishes_after_conversation_worker_and_keeps_prompt_open(ui):
    """Finish an active command's prompt after its conversation worker returns."""
    controller = SessionController(ui)
    command_started, prompt_returned, command_release = (Event() for _ in range(3))
    prompt_open = asyncio.Event()
    inputs = asyncio.Queue()
    answers = []

    def choose(context: CommandContext) -> None:
        """Wait for one choice and hold command completion until the test releases it."""
        command_started.set()
        answers.append(
            context.interaction.prompt("Choose a scope:", choices={"session": "Session"})
        )
        prompt_returned.set()
        command_release.wait()

    manager = CommandManager(
        commands=(
            CommandRegistration(choose, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)

    async def prompt(*args, **_kwargs):
        """Return ordinary input or wait for the blocking scope response."""
        if args == ("Choose a scope:",):
            prompt_open.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    def conversation(receive):
        """Return when active command dispatch proves its prompt has started."""
        receive(DummyCompleter())
        command_started.wait()

    async def scenario():
        """Resolve the prompt after conversation execution ends, then finish the command."""
        with controller.control.execution():
            task = asyncio.create_task(controller.run(conversation))
            try:
                inputs.put_nowait("start")
                inputs.put_nowait("/choose")
                await prompt_open.wait()
                inputs.put_nowait("session")
                await asyncio.to_thread(prompt_returned.wait)
                command_release.set()
                await task
            finally:
                command_release.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == ["session"]


def test_stopping_releases_an_active_command_confirmation(ui):
    """Shutdown cancels an active command confirmation without granting approval."""
    controller = SessionController(ui)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()
    answers = []

    def approve(context: CommandContext) -> None:
        """Request approval through the controller mailbox."""
        answers.append(context.interaction.confirm("Continue?", default=False))

    manager = CommandManager(
        commands=(
            CommandRegistration(approve, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Keep confirmation open until controller shutdown cancels it."""
        question_open.set()
        await asyncio.Future()

    def info(message=""):
        """Signal when conversation execution has become active."""
        if message == "working":
            active.set()

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = info

    def conversation(receive):
        """Hold active execution while the confirmation command is entered."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Stop after the command confirmation opens and wait for clean settlement."""
        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/approve")
            await question_open.wait()
            release.set()
            controller.stop()
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == []
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)


def test_command_dispatch_preserves_unexpected_keyboard_interrupt(ui):
    """Propagate an unexpected interruption instead of treating it as shutdown control flow."""
    controller = SessionController(ui)

    def interrupt() -> None:
        """Raise the keyboard interruption being handled by command dispatch."""
        raise KeyboardInterrupt

    manager = CommandManager(
        commands=(interrupt,),
        interaction=ui,
    )
    with pytest.raises(KeyboardInterrupt):
        controller._run_command(manager, "/interrupt")


def test_interrupted_active_command_does_not_stop_the_controller(ui):
    """A run interruption releases a blocking command and leaves shutdown unset."""
    controller = SessionController(ui)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()

    def approve(context: CommandContext) -> None:
        """Request approval through the controller mailbox."""
        context.interaction.confirm("Continue?", default=False)

    manager = CommandManager(
        commands=(
            CommandRegistration(approve, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Wait until an active run interruption cancels this confirmation."""
        question_open.set()
        await asyncio.Future()

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = lambda message="": active.set() if message == "working" else None

    def conversation(receive):
        """Hold active execution until the confirmation request is interrupted."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Interrupt the pending approval and allow both owners to settle normally."""
        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/approve")
            await question_open.wait()
            assert controller.request_interrupt() is True
            release.set()
            await task
            assert controller.control.stopped is False
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert controller.running is False
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)


def test_failed_active_command_is_reported_during_controller_settlement(ui):
    """Preserve a UI failure and note a distinct active-command failure during settlement."""
    controller = SessionController(ui)
    release = Event()
    active, question_open = asyncio.Event(), asyncio.Event()

    def fail(context: CommandContext) -> None:
        """Replace an interaction failure with a command failure for cleanup reporting."""
        try:
            context.interaction.confirm("Continue?", default=False)
        except RuntimeError as error:
            raise ValueError("command failed") from error

    manager = CommandManager(
        commands=(
            CommandRegistration(fail, capabilities=frozenset({CommandCapability.RUN_ACTIVE})),
        ),
        interaction=controller.interaction,
    )
    controller.bind_command_manager(manager)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    async def confirm(*_args, **_kwargs):
        """Release the conversation worker before reporting a simulated UI failure."""
        question_open.set()
        release.set()
        raise RuntimeError("confirmation UI failed")

    ui.confirm_async.side_effect = confirm
    ui.info.side_effect = lambda message="": active.set() if message == "working" else None

    def conversation(receive):
        """Wait for the command interaction while the conversation remains active."""
        receive(DummyCompleter())
        with controller.control.execution():
            controller.interaction.info("working")
            release.wait()

    async def scenario():
        """Raise the UI failure after controller cleanup settles both owners."""
        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await active.wait()
            inputs.put_nowait("/fail")
            await question_open.wait()
            with pytest.raises(RuntimeError, match="confirmation UI failed") as error:
                await task
            assert any(
                "Command also stopped with ValueError." in note for note in error.value.__notes__
            )
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


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

    assert {"interrupt", "respond", "status", "steer", "queue"}.isdisjoint(names())
    controller._loop = Mock()
    with controller.control.execution():
        active_names = names()
        assert {"interrupt", "status", "steer", "queue"} <= active_names
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
    assert {"interrupt", "respond", "status", "steer", "queue"}.isdisjoint(names())


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


@pytest.mark.parametrize(
    ("kind", "kwargs", "expected"),
    [
        (EVENT_CONFIRM, {}, False),
        (EVENT_CONFIRM, {"default": True}, True),
        (EVENT_PROMPT, {"default": "fallback"}, "fallback"),
    ],
)
def test_empty_direct_response_uses_the_prompt_default(ui, kind, kwargs, expected):
    """An empty direct answer applies a declared confirmation or prompt default."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    assert controller.request_response("") is ResponseOutcome.ANSWERED
    assert request.result.result() == expected


def test_empty_direct_response_without_a_default_is_rejected(ui):
    """An empty free-form answer stays pending when the prompt has no default."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs={}, kind=EVENT_PROMPT)
    controller._pending = request

    assert controller.request_response("") is ResponseOutcome.INVALID
    assert controller._pending is request


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [(EVENT_PROMPT, {}), (EVENT_CONFIRM, {})],
)
def test_direct_response_does_not_bypass_an_active_native_prompt(ui, kind, kwargs):
    """Direct responses cannot resolve a request while its native UI owns the input."""
    controller = SessionController(ui)
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs=kwargs, kind=kind)
    controller._pending = request

    async def scenario():
        """Keep a live async input task while rejecting a competing direct answer."""
        answer_input = asyncio.get_running_loop().create_future()
        controller._answer_input = answer_input
        try:
            assert controller.request_response("answer") is ResponseOutcome.PROMPT_REQUIRED
            assert controller._pending is request
            assert not request.result.done()
        finally:
            answer_input.cancel()
            controller._answer_input = None

    asyncio.run(scenario())


def test_request_response_without_value_keeps_the_native_prompt_active(ui):
    """A bare response request leaves an unanswered question active for its native prompt."""
    controller = SessionController(ui)
    assert controller.request_response() is ResponseOutcome.NO_REQUEST
    request = InteractionCall(function=ui.prompt, args=("Question",), kwargs={}, kind=EVENT_PROMPT)
    controller._pending = request

    assert controller.request_response() is ResponseOutcome.PROMPTED
    assert controller._pending is request


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


def test_queue_requires_a_run_and_preserves_command_like_text(ui):
    """Queued messages preserve text that resembles commands."""
    controller = SessionController(ui)

    assert controller.queue("/exit", ui) is False
    with controller.control.execution():
        assert controller.queue("/exit", ui) is True

    priority, _sequence, message = controller._inbox.get_nowait()
    assert priority == 1
    assert message == SteeringMessage("/exit")
    assert isinstance(message, SteeringMessage)


def test_default_steering_admission_falls_back_to_ordinary_input_while_idle(ui):
    """A steering admission racing with an idle boundary remains ordinary input."""
    controller = SessionController(ui)

    assert controller._enqueue_steering("message") is True

    _priority, _sequence, message = controller._inbox.get_nowait()
    assert message == "message"
    assert not isinstance(message, SteeringMessage)


def test_steering_admission_without_a_controller_loop_still_queues(ui):
    """An active execution can admit steering before controller startup."""
    controller = SessionController(ui)

    with controller.control.execution():
        assert controller._enqueue_steering("message") is True

    _priority, _sequence, message = controller._inbox.get_nowait()
    assert message == SteeringMessage("message")


def test_steering_admission_from_a_non_ui_thread_wakes_the_controller_loop(ui):
    """Cross-thread steering admission schedules its mailbox wake on the owning loop."""
    controller = SessionController(ui)

    async def scenario():
        """Use a live loop with a distinct UI-thread marker to exercise thread-safe wakeup."""
        controller._loop = asyncio.get_running_loop()
        controller._ui_thread = -1
        with controller.control.execution():
            assert controller._enqueue_steering("message") is True
        await asyncio.sleep(0)
        return controller._input_signal.is_set()

    assert asyncio.run(scenario()) is True


def test_controller_accepts_messages_and_status_during_serialized_work(ui):
    """UI input remains live while execution retains one owner and processes queued inputs FIFO."""
    controller = SessionController(ui)
    release = Event()
    received, execution_threads, output_threads = [], [], []

    async def scenario():
        """Queue messages behind a controlled execution boundary."""
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


def test_ordinary_input_during_active_run_is_deferred_steering(ui):
    """Active ordinary input finishes the current operation, then returns as a correction."""
    controller = SessionController(ui)
    release = Event()
    received = []

    async def scenario():
        """Submit a correction while an execution scope is active."""
        inputs = asyncio.Queue()
        working, accepted, consumed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)
        ui.info.side_effect = lambda message="": (
            working.set()
            if message == "working"
            else accepted.set()
            if message.startswith("Steering accepted")
            else None
        )

        def conversation(receive):
            """Complete the current boundary without converting steering into interruption."""
            received.append(receive(DummyCompleter()))
            with controller.control.execution():
                controller.interaction.info("working")
                release.wait()
                assert controller.control.steering_pending is True
                assert controller.control.interrupted is False
                controller.control.checkpoint()
            received.append(receive(DummyCompleter()))
            consumed.set()
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("first")
            await working.wait()
            inputs.put_nowait("correction")
            await accepted.wait()
            assert controller.status_snapshot().state == "Steering"
            release.set()
            await consumed.wait()
            inputs.put_nowait(False)
            await task
        finally:
            release.set()
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert received == ["first", SteeringMessage("correction")]
    assert controller.control.steering_pending is False


@pytest.mark.parametrize(
    ("options", "answer", "expected"),
    [
        ({}, "a typed clarification", "a typed clarification"),
        ({"completer": WordCompleter(["custom answer"])}, "custom answer", "custom answer"),
        ({"exit_commands": ("quit",)}, "a typed clarification", "a typed clarification"),
        ({"choices": {"first": "First", "second": "Second"}}, "second", "second"),
    ],
)
def test_controller_uses_the_native_prompt_for_blocking_interactions(ui, options, answer, expected):
    """Blocking questions preserve prompt options for the source's native async UI."""
    controller = SessionController(ui)
    responses = []

    async def scenario():
        """Open a blocking question in the regular editor and submit its response."""
        inputs = asyncio.Queue()
        question_visible, answered = asyncio.Event(), asyncio.Event()

        async def prompt(*args, **kwargs):
            """Answer the blocking request while ordinary input stays in the message editor."""
            if args == ("Choose a target?",):
                question_visible.set()
                return answer
            return await inputs.get()

        ui.prompt_async.side_effect = prompt

        def info(message=""):
            """Observe completion of the blocking request."""
            if message == "answered":
                answered.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Pause the owner on a prompt and then continue its input loop."""
            receive(DummyCompleter())
            responses.append(controller.interaction.prompt("Choose a target?", **options))
            controller.interaction.info("answered")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await question_visible.wait()
            await answered.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert responses == [expected]
    question_calls = [call for call in ui.prompt_async.await_args_list if call.args]
    assert len(question_calls) == 1
    assert question_calls[0].args == ("Choose a target?",)
    assert question_calls[0].kwargs["choices"] == options.get("choices")
    assert question_calls[0].kwargs["completer"] is options.get("completer")
    if options.get("exit_commands"):
        assert question_calls[0].kwargs["exit_commands"] == ("quit",)
    ui.confirm_async.assert_not_awaited()
    ui.prompt.assert_not_called()


def test_controller_answers_blocking_prompt_when_no_editor_task_is_active(ui):
    """A startup question uses the source's native async prompt before chat input starts."""
    controller = SessionController(ui)
    ui.prompt_async.return_value = "answer"
    responses = []

    def conversation(_receive):
        """Request clarification before opening the ordinary message editor."""
        responses.append(controller.interaction.prompt("Clarify this value?"))

    asyncio.run(controller.run(conversation))

    assert responses == ["answer"]
    ui.prompt_async.assert_awaited_once()
    assert ui.prompt_async.await_args.args == ("Clarify this value?",)


def test_pending_confirmation_uses_the_native_confirmation_ui(ui):
    """A confirmation delegates unchanged to the source's native confirmation interaction."""
    controller = SessionController(ui)
    ui.confirm_async.return_value = False
    decisions = []

    def conversation(_receive):
        """Ask for approval before opening the ordinary message editor."""
        decisions.append(controller.interaction.confirm("Continue?", default=False))

    asyncio.run(controller.run(conversation))

    assert decisions == [False]
    ui.confirm_async.assert_awaited_once_with("Continue?", default=False)
    ui.prompt_async.assert_not_awaited()


def test_pending_question_preserves_choices_for_the_native_prompt(ui):
    """Blocking choice prompts pass their choices to the source UI without rewriting them."""
    controller = SessionController(ui)
    responses = []

    async def scenario():
        """Resolve a selected answer through the source's regular choice prompt."""
        inputs = asyncio.Queue()
        question_open, answered = asyncio.Event(), asyncio.Event()

        async def prompt(*args, **kwargs):
            """Return the selected choice key through the native prompt interface."""
            if args == ("Clarify the target?",):
                question_open.set()
                return "second"
            return await inputs.get()

        ui.prompt_async.side_effect = prompt

        def info(message=""):
            """Observe the completed interaction."""
            if message == "answered":
                answered.set()

        ui.info.side_effect = info

        def conversation(receive):
            """Ask for a selected response and then return to ordinary conversation input."""
            receive(DummyCompleter())
            responses.append(
                controller.interaction.prompt(
                    "Clarify the target?", choices={"first": "First", "second": "Second"}
                )
            )
            controller.interaction.info("answered")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await question_open.wait()
            await answered.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert responses == ["second"]
    question_call = next(call for call in ui.prompt_async.await_args_list if call.args)
    assert question_call.args == ("Clarify the target?",)
    assert question_call.kwargs["choices"] == {"first": "First", "second": "Second"}
    assert question_call.kwargs["completer"] is None


def test_declining_a_pending_question_keeps_the_conversation_available(ui):
    """Declining a native confirmation does not close the regular conversation editor."""
    controller = SessionController(ui)
    answers, messages = [], []

    async def scenario():
        """Decline the active request and then submit another ordinary message."""
        inputs = asyncio.Queue()
        question_open, continued, completed = (asyncio.Event() for _ in range(3))

        async def confirm(*_args, **_kwargs):
            """Return a negative decision through the native confirmation UI."""
            question_open.set()
            return False

        read_inputs(ui, inputs)
        ui.confirm_async.side_effect = confirm
        ui.info.side_effect = lambda message="": (
            continued.set()
            if message == "continued"
            else completed.set()
            if message == "done"
            else None
        )

        def conversation(receive):
            """Continue reading after receiving a negative result from the interaction."""
            receive(DummyCompleter())
            answers.append(controller.interaction.confirm("Approve?"))
            controller.interaction.info("continued")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("done")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await question_open.wait()
            await continued.wait()
            assert controller.control.stopped is False
            inputs.put_nowait("later")
            await completed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())

    assert answers == [False]
    assert messages == ["later"]


def test_stop_discards_input_completed_at_the_shutdown_boundary(ui):
    """Input that finishes as shutdown begins is not admitted to conversation execution."""
    controller = SessionController(ui)
    messages = []

    async def prompt(*_args, **_kwargs):
        """Request shutdown immediately before returning one late editor value."""
        controller.stop()
        return "late input"

    ui.prompt_async.side_effect = prompt

    def conversation(receive):
        """Wait for input and record the shutdown signal instead of late text."""
        messages.append(receive(DummyCompleter()))

    asyncio.run(controller.run(conversation))

    assert messages == [False]


def test_interrupt_cancels_a_modal_prompt_and_restores_the_editor_draft(ui):
    """Interruption closes masked prompts without losing the suspended chat draft."""
    controller = SessionController(ui)
    messages = []

    async def scenario():
        """Interrupt a masked request and continue through the restored main editor."""
        inputs = asyncio.Queue()
        prompt_open, interrupted, processed = (asyncio.Event() for _ in range(3))
        ui.current_input.return_value = "unfinished draft"

        async def prompt(*_args, **kwargs):
            """Block on the masked prompt while normal messages use the input queue."""
            if kwargs.get("secret"):
                prompt_open.set()
                await asyncio.Future()
            return await inputs.get()

        ui.prompt_async.side_effect = prompt
        ui.info.side_effect = lambda message="": (
            interrupted.set()
            if message == "interrupted"
            else processed.set()
            if message == "processed"
            else None
        )

        def conversation(receive):
            """Catch an interrupted secret request and accept a later chat message."""
            receive(DummyCompleter())
            with controller.control.execution(), pytest.raises(RunInterrupted):
                controller.interaction.prompt("Credential:", secret=True, exit_commands=())
            controller.interaction.info("interrupted")
            messages.append(receive(DummyCompleter()))
            controller.interaction.info("processed")
            receive(DummyCompleter())

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await prompt_open.wait()
            assert controller.request_interrupt() is True
            await interrupted.wait()
            inputs.put_nowait("later")
            await processed.wait()
            assert ui.set_current_input.call_args.args == ("",)
            assert any(
                call.kwargs.get("initial") == "unfinished draft"
                for call in ui.prompt_async.await_args_list
            )
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert messages == ["later"]


def test_submitted_chat_remains_queued_when_a_modal_question_arrives(ui):
    """Input completed before a blocking question is delivered stays ordinary conversation text."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Credential:",),
        kwargs={"secret": True},
        kind=EVENT_PROMPT,
    )
    controller._ready = True
    controller._completer = DummyCompleter()

    async def scenario():
        """Complete normal input and modal interaction together through the controller loop."""
        controller._loop = asyncio.get_running_loop()
        controller._events = asyncio.Queue()
        controller._events.put_nowait(request)
        inputs = []

        async def prompt(*_args, **_kwargs):
            """Return queued chat first and a masked response to the blocking request second."""
            inputs.append("ordinary chat" if not inputs else "private response")
            return inputs[-1]

        ui.prompt_async.side_effect = prompt
        controller.bind_command_manager(
            CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
        )
        worker = asyncio.ensure_future(asyncio.wrap_future(request.result))

        await controller._drive(worker, route_idle_commands=False)

    asyncio.run(scenario())
    assert controller._inbox.get_nowait()[2] == "ordinary chat"
    assert controller._pending is None
    assert request.result.result() == "private response"


def test_cancelled_editor_task_does_not_block_a_modal_question(ui):
    """A cancelled editor task is ignored while its blocking interaction is answered."""
    controller = SessionController(ui)
    request = InteractionCall(
        function=ui.prompt,
        args=("Credential:",),
        kwargs={"secret": True},
        kind=EVENT_PROMPT,
    )
    controller._ready = True
    controller._completer = DummyCompleter()
    prompt_calls = 0

    async def prompt(*_args, **_kwargs):
        """Cancel ordinary editor input, then answer the masked interaction."""
        nonlocal prompt_calls
        prompt_calls += 1
        if prompt_calls == 1:
            raise asyncio.CancelledError
        return "private response"

    async def scenario():
        """Deliver the modal request alongside an editor task that cancels immediately."""
        controller._loop = asyncio.get_running_loop()
        controller._events = asyncio.Queue()
        controller._events.put_nowait(request)
        ui.prompt_async.side_effect = prompt
        controller.bind_command_manager(
            CommandManager(providers=(ConversationCommands(controller),), interaction=ui)
        )
        worker = asyncio.ensure_future(asyncio.wrap_future(request.result))

        await controller._drive(worker, route_idle_commands=False)

    asyncio.run(scenario())
    assert request.result.result() == "private response"
    assert prompt_calls == 3
    assert [call.kwargs.get("secret", False) for call in ui.prompt_async.await_args_list] == [
        False,
        True,
        False,
    ]


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
            with controller.control.execution():
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
        """Stop after the native confirmation prompt opens but before the user answers."""
        inputs = asyncio.Queue()
        prompted = asyncio.Event()
        read_inputs(ui, inputs)

        async def confirm(*_args, **_kwargs):
            """Signal that approval input opened, then wait for its cancellation."""
            prompted.set()
            return await inputs.get()

        ui.confirm_async.side_effect = confirm

        def conversation(receive):
            """Request a decision from the independent approval channel."""
            receive(DummyCompleter())
            decisions.append(controller.interaction.confirm("Approve?", default=True))

        task = asyncio.create_task(controller.run(conversation))
        try:
            inputs.put_nowait("start")
            await prompted.wait()
            controller.stop()
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    ui.confirm_async.assert_awaited_once_with("Approve?", default=True)


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
        held, finished = asyncio.Event(), asyncio.Event()
        read_inputs(ui, inputs)

        def output(message=""):
            """Observe terminal ownership without sharing domain objects."""
            if message == "held":
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


def test_async_controller_keeps_conversation_on_ui_loop_and_routes_approval(ui):
    """Async work can await a separate approval while the UI driver owns input."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    ui.confirm_async.return_value = True
    owner_thread = []

    async def conversation(receive):
        """Read admitted input, then request approval through the controller mailbox."""
        owner_thread.append(get_ident())
        assert await receive(DummyCompleter()) == "hello"
        assert await controller.interaction.confirm_async("Approve operation?") is True
        controller.interaction.info("approved")

    ui_thread = get_ident()

    async def scenario():
        """Run the async owner and submit its first ordinary message."""
        task = asyncio.create_task(controller.run_async(conversation))
        await asyncio.sleep(0)
        await inputs.put("hello")
        await task

    asyncio.run(scenario())

    assert owner_thread == [ui_thread]
    ui.confirm_async.assert_awaited_once_with("Approve operation?", default=False)
    ui.info.assert_any_call("approved")
    assert controller.running is False


def test_async_controller_processes_status_while_conversation_is_suspended(ui):
    """A read-only control command remains available during async work."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    working, status_seen, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    displayed = []

    def record_info(message=""):
        """Capture output and signal only after the status response is displayed."""
        displayed.append(message)
        if message.endswith("queued inputs."):
            status_seen.set()

    ui.info.side_effect = record_info

    async def conversation(receive):
        """Wait at an async operation boundary after consuming its initiating message."""
        assert await receive(DummyCompleter()) == "hello"
        working.set()
        await release.wait()

    async def scenario():
        """Keep the agent suspended while the UI executes its status command."""
        task = asyncio.create_task(controller.run_async(conversation))
        await asyncio.sleep(0)
        await inputs.put("hello")
        await working.wait()
        await inputs.put("/status")
        await status_seen.wait()
        release.set()
        await task

    asyncio.run(scenario())

    assert any(message.endswith("queued inputs.") for message in displayed)


def test_async_controller_rejects_reuse_and_suppresses_requested_keyboard_interrupt(ui):
    """The async controller remains single-use and normalizes stopped input reads."""
    controller = SessionController(ui)
    controller.stop()

    async def conversation(receive):
        """Observe the stop sentinel without running another operation."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Run once after stop, then verify the single-use lifecycle guard."""
        await controller.run_async(conversation)
        with pytest.raises(RuntimeError, match="only run once"):
            await controller.run_async(conversation)

    asyncio.run(scenario())

    ui.conversation_ended.assert_called_once_with()


def test_async_controller_wakes_a_pending_receive_when_stopped(ui):
    """A stop request wakes the async mailbox reader and rejects a later message."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    input_prompted = asyncio.Event()

    async def prompt(*args, **kwargs):
        """Signal that the owner is waiting for admitted input."""
        input_prompted.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    async def conversation(receive):
        """Wait for a message that the stop request replaces with its sentinel."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Request shutdown only after the async receive is waiting."""
        task = asyncio.create_task(controller.run_async(conversation))
        await input_prompted.wait()
        controller.stop()
        await task

    asyncio.run(scenario())

    assert controller.control.stopped is True


def test_async_controller_rechecks_queue_after_a_racing_admission(ui):
    """A queued input arriving at the clear-and-wait boundary is not lost."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)

    class RacingQueue(PriorityQueue):
        """Force one empty read followed by a visible queued input."""

        _first_read: bool

        def __init__(self):
            """Initialize a pre-admitted message and one synthetic read race."""
            super().__init__()
            self._first_read = True
            self.put((1, 0, "raced"))

        def get_nowait(self):
            """Return an empty result once so the receiver checks the queue again."""
            if self._first_read:
                self._first_read = False
                raise Empty
            return super().get_nowait()

        def empty(self):
            """Expose the raced message immediately after the synthetic empty result."""
            if not self._first_read:
                return False
            return super().empty()

    controller._inbox = RacingQueue()
    received = []

    async def conversation(receive):
        """Read the input admitted just before the mailbox wait."""
        received.append(await receive(DummyCompleter()))

    asyncio.run(controller.run_async(conversation))

    assert received == ["raced"]


def test_async_controller_handles_inline_output_and_rejects_sync_input(ui):
    """Async owners present output inline and cannot issue blocking UI input calls."""
    controller = SessionController(ui)
    ui.info.side_effect = ValueError("display failed")
    terminal_exit = InteractionCall(lambda: None, (), {}, kind="terminal_exit")
    controller._terminal_owned = True

    async def conversation(_receive):
        """Exercise inline output failure, terminal cleanup, and sync-input rejection."""
        with pytest.raises(ValueError, match="display failed"):
            controller._send(InteractionCall(ui.info, ("bad",), {}))
        with pytest.raises(RuntimeError, match="Synchronous mailbox input"):
            controller._send(InteractionCall(ui.confirm, ("Approve?",), {}, kind=EVENT_CONFIRM))
        controller._send(terminal_exit)

    asyncio.run(controller.run_async(conversation))

    assert terminal_exit.result.result() is None
    assert controller._terminal_owned is False


def test_async_controller_settles_failures_and_reports_discarded_input(ui):
    """Execution failures remain primary while queue-reporting errors are attached."""
    controller = SessionController(ui)
    controller._inbox.put((1, 0, "queued"))
    ui.warning.side_effect = OSError("warning failed")

    async def conversation(_receive):
        """Fail the async owner after the controller has started it."""
        raise RuntimeError("operation failed")

    with pytest.raises(RuntimeError, match="operation failed") as error:
        asyncio.run(controller.run_async(conversation))

    assert any(
        "Queue cancellation reporting also failed with OSError" in note
        for note in error.value.__notes__
    )
    assert controller.running is False


def test_async_controller_propagates_unsolicited_keyboard_interrupt(ui):
    """Keyboard interrupts remain visible unless the controller requested shutdown."""
    controller = SessionController(ui)

    async def conversation(_receive):
        """Raise an unsolicited interrupt from the conversation owner."""
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(controller.run_async(conversation))


def test_async_controller_waits_for_owned_work_after_repeated_cancellation(ui):
    """A second cancellation cannot abandon async work still owned by the controller."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    operation_started, cleanup_started, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    settle = controller._settle

    async def observe_settlement(worker):
        """Expose the cleanup boundary before delegating to the real settlement loop."""
        cleanup_started.set()
        return await settle(worker)

    controller._settle = observe_settlement

    async def conversation(receive):
        """Keep owned work active until the test releases it."""
        assert await receive(DummyCompleter()) == "hello"
        operation_started.set()
        await release.wait()

    async def scenario():
        """Cancel twice while cleanup waits, then allow the operation to settle."""
        await inputs.put("hello")
        task = asyncio.create_task(controller.run_async(conversation))
        await operation_started.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert controller.running is False


def test_async_controller_preserves_primary_and_secondary_failures(ui):
    """UI, command, and conversation failures remain visible with the primary error first."""
    controller = SessionController(ui)
    command_failure = RuntimeError("command failed")

    async def fail_after_owner(worker, *, route_idle_commands):
        """Wait for the conversation failure, then report a separate UI failure."""
        await asyncio.gather(worker, return_exceptions=True)
        raise ValueError("driver failed")

    async def settle_with_command_failure(_worker):
        """Return a late command failure for the controller to attach."""
        return command_failure

    async def conversation(_receive):
        """Fail the conversation owner independently of the UI driver."""
        raise OSError("conversation failed")

    controller._drive = fail_after_owner
    controller._settle = settle_with_command_failure

    with pytest.raises(ValueError, match="driver failed") as error:
        asyncio.run(controller.run_async(conversation))

    assert any("Command also stopped with RuntimeError" in note for note in error.value.__notes__)
    assert any("Conversation also stopped with OSError" in note for note in error.value.__notes__)


def test_async_controller_reports_discarded_input_failure_without_primary_error(ui):
    """A failure reporting discarded queued input surfaces after normal settlement."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    ui.warning.side_effect = OSError("warning failed")
    started, stopping = asyncio.Event(), asyncio.Event()

    def observe_info(message=""):
        """Signal when the UI accepted its terminal-exit response."""
        if message.startswith("Stopping after"):
            stopping.set()

    ui.info.side_effect = observe_info

    async def conversation(receive):
        """Wait for UI exit while a previously queued input is discarded."""
        assert await receive(DummyCompleter()) == "hello"
        controller._inbox.put((1, 0, "queued"))
        started.set()
        await stopping.wait()

    async def scenario():
        """Submit a message, then exit while the owner is doing asynchronous work."""
        task = asyncio.create_task(controller.run_async(conversation))
        await inputs.put("hello")
        await started.wait()
        await inputs.put(False)
        with pytest.raises(OSError, match="warning failed"):
            await task

    asyncio.run(scenario())


def test_async_controller_stop_without_start_and_requested_keyboard_interrupt(ui):
    """Stop-before-start wakes no foreign loop and suppresses its owned interrupt."""
    controller = SessionController(ui)
    assert controller._stop() == 0

    async def conversation(_receive):
        """Raise an interrupt after shutdown has already been requested."""
        raise KeyboardInterrupt

    asyncio.run(controller.run_async(conversation))

    ui.conversation_ended.assert_called_once_with()


def test_async_controller_stop_signals_from_a_foreign_thread_identity(ui):
    """The stop signal uses thread-safe scheduling when the caller is not the UI owner."""
    controller = SessionController(ui)
    inputs = asyncio.Queue()
    read_inputs(ui, inputs)
    input_prompted = asyncio.Event()

    async def prompt(*args, **kwargs):
        """Signal that the conversation owner is waiting for a message."""
        input_prompted.set()
        return await inputs.get()

    ui.prompt_async.side_effect = prompt

    async def conversation(receive):
        """Wait for the stop sentinel."""
        assert await receive(DummyCompleter()) is False

    async def scenario():
        """Exercise the thread-safe input wakeup while remaining on one event loop."""
        task = asyncio.create_task(controller.run_async(conversation))
        await input_prompted.wait()
        controller._ui_thread = -1
        controller._stop()
        await task

    asyncio.run(scenario())


def test_async_controller_marks_inline_conversation_end_as_delivered(ui):
    """An inline conversation-ended event is not emitted twice after shutdown."""
    controller = SessionController(ui)
    controller.stop()

    async def conversation(_receive):
        """Deliver the terminal event through the async owner's UI thread."""
        controller.interaction.conversation_ended()

    asyncio.run(controller.run_async(conversation))

    ui.conversation_ended.assert_called_once_with()


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


def test_controller_external_stop_cancels_an_active_secret_prompt(ui):
    """Shutdown cancels a masked request before releasing exclusive prompt ownership."""
    controller = SessionController(ui)

    async def scenario():
        """Stop while a masked prompt is actively waiting for terminal input."""
        opened = asyncio.Event()

        async def answer(*_args, **kwargs):
            """Keep the masked request open until the controller cancels its input task."""
            if kwargs.get("secret"):
                opened.set()
                await asyncio.Event().wait()

        ui.prompt_async.side_effect = answer
        task = asyncio.create_task(
            controller.run(
                lambda _receive: controller.interaction.prompt("Credential:", secret=True)
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
    assert any(call.kwargs.get("secret") for call in ui.prompt_async.await_args_list)


@pytest.mark.parametrize("failure", [asyncio.CancelledError, ValueError])
def test_controller_propagates_failures_from_a_masked_prompt(ui, failure):
    """Masked prompt failures are returned to the execution owner without being swallowed."""
    controller = SessionController(ui)
    ui.prompt_async.side_effect = failure

    with pytest.raises(failure):
        asyncio.run(
            controller.run(
                lambda _receive: controller.interaction.prompt("Credential:", secret=True)
            )
        )

    assert controller.running is False


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


def test_steering_precedes_queued_messages_and_keeps_corrections_in_submission_order(ui):
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
            for value in (
                "/queue queued one",
                "/queue queued two",
                "/steer correction",
                "/steer /new",
            ):
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
    assert messages == ["first", "correction", "/new", "queued one", "queued two"]
    assert all(isinstance(value, SteeringMessage) for value in messages[1:3])


def test_run_interruption_revokes_pending_approval_without_ending_the_chat(ui):
    """An active approval is revoked by run control and leaves the chat available."""
    controller = SessionController(ui)
    decisions, messages = [], []

    async def scenario():
        """Interrupt the native question, then accept a fresh conversation message."""
        inputs = asyncio.Queue()
        prompted, finished, processed = (asyncio.Event() for _ in range(3))
        read_inputs(ui, inputs)

        async def confirm(*_args, **_kwargs):
            """Signal that approval input opened, then wait until run control cancels it."""
            prompted.set()
            await asyncio.Event().wait()

        ui.confirm_async.side_effect = confirm

        def info(text=""):
            """Observe the execution owner after its approval request is revoked."""
            if text == "interrupted":
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
            await prompted.wait()
            assert controller.request_interrupt() is True
            await finished.wait()
            assert controller.control.stopped is False
            inputs.put_nowait("later")
            await processed.wait()
            inputs.put_nowait("/exit")
            await task
        finally:
            controller.stop()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert decisions == []
    assert messages == ["later"]
    ui.confirm_async.assert_awaited_once_with("Approve?", default=True)


def test_rejected_steering_does_not_interrupt_or_displace_a_full_mailbox(ui):
    """Capacity refusal leaves the active run and all sixteen accepted queued messages intact."""
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
                inputs.put_nowait(f"/queue queued {index}")
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
    assert received == [f"queued {index}" for index in range(16)]
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
