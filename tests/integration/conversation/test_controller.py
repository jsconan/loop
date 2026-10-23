"""Exercise mailbox execution with native prompt input and captured response output."""

import asyncio
from io import StringIO
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, DummyCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from loop import (
    AnswerCompleted,
    CommandContext,
    CommandManager,
    ConsoleInteraction,
    Loop,
    Message,
    ModelInfo,
    ResponseCompleted,
    SessionController,
    SessionManager,
)
from loop.configuration import ConfigurationCommands, ConfigurationManager
from loop.session import LocalSessionNameGenerator

pytestmark = pytest.mark.integration


def test_async_loop_commands_can_reserve_the_native_terminal(tmp_path):
    """An idle synchronous command hands off the terminal while its async loop awaits it."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(session=prompt, console=Console(file=output), markdown=False)
        controller = SessionController(source)
        loop = Loop.create_default(
            SimpleNamespace(default_model="fixture", get_context_window=lambda _model: None),
            interaction=controller.interaction,
            run_control=controller.control,
            working_directory=tmp_path,
            session_name_generator=LocalSessionNameGenerator(),
        )
        controller.bind_command_manager(loop.command_manager)

        def terminal_probe(context: CommandContext) -> None:
            """Reserve and release terminal input without privileged or external operations."""
            with context.interaction.terminal_context():
                context.interaction.info("terminal reserved")
            context.interaction.info("terminal released")

        loop.command_manager.register(terminal_probe)

        async def scenario():
            """Dispatch through the actual async loop and stop after terminal ownership returns."""
            completed = asyncio.Event()
            original_info = source.info

            def info(message=""):
                """Render output and observe the terminal's release on its UI owner."""
                original_info(message)
                if message == "terminal released":
                    completed.set()

            source.info = info
            task = asyncio.create_task(
                controller.run_async(
                    lambda receive: loop.run_async(input_source=receive),
                    route_idle_commands=True,
                )
            )
            ready = asyncio.create_task(completed.wait())
            try:
                pipe.send_text("/terminal_probe\n")
                done, _ = await asyncio.wait(
                    {task, ready}, timeout=5, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    await task
                assert completed.is_set(), "Terminal command did not finish"
                pipe.send_text("/exit\n")
                await asyncio.wait_for(task, timeout=5)
            finally:
                controller.stop()
                ready.cancel()
                await asyncio.gather(task, ready, return_exceptions=True)

        asyncio.run(scenario())

    assert "terminal reserved" in output.getvalue()
    assert "terminal released" in output.getvalue()


def test_async_loop_dispatches_model_commands_without_blocking_the_ui(tmp_path):
    """Model catalog commands use the bounded worker and mailbox in the async application."""
    output = StringIO()
    backend_threads = []
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(session=prompt, console=Console(file=output), markdown=False)
        controller = SessionController(source)
        table_rendered = asyncio.Event()
        model_selected = asyncio.Event()
        original_table = source.table
        original_info = source.info

        def get_models():
            """Return a local model fixture for synchronous callers."""
            backend_threads.append(get_ident())
            return [ModelInfo(id="fixture")]

        async def get_models_async():
            """Record the owner thread and return a local asynchronous catalog fixture."""
            backend_threads.append(get_ident())
            return [ModelInfo(id="fixture")]

        async def get_context_window_async(_model):
            """Return local async context metadata for the selected fixture."""

        def observed_table(*args, **kwargs):
            """Render the catalog and signal that the mailbox command completed."""
            original_table(*args, **kwargs)
            table_rendered.set()

        def observed_info(message=""):
            """Render selection status and signal when model selection completes."""
            original_info(message)
            if message == "Using model: fixture":
                model_selected.set()

        source.table = observed_table
        source.info = observed_info
        loop = Loop.create_default(
            SimpleNamespace(
                default_model="fixture",
                get_context_window=lambda _model: None,
                get_models=get_models,
                get_context_window_async=get_context_window_async,
                get_models_async=get_models_async,
            ),
            interaction=controller.interaction,
            run_control=controller.control,
            working_directory=tmp_path,
            session_name_generator=LocalSessionNameGenerator(),
        )
        controller.bind_command_manager(loop.command_manager)

        async def scenario():
            """Run the normal async controller and submit the model catalog command."""
            owner_thread = get_ident()
            task = asyncio.create_task(
                controller.run_async(
                    lambda receive: loop.run_async(input_source=receive),
                    route_idle_commands=True,
                )
            )
            try:
                pipe.send_text("/models\n")
                await asyncio.wait_for(table_rendered.wait(), timeout=5)
                pipe.send_text("/model fixture\n")
                await asyncio.wait_for(model_selected.wait(), timeout=5)
                pipe.send_text("/exit\n")
                await asyncio.wait_for(task, timeout=5)
                return owner_thread
            finally:
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        owner_thread = asyncio.run(scenario())

    assert len(backend_threads) == 1
    assert backend_threads == [owner_thread]
    assert "fixture" in output.getvalue()
    assert "Using model: fixture" in output.getvalue()


def test_async_model_check_can_select_a_fallback_without_stopping_the_loop(tmp_path):
    """A model-check fallback prompt completes through the async command mailbox."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(session=prompt, console=Console(file=output), markdown=False)
        controller = SessionController(source)
        prompt_opened = asyncio.Event()
        model_selected = asyncio.Event()
        original_prompt_async = source.prompt_async
        original_info = source.info

        async def observed_prompt_async(*args, **kwargs):
            """Choose a replacement after model recovery asks the async UI."""
            if args and args[0] == "Select a replacement model:":
                prompt_opened.set()
                return "replacement"
            return await original_prompt_async(*args, **kwargs)

        def observed_info(message=""):
            """Render command output and observe successful fallback selection."""
            original_info(message)
            if message == "Using model: replacement":
                model_selected.set()

        async def get_models_async():
            """Return the local async catalog used by model-check recovery."""
            return [ModelInfo(id="replacement"), ModelInfo(id="other")]

        async def get_context_window_async(_model):
            """Return local async context metadata for the selected replacement."""
            return 4096

        source.prompt_async = observed_prompt_async
        source.info = observed_info
        loop = Loop.create_default(
            SimpleNamespace(
                default_model=None,
                get_context_window=lambda _model: 4096,
                get_models=lambda: [ModelInfo(id="replacement"), ModelInfo(id="other")],
                get_context_window_async=get_context_window_async,
                get_models_async=get_models_async,
            ),
            interaction=controller.interaction,
            run_control=controller.control,
            working_directory=tmp_path,
            session_name_generator=LocalSessionNameGenerator(),
        )
        controller.bind_command_manager(loop.command_manager)

        async def scenario():
            """Run model recovery through the production controller and stop cleanly."""
            task = asyncio.create_task(
                controller.run_async(
                    lambda receive: loop.run_async(input_source=receive),
                    route_idle_commands=True,
                )
            )
            try:
                pipe.send_text("/check\n")
                await asyncio.wait_for(prompt_opened.wait(), timeout=5)
                await asyncio.wait_for(model_selected.wait(), timeout=5)
                pipe.send_text("/exit\n")
                await asyncio.wait_for(task, timeout=5)
            finally:
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    assert "Using model: replacement" in output.getvalue()
    assert loop.model == "replacement"


@pytest.mark.parametrize("mode", ["queue", "interrupt", "steer", "default-steer"])
def test_native_input_queues_and_controls_active_model_execution(tmp_path, mode):
    """A real editor queues durable input and controls execution while its model is active."""
    output = StringIO()
    release = Event()
    requests = []
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(session=prompt, console=Console(file=output), markdown=False)
        controller = SessionController(source)

        async def scenario():
            """Submit queued input before releasing the first model operation."""
            working, queued, completed, interrupted = (asyncio.Event() for _ in range(4))
            original_info, original_metrics = source.info, source.run_metrics
            metrics = []

            def info(text=""):
                """Capture real output and expose deterministic user-visible boundaries."""
                original_info(text)
                if text == "model waiting":
                    working.set()
                elif text.startswith(("Queued input", "Steering accepted")):
                    queued.set()
                elif text.startswith("Interrupt requested"):
                    interrupted.set()

            def run_metrics(value):
                """Observe the second durable run completion before requesting shutdown."""
                original_metrics(value)
                metrics.append(value)
                if len(metrics) == (3 if mode == "steer" else 2):
                    completed.set()

            def get_response(**kwargs):
                """Simulate model I/O without any remote service or real network."""
                requests.append(kwargs["input"])
                if len(requests) == 1:
                    controller.interaction.info("model waiting")
                    release.wait()
                answer = f"answer {len(requests)}"
                return [
                    AnswerCompleted(text=answer),
                    ResponseCompleted(
                        answer=answer, items=(Message(role="assistant", content=answer),)
                    ),
                ]

            source.info, source.run_metrics = info, run_metrics
            backend = SimpleNamespace(
                default_model="local",
                get_context_window=lambda _model: None,
                prepare_context=list,
                get_response=get_response,
            )
            sessions = SessionManager(interaction=controller.interaction)
            loop = Loop.create_default(
                backend,
                interaction=controller.interaction,
                run_control=controller.control,
                session_manager=sessions,
                working_directory=tmp_path,
                session_name_generator=LocalSessionNameGenerator(),
            )
            controller.bind_session_manager(sessions)
            task = asyncio.create_task(
                controller._run_sync_callback_async(lambda receive: loop.run(input_source=receive))
            )
            try:
                pipe.send_text("first\n")
                await working.wait()
                pipe.send_text("second\n" if mode == "default-steer" else "/queue second\n")
                await asyncio.wait_for(queued.wait(), timeout=5)
                assert len(requests) == 1
                if mode in {"interrupt", "steer"}:
                    pipe.send_text("/steer correction\n" if mode == "steer" else "/interrupt\n")
                    await interrupted.wait()
                release.set()
                await completed.wait()
                pipe.send_text("/exit\n")
                await task
                return loop
            finally:
                release.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        loop = asyncio.run(scenario())

    if mode in {"queue", "default-steer"}:
        expected = ["first", "answer 1", "second", "answer 2"]
    elif mode == "interrupt":
        expected = ["first", "second", "answer 2"]
    else:
        expected = ["first", "correction", "answer 2", "second", "answer 3"]
    assert [item.content for item in loop.messages] == expected
    assert loop.session.recovery_state() is None
    assert loop.session.pending_inputs == ()
    assert "Queued input" in output.getvalue() or "Steering accepted" in output.getvalue()
    assert "answer 2" in output.getvalue()
    assert controller.running is False


def test_idle_config_command_opens_its_choice_prompt_immediately(tmp_path):
    """The config scope picker opens immediately and applies its selected scope."""
    output = StringIO()
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    with create_pipe_input() as pipe:
        session = PromptSession(input=pipe, output=DummyOutput())
        choices_session = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(
            session=session,
            choices_session=choices_session,
            console=Console(file=output),
        )
        controller = SessionController(source)
        choice_open = asyncio.Event()
        selection_reported = asyncio.Event()
        original_prompt_async = choices_session.prompt_async

        async def observed_prompt_async(*args, **kwargs):
            """Signal when the configuration question opens in the native choices editor."""
            if args and str(args[0]).strip() == "Choose configuration scope:":
                choice_open.set()
            return await original_prompt_async(*args, **kwargs)

        choices_session.prompt_async = observed_prompt_async

        provider = ConfigurationCommands(configuration, lambda _path, _settings: "applied now")
        manager = CommandManager(providers=(provider,), interaction=controller.interaction)
        controller.bind_command_manager(manager)
        original_info = source.info

        def info(message=""):
            """Render output and observe completion of the command selection."""
            original_info(message)
            if message.startswith("Updated loop.max_agent_turns"):
                selection_reported.set()

        source.info = info

        def conversation(receive):
            """Dispatch idle commands on the conversation owner."""
            while (value := receive(DummyCompleter())) is not False:
                manager.handle_user_command(value)

        async def scenario():
            """Select a choice directly after the command opens its prompt."""
            task = asyncio.create_task(
                controller._run_sync_callback_async(conversation, route_idle_commands=True)
            )
            try:
                pipe.send_text("/config set loop.max_agent_turns 100\n")
                await choice_open.wait()
                pipe.send_text("s\n")
                await selection_reported.wait()
                controller.stop()
                await task
            finally:
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    assert configuration.effective.loop.max_agent_turns == 100
    assert configuration.source_for("loop.max_agent_turns") == "session"
    assert "Input requested:" not in output.getvalue()
    assert "Save for current user" in output.getvalue()
    assert "This session only" in output.getvalue()


def test_active_config_command_opens_its_choice_prompt_immediately(tmp_path):
    """An active config command selects its scope in the native choices editor."""
    output = StringIO()
    release = Event()
    configuration = ConfigurationManager(tmp_path / "config.toml")
    configuration.load()
    with create_pipe_input() as pipe:
        session = PromptSession(input=pipe, output=DummyOutput())
        choices_session = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(
            session=session,
            choices_session=choices_session,
            console=Console(file=output),
        )
        controller = SessionController(source)
        choice_open = asyncio.Event()
        selection_reported = asyncio.Event()
        working = asyncio.Event()
        original_prompt_async = choices_session.prompt_async

        async def observed_prompt_async(*args, **kwargs):
            """Signal when the active command's native scope selector opens."""
            if args and str(args[0]).strip() == "Choose configuration scope:":
                choice_open.set()
            return await original_prompt_async(*args, **kwargs)

        choices_session.prompt_async = observed_prompt_async

        provider = ConfigurationCommands(configuration, lambda _path, _settings: "applied now")
        manager = CommandManager(providers=(provider,), interaction=controller.interaction)
        controller.bind_command_manager(manager)
        original_info = source.info

        def info(message=""):
            """Render output and observe agent activity and command completion."""
            original_info(message)
            if message == "agent working":
                working.set()
            elif message.startswith("Updated loop.max_agent_turns"):
                selection_reported.set()

        source.info = info

        def conversation(receive):
            """Keep the conversation in an active run while configuration is changed."""
            assert receive(DummyCompleter()) == "start"
            with controller.control.execution():
                controller.interaction.info("agent working")
                release.wait()
            receive(DummyCompleter())

        async def scenario():
            """Select a scope while the active agent keeps running."""
            task = asyncio.create_task(controller._run_sync_callback_async(conversation))
            try:
                pipe.send_text("start\n")
                await working.wait()
                pipe.send_text("/config set loop.max_agent_turns 100\n")
                await choice_open.wait()
                pipe.send_text("s\n")
                await selection_reported.wait()
                release.set()
                controller.stop()
                await task
            finally:
                release.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    assert configuration.effective.loop.max_agent_turns == 100
    assert configuration.source_for("loop.max_agent_turns") == "session"
    assert "Input requested:" not in output.getvalue()
    assert "Save for current user" in output.getvalue()
    assert "This session only" in output.getvalue()


def test_pending_approval_uses_native_confirmation_ui_and_preserves_the_draft():
    """A confirmation uses its native choices UI and restores the unsent chat draft."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        choices = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(
            session=prompt,
            choices_session=choices,
            console=Console(file=output),
        )
        controller = SessionController(source)
        messages, approvals = [], []
        decision_open = asyncio.Event()

        async def scenario():
            """Answer the editor approval prompt, then submit the restored chat draft."""
            processed = asyncio.Event()
            ready, drafted = asyncio.Event(), asyncio.Event()
            ask = Event()
            original_info = source.info

            def info(text=""):
                """Render output and observe the agent's question setup."""
                original_info(text)
                if text == "processed":
                    processed.set()
                elif text == "ready for request":
                    ready.set()

            def changed(buffer):
                """Observe a real unsent draft before the request is announced."""
                if buffer.text == "unfinished draft":
                    drafted.set()

            prompt.default_buffer.on_text_changed += changed
            source.info = info
            original_prompt_async = choices.prompt_async

            async def observed_prompt_async(*args, **kwargs):
                """Signal when the approval question opens in the native confirmation editor."""
                if args and str(args[0]).startswith("Write? [y/n]"):
                    decision_open.set()
                return await original_prompt_async(*args, **kwargs)

            choices.prompt_async = observed_prompt_async

            def conversation(receive):
                """Separate queued conversation text from an explicit fresh approval."""
                while (message := receive(DummyCompleter())) is not False:
                    messages.append(message)
                    if message == "start":
                        controller.interaction.info("ready for request")
                        ask.wait()
                        approvals.append(controller.interaction.confirm("Write?", default=False))
                    else:
                        controller.interaction.info("processed")

            task = asyncio.create_task(controller._run_sync_callback_async(conversation))
            try:
                pipe.send_text("start\n")
                await asyncio.wait_for(ready.wait(), timeout=5)
                pipe.send_text("unfinished draft")
                await asyncio.wait_for(drafted.wait(), timeout=5)
                ask.set()
                await asyncio.wait_for(decision_open.wait(), timeout=5)
                completer = choices.default_buffer.completer
                assert [
                    completion.text
                    for completion in completer.get_completions(Document("y"), CompleteEvent())
                ] == ["yes"]
                assert [
                    completion.text
                    for completion in completer.get_completions(Document("n"), CompleteEvent())
                ] == ["no"]
                assert "Input requested:" not in output.getvalue()
                assert "Press Enter or use /respond" not in output.getvalue()
                assert approvals == []
                assert prompt.default_buffer.text == ""
                pipe.send_text("n\n\n")
                await asyncio.wait_for(processed.wait(), timeout=5)
                pipe.send_text("/exit\n")
                await task
            finally:
                ask.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    assert messages == ["start", "unfinished draft"]
    assert approvals == [False]
