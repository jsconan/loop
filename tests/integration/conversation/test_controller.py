"""Exercise mailbox execution with native prompt input and captured response output."""

import asyncio
from io import StringIO
from threading import Event
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import DummyCompleter
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from loop import (
    AnswerCompleted,
    ConsoleInteraction,
    Loop,
    Message,
    ResponseCompleted,
    SessionController,
)
from loop.session import LocalSessionNameGenerator

pytestmark = pytest.mark.integration


def test_native_input_queues_a_second_message_during_model_execution(tmp_path):
    """A real editor remains usable while the conversation owner waits for its model."""
    output = StringIO()
    release = Event()
    requests = []
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(session=prompt, console=Console(file=output), markdown=False)
        controller = SessionController(source)

        async def scenario():
            """Submit follow-up input before releasing the first model operation."""
            working, queued, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
            original_info, original_metrics = source.info, source.run_metrics
            metrics = []

            def info(text=""):
                """Capture real output and expose deterministic user-visible boundaries."""
                original_info(text)
                if text == "model waiting":
                    working.set()
                elif text.startswith("Queued input"):
                    queued.set()

            def run_metrics(value):
                """Observe the second durable run completion before requesting shutdown."""
                original_metrics(value)
                metrics.append(value)
                if len(metrics) == 2:
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
            loop = Loop.create_default(
                backend,
                interaction=controller.interaction,
                run_control=controller.control,
                working_directory=tmp_path,
                session_name_generator=LocalSessionNameGenerator(),
            )
            task = asyncio.create_task(
                controller.run(lambda receive: loop.run(input_source=receive))
            )
            try:
                pipe.send_text("first\n")
                await working.wait()
                pipe.send_text("second\n")
                await queued.wait()
                assert len(requests) == 1
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

    assert [item.content for item in loop.messages] == ["first", "answer 1", "second", "answer 2"]
    assert "Queued input" in output.getvalue()
    assert "answer 2" in output.getvalue()
    assert controller.running is False


def test_native_approval_preserves_the_editor_draft_and_queued_chat():
    """An approval request does not cancel the editor or consume draft text as a decision."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        choices = PromptSession(input=pipe, output=DummyOutput())
        source = ConsoleInteraction(
            session=prompt, choices_session=choices, console=Console(file=output)
        )
        controller = SessionController(source)
        messages, approvals = [], []

        async def scenario():
            """Complete a preserved draft, then explicitly open and decline approval."""
            pending, queued, decision_open, processed = (
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
                asyncio.Event(),
            )
            ready, drafted = asyncio.Event(), asyncio.Event()
            ask = Event()
            original_info = source.info

            def info(text=""):
                """Render output and observe request transitions on the actual UI."""
                original_info(text)
                if text.startswith("Input requested"):
                    pending.set()
                elif text.startswith("Queued input"):
                    queued.set()
                elif text == "processed":
                    processed.set()
                elif text == "ready for request":
                    ready.set()

            def changed(buffer):
                """Observe a real unsent draft before the request is announced."""
                if buffer.text == "unfinished draft":
                    drafted.set()

            prompt.default_buffer.on_text_changed += changed
            source.info = info
            choices.app.pre_run_callables.append(decision_open.set)

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

            task = asyncio.create_task(controller.run(conversation))
            try:
                pipe.send_text("start\n")
                await ready.wait()
                pipe.send_text("unfinished draft")
                await drafted.wait()
                ask.set()
                await pending.wait()
                pipe.send_text("\n")
                await queued.wait()
                assert approvals == []
                pipe.send_text("/respond\n")
                await decision_open.wait()
                pipe.send_text("n\n")
                await processed.wait()
                pipe.send_text("/exit\n")
                await task
            finally:
                ask.set()
                controller.stop()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(scenario())

    assert messages == ["start", "unfinished draft"]
    assert approvals == [False]
