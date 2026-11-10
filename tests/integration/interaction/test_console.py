"""Exercise native prompt-toolkit async input with owned pipes and captured output."""

import asyncio
import os
from contextlib import aclosing
from io import StringIO

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from loop import (
    AnswerDelta,
    ConsoleInteraction,
    PermissionManager,
    Response,
    ResponseCompleted,
    SessionManager,
)
from loop.execution.sandbox import SandboxRequest

pytestmark = pytest.mark.integration


def test_sandbox_prompt_displays_details_only_when_requested(tmp_path):
    """Actual choice input can show and hide full scope before denying the command."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        interaction = ConsoleInteraction(
            session=prompt,
            choices_session=prompt,
            console=Console(file=output, width=200),
            markdown=False,
        )
        manager = PermissionManager(tmp_path, interaction=interaction)
        request = SandboxRequest.create(
            source="printf x > output",
            cwd=tmp_path,
            workspace=tmp_path,
            read_roots=(),
            write_roots=(tmp_path,),
            network=False,
            environment={"PATH": "/usr/bin:/bin"},
            policy_version="test",
            deadline=130.0,
            workspace_id="test",
        )

        async def scenario():
            """Toggle details twice, then deny through the same prompt session."""
            pipe.send_text("D\nD\nN\n")
            return await asyncio.wait_for(
                manager.authorize_sandboxed_command_async(request, tool_id="run_command"), 5
            )

        assert not asyncio.run(scenario())

    rendered = output.getvalue()
    assert "Show full scope" in rendered
    assert "Show reduced scope" in rendered
    assert "Full scope:" in rendered
    assert "working_directory" in rendered
    assert rendered.count("Command:") == 3
    assert rendered.count("Reason:") == 3


def test_host_approval_capability_uses_the_actual_input_descriptor():
    """An owned pseudo-terminal qualifies, while a prompt-toolkit pipe does not."""
    master, slave = os.openpty()
    try:
        with os.fdopen(slave, "r", closefd=False) as stream:
            terminal_input = create_input(stdin=stream)
            prompt = PromptSession(input=terminal_input, output=DummyOutput())
            interaction = ConsoleInteraction(session=prompt, choices_session=prompt)
            assert interaction.supports_fresh_host_approval
        with create_pipe_input() as pipe:
            prompt = PromptSession(input=pipe, output=DummyOutput())
            interaction = ConsoleInteraction(session=prompt, choices_session=prompt)
            assert not interaction.supports_fresh_host_approval
    finally:
        os.close(slave)
        os.close(master)


def test_async_response_presentation_keeps_native_input_available():
    """A real editable prompt stays active while a separate response stream is rendered."""
    output = StringIO()
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        interaction = ConsoleInteraction(
            session=prompt,
            choices_session=prompt,
            console=Console(file=output),
            markdown=False,
        )
        manager = SessionManager(interaction=interaction)

        async def scenario():
            """Wait for native prompt startup before rendering and submitting input."""
            started = asyncio.Event()
            prompt.app.pre_run_callables.append(started.set)
            input_task = asyncio.create_task(interaction.prompt_async())
            try:
                await started.wait()

                async def events():
                    """Supply a complete response independently of the input source."""
                    yield AnswerDelta(text="Working")
                    yield ResponseCompleted(answer="Working")

                async with aclosing(events()) as stream:
                    response = await manager.response_async(stream)
                assert not input_task.done()
                pipe.send_text("  next request  \n")
                return response, await input_task
            finally:
                input_task.cancel()
                await asyncio.gather(input_task, return_exceptions=True)

        response, message = asyncio.run(scenario())

    assert response == Response(answer="Working", reasoning="")
    assert message == "next request"
    assert output.getvalue() == "\nAnswer:\nWorking\n"
    assert manager.messages == []


def test_cancelled_native_async_prompt_releases_input_for_the_next_prompt():
    """Cancelling a real prompt cleans up its reader so the same session remains usable."""
    with create_pipe_input() as pipe:
        prompt = PromptSession(input=pipe, output=DummyOutput())
        interaction = ConsoleInteraction(session=prompt, choices_session=prompt)

        async def scenario():
            """Cancel at native startup, then reuse the owned input session."""
            started = asyncio.Event()
            prompt.app.pre_run_callables.append(started.set)
            task = asyncio.create_task(interaction.prompt_async())
            try:
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            pipe.send_text("next\n")
            return await interaction.prompt_async()

        assert asyncio.run(scenario()) == "next"
