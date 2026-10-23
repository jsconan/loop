"""Tests for individual schema-backed command declarations."""

import asyncio
from threading import Event, get_ident
from typing import Annotated
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from loop import (
    Command,
    CommandArgumentError,
    CommandCompletion,
    CommandContext,
    CommandManager,
    CommandRegistration,
    CompletionValue,
    Interaction,
    ProblemException,
    command,
)
from loop.commands.models import CommandRemainder
from loop.commands.utils import get_command_arguments_model


def make_command(function, name: str = "select") -> Command:
    """Build a command schema from a test function."""
    return Command(
        function,
        name=name,
        description="Select.",
        arguments_model=get_command_arguments_model(function, name),
    )


def test_command_binds_positional_and_named_arguments_and_decodes_each_value():
    """Commands bind mixed tokens and independently decode values for annotated fields."""
    selected = []

    def select(
        count: int,
        label: str,
        enabled: bool,
        numbers: list[int],
    ) -> None:
        selected.append((count, label, enabled, numbers))

    command = make_command(select)
    assert command.call("3 enabled=true numbers='[1, 2]' label=true") is None
    assert selected == [(3, "true", True, [1, 2])]


def test_call_async_keeps_the_owner_loop_responsive_for_context_free_sync_commands():
    """Direct async dispatch runs context-free synchronous callbacks in a worker."""
    started = Event()
    release = Event()
    worker_threads = []

    def select() -> str:
        """Wait for the test owner before returning a value."""
        worker_threads.append(get_ident())
        started.set()
        release.wait(timeout=1)
        return "selected"

    async def invoke():
        owner_thread = get_ident()
        pending = asyncio.create_task(make_command(select).call_async(""))
        assert await asyncio.wait_for(asyncio.to_thread(started.wait, 1), timeout=2)
        await asyncio.sleep(0)
        assert not pending.done()
        release.set()
        assert await pending == "selected"
        assert worker_threads != [owner_thread]

    asyncio.run(invoke())


def test_direct_async_context_command_requires_the_manager_relay():
    """A direct async call cannot move a live command context to a worker."""

    def contextual(context: CommandContext) -> None:
        """Return after receiving the command context."""
        del context

    command = make_command(contextual)
    context = CommandContext("select", Mock(spec=Interaction))

    with pytest.raises(ProblemException) as error:
        asyncio.run(command.call_async("", context))

    assert error.value.problem.code == "command.context_relay_unavailable"


def test_direct_async_command_rejects_a_future_owned_by_another_loop():
    """A synchronous callback cannot return a future from a foreign event loop."""
    foreign_loop = asyncio.new_event_loop()
    foreign_future = foreign_loop.create_future()
    foreign_future.set_result("unowned")

    def return_foreign_future():
        """Return a completed future attached to another loop."""
        return foreign_future

    try:
        with pytest.raises(ProblemException) as error:
            asyncio.run(make_command(return_foreign_future).call_async(""))
    finally:
        foreign_loop.close()

    assert error.value.problem.code == "command.loop_mismatch"


def test_command_preserves_raw_strings_and_reports_annotation_errors():
    """Non-JSON text remains a string and invalid decoded structures fail model validation."""

    def select(label: str, numbers: list[int]) -> None:
        pass

    command = make_command(select)
    with pytest.raises(ValidationError):
        command.call("plain null")


def test_command_calls_parameterless_function_and_injects_context_when_declared():
    """Empty payloads work and context is injected only as the leading declared parameter."""
    calls = []

    def plain() -> None:
        calls.append("plain")

    def contextual(context: CommandContext) -> None:
        calls.append(context.name)

    interaction = Mock(spec=Interaction)
    assert make_command(plain).call("") is None
    assert make_command(contextual).call("", CommandContext("select", interaction)) is None
    assert calls == ["plain", "select"]
    with pytest.raises(ValueError, match="requires a CommandContext"):
        make_command(contextual).call("")


def test_command_collects_all_tokens_in_a_declared_remainder():
    """A final remainder field receives positional and name-like tokens unchanged."""
    calls = []

    def collect(arguments: Annotated[tuple[str, ...], CommandRemainder()] = ()) -> None:
        calls.append(arguments)

    make_command(collect).call("first name=value")

    assert calls == [("first", "name=value")]


def test_command_declaration_can_be_attached_to_and_retrieved_from_a_function():
    """Command declarations can be stored on a function and retrieved unchanged."""

    def select() -> None:
        """Select a value."""

    declaration = Command(select)

    Command.set_declaration(select, declaration)

    assert Command.get_declaration(select) is declaration


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("unknown=1", "Unknown parameter 'unknown'"),
        ("first=1 first=2", "supplied more than once"),
        ("1 2 3", "Too many positional arguments"),
        ('{"first": 1, "second": 2}', "Too many positional arguments"),
        ('"unterminated', "Could not parse arguments"),
    ],
)
def test_command_rejects_invalid_argument_syntax_and_binding(arguments, message):
    """Commands reject unknown, duplicate, excess, and malformed argument input."""

    def pair(first: int, second: int) -> None:
        pass

    with pytest.raises(CommandArgumentError, match=message):
        make_command(pair).call(arguments)


def test_command_decorator_declares_passive_metadata_for_registration():
    """Passive declarations preserve identity while managers resolve their metadata."""
    completion = CommandCompletion(values=(CompletionValue("one"),))

    @command(name="choose", description="Choose explicitly.", completion=completion)
    def select(value: str) -> None:
        """Select a value."""

    manager = CommandManager((select,))

    assert manager.commands[-1].name == "choose"
    assert manager.commands[-1].description == "Choose explicitly."
    assert manager.commands[-1].completion is completion
    with pytest.raises(ValueError, match="already declared"):
        command(select)


def test_passive_command_resolves_to_an_independent_registered_copy():
    """A passive command resolves metadata and validation without mutating its declaration."""

    def select(value: str) -> None:
        """Select a value."""

    declaration = Command(select)
    registered = declaration.registered(name="choose")

    assert declaration.arguments_model is None
    assert registered.name == "choose"
    assert registered.description == "Select a value."
    assert registered.arguments_model is not None
    with pytest.raises(ValueError, match="must be registered"):
        declaration.call("value")


def test_registration_metadata_overrides_passive_declarations():
    """Container registrations can specialize passive command declarations."""

    @command(name="declared")
    def select() -> None:
        """Select a value."""

    manager = CommandManager(
        (CommandRegistration(select, name="local", description="Local selection."),)
    )

    assert manager.commands[-1].name == "local"
    assert manager.commands[-1].description == "Local selection."
