"""Tests for command completion."""

from enum import StrEnum
from typing import Annotated, Literal
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import (
    Command,
    CommandCompletion,
    CommandCompletionAdapter,
    CompletionManager,
    CompletionMatch,
    CompletionProviderRegistration,
    CompletionValue,
    SchemaCompletionProviderRegistration,
)
from loop.commands.utils import get_command_arguments_model


def complete(completer: CompletionManager, text: str):
    """Return all completions produced for text with its cursor at the end."""
    return list(completer.get_completions(Document(text), Mock()))


def command_for(function, *, completion=None) -> Command:
    """Build a command declaration from a typed function."""
    return Command(
        function,
        name=function.__name__,
        description=function.__doc__ or "Command.",
        arguments_model=get_command_arguments_model(function, function.__name__),
        completion=completion,
    )


def command_completer(
    *commands: Command, providers=None, schema_providers=None
) -> CompletionManager:
    """Build a manager containing one command completion adapter."""

    class Provider:
        """Expose test-local named completion sources."""

        def get_completion_providers(self):
            """Return configured value and schema sources."""
            return (
                *(
                    CompletionProviderRegistration(name, provider)
                    for name, provider in (providers or {}).items()
                ),
                *(
                    SchemaCompletionProviderRegistration(name, provider)
                    for name, provider in (schema_providers or {}).items()
                ),
            )

    return CompletionManager((CommandCompletionAdapter(lambda: commands, providers=(Provider(),)),))


def test_command_names_match_fragments_and_replace_the_complete_slash_token():
    """Command completion matches middle fragments and preserves descriptive metadata."""

    def permissions() -> None:
        """Manage permissions."""

    results = complete(command_completer(command_for(permissions)), "/miss")

    assert [(item.text, item.start_position) for item in results] == [("/permissions", -5)]
    assert results[0].display_meta_text == "Manage permissions."


def test_schema_completion_infers_enum_literal_union_and_boolean_values():
    """Finite annotations produce alphabetized command values without custom metadata."""

    class Mode(StrEnum):
        READ_ONLY = "read_only"
        WRITE = "workspace_write"

    def mode(value: Mode) -> None:
        """Select a mode."""

    def level(value: Literal["low", "high"] | None) -> None:
        """Select a level."""

    def enabled(value: bool) -> None:
        """Toggle a feature."""

    completer = command_completer(command_for(mode), command_for(level), command_for(enabled))

    assert [item.text for item in complete(completer, "/mode work")] == ["workspace_write"]
    assert [item.text for item in complete(completer, "/level h")] == ["high"]
    assert [item.text for item in complete(completer, "/enabled ")] == ["false", "true", "value="]


def test_schema_completion_accepts_annotated_dynamic_metadata():
    """Annotated fields provide dynamic values for positional and named completion."""
    dynamic = CommandCompletion(provider=lambda: (CompletionValue("session-two", "session"),))

    def load(session_id: Annotated[str, dynamic]) -> None:
        """Load a session."""

    completer = command_completer(command_for(load))

    assert [item.text for item in complete(completer, "/load two")] == ["session-two"]
    assert [item.text for item in complete(completer, "/load session_id=two")] == [
        "session_id=session-two"
    ]


def test_schema_completion_offers_named_fields_and_tracks_positional_binding():
    """Schema completion offers remaining fields and values after mixed bindings."""

    def text(value: str) -> None:
        """Accept text."""

    def pair(first: Literal["a"], second: Literal["b"]) -> None:
        """Accept a pair."""

    completer = command_completer(command_for(text), command_for(pair))

    assert [item.text for item in complete(completer, "/text ")] == ["value="]
    assert [item.text for item in complete(completer, "/pair ")] == ["a", "first=", "second="]
    assert [item.text for item in complete(completer, "/pair a ")] == ["b", "second="]
    assert [item.text for item in complete(completer, "/pair second=b ")] == ["a", "first="]
    assert [item.text for item in complete(completer, "/pair second=")] == ["second=b"]
    assert complete(completer, "/pair unknown=a ") == []
    assert complete(completer, "/pair first=a first=") == []
    assert complete(completer, "/pair a b ") == []
    assert complete(completer, '/pair "') == []
    assert complete(completer, "/pair a b extra ") == []
    assert complete(completer, "/text value=") == []


def test_nested_command_completion_uses_dynamic_values_and_continuations():
    """A declarative grammar follows selected branches and runtime provider values."""
    final = CommandCompletion(values=(CompletionValue("last", "final value"),))
    dynamic = CommandCompletion(provider="tools", next=final)
    grammar = CommandCompletion(values=(CompletionValue("add"),), children={"add": dynamic})

    def configure(value: str) -> None:
        """Configure a value."""

    completer = command_completer(
        command_for(configure, completion=grammar),
        providers={"tools": lambda: (CompletionValue("read_file", "tool"),)},
    )

    assert [item.text for item in complete(completer, "/configure add read")] == ["read_file"]
    results = complete(completer, "/configure add read_file l")
    assert [item.text for item in results] == ["last"]
    assert results[0].display_meta_text == "final value"


def test_nested_command_completion_switches_to_a_selected_dynamic_schema():
    """A grammar alphabetizes fields from the model selected by its leading token."""

    def invoke(tool: str) -> None:
        """Invoke a tool."""

    def selected(count: int, mode: Literal["fast", "safe"] = "safe") -> None:
        """Define selected tool arguments."""

    grammar = CommandCompletion(
        provider="tools", next=CommandCompletion(schema_provider="tool_arguments")
    )
    completer = command_completer(
        command_for(invoke, completion=grammar),
        providers={"tools": lambda: (CompletionValue("selected"),)},
        schema_providers={
            "tool_arguments": lambda tokens: (
                get_command_arguments_model(selected, "selected")
                if tokens == ("selected",)
                else None
            )
        },
    )
    assert [item.text for item in complete(completer, "/invoke selected ")] == ["count=", "mode="]
    assert [item.text for item in complete(completer, "/invoke selected 2 mode=f")] == [
        "mode=fast",
        "mode=safe",
    ]
    assert complete(completer, "/invoke missing ") == []

    direct = command_completer(
        command_for(
            invoke,
            completion=CommandCompletion(
                provider="tools",
                next=CommandCompletion(
                    schema_provider=lambda _tokens: get_command_arguments_model(
                        selected, "selected"
                    )
                ),
            ),
        ),
        providers={"tools": lambda: (CompletionValue("selected"),)},
    )
    assert complete(direct, "/invoke selected ")


def test_command_adapter_validates_named_provider_registrations():
    """Providers are duck typed while named sources reject invalid registrations."""

    class Provider:
        """Expose supplied completion registrations."""

        def __init__(self, *registrations) -> None:
            self.registrations = registrations

        def get_completion_providers(self):
            """Return supplied completion registrations."""
            return self.registrations

    adapter = CommandCompletionAdapter(lambda: (), providers=(object(),))
    with pytest.raises(ValueError, match="must not be empty"):
        adapter.register(CompletionProviderRegistration("", lambda: ()))
    registration = CompletionProviderRegistration("values", lambda: ())
    adapter.register_provider(Provider(registration))
    with pytest.raises(ValueError, match="already registered"):
        adapter.register_providers((Provider(registration),))
    schema = SchemaCompletionProviderRegistration("values", lambda _tokens: None)
    adapter.register(schema)
    with pytest.raises(ValueError, match="already registered"):
        adapter.register(schema)


def test_invalid_command_paths_quotes_and_provider_failures_return_no_values():
    """Incomplete grammar paths and unavailable dynamic sources never disrupt input."""

    def unavailable():
        """Represent a temporarily unavailable dynamic source."""
        raise RuntimeError("unavailable")

    grammar = CommandCompletion(
        values=(CompletionValue("known"),),
        children={"known": CommandCompletion(provider=unavailable)},
    )

    def choose(value: str) -> None:
        """Choose a value."""

    completer = command_completer(command_for(choose, completion=grammar))
    assert complete(completer, "/missing ") == []
    assert complete(completer, '/choose "') == []
    assert complete(completer, "/choose unknown ") == []
    assert complete(completer, "/choose known ") == []


def test_command_adapter_declarations_expose_markers_and_dynamic_keywords():
    """Command adapters declare their marker and current command names."""
    commands = [command_for(lambda: None)]
    adapter = CommandCompletionAdapter(lambda: commands)

    assert adapter.front_markers == ("/",)
    assert adapter.keywords == ("<lambda>",)
    assert adapter.complete(CompletionMatch("", "")) == ()


@pytest.mark.parametrize("marker", ["ab", "a", " "])
def test_command_adapter_rejects_invalid_markers(marker):
    """Command capabilities reject ambiguous alphanumeric activators."""
    with pytest.raises(ValueError, match="one non-alphanumeric"):
        CommandCompletionAdapter(lambda: (), marker=marker)
