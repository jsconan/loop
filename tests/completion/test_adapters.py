"""Tests for declarative interactive completion adapters."""

from enum import StrEnum
from typing import Annotated, Literal
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import (
    Command,
    CommandCompletion,
    CommandCompletionAdapter,
    CommandManager,
    CompletionAdapter,
    CompletionManager,
    CompletionMatch,
    CompletionValue,
    MarkerCompletionAdapter,
    ProjectPathCompletionAdapter,
    Skill,
)
from loop.commands.utils import get_command_arguments_model


def complete(completer: CompletionManager, text: str):
    """Return all completions produced for text with its cursor at the end."""
    return list(completer.get_completions(Document(text), Mock()))


def command_for(function, *, completion=None) -> Command:
    """Build a command declaration from a typed function."""
    return Command(
        function.__name__,
        function.__doc__ or "Command.",
        function,
        get_command_arguments_model(function, function.__name__),
        completion,
    )


def command_completer(*commands: Command, providers=None) -> CompletionManager:
    """Build a manager containing one command completion adapter."""
    return CompletionManager((CommandCompletionAdapter(lambda: commands, providers=providers),))


def test_command_names_match_fragments_and_replace_the_complete_slash_token():
    """Command completion matches middle fragments and preserves descriptive metadata."""

    def permissions() -> None:
        """Manage permissions."""

    results = complete(command_completer(command_for(permissions)), "/miss")

    assert [(item.text, item.start_position) for item in results] == [("/permissions", -5)]
    assert results[0].display_meta_text == "Manage permissions."


def test_schema_completion_infers_enum_literal_union_and_boolean_values():
    """Single-argument finite annotations produce command values without custom metadata."""

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
    assert [item.text for item in complete(completer, "/enabled ")] == ["true", "false"]


def test_schema_completion_accepts_annotated_dynamic_metadata():
    """Annotated fields can declare dynamic completion without changing command registration."""
    dynamic = CommandCompletion(provider=lambda: (CompletionValue("session-two", "session"),))

    def load(session_id: Annotated[str, dynamic]) -> None:
        """Load a session."""

    completer = command_completer(command_for(load))

    assert [item.text for item in complete(completer, "/load two")] == ["session-two"]


def test_schema_completion_ignores_open_and_multi_argument_schemas():
    """Unbounded strings and multi-field JSON commands do not invent argument values."""

    def text(value: str) -> None:
        """Accept text."""

    def pair(first: Literal["a"], second: Literal["b"]) -> None:
        """Accept a pair."""

    completer = command_completer(command_for(text), command_for(pair))

    assert complete(completer, "/text ") == []
    assert complete(completer, "/pair ") == []


def test_nested_command_completion_uses_dynamic_values_and_continuations():
    """A declarative grammar follows selected branches and runtime provider values."""
    final = CommandCompletion(values=(CompletionValue("last", "final value"),))
    dynamic = CommandCompletion(
        provider="tools",
        next=final,
    )
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


def test_file_mentions_match_path_fragments_rank_basenames_and_respect_ignores(tmp_path):
    """File mentions search visible relative paths and rank basename prefixes first."""
    source = tmp_path / "src" / "commands"
    source.mkdir(parents=True)
    (source / "command_manager.py").write_text("", encoding="utf-8")
    (tmp_path / "manager_notes.txt").write_text("", encoding="utf-8")
    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "manager_secret.py").write_text("", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    completer = CompletionManager((ProjectPathCompletionAdapter("@", tmp_path),))

    results = complete(completer, "review @manager")

    assert [item.text for item in results] == [
        "@manager_notes.txt",
        "@src/commands/command_manager.py",
    ]
    assert all(item.start_position == -8 for item in results)
    assert results[0].display_meta_text == "file"
    assert "ignored/manager_secret.py" not in [item.text for item in results]
    assert [item.text for item in complete(completer, "@src")][0] == "@src/"
    assert [item.text for item in complete(completer, "@commands")] == [
        "@src/commands/",
        "@src/commands/command_manager.py",
    ]


def test_file_mentions_include_directories_cache_the_snapshot_and_handle_missing_roots(tmp_path):
    """Directory candidates remain navigable and one prompt reuses its indexed snapshot."""
    folder = tmp_path / "nested"
    folder.mkdir()
    completer = CompletionManager((ProjectPathCompletionAdapter("@", tmp_path),), max_results=1)

    assert [item.text for item in complete(completer, "@nest")] == ["@nested/"]
    assert (
        complete(
            CompletionManager((ProjectPathCompletionAdapter("@", tmp_path / "missing"),)),
            "@anything",
        )
        == []
    )
    (tmp_path / "new.txt").write_text("", encoding="utf-8")
    assert complete(completer, "@new") == []


def test_skill_mentions_work_in_prose_and_require_a_token_boundary(tmp_path):
    """Skill mentions replace only an active bounded token and expose descriptions."""
    skill = Skill("coding", "Implement Python code.", tmp_path / "SKILL.md")
    completer = CompletionManager(
        (MarkerCompletionAdapter("$", lambda: (CompletionValue(skill.name, skill.description),)),)
    )

    results = complete(completer, "Please use ($din")

    assert [(item.text, item.start_position) for item in results] == [("$coding", -4)]
    assert results[0].display_meta_text == "Implement Python code."
    assert complete(completer, "price$din") == []
    assert complete(completer, "plain text") == []


def test_adapter_declarations_expose_markers_and_dynamic_keywords():
    """Adapters declare their activators without manager-specific domain knowledge."""

    class KeywordAdapter(CompletionAdapter):
        def match(self, document):
            """Remain inactive for this declaration-only capability."""
            return None

        def complete(self, match):
            """Return no values for this declaration-only capability."""
            return ()

    commands = [command_for(lambda: None)]
    command_adapter = CommandCompletionAdapter(lambda: commands)
    marker_adapter = MarkerCompletionAdapter("$", lambda: ())

    assert KeywordAdapter().front_markers == ()
    assert KeywordAdapter().keywords == ()
    assert marker_adapter.front_markers == ("$",)
    assert command_adapter.front_markers == ("/",)
    assert command_adapter.keywords == ("<lambda>",)
    assert command_adapter.complete(CompletionMatch("", "")) == ()
    assert complete(CompletionManager((marker_adapter, command_adapter)), "plain") == []


@pytest.mark.parametrize("marker", ["ab", "a", " "])
@pytest.mark.parametrize("adapter", [MarkerCompletionAdapter, ProjectPathCompletionAdapter])
def test_marker_adapters_reject_invalid_markers(adapter, marker, tmp_path):
    """Marker capabilities reject ambiguous alphanumeric activators."""
    argument = (lambda: ()) if adapter is MarkerCompletionAdapter else tmp_path

    with pytest.raises(ValueError, match="one non-alphanumeric"):
        adapter(marker, argument)


@pytest.mark.parametrize("marker", ["ab", "a", " "])
def test_command_adapter_rejects_invalid_markers(marker):
    """Command capabilities reject ambiguous alphanumeric activators."""
    with pytest.raises(ValueError, match="one non-alphanumeric"):
        CommandCompletionAdapter(lambda: (), marker=marker)


def test_registered_permissions_grammar_completes_modes_decisions_tools_and_capabilities():
    """The built-in permissions mini-language exposes every known positional domain."""
    manager = CommandManager()
    completer = CompletionManager(
        (
            CommandCompletionAdapter(
                lambda: manager.commands,
                providers={"tools": lambda: (CompletionValue("read_text_file", "tool"),)},
            ),
        )
    )

    assert [item.text for item in complete(completer, "/permissions mode work")] == [
        "workspace_write"
    ]
    assert [item.text for item in complete(completer, "/permissions add al")] == ["allow"]
    assert [item.text for item in complete(completer, "/permissions add allow read")] == [
        "read_text_file"
    ]
    capabilities = complete(completer, "/permissions add allow read_text_file filesystem.w")
    assert [item.text for item in capabilities] == ["filesystem.write"]
    assert [
        item.text
        for item in complete(completer, "/permissions add allow read_text_file filesystem.write ")
    ] == ["*"]
