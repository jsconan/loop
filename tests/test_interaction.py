"""Tests for terminal-backed user interaction."""

from unittest.mock import Mock

import pytest
from rich.prompt import Confirm

from loop.interaction import ConsoleInteraction


def test_input_reads_from_the_terminal():
    """Input owns its prompt and strips the terminal input."""
    session = Mock()
    session.prompt.return_value = "  answer  \n"

    assert ConsoleInteraction(session=session).input() == "answer"
    session.prompt.assert_called_once_with("\nYou: ")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("reasoning", "\nThinking...\n\ncomplete\n"),
        ("answer", "\nAnswer:complete\n"),
    ],
)
def test_model_output_has_a_console_presentation(capsys, method, expected):
    """The console owns presentation of complete model output."""
    getattr(ConsoleInteraction(), method)("complete")

    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("reasoning_delta", "\nThinking...\n\npartial"),
        ("answer_delta", "\nAnswer:partial"),
    ],
)
def test_model_output_formats_stream_starts(capsys, method, expected):
    """The console owns presentation of model-output stream boundaries."""
    getattr(ConsoleInteraction(), method)("partial", start=True)

    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("method", ["reasoning_delta", "answer_delta"])
def test_model_output_continues_streams_without_repeating_headings(capsys, method):
    """Continued deltas are appended without another heading."""
    getattr(ConsoleInteraction(), method)("partial")

    assert capsys.readouterr().out == "partial"


@pytest.mark.parametrize(
    ("method", "message", "expected"),
    [
        ("error", "failed", "Error: failed\n"),
        ("warning", "careful", "Warning: careful\n"),
        ("info", "working", "working\n"),
    ],
)
def test_classified_messages_have_console_presentations(capsys, method, message, expected):
    """Classified messages retain recognizable terminal presentations."""
    getattr(ConsoleInteraction(), method)(message)

    assert capsys.readouterr().out == expected


def test_info_supports_a_blank_separator(capsys):
    """Neutral information can emit a blank line between streamed sections."""
    ConsoleInteraction().info()

    assert capsys.readouterr().out == "\n"


def test_tool_call_displays_its_name_and_arguments(capsys):
    """Tool calls receive a dedicated terminal presentation."""
    ConsoleInteraction().tool_call("search", '{"query":"term"}')

    assert capsys.readouterr().out == '\n[TOOL CALL]: search({"query":"term"})\n'


def test_debug_formats_raw_values(capsys):
    """The console owns type labeling and pretty-printing of diagnostic values."""
    ConsoleInteraction().debug({"details": [1, 2]})

    output = capsys.readouterr().out
    assert "[DEBUG EVENT]: <class 'dict'>" in output
    assert "{'details': [1, 2]}" in output


def test_output_styles_prioritize_answers_over_diagnostics():
    """Answers are emphasized while reasoning, tool calls, and debug output are dimmed."""
    console = Mock()
    interaction = ConsoleInteraction(console=console)

    interaction.answer("result")
    interaction.reasoning("thought")
    interaction.tool_call("search", "{}")
    interaction.debug({"detail": True})

    styles = [call.kwargs.get("style") for call in console.print.call_args_list]
    assert styles[:2] == ["bold bright_green", "bold"]
    assert styles[2:] == ["dim cyan", "dim", "dim magenta", "dim blue", "dim"]


def test_conversation_events_have_console_presentations(capsys):
    """The console owns validation, progress, response, and termination formatting."""
    interaction = ConsoleInteraction()
    interaction.invalid_input()
    interaction.response_finished()
    interaction.conversation_ended()

    assert capsys.readouterr().out == "Warning: Please enter a message!\n\n\nConversation ended.\n"


@pytest.mark.parametrize(("default", "expected"), [(False, True), (True, False)])
def test_confirm_uses_rich_with_the_configured_default(monkeypatch, default, expected):
    """Confirmation delegates its question and default to Rich."""
    ask = Mock(return_value=expected)
    console = Mock()
    monkeypatch.setattr(Confirm, "ask", ask)
    interaction = ConsoleInteraction(console=console)

    assert interaction.confirm("Continue?", default=default) is expected
    ask.assert_called_once_with("Continue?", default=default, console=console)
