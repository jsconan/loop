"""Tests for terminal-backed user interaction."""

from unittest.mock import Mock

import pytest
from rich.prompt import Confirm

from loop import Command
from loop.commands.utils import get_command_arguments_model
from loop.interaction import ConsoleInteraction


def test_input_reads_a_trimmed_message_with_a_custom_prompt():
    """Input forwards its prompt message and strips the terminal response."""
    session = Mock()
    session.prompt.return_value = "  answer  \n"

    assert ConsoleInteraction(session=session).input(message="Question: ") == "answer"
    session.prompt.assert_called_once()
    assert session.prompt.call_args.args == ("Question: ",)


def test_input_offers_command_names_and_descriptions_for_completion():
    """Terminal input derives slash completion metadata from available commands."""
    session = Mock()
    session.prompt.return_value = "answer"

    def function() -> None:
        pass

    commands = (
        Command("help", "Show help.", function, get_command_arguments_model(function, "help")),
    )

    ConsoleInteraction(session=session).input(commands=commands)

    completer = session.prompt.call_args.kwargs["completer"]
    completions = list(completer.get_completions(Mock(text_before_cursor="/h"), Mock()))
    assert [completion.text for completion in completions] == ["/help"]
    assert completions[0].display_meta_text == "Show help."


def test_input_reprompts_for_blank_input(capsys):
    """Input warns and prompts again until the terminal provides a message."""
    session = Mock()
    session.prompt.side_effect = ["   ", "answer"]

    assert ConsoleInteraction(session=session).input() == "answer"
    assert capsys.readouterr().out == "Warning: Please enter a message!\n"
    assert session.prompt.call_count == 2


@pytest.mark.parametrize("command", ["exit", "QUIT", " Bye ", "q"])
def test_input_returns_false_for_exit_commands(command):
    """Input recognizes supported exit commands without case or surrounding whitespace."""
    session = Mock()
    session.prompt.return_value = command

    assert ConsoleInteraction(session=session).input() is False


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_input_returns_false_when_the_prompt_is_interrupted(error):
    """Input treats terminal interruption and end-of-file as exit requests."""
    session = Mock()
    session.prompt.side_effect = error

    assert ConsoleInteraction(session=session).input() is False


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

    assert capsys.readouterr().out == '\n[TOOL CALL]: search(query="term")\n'


def test_tool_call_truncates_each_long_argument_value_in_the_middle(capsys):
    """Tool calls retain the beginning and end of long argument values."""
    ConsoleInteraction().tool_call("write", '{"content":"0123456789abcdefghijklmnop"}')

    assert capsys.readouterr().out == '\n[TOOL CALL]: write(content="0123456789…hijklmnop")\n'


@pytest.mark.parametrize(
    ("model", "context_tokens", "context_window", "expected"),
    [
        (
            None,
            None,
            262144,
            "Model: ? · Context: ? / 262,144 tokens\n",
        ),
        (
            "served-model",
            18432,
            262144,
            "Model: served-model · Context: 18,432 / 262,144 tokens\n",
        ),
        (
            "local-model",
            12,
            None,
            "Model: local-model · Context: 12 / ? tokens\n",
        ),
    ],
)
def test_token_usage_has_a_console_presentation(
    capsys, model, context_tokens, context_window, expected
):
    """Token usage presents the current model and context against maximum context."""
    ConsoleInteraction().token_usage(model, context_tokens, context_window)
    assert capsys.readouterr().out == expected


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


def test_response_scope_terminates_streamed_output(capsys):
    """A response scope terminates streamed output before subsequent presentation."""
    interaction = ConsoleInteraction()
    with interaction.response():
        interaction.answer_delta("partial", start=True)
    interaction.conversation_ended()

    assert capsys.readouterr().out == "\nAnswer:partial\n\nConversation ended.\n"


def test_response_scope_does_not_extend_complete_output(capsys):
    """A response scope adds no separator when output already terminates itself."""
    interaction = ConsoleInteraction()
    with interaction.response():
        interaction.answer("complete")

    assert capsys.readouterr().out == "\nAnswer:complete\n"


def test_response_scope_terminates_streamed_output_after_an_error(capsys):
    """A response scope terminates streamed output when presentation fails."""
    interaction = ConsoleInteraction()

    with pytest.raises(RuntimeError, match="failed"):
        with interaction.response():
            interaction.answer_delta("partial", start=True)
            raise RuntimeError("failed")

    assert capsys.readouterr().out == "\nAnswer:partial\n"


@pytest.mark.parametrize(("default", "expected"), [(False, True), (True, False)])
def test_confirm_uses_rich_with_the_configured_default(monkeypatch, default, expected):
    """Confirmation delegates its question and default to Rich."""
    ask = Mock(return_value=expected)
    console = Mock()
    monkeypatch.setattr(Confirm, "ask", ask)
    interaction = ConsoleInteraction(console=console)

    assert interaction.confirm("Continue?", default=default) is expected
    ask.assert_called_once_with("Continue?", default=default, console=console)
