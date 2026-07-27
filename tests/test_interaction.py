"""Tests for terminal-backed user interaction."""

import pytest

from loop.interaction import ConsoleInteraction


def test_input_reads_from_the_terminal(monkeypatch):
    """Input owns its prompt and strips the terminal input."""
    def terminal_input(message):
        assert message == "\nYou: "
        return "  answer  \n"

    monkeypatch.setattr("builtins.input", terminal_input)

    assert ConsoleInteraction().input() == "answer"


@pytest.mark.parametrize(
    ("method", "heading"),
    [("reasoning", "[THOUGHT PROCESS]"), ("answer", "[ANSWER]")],
)
def test_model_output_has_a_console_presentation(capsys, method, heading):
    """The console owns presentation of complete model output."""
    getattr(ConsoleInteraction(), method)("complete")

    assert capsys.readouterr().out == f"\n{heading}:\ncomplete\n"


@pytest.mark.parametrize(
    ("method", "heading"),
    [("reasoning_delta", "[THOUGHT PROCESS]"), ("answer_delta", "[ANSWER]")],
)
def test_model_output_formats_stream_starts(capsys, method, heading):
    """The console owns presentation of model-output stream boundaries."""
    getattr(ConsoleInteraction(), method)("partial", start=True)

    assert capsys.readouterr().out == f"\n{heading}:\npartial"


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


def test_conversation_events_have_console_presentations(capsys):
    """The console owns validation, progress, response, and termination formatting."""
    interaction = ConsoleInteraction()
    interaction.invalid_input()
    interaction.thinking()
    interaction.response_finished()
    interaction.conversation_ended()

    assert capsys.readouterr().out == (
        "Warning: Please enter a message!\n\nThinking...\n\n\nConversation ended.\n"
    )


@pytest.mark.parametrize(
    ("answer", "default", "expected", "suffix"),
    [
        ("y", False, True, "[y/N]"),
        (" YES ", False, True, "[y/N]"),
        ("n", True, False, "[Y/n]"),
        ("anything", True, False, "[Y/n]"),
        ("", False, False, "[y/N]"),
        ("   ", True, True, "[Y/n]"),
    ],
)
def test_confirm_parses_answers_and_empty_defaults(
    monkeypatch, answer, default, expected, suffix
):
    """Confirmation accepts yes and applies its configured empty-answer default."""
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or answer)

    assert ConsoleInteraction().confirm("Continue?", default=default) is expected
    assert prompts == [f"Continue? {suffix}: "]
