"""Tests for terminal-backed user interaction."""

import pytest

from loop.interaction import ConsoleInteraction


def test_prompt_reads_from_the_terminal(monkeypatch):
    """Prompt forwards its message and returns terminal input."""
    def terminal_input(message):
        return f"answer to {message}"

    monkeypatch.setattr("builtins.input", terminal_input)

    assert ConsoleInteraction().prompt("Question? ") == "answer to Question? "


def test_write_forwards_print_options(capsys):
    """Write supports the end and flush options required by streaming output."""
    ConsoleInteraction().write("partial", end="", flush=True)

    assert capsys.readouterr().out == "partial"


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
