"""Tests for terminal-backed user interaction."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.completion import DummyCompleter
from rich.console import Console
from rich.prompt import Confirm

from loop.interaction import ConsoleInteraction


def test_input_reads_a_trimmed_message_with_a_custom_prompt():
    """Input forwards its prompt message and strips the terminal response."""
    session = Mock()
    session.prompt.return_value = "  answer  \n"

    assert ConsoleInteraction(session=session).input(message="Question: ") == "answer"
    session.prompt.assert_called_once()
    assert session.prompt.call_args.args == ("Question: ",)


def test_input_forwards_an_explicit_completer_to_the_prompt_session():
    """Terminal input forwards the caller's capability manager unchanged."""
    session = Mock()
    session.prompt.return_value = "answer"
    completer = DummyCompleter()

    ConsoleInteraction(session=session).input(completer=completer)

    assert session.prompt.call_args.kwargs["completer"] is completer
    assert session.prompt.call_args.kwargs["complete_in_thread"] is True


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


def test_table_displays_a_titled_object_catalog(capsys):
    """Tables render their title, prefix, ordered attributes, and converted values."""
    items = [
        SimpleNamespace(name="alpha", description="first", count=1),
        SimpleNamespace(name="beta", description="second", count=2),
    ]

    ConsoleInteraction(console=Console(width=200)).table(
        items,
        title="Items:",
        prefix="/",
        columns=("name", "count"),
    )

    output = capsys.readouterr().out
    assert output.startswith("Items:\n")
    assert "/alpha" in output
    assert "1" in output
    assert "/beta" in output
    assert "2" in output
    assert "first" not in output


def test_table_handles_missing_attributes_empty_columns_and_row_limits(capsys):
    """Tables tolerate absent values and columns while honoring a row limit."""
    interaction = ConsoleInteraction(console=Console(width=200))

    interaction.table(
        [SimpleNamespace(name="first"), SimpleNamespace(name="second")],
        max_rows=1,
    )
    interaction.table([SimpleNamespace(name="hidden")], columns=())

    output = capsys.readouterr().out
    assert "first" in output
    assert "second" not in output
    assert "hidden" not in output


def test_table_constrains_output_width_and_can_use_the_console_width(capsys):
    """Tables cap rows at an explicit width and accept the console width when unbounded."""
    item = SimpleNamespace(name="item", description="x" * 100)
    interaction = ConsoleInteraction(console=Console(width=200))

    interaction.table([item], max_width=20)
    constrained = capsys.readouterr().out
    interaction.table([item], max_width=None)
    unconstrained = capsys.readouterr().out

    assert all(len(line) <= 20 for line in constrained.splitlines())
    assert "x" * 100 in unconstrained


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"max_width": 0}, "max_width must be positive."),
        ({"max_rows": -1}, "max_rows cannot be negative."),
    ],
)
def test_table_rejects_invalid_limits(limits, message):
    """Tables reject invalid width and row constraints."""
    with pytest.raises(ValueError, match=message):
        ConsoleInteraction().table([], **limits)


def test_tool_call_displays_its_name_and_arguments(capsys):
    """Tool calls receive a dedicated terminal presentation."""
    ConsoleInteraction().tool_call("search", '{"query":"term"}')

    assert capsys.readouterr().out == '\n[TOOL CALL]: search(query="term")\n'


def test_tool_call_truncates_each_long_argument_value_in_the_middle(capsys):
    """Tool calls retain the beginning and end of long argument values."""
    ConsoleInteraction().tool_call("write", '{"content":"0123456789abcdefghijklmnop"}')

    assert capsys.readouterr().out == '\n[TOOL CALL]: write(content="0123456789…hijklmnop")\n'


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("plain text", "plain text\n"),
        ('"Unicode: \\u2603"', "Unicode: ☃\n"),
    ],
)
def test_tool_result_displays_text_without_transport_quoting(capsys, result, expected):
    """Tool results preserve plain text and unwrap JSON-encoded strings."""
    ConsoleInteraction().tool_result("example", result)

    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("result", "expected_parts"),
    [
        ('{"name":"loop","items":[1,true,null]}', ('"name": "loop"', '"items": [', "true")),
        ('[{"id":1},{"id":2}]', ('[', '"id": 1', '"id": 2')),
        ("42", ("42",)),
        ("null", ("null",)),
    ],
)
def test_tool_result_pretty_prints_structured_json(capsys, result, expected_parts):
    """Structured and primitive JSON results receive readable JSON presentation."""
    ConsoleInteraction().tool_result("example", result)

    output = capsys.readouterr().out
    assert all(part in output for part in expected_parts)


def test_tool_result_classifies_error_envelopes_and_displays_details(capsys):
    """Serialized tool errors emphasize their message and retain diagnostic details."""
    result = '{"error":"invalid_arguments","message":"Invalid input.","details":[{"field":"x"}]}'

    ConsoleInteraction().tool_result("example", result)

    output = capsys.readouterr().out
    assert output.startswith("Error: Invalid input.\n")
    assert '"details": [' in output
    assert '"field": "x"' in output
    assert "invalid_arguments" not in output


def test_tool_result_classifies_error_envelopes_without_details(capsys):
    """Minimal serialized tool errors display only their human-readable message."""
    ConsoleInteraction().tool_result(
        "example", '{"error":"unknown_tool","message":"Unavailable."}'
    )

    assert capsys.readouterr().out == "Error: Unavailable.\n"


def test_tool_result_displays_local_file_content_with_source_and_line_numbers(capsys):
    """Local bounded content shows its path, byte range, and source line numbers."""
    result = json.dumps(
        {
            "path": "src/example.py",
            "content": "first\nsecond\n",
            "size_bytes": 40,
            "start_byte": 10,
            "end_byte": 23,
            "included_bytes": 13,
            "truncated": True,
            "truncation_reason": "lines",
            "start_line": 3,
            "end_line": 4,
            "next_start_line": 5,
        }
    )

    ConsoleInteraction().tool_result("read_text_file", result)

    output = capsys.readouterr().out
    assert "src/example.py · bytes 10–23 of 40 · truncated (lines)" in output
    assert "3 first" in output
    assert "4 second" in output


@pytest.mark.parametrize("name", ["fetch_content", "read_cached_content"])
def test_tool_result_displays_cached_content_with_source_and_handle(capsys, name):
    """Fetched and cached bounded content show their origin, handle, and body."""
    result = json.dumps(
        {
            "handle": "content-123",
            "source": "https://example.com/article.txt",
            "content": "article body",
            "size_bytes": 12,
            "start_byte": 0,
            "end_byte": 12,
            "included_bytes": 12,
            "truncated": False,
        }
    )

    ConsoleInteraction().tool_result(name, result)

    output = capsys.readouterr().out
    assert "https://example.com/article.txt · bytes 0–12 of 12 · handle content-123" in output
    assert "article body" in output


def test_tool_result_displays_folder_entries_as_a_hierarchical_tree(capsys):
    """Folder listings group recursive paths beneath inferred parent folders."""
    result = json.dumps(
        [
            {"path": "README.md", "type": "file"},
            {"path": "src/loop/main.py", "type": "file"},
            {"path": "src/loop/tools", "type": "folder"},
        ]
    )

    ConsoleInteraction().tool_result("list_folder", result)

    output = capsys.readouterr().out
    assert output == (
        ".\n"
        "├── README.md\n"
        "└── src\n"
        "    └── loop\n"
        "        ├── main.py\n"
        "        └── tools\n"
    )


def test_tool_result_displays_an_empty_folder_as_a_tree(capsys):
    """An empty list from list_folder remains identifiable as an empty folder tree."""
    ConsoleInteraction().tool_result("list_folder", "[]")

    assert capsys.readouterr().out == ".\n"


@pytest.mark.parametrize(
    ("name", "result"),
    [
        ("list_folder", '[{"path":"missing-type"}]'),
        ("read_text_file", "[]"),
        (
            "read_text_file",
            '{"path":"file.txt","content":"incomplete"}',
        ),
        (
            "fetch_content",
            '{"source":"https://example.com","content":"incomplete"}',
        ),
    ],
)
def test_tool_result_falls_back_for_malformed_well_known_formats(capsys, name, result):
    """Malformed recognized-tool results retain the generic JSON presentation."""
    ConsoleInteraction().tool_result(name, result)

    assert capsys.readouterr().out.strip()


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
