"""Tests for terminal-backed user interaction."""

import json
import sys
import termios
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.completion import CompleteEvent, DummyCompleter
from prompt_toolkit.document import Document
from rich.console import Console
from rich.prompt import Confirm

from loop import (
    ModelCallMetrics,
    Problem,
    RunMetrics,
    ToolResultPresentation,
    ToolResultPresentationSpec,
    constants,
)
from loop.interaction import ConsoleInteraction
from loop.models import Usage


def test_prompt_reads_a_trimmed_message_with_a_custom_prompt():
    """Input forwards its prompt message and strips the terminal response."""
    session = Mock()
    session.prompt.return_value = "  answer  \n"

    assert ConsoleInteraction(session=session).prompt(message="Question:") == "answer"
    session.prompt.assert_called_once()
    assert session.prompt.call_args.args == ("Question: ",)


def test_prompt_forwards_an_explicit_completer_to_the_prompt_session():
    """Terminal input forwards the caller's capability manager unchanged."""
    session = Mock()
    session.prompt.return_value = "answer"
    completer = DummyCompleter()

    ConsoleInteraction(session=session).prompt(completer=completer)

    assert session.prompt.call_args.kwargs["completer"] is completer
    assert session.prompt.call_args.kwargs["complete_in_thread"] is True


def test_prompt_displays_a_short_choice_list_and_returns_a_mapped_value_for_a_number(capsys):
    """Short catalogs render vertically and return the key associated with a selected number."""
    session = Mock()
    session.prompt.return_value = "2"

    result = ConsoleInteraction(console=Console(width=80), session=session).prompt(
        "Select:",
        choices={"first": "First choice", "second": "Second choice"},
    )

    assert result == "second"
    output = capsys.readouterr().out
    assert output.splitlines() == ["1. First choice", "2. Second choice"]
    assert "complete_while_typing" not in session.prompt.call_args.kwargs


def test_prompt_displays_large_choice_catalogs_in_columns(capsys):
    """Catalogs larger than nine choices retain the terminal-width-aware column layout."""
    session = Mock()
    session.prompt.return_value = "10"
    choices = [f"Choice {index}" for index in range(1, constants.COLUMNS_THRESHOLD + 2)]

    assert ConsoleInteraction(console=Console(width=80), session=session).prompt(
        choices=choices
    ) == ("Choice 10")

    output = capsys.readouterr().out
    assert "1. Choice 1" in output
    assert len(output.splitlines()) < len(choices)


def test_prompt_confirms_a_single_choice_without_opening_an_input_session(monkeypatch):
    """A singleton catalog uses confirmation and returns its mapped value when approved."""
    session = Mock()
    confirm = Mock(return_value=True)
    monkeypatch.setattr(ConsoleInteraction, "confirm", confirm)

    assert (
        ConsoleInteraction(session=session).prompt(choices={"model-id": "Model label"})
        == "model-id"
    )
    confirm.assert_called_once_with("Use 'Model label'?", default=False)
    session.prompt.assert_not_called()


def test_prompt_returns_false_when_its_single_choice_is_declined(monkeypatch):
    """Declining a singleton catalog has the same outcome as leaving a choice prompt."""
    monkeypatch.setattr(ConsoleInteraction, "confirm", Mock(return_value=False))

    assert ConsoleInteraction().prompt(choices=["Only choice"]) is False


def test_prompt_returns_a_choice_value_entered_without_its_number():
    """Case-insensitive typed labels return their mapped values."""
    session = Mock()
    session.prompt.return_value = "SECOND CHOICE"

    assert (
        ConsoleInteraction(session=session).prompt(
            choices={"first": "First choice", "second": "Second choice"}
        )
        == "second"
    )


def test_prompt_autocompletes_display_labels():
    """Choice completion replaces a typed label fragment with the displayed value."""
    session = Mock()
    session.prompt.return_value = "First choice"

    ConsoleInteraction(session=session).prompt(choices=["First choice", "Second choice"])

    completer = session.prompt.call_args.kwargs["completer"]
    completions = list(completer.get_completions(Document("sec"), CompleteEvent()))

    assert [(completion.text, completion.start_position) for completion in completions] == [
        ("Second choice", -3)
    ]


def test_prompt_reprompts_until_a_choice_is_selected(capsys):
    """Choice prompts reject values outside the displayed catalog."""
    session = Mock()
    session.prompt.side_effect = ["0", "unknown", "first"]

    assert ConsoleInteraction(session=session).prompt(choices=["first", "second"]) == "first"
    assert capsys.readouterr().out.endswith(
        "Warning: Select one of the listed choices by number or value.\n"
    )
    assert session.prompt.call_count == 3


@pytest.mark.parametrize(
    ("choices", "message"),
    [
        ([], "choices cannot be empty."),
        ([""], "choice labels cannot be empty."),
        (["same", "SAME"], "choice labels must be unique ignoring case."),
        (["1"], "choice labels cannot conflict with selection numbers."),
    ],
)
def test_prompt_rejects_ambiguous_choices(choices, message):
    """Choice prompts reject catalog shapes that cannot be selected unambiguously."""
    with pytest.raises(ValueError, match=message):
        ConsoleInteraction().prompt(choices=choices)


def test_prompt_rejects_a_custom_completer_when_it_manages_choices():
    """Choice prompts own completion so accepted input stays constrained to the catalog."""
    with pytest.raises(ValueError, match="choices cannot be combined"):
        ConsoleInteraction().prompt(choices=["first"], completer=DummyCompleter())


def test_columns_displays_unumbered_mapping_labels_without_exposing_keys(capsys):
    """Column display keeps mapping keys separate from their user-facing labels."""
    ConsoleInteraction(console=Console(width=80)).columns(
        {"internal-one": "First", "internal-two": "Second"}
    )

    output = capsys.readouterr().out
    assert "First" in output
    assert "Second" in output
    assert "internal-one" not in output


def test_list_displays_numbered_mapping_labels_without_exposing_keys(capsys):
    """List display keeps mapping keys separate from their numbered user-facing labels."""
    ConsoleInteraction(console=Console(width=80)).list(
        {"internal-one": "First", "internal-two": "Second"}, marker="numbered"
    )

    assert capsys.readouterr().out.splitlines() == ["1. First", "2. Second"]


def test_list_displays_arbitrary_values_with_bullet_markers(capsys):
    """Bullet lists accept tool-shaped scalar values without choice validation."""
    ConsoleInteraction().list(["first", 2, True, None, "first"], marker="bullet")

    assert capsys.readouterr().out.splitlines() == [
        "• first",
        "• 2",
        "• True",
        "• None",
        "• first",
    ]


def test_list_rejects_an_unknown_marker():
    """List rendering rejects marker styles outside its public contract."""
    with pytest.raises(ValueError, match="marker must be plain, numbered, or bullet"):
        ConsoleInteraction().list(["first"], marker="unknown")  # type: ignore[arg-type]


def test_prompt_reprompts_for_blank_input(capsys):
    """Input warns and prompts again until the terminal provides a message."""
    session = Mock()
    session.prompt.side_effect = ["   ", "answer"]

    assert ConsoleInteraction(session=session).prompt() == "answer"
    assert capsys.readouterr().out == "Warning: Please enter a message!\n"
    assert session.prompt.call_count == 2


def test_prompt_returns_false_for_its_default_exit_command():
    """Input treats the default exit command as an exit request."""
    session = Mock()
    session.prompt.return_value = " Q "

    assert ConsoleInteraction(session=session).prompt() is False


@pytest.mark.parametrize(
    ("command", "exit_commands"),
    [
        (" QuIt ", ("EXIT", "QUIT")),
        ("EXIT", "exit"),
    ],
)
def test_prompt_returns_false_for_configured_exit_commands(command, exit_commands):
    """Configured strings and iterables match commands without case or surrounding whitespace."""
    session = Mock()
    session.prompt.return_value = command

    assert ConsoleInteraction(session=session).prompt(exit_commands=exit_commands) is False


@pytest.mark.parametrize("exit_commands", [None, "", ()])
def test_prompt_accepts_messages_when_exit_commands_are_disabled(exit_commands):
    """Absent or empty exit commands leave otherwise valid input available to callers."""
    session = Mock()
    session.prompt.return_value = "q"

    assert ConsoleInteraction(session=session).prompt(exit_commands=exit_commands) == "q"


@pytest.mark.parametrize("error", [KeyboardInterrupt, EOFError])
def test_prompt_returns_false_when_the_prompt_is_interrupted(error):
    """Input treats terminal interruption and end-of-file as exit requests."""
    session = Mock()
    session.prompt.side_effect = error

    assert ConsoleInteraction(session=session).prompt() is False


def test_user_message_has_a_console_presentation(capsys):
    """Completed user messages retain the interactive prompt label."""
    ConsoleInteraction().user("complete")

    assert capsys.readouterr().out == "\nYou: complete\n"


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


def test_report_renders_a_problem_with_its_reference_and_retry_guidance(capsys):
    """Problem reports consistently present safe detail and an occurrence reference."""
    ConsoleInteraction().report(
        Problem(
            code="example.failed",
            title="Example failed",
            detail="Try again later.",
            retryable=True,
            instance="err_test",
        )
    )

    output = capsys.readouterr().out
    assert "Example failed" in output
    assert "Try again later." in output
    assert "Reference: err_test" in output
    assert "can be retried" in output


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


def test_table_displays_mapping_rows_with_the_standard_table_style(capsys):
    """Tables use the same headed layout for mapping rows as tool results."""
    ConsoleInteraction(console=Console(width=200)).table(
        [{"name": "alpha", "enabled": True}], columns=("name", "enabled")
    )

    output = capsys.readouterr().out
    assert "name" in output
    assert "enabled" in output
    assert "alpha" in output
    assert "True" in output


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


def tool_output(value):
    """Serialize a successful application tool result envelope."""
    return json.dumps({"ok": True, "result": value})


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("plain text", "plain text\n"),
        ("Unicode: ☃", "Unicode: ☃\n"),
    ],
)
def test_tool_result_displays_text_without_transport_quoting(capsys, result, expected):
    """Tool results preserve plain text and unwrap JSON-encoded strings."""
    ConsoleInteraction().tool_result(tool_output(result))

    assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("result", "expected_parts"),
    [
        ('{"name":"loop","items":[1,true,null]}', ('"name": "loop"', '"items": [', "true")),
        ('[{"id":1},{"id":2}]', ("[", '"id": 1', '"id": 2')),
        ("42", ("42",)),
        ("null", ("null",)),
    ],
)
def test_tool_result_pretty_prints_structured_json(capsys, result, expected_parts):
    """Structured and primitive JSON results receive readable JSON presentation."""
    ConsoleInteraction().tool_result(tool_output(json.loads(result)), ToolResultPresentationSpec())

    output = capsys.readouterr().out
    assert all(part in output for part in expected_parts)


def test_tool_result_renders_problem_envelopes_without_raw_metadata(capsys):
    """Serialized tool problems use one panel and hide machine-oriented metadata."""
    result = json.dumps(
        {
            "ok": False,
            "problem": {
                "code": "tool.invalid_arguments",
                "title": "Invalid input",
                "detail": "The supplied value is invalid.",
                "severity": "error",
                "retryable": False,
                "operation": "example",
                "instance": "err_test",
                "metadata": {"fields": [{"field": "x"}]},
            },
        }
    )

    ConsoleInteraction().tool_result(result, ToolResultPresentationSpec())

    output = capsys.readouterr().out
    assert "Invalid input" in output
    assert "The supplied value is invalid." in output
    assert "Reference: err_test" in output
    assert '"fields"' not in output


def test_tool_result_rejects_values_without_an_envelope():
    """Raw values cannot bypass the application tool result contract."""
    with pytest.raises(TypeError, match="result envelope"):
        ConsoleInteraction().tool_result('{"message":"Unavailable."}')


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

    ConsoleInteraction().tool_result(
        tool_output(json.loads(result)),
        ToolResultPresentationSpec(kind=ToolResultPresentation.TEXT),
    )

    output = capsys.readouterr().out
    assert "src/example.py · bytes 10–23 of 40 · truncated (lines)" in output
    assert "3 first" in output
    assert "4 second" in output


def test_tool_result_displays_cached_content_with_source_and_handle(capsys):
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

    ConsoleInteraction().tool_result(
        tool_output(json.loads(result)),
        ToolResultPresentationSpec(kind=ToolResultPresentation.TEXT),
    )

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

    ConsoleInteraction().tool_result(
        tool_output(json.loads(result)),
        ToolResultPresentationSpec(kind=ToolResultPresentation.TREE),
    )

    output = capsys.readouterr().out
    assert output == (
        ".\n├── README.md\n└── src\n    └── loop\n        ├── main.py\n        └── tools\n"
    )


def test_tool_result_displays_an_empty_folder_as_a_tree(capsys):
    """An empty list from list_folder remains identifiable as an empty folder tree."""
    ConsoleInteraction().tool_result(
        tool_output([]), ToolResultPresentationSpec(kind=ToolResultPresentation.TREE)
    )

    assert capsys.readouterr().out == ".\n"


@pytest.mark.parametrize(
    ("presentation", "result"),
    [
        (ToolResultPresentation.TREE, '[{"path":"missing-type"}]'),
        (ToolResultPresentation.TEXT, "[]"),
        (
            ToolResultPresentation.TEXT,
            '{"path":"file.txt","content":"incomplete"}',
        ),
        (
            ToolResultPresentation.TEXT,
            '{"source":"https://example.com","content":"incomplete"}',
        ),
    ],
)
def test_tool_result_falls_back_for_malformed_well_known_formats(capsys, presentation, result):
    """Malformed recognized-tool results retain the generic JSON presentation."""
    ConsoleInteraction().tool_result(
        tool_output(json.loads(result)), ToolResultPresentationSpec(kind=presentation)
    )

    assert capsys.readouterr().out.strip()


def test_tool_result_renders_nested_tables_with_declared_columns_and_title(capsys):
    """Table presentations select nested rows and use declared headings."""
    ConsoleInteraction().tool_result(
        tool_output({"skills": [{"name": "coding", "activated": True}]}),
        ToolResultPresentationSpec(
            kind=ToolResultPresentation.TABLE,
            value_path=("skills",),
            columns=("name", "activated"),
            title="Skills:",
        ),
    )

    output = capsys.readouterr().out
    assert "Skills:" in output
    assert "name" in output
    assert "coding" in output
    assert "True" in output


def test_tool_result_renders_scalar_lists_as_bullet_lists(capsys):
    """List presentations delegate scalar values to the shared bullet-list renderer."""
    ConsoleInteraction().tool_result(
        tool_output(["first", 2, True, None]),
        ToolResultPresentationSpec(kind=ToolResultPresentation.LIST),
    )

    assert capsys.readouterr().out == "• first\n• 2\n• True\n• None\n"


def test_json_writes_a_structured_value(capsys):
    """The public JSON renderer formats arbitrary JSON-compatible data."""
    ConsoleInteraction().json({"ready": True})

    assert '"ready": true' in capsys.readouterr().out


def test_tree_writes_typed_path_entries(capsys):
    """The public tree renderer presents nested typed paths hierarchically."""
    ConsoleInteraction().tree(
        [
            {"path": "src/main.py", "type": "file"},
            {"path": "src/tools", "type": "folder"},
        ]
    )

    assert capsys.readouterr().out == ".\n└── src\n    ├── main.py\n    └── tools\n"


def test_content_writes_only_highlighted_text_without_metadata(capsys):
    """The public content renderer omits source metadata and line numbers by default."""
    ConsoleInteraction().content("ready = True", identifier="example.py")

    output = capsys.readouterr().out
    assert "ready = True" in output
    assert "bytes" not in output


def test_tool_result_pretty_prints_explicit_nested_json(capsys):
    """JSON presentations can select and render a nested value."""
    ConsoleInteraction().tool_result(
        tool_output({"result": {"ready": True}}),
        ToolResultPresentationSpec(
            kind=ToolResultPresentation.JSON,
            value_path=("result",),
        ),
    )

    output = capsys.readouterr().out
    assert '"ready": true' in output
    assert '"result"' not in output


@pytest.mark.parametrize("result", ['{"other":true}', "42"])
def test_tool_result_falls_back_to_the_root_for_an_invalid_value_path(capsys, result):
    """Missing or non-mapping presentation paths preserve the complete result."""
    ConsoleInteraction().tool_result(
        tool_output(json.loads(result)),
        ToolResultPresentationSpec(
            kind=ToolResultPresentation.JSON,
            value_path=("missing",),
        ),
    )

    assert capsys.readouterr().out.strip()


def test_tool_result_table_without_columns_falls_back_to_json(capsys):
    """A table lacking column declarations remains visible as structured JSON."""
    ConsoleInteraction().tool_result(
        tool_output([{"name": "loop"}]),
        ToolResultPresentationSpec(kind=ToolResultPresentation.TABLE),
    )

    assert '"name": "loop"' in capsys.readouterr().out


def test_run_metrics_have_a_complete_console_presentation(capsys):
    """Run statistics present context, counts, usage, duration, and throughput."""
    usage = Usage(input_tokens=100, output_tokens=20, cached_tokens=40, reasoning_tokens=5)
    metrics = RunMetrics(
        active_duration_seconds=2.5,
        model_duration_seconds=2,
        tool_duration_seconds=0.5,
        model_calls=(ModelCallMetrics(model="served-model", duration_seconds=2, usage=usage),),
        message_count=2,
        item_count=4,
        usage=usage,
        model="served-model",
        context_tokens=18_432,
        context_window=262_144,
    )

    ConsoleInteraction().run_metrics(metrics)

    assert capsys.readouterr().out == (
        "Model: served-model · Context: 18,432 / 262,144 tokens\n"
        "Run: 1 model calls · 2 messages · 4 items · 2.50s active\n"
        "Tokens: 100 input · 20 output · 40 cached · 5 reasoning\n"
        "Performance: 2.00s model · 0.50s tools · 10.0 output tokens/s\n"
    )


def test_run_metrics_omit_unknown_usage_and_throughput(capsys):
    """Unknown token counts do not produce fabricated usage or throughput values."""
    metrics = RunMetrics(
        active_duration_seconds=0,
        model_duration_seconds=0,
        tool_duration_seconds=0,
        message_count=0,
        item_count=0,
        context_window=100,
    )

    ConsoleInteraction().run_metrics(metrics)

    assert capsys.readouterr().out == (
        "Model: ? · Context: ? / 100 tokens\n"
        "Run: 0 model calls · 0 messages · 0 items · 0.00s active\n"
        "Performance: 0.00s model · 0.00s tools\n"
    )


def test_permission_replay_uses_the_stored_prompt_or_a_fallback(capsys):
    """Permission replay preserves exact prompts and handles absent historical text."""
    interaction = ConsoleInteraction()
    interaction.permission("Proceed?", "allow")
    interaction.permission("Permission requested.", "allow")

    assert capsys.readouterr().out == "Proceed? [allow]\nPermission requested. [allow]\n"


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
    with interaction.response_context():
        interaction.answer_delta("partial", start=True)
    interaction.conversation_ended()

    assert capsys.readouterr().out == "\nAnswer:partial\n\nConversation ended.\n"


def test_response_scope_does_not_extend_complete_output(capsys):
    """A response scope adds no separator when output already terminates itself."""
    interaction = ConsoleInteraction()
    with interaction.response_context():
        interaction.answer("complete")

    assert capsys.readouterr().out == "\nAnswer:complete\n"


def test_response_scope_terminates_streamed_output_after_an_error(capsys):
    """A response scope terminates streamed output when presentation fails."""
    interaction = ConsoleInteraction()

    with pytest.raises(RuntimeError, match="failed"), interaction.response_context():
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


def test_confirm_discards_terminal_input_before_asking_its_question(monkeypatch):
    """Confirmation discards queued terminal input before delegating to Rich."""
    events = []
    console = Mock()
    ask = Mock(side_effect=lambda *args, **kwargs: events.append("ask") or True)
    monkeypatch.setattr(Confirm, "ask", ask)
    stdin = Mock()
    stdin.isatty.return_value = True
    stdin.fileno.return_value = 42
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(
        termios,
        "tcflush",
        lambda descriptor, queue: events.append(("flush", descriptor, queue)),
    )

    assert ConsoleInteraction(console=console).confirm("Continue?") is True
    assert events == [("flush", 42, termios.TCIFLUSH), "ask"]


def test_confirm_does_not_flush_redirected_input(monkeypatch):
    """Confirmation preserves input when standard input is not an interactive terminal."""
    console = Mock()
    console.input.return_value = "n"
    stdin = Mock()
    stdin.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", stdin)
    flush = Mock()
    monkeypatch.setattr(termios, "tcflush", flush)

    assert ConsoleInteraction(console=console).confirm("Continue?") is False
    flush.assert_not_called()


def test_confirm_reads_input_when_the_terminal_cannot_be_flushed(monkeypatch):
    """Confirmation remains usable when its terminal does not support queue flushing."""
    console = Mock()
    ask = Mock(return_value=True)
    monkeypatch.setattr(Confirm, "ask", ask)
    stdin = Mock()
    stdin.isatty.return_value = True
    stdin.fileno.return_value = 42
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(termios, "tcflush", Mock(side_effect=OSError("unsupported")))

    assert ConsoleInteraction(console=console).confirm("Continue?") is True
    ask.assert_called_once_with("Continue?", default=False, console=console)
