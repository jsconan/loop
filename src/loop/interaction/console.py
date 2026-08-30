"""Define user interaction abstractions and tool invocation context."""

from __future__ import annotations

import json
import sys
import termios
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from itertools import islice
from pprint import pformat
from typing import Any, Literal

from markdown_it import MarkdownIt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter
from rich.columns import Columns
from rich.console import Console
from rich.constrain import Constrain
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .. import constants
from ..errors import Problem
from ..models import (
    RAW_TOOL_RESULT_PRESENTATION,
    RunMetrics,
    ToolResultPresentation,
    ToolResultPresentationSpec,
)
from ..utils import ChoiceItem, choice_items, format_tool_call_arguments
from .interaction import Interaction
from .models import ListMarker


class MarkdownStream:
    """Append structurally complete Markdown blocks to a console.

    Earlier reference-style links are not held for definitions in later blocks. Supporting that
    uncommon CommonMark feature would require buffering the entire response and defeat streaming.
    """

    _parser = MarkdownIt().enable("strikethrough").enable("table")

    def __init__(self, console: Console, *, style: str) -> None:
        self._console = console
        self._style = style
        self._pending = ""

    def write(self, delta: str) -> None:
        """Buffer a delta and print any structurally stable prefix."""
        self._pending += delta
        if not self._pending.endswith(("\n", "\r")):
            return
        boundary = self._stable_boundary()
        if boundary is None:
            return
        stable = self._pending[:boundary]
        self._pending = self._pending[boundary:]
        self._console.print(Markdown(stable, style=self._style))

    def finish(self) -> None:
        """Print the final buffered block, treating end-of-stream as its boundary."""
        if self._pending:
            self._console.print(Markdown(self._pending, style=self._style))
        self._pending = ""

    def _stable_boundary(self) -> int | None:
        """Return the source offset before the final top-level Markdown block."""
        block_starts = [
            token.map[0]
            for token in self._parser.parse(self._pending)
            if token.level == 0 and token.map is not None
        ]
        if len(block_starts) < 2:
            return None
        final_start = block_starts[-1]
        lines = self._pending.splitlines(keepends=True)
        return sum(len(line) for line in lines[:final_start])


class ConsoleInteraction(Interaction):
    """Interact with a user through a rich, editable process terminal.

    Args:
        console (Console | None): Rich console used for terminal output. Defaults to a new console.
        session (PromptSession[str] | None): Prompt session used for editable user input. Defaults
            to a new session.
        choices_session (PromptSession[str] | None): Prompt session used for editable user input
            with selectable choices. Defaults to a new session.
        markdown (bool): Whether to render model output as Markdown.
            Defaults to ``RENDER_MARKDOWN``.
    """

    _console: Console
    _session: PromptSession[str]
    _choices_session: PromptSession[str]
    _markdown: bool
    _stream_kind: Literal["reasoning", "answer"] | None
    _stream: MarkdownStream | None
    _plain_stream_has_output: bool

    def __init__(
        self,
        *,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
        choices_session: PromptSession[str] | None = None,
        markdown: bool = constants.RENDER_MARKDOWN,
    ) -> None:
        self._console = console or Console()
        self._session = session or PromptSession()
        self._choices_session = choices_session or PromptSession()
        self._markdown = markdown
        self._stream_kind = None
        self._stream = None
        self._plain_stream_has_output = False

    @contextmanager
    def response_context(self) -> Generator[None]:
        """Present one model response and finalize streamed output.

        Yields:
            None: Control while the response is being presented.
        """
        try:
            yield
        finally:
            self._finish_markdown_stream()

    def _finish_markdown_stream(self) -> None:
        """Commit and clear the active model-output stream."""
        if self._stream is not None:
            self._stream.finish()
        elif self._plain_stream_has_output:
            self._console.print()
        self._stream_kind = None
        self._stream = None
        self._plain_stream_has_output = False

    def _write_markdown_delta(
        self,
        delta: str,
        *,
        kind: Literal["reasoning", "answer"],
    ) -> None:
        """Append stable Markdown blocks without rewriting terminal history."""
        if self._stream is None or self._stream_kind != kind:
            self._finish_markdown_stream()
            if kind == "reasoning":
                self._reasoning_heading()
            else:
                self._answer_heading()
            self._stream_kind = kind
            self._stream = MarkdownStream(
                self._console,
                style="dim" if kind == "reasoning" else "none",
            )
        self._stream.write(delta)

    def _write_plain_delta(self, delta: str, *, kind: Literal["reasoning", "answer"]) -> None:
        """Write a streamed model delta immediately without Markdown parsing."""
        if self._stream_kind != kind:
            self._finish_markdown_stream()
            if kind == "reasoning":
                self._reasoning_heading()
            else:
                self._answer_heading()
            self._stream_kind = kind
        self._console.print(
            delta,
            end="",
            style="dim" if kind == "reasoning" else None,
            markup=False,
        )
        self._plain_stream_has_output |= bool(delta)

    # pylint: disable-next=too-many-branches
    def prompt(
        self,
        message: str | None = None,
        completer: Completer | None = None,
        exit_commands: str | Iterable[str] | None = "q",
        choices: Iterable[str | ChoiceItem] | Mapping[object, str] | None = None,
        index: Iterable[str] | Mapping[object, str] | None = None,
        default: object | None = None,
        secret: bool = False,
    ) -> object | False:
        """Prompt for a non-empty user message or an exit command.

        The choice, completion, cancellation, and default-input paths deliberately share one loop.

        Args:
            message (str | None): Prompt message displayed before reading input.
                Defaults to ``None`` for the default prompt.
            completer (Completer | None): Optional input completer. Defaults to no completion.
            exit_commands (str | Iterable[str] | None): Optional list of exit terms that end the
                prompt. Defaults to ``"q"``.
            choices (Iterable[str | ChoiceItem] | Mapping[object, str] | None): Optional selectable
                values. Mapping keys are returned while their values are displayed and accepted as
                input. One choice is confirmed directly, two through nine are shown as a numbered
                list, and larger catalogs are arranged in columns. Defaults to ``None``.
            index (Iterable[str] | Mapping[object, str] | None): Optional selection indexes to
                display alongside each choice. If omitted, indexes are automatically generated as
                numeric indexes. Defaults to ``None``.
            default (object | None): Value returned for empty input, or ``None`` to require
                non-empty input. Defaults to ``None``.
            secret (bool): Whether entered text is masked. Defaults to ``False``.

        Returns:
            object | False: The selected value or entered message, or ``False`` when the user
            requests to exit.

        Raises:
            ValueError: If ``choices`` or ``index`` are invalid, or choices are used with a custom
                ``completer``.
        """
        if index is not None and choices is None:
            raise ValueError("index requires choices.")
        normalized_choices = choice_items(choices, index=index) if choices is not None else None
        choice_index = (
            {choice.index.casefold(): choice.value for choice in normalized_choices}
            if normalized_choices is not None
            else None
        )
        if isinstance(exit_commands, str):
            exit_commands = exit_commands.strip()
            if exit_commands:
                exit_commands = (exit_commands.casefold(),)
        elif exit_commands:
            exit_commands = tuple(
                command.strip().casefold() for command in exit_commands if command.strip()
            )
        if not exit_commands:
            exit_commands = ()
        if choice_index is not None and set(choice_index) & set(exit_commands):
            raise ValueError("selection indexes cannot conflict with exit commands.")
        session = self._session
        if normalized_choices is not None:
            session = self._choices_session
            if completer is not None:
                raise ValueError("choices cannot be combined with a custom completer.")
            if len(normalized_choices) == 1:
                choice = normalized_choices[0]
                self.info(f"\nOnly one choice available: {choice.name}")
                return (
                    choice.value if self.confirm(f"Use '{choice.name}'?", default=False) else False
                )
            displayed_choices = {choice.value: choice.label for choice in normalized_choices}
            if len(normalized_choices) < constants.COLUMNS_THRESHOLD:
                self.list(displayed_choices)
            else:
                self.columns(displayed_choices)
            completer = WordCompleter(
                [choice.name for choice in normalized_choices],
                ignore_case=True,
                sentence=True,
                match_middle=True,
            )
        if message is None:
            message = "\nYou:"
        message = str(message).rstrip()
        while True:
            try:
                user_input = session.prompt(
                    f"{message} " if message else "",
                    completer=completer,
                    complete_in_thread=True,
                    is_password=secret,
                ).strip()
            except (KeyboardInterrupt, EOFError):
                return False
            if not user_input:
                if default is not None:
                    return default
                self.warning("Please enter a message!")
                continue
            if user_input.casefold() in exit_commands:
                return False
            if normalized_choices is not None:
                if user_input.casefold() in choice_index:
                    return choice_index[user_input.casefold()]
                for choice in normalized_choices:
                    if user_input.casefold() == choice.name.casefold():
                        return choice.value
                self.warning("Select one of the listed choices by index or value.")
                continue
            return user_input

    def user(self, message: str) -> None:
        """Write a completed user message to the terminal.

        Args:
            message (str): Complete user message text to write.
        """
        self._console.print("\nYou:", end=" ", style="bold bright_blue", markup=False)
        self._console.print(message, markup=False, highlight=False)

    def _reasoning_heading(self) -> None:
        """Write a reasoning heading to the terminal."""
        self._console.print("\nThinking...\n", style="dim cyan", markup=False)

    def reasoning(self, message: str) -> None:
        """Render completed model reasoning as Markdown in the terminal.

        Args:
            message (str): Complete reasoning text to write.
        """
        self._finish_markdown_stream()
        self._reasoning_heading()
        self._console.print(
            Markdown(message, style="dim") if self._markdown else message,
            markup=False,
        )

    def reasoning_delta(self, delta: str) -> None:
        """Buffer streamed reasoning and append complete Markdown blocks.

        Args:
            delta (str): Incremental reasoning text to write.
        """
        if self._markdown:
            self._write_markdown_delta(delta, kind="reasoning")
        else:
            self._write_plain_delta(delta, kind="reasoning")

    def _answer_heading(self) -> None:
        """Write an answer heading to the terminal."""
        self._console.print("\nAnswer:", style="bold bright_green", markup=False)

    def answer(self, message: str) -> None:
        """Render a completed model answer as Markdown in the terminal.

        Args:
            message (str): Complete answer text to write.
        """
        self._finish_markdown_stream()
        self._answer_heading()
        self._console.print(Markdown(message) if self._markdown else message, markup=False)

    def answer_delta(self, delta: str) -> None:
        """Buffer streamed answers and append complete Markdown blocks.

        Args:
            delta (str): Incremental answer text to write.
        """
        if self._markdown:
            self._write_markdown_delta(delta, kind="answer")
        else:
            self._write_plain_delta(delta, kind="answer")

    def report(self, problem: Problem) -> None:
        """Write a structured problem to the terminal.

        Args:
            problem (Problem): Safe problem to render with a stable reference.
        """
        color = "yellow" if problem.severity == "warning" else "red"
        content = Text(problem.detail)
        content.append(f"\n\nReference: {problem.instance}", style="dim")
        if problem.retryable:
            content.append("\nThis operation can be retried.", style="dim")
        self._console.print(Panel(content, title=problem.title, border_style=color))

    def error(self, message: str) -> None:
        """Write an error to the terminal.

        Args:
            message (str): Error text to write.
        """
        self._console.print(f"Error: {message}", style="bold red", markup=False)

    def warning(self, message: str) -> None:
        """Write a warning to the terminal.

        Args:
            message (str): Warning text to write.
        """
        self._console.print(f"Warning: {message}", style="bold orange1", markup=False)

    def debug(self, value: Any) -> None:
        """Write diagnostic output to the terminal.

        Args:
            value (Any): Diagnostic value to write.
        """
        self._console.print(f"\n[DEBUG EVENT]: {type(value)}", style="dim blue", markup=False)
        self._console.print(pformat(value), style="dim", markup=False, highlight=False)

    def info(self, message: str = "") -> None:
        """Write neutral status information to the terminal.

        Args:
            message (str): Status text to write, or an empty string for a blank line.
        """
        self._console.print(message, markup=False, highlight=False)

    def table(
        self,
        items: Iterable[object | Mapping[str, Any]],
        *,
        title: str | None = None,
        prefix: str = "  ",
        columns: Iterable[str] = ("name", "description"),
        max_width: int | None = constants.TABULAR_MAX_WIDTH,
        max_rows: int | None = None,
    ) -> None:
        """Write mapping fields or object attributes as a table.

        Args:
            items (Iterable[object | Mapping[str, Any]]): Rows whose mappings or attributes
                provide the displayed values.
            title (str | None): Optional title to write above the table.
            prefix (str): Text to prepend to the first value in each row.
            columns (Iterable[str]): Attribute names to display as columns.
            max_width (int | None): Maximum table width in characters. Defaults to
                ``TABULAR_MAX_WIDTH``. Pass ``None`` to use the console's available width.
            max_rows (int | None): Maximum number of objects to display. Displays every object
                when unset.

        Raises:
            ValueError: If ``max_width`` is not positive or ``max_rows`` is negative.
        """
        if max_width is not None and max_width < 1:
            raise ValueError("max_width must be positive.")
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows cannot be negative.")

        if title:
            self.info(title)
        column_names = tuple(columns)
        table = Table(*column_names)
        for column in table.columns:
            column.overflow = "ellipsis"
        visible_items = items if max_rows is None else islice(items, max_rows)
        for item in visible_items:
            values = [Text(str(self._value_for(item, column))) for column in column_names]
            if values:
                values[0].plain = f"{prefix}{values[0].plain}"
            table.add_row(*values)
        self._console.print(Constrain(table, max_width))

    def list(
        self,
        values: Iterable[object] | Mapping[object, str],
        *,
        marker: ListMarker = "plain",
    ) -> None:
        """Write values as a vertical list.

        Args:
            values (Iterable[object] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            marker (ListMarker): Prefix style for each displayed
                value. Defaults to ``"plain"``.
        """
        self._console.print("\n".join(self._marked_values(values, marker)))

    def columns(
        self,
        values: Iterable[object] | Mapping[object, str],
        *,
        marker: ListMarker = "plain",
    ) -> None:
        """Write values in terminal-width-aware columns.

        Args:
            values (Iterable[object] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            marker (ListMarker): Prefix style for each displayed
                value. Defaults to ``"plain"``.
        """
        renderables = [Text(value) for value in self._marked_values(values, marker)]
        self._console.print(
            Columns(renderables, padding=(0, 2), equal=True, expand=True, column_first=True)
        )

    def json(self, value: Any) -> None:
        """Write a structured value as formatted JSON.

        Args:
            value (Any): JSON-compatible value to write.
        """
        self._console.print(JSON.from_data(value))

    def tree(self, entries: Iterable[Mapping[str, str]]) -> None:
        """Write typed path entries as a hierarchical tree.

        Args:
            entries (Iterable[Mapping[str, str]]): Entries containing ``path`` and ``type``
                fields, where type is ``"file"`` or ``"folder"``.
        """
        root = Tree(Text("."), guide_style="dim")
        folders = {(): root}
        entry_types = {tuple(entry["path"].split("/")): entry["type"] for entry in entries}
        paths = set(entry_types)
        paths.update(path[:index] for path in entry_types for index in range(1, len(path)))

        for path in sorted(paths, key=lambda item: (len(item), item)):
            parent = folders[path[:-1]]
            is_folder = entry_types.get(path, "folder") == "folder"
            node = parent.add(Text(path[-1], style="bold blue" if is_folder else None))
            if is_folder:
                folders[path] = node
        self._console.print(root)

    def content(
        self,
        text: str,
        *,
        identifier: str,
        start_line: int | None = None,
    ) -> None:
        """Write syntax-highlighted textual content.

        Args:
            text (str): Textual content to write.
            identifier (str): Source name used to select a syntax lexer.
            start_line (int | None): First source line number, or ``None`` to omit line numbers.
        """
        self._console.print(
            Syntax(
                text,
                Syntax.guess_lexer(identifier, text),
                line_numbers=start_line is not None,
                start_line=start_line or 1,
                word_wrap=True,
            )
        )

    def tool_call(self, name: str, arguments: str) -> None:
        """Write a model-requested tool call to the terminal.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """
        self._console.print(
            f"\n[TOOL CALL]: {name}({format_tool_call_arguments(arguments)})",
            style="dim magenta",
            markup=False,
        )

    def tool_result(
        self,
        result: str,
        presentation: ToolResultPresentationSpec = RAW_TOOL_RESULT_PRESENTATION,
    ) -> None:
        """Write a serialized tool result in a user-readable form.

        Args:
            result (str): Serialized result returned by the tool.
            presentation (ToolResultPresentationSpec): Semantic presentation requested by the
                tool execution. Defaults to generic raw presentation.
        """
        value = json.loads(result)
        if not isinstance(value, dict) or not isinstance(value.get("ok"), bool):
            raise TypeError("Tool result must use the application result envelope.")
        if not value["ok"]:
            self.report(Problem.model_validate(value.get("problem")))
            return
        value = value.get("result")
        selected = self._value_at(value, presentation.value_path)
        if presentation.title:
            self.info(presentation.title)
        if presentation.kind is ToolResultPresentation.TREE and self._is_folder_result(selected):
            self.tree(selected)
            return
        if presentation.kind is ToolResultPresentation.TEXT and self._is_content_result(selected):
            identifier = selected.get("path", selected.get("source"))
            self._display_content_metadata(selected, identifier=identifier)
            return
        if presentation.kind is ToolResultPresentation.TABLE and self._is_table_result(selected):
            if presentation.columns:
                self.table(selected, columns=presentation.columns)
                return
            self.json(selected)
            return
        if presentation.kind is ToolResultPresentation.LIST and self._is_list_result(selected):
            self.list(selected, marker="bullet")
            return
        if presentation.kind is ToolResultPresentation.JSON:
            self.json(selected)
            return
        if isinstance(value, str):
            self.info(value)
        else:
            self.json(value)

    @staticmethod
    def _value_at(value: Any, path: tuple[str, ...]) -> Any:
        """Return the nested mapping value at a presentation path, or the root on mismatch."""
        selected = value
        for key in path:
            if not isinstance(selected, dict) or key not in selected:
                return value
            selected = selected[key]
        return selected

    @staticmethod
    def _is_folder_result(value: Any) -> bool:
        """Return whether a value is a typed folder-entry list."""
        return isinstance(value, list) and all(
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and entry.get("type") in {"file", "folder"}
            for entry in value
        )

    @staticmethod
    def _is_content_result(value: Any) -> bool:
        """Return whether a value is a local or cached bounded-content envelope."""
        if not isinstance(value, dict):
            return False
        required = {"content", "size_bytes", "start_byte", "end_byte", "truncated"}
        has_identity = isinstance(value.get("path", value.get("source")), str)
        return required <= value.keys() and has_identity and isinstance(value["content"], str)

    @staticmethod
    def _is_table_result(value: Any) -> bool:
        """Return whether a value is a list of string-keyed records."""
        return isinstance(value, list) and all(
            isinstance(item, dict) and all(isinstance(key, str) for key in item) for item in value
        )

    @staticmethod
    def _is_list_result(value: Any) -> bool:
        """Return whether a value is a list of scalar labels."""
        return isinstance(value, list) and all(
            item is None or isinstance(item, (bool, int, float, str)) for item in value
        )

    def _display_content_metadata(self, value: dict[str, Any], *, identifier: str) -> None:
        """Write bounded-content metadata before its separately rendered body."""
        range_text = f"bytes {value['start_byte']}–{value['end_byte']} of {value['size_bytes']}"
        metadata = Text(identifier, style="bold cyan")
        metadata.append(f" · {range_text}", style="dim")
        if value["truncated"]:
            reason = value.get("truncation_reason", "limit")
            metadata.append(f" · truncated ({reason})", style="yellow")
        if "handle" in value:
            metadata.append(f" · handle {value['handle']}", style="dim")
        self._console.print(metadata)
        self.content(value["content"], identifier=identifier, start_line=value.get("start_line"))

    @staticmethod
    def _value_for(item: object | Mapping[str, Any], column: str) -> Any:
        """Return one table cell value from a mapping row or an object row."""
        return item.get(column, "") if isinstance(item, Mapping) else getattr(item, column, "")

    @staticmethod
    def _marked_values(
        values: Iterable[object] | Mapping[object, str],
        marker: ListMarker,
    ) -> list[str]:
        """Return display values with a requested list marker."""
        if marker not in {"plain", "numbered", "bullet"}:
            raise ValueError("marker must be plain, numbered, or bullet.")
        labels = values.values() if isinstance(values, Mapping) else values
        prefix = "• " if marker == "bullet" else ""
        return [
            f"{index}. {label}" if marker == "numbered" else f"{prefix}{label}"
            for index, label in enumerate(labels, start=1)
        ]

    def run_metrics(self, metrics: RunMetrics) -> None:
        """Write one completed agent run's durable statistics.

        Args:
            metrics (RunMetrics): Complete run statistics and current context occupancy.
        """
        current_model = metrics.model or "?"
        used = f"{metrics.context_tokens:,}" if metrics.context_tokens is not None else "?"
        capacity = f"{metrics.context_window:,}" if metrics.context_window is not None else "?"
        self._console.print(
            f"Model: {current_model} · Context: {used} / {capacity} tokens",
            style="dim cyan",
            markup=False,
            soft_wrap=True,
        )
        self._console.print(
            f"Run: {len(metrics.model_calls)} model calls · {metrics.message_count} messages · "
            f"{metrics.item_count} items · {metrics.active_duration_seconds:.2f}s active",
            style="dim cyan",
            markup=False,
            soft_wrap=True,
        )
        usage = metrics.usage
        tokens = []
        for label, value in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cached", usage.cached_tokens),
            ("reasoning", usage.reasoning_tokens),
        ):
            if value is not None:
                tokens.append(f"{value:,} {label}")
        if tokens:
            self._console.print(
                f"Tokens: {' · '.join(tokens)}",
                style="dim cyan",
                markup=False,
                soft_wrap=True,
            )
        performance = (
            f"Performance: {metrics.model_duration_seconds:.2f}s model · "
            f"{metrics.tool_duration_seconds:.2f}s tools"
        )
        if usage.output_tokens is not None and metrics.model_duration_seconds > 0:
            performance += (
                f" · {usage.output_tokens / metrics.model_duration_seconds:.1f} output tokens/s"
            )
        self._console.print(
            performance,
            style="dim cyan",
            markup=False,
            soft_wrap=True,
        )

    def permission(self, prompt: str, decision: str) -> None:
        """Write one originally prompted permission decision.

        Args:
            prompt (str): Exact original prompt or a replay fallback.
            decision (str): Effective decision label.
        """
        self._console.print(
            f"{prompt} [{decision}]",
            style="dim yellow",
            markup=False,
            soft_wrap=True,
        )

    def conversation_ended(self) -> None:
        """Report conversation termination in the terminal."""
        self.info("\nConversation ended.")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask for a yes-or-no answer and apply an empty-answer default.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the operation.
        """
        self._discard_pending_terminal_input()
        return Confirm.ask(message, default=default, console=self._console)

    @staticmethod
    def _discard_pending_terminal_input() -> None:
        """Discard unread input from an interactive terminal when supported."""
        if not sys.stdin.isatty():
            return
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (AttributeError, OSError, ValueError):
            pass
