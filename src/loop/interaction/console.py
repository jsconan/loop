"""Define user interaction abstractions and tool invocation context."""

from __future__ import annotations

import json
import sys
import termios
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from pprint import pformat
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter
from rich.columns import Columns
from rich.console import Console
from rich.constrain import Constrain
from rich.json import JSON
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .. import constants
from ..models import RunMetrics
from ..utils import format_tool_call_arguments
from .interaction import Interaction


class ConsoleInteraction(Interaction):
    """Interact with a user through a rich, editable process terminal.

    Args:
        console (Console | None): Rich console used for terminal output. Defaults to a new console.
        session (PromptSession[str] | None): Prompt session used for editable user input. Defaults
            to a new session.
    """

    def __init__(
        self,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
    ) -> None:
        self._console = console or Console()
        self._session = session or PromptSession()
        self._streamed_output = False

    @contextmanager
    def response_context(self) -> Generator[None]:
        """Present one model response and finalize streamed output.

        Yields:
            None: Control while the response is being presented.
        """
        self._streamed_output = False
        try:
            yield
        finally:
            if self._streamed_output:
                self.info()
            self._streamed_output = False

    def prompt(
        self,
        message: str | None = None,
        completer: Completer | None = None,
        exit_commands: str | Iterable[str] | None = "q",
        choices: Iterable[str] | Mapping[object, str] | None = None,
    ) -> object | False:
        """Prompt for a non-empty user message or an exit command.

        Args:
            message (str | None): Prompt message displayed before reading input.
                Defaults to ``None`` for the default prompt.
            completer (Completer | None): Optional input completer. Defaults to no completion.
            exit_commands (str | Iterable[str] | None): Optional list of exit terms that end the
                prompt. Defaults to ``"q"``.
            choices (Iterable[str] | Mapping[object, str] | None): Optional selectable values.
                Mapping keys are returned while their values are displayed and accepted as input.
                Defaults to ``None``.

        Returns:
            object | False: The selected value or entered message, or ``False`` when the user
            requests to exit.

        Raises:
            ValueError: If ``choices`` are invalid or used with a custom ``completer``.
        """
        choice_items = self._choice_items(choices) if choices is not None else None
        if choice_items is not None:
            if completer is not None:
                raise ValueError("choices cannot be combined with a custom completer.")
            self.columns(dict(choice_items), numbered=True)
            completer = WordCompleter(
                [label for _, label in choice_items],
                ignore_case=True,
                sentence=True,
                match_middle=True,
            )
        if message is None:
            message = "\nYou: "
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
        while True:
            try:
                user_input = self._session.prompt(
                    message,
                    completer=completer,
                    complete_in_thread=True,
                ).strip()
            except KeyboardInterrupt, EOFError:
                return False
            if not user_input:
                self.warning("Please enter a message!")
                continue
            if user_input.casefold() in exit_commands:
                return False
            if choice_items is not None:
                if user_input.isdecimal():
                    index = int(user_input) - 1
                    if 0 <= index < len(choice_items):
                        return choice_items[index][0]
                for value, label in choice_items:
                    if user_input.casefold() == label.casefold():
                        return value
                self.warning("Select one of the listed choices by number or value.")
                continue
            return user_input

    @staticmethod
    def _choice_items(
        values: Iterable[str] | Mapping[object, str],
    ) -> tuple[tuple[object, str], ...]:
        """Normalize selectable values and enforce unambiguous input."""
        items = (
            tuple(values.items())
            if isinstance(values, Mapping)
            else tuple((value, value) for value in values)
        )
        if not items:
            raise ValueError("choices cannot be empty.")
        labels = tuple(label for _, label in items)
        if any(not label for label in labels):
            raise ValueError("choice labels cannot be empty.")
        if len({label.casefold() for label in labels}) != len(labels):
            raise ValueError("choice labels must be unique ignoring case.")
        numbers = {str(index) for index in range(1, len(items) + 1)}
        if any(label in numbers for label in labels):
            raise ValueError("choice labels cannot conflict with selection numbers.")
        return items

    def columns(
        self,
        values: Iterable[str] | Mapping[object, str],
        *,
        numbered: bool = False,
    ) -> None:
        """Write values in terminal-width-aware columns.

        Args:
            values (Iterable[str] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            numbered (bool): Whether to prefix displayed values with one-based numbers.

        Raises:
            ValueError: If no values are supplied, a label is empty, labels are duplicated, or a
                label conflicts with a displayed number.
        """
        items = self._choice_items(values)
        renderables = [
            Text(f"{index}. {label}" if numbered else label)
            for index, (_, label) in enumerate(items, start=1)
        ]
        self._console.print(
            Columns(renderables, padding=(0, 2), equal=True, expand=True, column_first=True)
        )

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
        """Write model reasoning to the terminal.

        Args:
            message (str): Complete reasoning text to write.
        """
        self._reasoning_heading()
        self._console.print(message, style="dim", markup=False, highlight=False)

    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model reasoning delta to the terminal.

        Args:
            delta (str): Incremental reasoning text to write.
            start (bool): Whether to write the reasoning heading before the delta.
        """
        if start:
            self._reasoning_heading()
        self._streamed_output = True
        self._console.print(
            delta, end="", style="dim", markup=False, highlight=False, soft_wrap=True
        )

    def _answer_heading(self) -> None:
        """Write an answer heading to the terminal."""
        self._console.print("\nAnswer:", end="", style="bold bright_green", markup=False)

    def answer(self, message: str) -> None:
        """Write a model answer to the terminal.

        Args:
            message (str): Complete answer text to write.
        """
        self._answer_heading()
        self._console.print(message, style="bold", markup=False, highlight=False)

    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Write a streamed model answer delta to the terminal.

        Args:
            delta (str): Incremental answer text to write.
            start (bool): Whether to write the answer heading before the delta.
        """
        if start:
            self._answer_heading()
        self._streamed_output = True
        self._console.print(
            delta, end="", style="bold", markup=False, highlight=False, soft_wrap=True
        )

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
        self._console.print(f"Warning: {message}", style="bold yellow", markup=False)

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
        items: list[object],
        *,
        title: str | None = None,
        prefix: str = "  ",
        columns: Iterable[str] = ("name", "description"),
        max_width: int | None = constants.TABULAR_MAX_WIDTH,
        max_rows: int | None = None,
    ) -> None:
        """Write object attributes as a table.

        Args:
            items (list[object]): Objects whose attributes provide the row values.
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
        table = Table.grid(padding=(0, 2))
        for _ in column_names:
            table.add_column(overflow="ellipsis")
        visible_items = items if max_rows is None else items[:max_rows]
        for item in visible_items:
            values = [Text(str(getattr(item, column, ""))) for column in column_names]
            if values:
                values[0].plain = f"{prefix}{values[0].plain}"
            table.add_row(*values)
        self._console.print(Constrain(table, max_width))

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

    def tool_result(self, name: str, result: str) -> None:
        """Write a serialized tool result in a user-readable form.

        Args:
            name (str): Name of the tool that produced the result.
            result (str): Serialized result returned by the tool.
        """
        try:
            value = json.loads(result)
        except json.JSONDecodeError:
            self.info(result)
            return

        if isinstance(value, str):
            self.info(value)
            return
        if isinstance(value, dict) and "error" in value and "message" in value:
            self.error(str(value["message"]))
            details = {key: item for key, item in value.items() if key not in {"error", "message"}}
            if details:
                self._console.print(JSON.from_data(details))
            return
        if name == "list_folder" and self._is_folder_result(value):
            self._display_folder_result(value)
            return
        if name == "read_text_file" and self._is_content_result(value, cached=False):
            self._display_content_result(value, identifier=value["path"])
            return
        if name in {"fetch_content", "read_cached_content"} and self._is_content_result(
            value, cached=True
        ):
            self._display_content_result(value, identifier=value["source"])
            return
        self._console.print(JSON.from_data(value))

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
    def _is_content_result(value: Any, *, cached: bool) -> bool:
        """Return whether a value is a local or cached bounded-content envelope."""
        if not isinstance(value, dict):
            return False
        required = {"content", "size_bytes", "start_byte", "end_byte", "truncated"}
        identity = {"handle", "source"} if cached else {"path"}
        return required | identity <= value.keys() and isinstance(value["content"], str)

    def _display_folder_result(self, entries: list[dict[str, Any]]) -> None:
        """Write typed folder entries as a hierarchical tree."""
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

    def _display_content_result(self, value: dict[str, Any], *, identifier: str) -> None:
        """Write bounded textual content with its source and range metadata."""
        range_text = f"bytes {value['start_byte']}–{value['end_byte']} of {value['size_bytes']}"
        metadata = Text(identifier, style="bold cyan")
        metadata.append(f" · {range_text}", style="dim")
        if value["truncated"]:
            reason = value.get("truncation_reason", "limit")
            metadata.append(f" · truncated ({reason})", style="yellow")
        if "handle" in value:
            metadata.append(f" · handle {value['handle']}", style="dim")
        self._console.print(metadata)
        self._console.print(
            Syntax(
                value["content"],
                Syntax.guess_lexer(identifier, value["content"]),
                line_numbers="start_line" in value,
                start_line=value.get("start_line", 1),
                word_wrap=True,
            )
        )

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
        except AttributeError, OSError, ValueError:
            pass
