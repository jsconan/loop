"""Define user interaction abstractions."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Any

from prompt_toolkit.completion import Completer

from .. import constants
from ..models import RAW_TOOL_RESULT_PRESENTATION, RunMetrics, ToolResultPresentationSpec


class Interaction(ABC):
    """Provide semantically classified user interaction independently of a UI."""

    @abstractmethod
    def response_context(self) -> AbstractContextManager[None]:
        """Create a presentation scope for one model response.

        Returns:
            AbstractContextManager[None]: Scope that finalizes response presentation on exit.
        """

    @abstractmethod
    def prompt(
        self,
        message: str | None = None,
        completer: Completer | None = None,
        exit_commands: str | Iterable[str] | None = None,
        choices: Iterable[str] | Mapping[object, str] | None = None,
    ) -> object | False:
        """Read a non-empty user message or an exit command.

        Args:
            message (str | None): Prompt message displayed before reading input.
                Defaults to ``None``.
            completer (Completer | None): Optional input completer. Defaults to no completion.
            exit_commands (str | Iterable[str] | None): Optional list of exit terms that end the
                prompt. Defaults to ``None``.
            choices (Iterable[str] | Mapping[object, str] | None): Optional selectable values.
                Mapping keys are returned while their values are displayed and accepted as input.
                Implementations may adapt their presentation to the catalog size. Defaults to
                ``None``.

        Returns:
            object | False: The selected value or stripped text entered by the user, or ``False``
            when the user requests to exit.
        """

    @abstractmethod
    def user(self, message: str) -> None:
        """Display a completed user message.

        Args:
            message (str): Complete user message text to display.
        """

    @abstractmethod
    def reasoning(self, message: str) -> None:
        """Display model reasoning output.

        Args:
            message (str): Complete reasoning text to display.
        """

    @abstractmethod
    def reasoning_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model reasoning delta.

        Args:
            delta (str): Incremental reasoning text to display.
            start (bool): Whether this is the first reasoning delta in the response.
        """

    @abstractmethod
    def answer(self, message: str) -> None:
        """Display model answer output.

        Args:
            message (str): Complete answer text to display.
        """

    @abstractmethod
    def answer_delta(self, delta: str, *, start: bool = False) -> None:
        """Display a streamed model answer delta.

        Args:
            delta (str): Incremental answer text to display.
            start (bool): Whether this is the first answer delta in the response.
        """

    @abstractmethod
    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message (str): Error text to display.
        """

    @abstractmethod
    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message (str): Warning text to display.
        """

    @abstractmethod
    def debug(self, value: Any) -> None:
        """Display diagnostic output.

        Args:
            value (Any): Diagnostic value to display.
        """

    @abstractmethod
    def info(self, message: str = "") -> None:
        """Display neutral status information.

        Args:
            message (str): Status text to display, or an empty string for a blank line.
        """

    @abstractmethod
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
        """Display object attributes as a table.

        Args:
            items (list[object]): Objects whose attributes provide the row values.
            title (str | None): Optional title to display above the table.
            prefix (str): Text to prepend to the first value in each row.
            columns (Iterable[str]): Attribute names to display as columns.
            max_width (int | None): Maximum table width in characters. Defaults to
                ``TABULAR_MAX_WIDTH``. Pass ``None`` to use the interaction's available width.
            max_rows (int | None): Maximum number of objects to display. Displays every object
                when unset.

        Raises:
            ValueError: If ``max_width`` is not positive or ``max_rows`` is negative.
        """

    @abstractmethod
    def columns(
        self,
        values: Iterable[str] | Mapping[object, str],
        *,
        numbered: bool = False,
    ) -> None:
        """Display values in terminal-width-aware columns.

        Args:
            values (Iterable[str] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            numbered (bool): Whether to prefix displayed values with one-based numbers.

        Raises:
            ValueError: If no values are supplied, a label is empty, labels are duplicated, or a
                label conflicts with a displayed number.
        """

    @abstractmethod
    def list(
        self,
        values: Iterable[str] | Mapping[object, str],
        *,
        numbered: bool = False,
    ) -> None:
        """Write values as a vertical list.

        Args:
            values (Iterable[str] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            numbered (bool): Whether to prefix displayed values with one-based numbers.

        Raises:
            ValueError: If no values are supplied, a label is empty, labels are duplicated, or a
                label conflicts with a displayed number.
        """

    @abstractmethod
    def tool_call(self, name: str, arguments: str) -> None:
        """Display a model-requested tool call.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """

    @abstractmethod
    def tool_result(
        self,
        result: str,
        presentation: ToolResultPresentationSpec = RAW_TOOL_RESULT_PRESENTATION,
    ) -> None:
        """Display a serialized tool result for a user.

        Args:
            result (str): Serialized result returned by the tool.
            presentation (ToolResultPresentationSpec): Semantic presentation requested by the
                tool execution. Defaults to generic raw presentation.
        """

    @abstractmethod
    def run_metrics(self, metrics: RunMetrics) -> None:
        """Display one completed agent run's durable statistics.

        Args:
            metrics (RunMetrics): Complete run statistics and current context occupancy.
        """

    @abstractmethod
    def permission(self, prompt: str, decision: str) -> None:
        """Display one originally prompted permission decision.

        Args:
            prompt (str): Exact original prompt or a replay fallback.
            decision (str): Effective decision label.
        """

    @abstractmethod
    def conversation_ended(self) -> None:
        """Display that the conversation has ended."""

    @abstractmethod
    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Ask the user to approve an operation.

        Args:
            message (str): Confirmation question to display.
            default (bool): Answer to use when the user enters no response.

        Returns:
            bool: Whether the user approved the operation.
        """
