"""Define user interaction abstractions."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer

from .. import constants
from ..models import (
    AnswerCompleted,
    AnswerDelta,
    ConversationItem,
    Message,
    Reasoning,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ResponseEvent,
    ToolCall,
    ToolCallCompleted,
    Usage,
)

if TYPE_CHECKING:
    from ..session.models import Compaction


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
    ) -> str | False:
        """Read a non-empty user message or an exit command.

        Args:
            message (str | None): Prompt message displayed before reading input.
                Defaults to ``None``.
            completer (Completer | None): Optional input completer. Defaults to no completion.
            exit_commands (str | Iterable[str] | None): Optional list of exit terms that end the
                prompt. Defaults to ``None``.

        Returns:
            str | False: The stripped text entered by the user, or ``False`` when the user
            requests to exit.
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
    def tool_call(self, name: str, arguments: str) -> None:
        """Display a model-requested tool call.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """

    @abstractmethod
    def tool_result(self, name: str, result: str) -> None:
        """Display a serialized tool result for a user.

        Args:
            name (str): Name of the tool that produced the result.
            result (str): Serialized result returned by the tool.
        """

    @abstractmethod
    def token_usage(
        self,
        model: str | None,
        context_tokens: int | None,
        context_window: int | None,
    ) -> None:
        """Display the current model and context occupancy.

        Args:
            model (str | None): Current model identifier, when known.
            context_tokens (int | None): Number of tokens currently in the context, when known.
            context_window (int | None): Maximum context size in tokens, when known.
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

    def response(self, events: Iterable[ResponseEvent], *, debug: bool = False) -> Response:
        """Display and collect normalized response events.

        Args:
            events (Iterable[ResponseEvent]): Response events to display and collect.
            debug (bool): Whether to display every raw response event.

        Returns:
            Response: The collected answer, reasoning, tool calls, items, usage, and model.
        """
        reasoning = ""
        answer = ""
        tool_calls = []
        items = ()
        usage = None
        model = None
        structured_output = None
        reasoning_started = False
        answer_started = False

        with self.response_context():
            for event in events:
                if debug:
                    self.debug(event)

                if isinstance(event, ReasoningDelta):
                    self.reasoning_delta(event.text, start=not reasoning_started)
                    reasoning_started = True
                    continue

                if isinstance(event, AnswerDelta):
                    self.answer_delta(event.text, start=not answer_started)
                    answer_started = True
                    continue

                if isinstance(event, ReasoningCompleted):
                    reasoning = event.text
                    self.reasoning(event.text)
                    continue

                if isinstance(event, AnswerCompleted):
                    answer = event.text
                    self.answer(event.text)
                    continue

                if isinstance(event, ToolCallCompleted):
                    tool_calls.append(event.call)
                    continue

                if isinstance(event, ResponseCompleted):
                    items = event.items
                    usage = event.usage
                    model = event.model
                    answer = event.answer
                    reasoning = event.reasoning
                    structured_output = event.structured_output

        return Response(
            answer=answer,
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
            items=items,
            usage=usage or Usage(),
            model=model,
            structured_output=structured_output,
        )

    def history(
        self,
        items: Iterable[ConversationItem],
        compactions: Iterable[Compaction] = (),
    ) -> None:
        """Display persisted conversation items as prior interaction.

        Args:
            items (Iterable[ConversationItem]): Ordered conversation items to replay.
            compactions (Iterable[Compaction]): Ordered compaction checkpoints to replay at their
                full-history boundaries.
        """

        def display_compaction(compaction: Compaction) -> None:
            before = compaction.input_tokens_before
            after = compaction.input_tokens_after
            if before is not None and after is not None and before != after:
                self.info(f"Compacted session context from {before:,} to {after:,} tokens.")
            else:
                self.info("Compacted session context.")

        checkpoints = iter(compactions)
        checkpoint = next(checkpoints, None)
        for index, item in enumerate(items):
            while checkpoint is not None and checkpoint.boundary == index:
                display_compaction(checkpoint)
                checkpoint = next(checkpoints, None)
            if isinstance(item, Message):
                display = self.user if item.role == "user" else self.answer
                display(item.content)
                continue
            if isinstance(item, Reasoning):
                self.reasoning(item.content)
                continue
            if isinstance(item, ToolCall):
                self.tool_call(item.name, item.arguments)
        while checkpoint is not None:
            display_compaction(checkpoint)
            checkpoint = next(checkpoints, None)
