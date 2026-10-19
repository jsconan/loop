"""Marshal execution-thread interaction onto its owning UI event loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import Any

from prompt_toolkit.completion import Completer

from .. import constants
from ..errors import Problem
from ..interaction import AsyncInteractionInput, Interaction
from ..interaction.models import ListMarker
from ..models import RAW_TOOL_RESULT_PRESENTATION, RunMetrics, ToolResultPresentationSpec
from ..utils import ChoiceItem
from .models import (
    EVENT_CONFIRM,
    EVENT_OUTPUT,
    EVENT_PROMPT,
    EVENT_TERMINAL,
    EVENT_TERMINAL_EXIT,
    InteractionCall,
)


class MailboxInteraction(Interaction, AsyncInteractionInput):
    """Route interaction through a controller while retaining direct setup interaction.

    Args:
        source (Interaction): Actual UI surface, owned by the controller's event loop.
    """

    _source: Interaction
    _dispatch: Callable[[InteractionCall], Any] | None
    _dispatch_async: Callable[[InteractionCall], Awaitable[Any]] | None

    def __init__(self, source: Interaction) -> None:
        self._source = source
        self._dispatch = None
        self._dispatch_async = None

    def bind(
        self,
        dispatch: Callable[[InteractionCall], Any] | None,
        dispatch_async: Callable[[InteractionCall], Awaitable[Any]] | None = None,
    ) -> None:
        """Bind active mailbox dispatch or restore direct setup interaction.

        Args:
            dispatch (Callable[[InteractionCall], Any] | None): Active controller dispatcher,
                or ``None`` outside its execution lifecycle.
            dispatch_async (Callable[[InteractionCall], Awaitable[Any]] | None): Optional async
                dispatcher for input requests. Defaults to direct source input when unbound.
        """
        self._dispatch = dispatch
        self._dispatch_async = dispatch_async

    def _call(
        self,
        function: Callable[..., Any],
        *args: Any,
        kind=EVENT_OUTPUT,
        **kwargs: Any,
    ) -> Any:
        """Execute locally during setup or wait for an acknowledged UI operation."""
        if self._dispatch is None:
            return function(*args, **kwargs)
        return self._dispatch(InteractionCall(function, args, kwargs, kind))

    async def _call_async(
        self,
        function: Callable[..., Any],
        *args: Any,
        kind=EVENT_PROMPT,
        **kwargs: Any,
    ) -> Any:
        """Request input through the controller's correlated async interaction channel."""
        if self._dispatch_async is None:
            return await function(*args, **kwargs)
        return await self._dispatch_async(InteractionCall(function, args, kwargs, kind))

    @contextmanager
    def response_context(self) -> Generator[None]:
        """Finalize a response presentation scope on its owning UI thread.

        Yields:
            None: Control while response output is being presented.
        """
        yield from self._scope(self._source.response_context)

    @contextmanager
    def terminal_context(self) -> Generator[None]:
        """Reserve the terminal before starting a directly interactive operation.

        Yields:
            None: Control while ordinary mailbox input is suspended.

        Raises:
            KeyboardInterrupt: If shutdown revokes the pending terminal handoff.
        """
        yield from self._scope(self._source.terminal_context, EVENT_TERMINAL, EVENT_TERMINAL_EXIT)

    def _scope(
        self,
        function: Callable,
        kind=EVENT_OUTPUT,
        exit_kind=EVENT_OUTPUT,
    ) -> Generator[None]:
        """Marshal context entry and exception-aware cleanup to the UI owner."""
        scope = self._call(function, kind=kind)
        self._call(scope.__enter__)
        try:
            yield
        except BaseException as error:
            if not self._call(
                scope.__exit__,
                type(error),
                error,
                error.__traceback__,
                kind=exit_kind,
            ):
                raise
        else:
            self._call(scope.__exit__, None, None, None, kind=exit_kind)

    def prompt(
        self,
        message: str | None = None,
        completer: Completer | None = None,
        exit_commands: str | Iterable[str] | None | constants.Omit = constants.OMIT,
        choices: Iterable[str | ChoiceItem] | Mapping[object, str] | None = None,
        index: Iterable[str] | Mapping[object, str] | None = None,
        default: object | None = None,
        secret: bool = False,
    ) -> object | False:
        """Request input through the source interaction's native asynchronous prompt.

        Args:
            message (str | None): Question to display, or ``None`` for the default prompt.
            completer (Completer | None): Optional completion provider.
            exit_commands (str | Iterable[str] | None | constants.Omit): Exit terms. Omission
                preserves the source UI's default; ``None`` disables exit terms.
            choices (Iterable[str | ChoiceItem] | Mapping[object, str] | None): Accepted choices,
                or ``None`` for unrestricted text.
            index (Iterable[str] | Mapping[object, str] | None): Optional choice indexes.
            default (object | None): Empty-input default, or ``None`` to require input.
            secret (bool): Whether input must be masked. Defaults to ``False``.

        Returns:
            object | False: Selected value or text, or ``False`` on terminal exit.

        Raises:
            ValueError: If the supplied prompt choices or completion options are invalid.
            KeyboardInterrupt: If the controller stops before input is provided.
        """
        options = {} if exit_commands is constants.OMIT else {"exit_commands": exit_commands}
        return self._call(
            self._source.prompt,
            message,
            completer=completer,
            choices=choices,
            index=index,
            default=default,
            secret=secret,
            kind=EVENT_PROMPT,
            **options,
        )

    async def prompt_async(
        self,
        message: str | None = None,
        completer: Completer | None = None,
        exit_commands: str | Iterable[str] | None = constants.OMIT,
        choices: Iterable[str | ChoiceItem] | Mapping[object, str] | None = None,
        index: Iterable[str] | Mapping[object, str] | None = None,
        default: object | None = None,
        secret: bool = False,
        initial: str | None = None,
    ) -> object | bool:
        """Read input through the UI owner without blocking the conversation event loop.

        Args:
            message (str | None): Prompt message, or ``None`` for the default prompt.
            completer (Completer | None): Optional completion provider.
            exit_commands (str | Iterable[str] | None | constants.Omit): Exit terms, or omission
                to preserve the source UI default.
            choices (Iterable[str | ChoiceItem] | Mapping[object, str] | None): Optional choices.
            index (Iterable[str] | Mapping[object, str] | None): Optional choice indexes.
            default (object | None): Value returned for empty input, or ``None`` to require input.
            secret (bool): Whether input must be masked. Defaults to ``False``.
            initial (str | None): Initial editor text. Defaults to ``None``.

        Returns:
            object | bool: Selected value or text, or ``False`` when input requests exit.

        Raises:
            ValueError: If the prompt choices or completion options are invalid.
            CancelledError: If the input task is cancelled.
        """
        options = {} if exit_commands is constants.OMIT else {"exit_commands": exit_commands}
        return await self._call_async(
            self._source.prompt_async,
            message,
            completer=completer,
            choices=choices,
            index=index,
            default=default,
            secret=secret,
            initial=initial,
            kind=EVENT_PROMPT,
            **options,
        )

    def confirm(self, message: str, *, default: bool = False) -> bool:
        """Request explicit approval through the source interaction's native confirmation UI.

        Args:
            message (str): Exact confirmation question.
            default (bool): Empty-input decision. Defaults to ``False``.

        Returns:
            bool: Whether the user explicitly approved the operation.

        Raises:
            KeyboardInterrupt: If the controller stops before approval is provided.
        """
        return self._call(self._source.confirm, message, default=default, kind=EVENT_CONFIRM)

    async def confirm_async(self, message: str, *, default: bool = False) -> bool:
        """Request explicit approval through the controller's correlated async channel.

        Args:
            message (str): Exact confirmation question.
            default (bool): Empty-input decision. Defaults to ``False``.

        Returns:
            bool: Whether the user explicitly approved the operation.

        Raises:
            CancelledError: If the confirmation task is cancelled.
        """
        return await self._call_async(
            self._source.confirm_async,
            message,
            default=default,
            kind=EVENT_CONFIRM,
        )

    def user(self, message: str) -> None:
        """Display a completed user message.

        Args:
            message (str): Complete user message text to display.
        """
        self._call(self._source.user, message)

    def reasoning(self, message: str) -> None:
        """Display model reasoning output.

        Args:
            message (str): Complete reasoning text to display.
        """
        self._call(self._source.reasoning, message)

    def reasoning_delta(self, delta: str) -> None:
        """Display a streamed model reasoning delta.

        Args:
            delta (str): Incremental reasoning text to display.
        """
        self._call(self._source.reasoning_delta, delta)

    def answer(self, message: str) -> None:
        """Display model answer output.

        Args:
            message (str): Complete answer text to display.
        """
        self._call(self._source.answer, message)

    def answer_delta(self, delta: str) -> None:
        """Display a streamed model answer delta.

        Args:
            delta (str): Incremental answer text to display.
        """
        self._call(self._source.answer_delta, delta)

    def report(self, problem: Problem) -> None:
        """Display a structured application problem.

        Args:
            problem (Problem): Safe problem to present consistently.
        """
        self._call(self._source.report, problem)

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message (str): Error text to display.
        """
        self._call(self._source.error, message)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message (str): Warning text to display.
        """
        self._call(self._source.warning, message)

    def debug(self, value: Any) -> None:
        """Display diagnostic output.

        Args:
            value (Any): Diagnostic value to display.
        """
        self._call(self._source.debug, value)

    def info(self, message: str = "") -> None:
        """Display neutral status information.

        Args:
            message (str): Status text to display, or an empty string for a blank line.
        """
        self._call(self._source.info, message)

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
        """Display mapping fields or object attributes as a table.

        Args:
            items (Iterable[object | Mapping[str, Any]]): Rows whose mappings or attributes
                provide the displayed values.
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
        self._call(
            self._source.table,
            items,
            title=title,
            prefix=prefix,
            columns=columns,
            max_width=max_width,
            max_rows=max_rows,
        )

    def columns(
        self,
        values: Iterable[object] | Mapping[object, str],
        *,
        marker: ListMarker = "plain",
    ) -> None:
        """Display values in terminal-width-aware columns.

        Args:
            values (Iterable[object] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            marker (ListMarker): Prefix style for each displayed
                value. Defaults to ``"plain"``.
        """
        self._call(self._source.columns, values, marker=marker)

    def list(
        self,
        values: Iterable[object] | Mapping[object, str],
        *,
        marker: ListMarker = "plain",
    ) -> None:
        """Display values as a vertical list.

        Args:
            values (Iterable[object] | Mapping[object, str]): Values to display. Mapping values are
                displayed while their keys remain available to callers that also retain the
                mapping.
            marker (ListMarker): Prefix style for each displayed
                value. Defaults to ``"plain"``.
        """
        self._call(self._source.list, values, marker=marker)

    def json(self, value: Any) -> None:
        """Display a structured value as formatted JSON.

        Args:
            value (Any): JSON-compatible value to display.
        """
        self._call(self._source.json, value)

    def tree(self, entries: Iterable[Mapping[str, str]]) -> None:
        """Display typed path entries as a hierarchical tree.

        Args:
            entries (Iterable[Mapping[str, str]]): Entries containing ``path`` and ``type``
                fields, where type is ``"file"`` or ``"folder"``.
        """
        self._call(self._source.tree, entries)

    def content(
        self,
        text: str,
        *,
        identifier: str,
        start_line: int | None = None,
    ) -> None:
        """Display syntax-highlighted textual content.

        Args:
            text (str): Textual content to display.
            identifier (str): Source name used to select a syntax lexer.
            start_line (int | None): First source line number, or ``None`` to omit line numbers.
        """
        self._call(self._source.content, text, identifier=identifier, start_line=start_line)

    def tool_call(self, name: str, arguments: str) -> None:
        """Display a model-requested tool call.

        Args:
            name (str): Name of the requested tool.
            arguments (str): JSON arguments supplied to the tool.
        """
        self._call(self._source.tool_call, name, arguments)

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
        self._call(self._source.tool_result, result, presentation)

    def run_metrics(self, metrics: RunMetrics) -> None:
        """Display one completed agent run's durable statistics.

        Args:
            metrics (RunMetrics): Complete run statistics and current context occupancy.
        """
        self._call(self._source.run_metrics, metrics)

    def permission(self, prompt: str, decision: str) -> None:
        """Display one originally prompted permission decision.

        Args:
            prompt (str): Exact original prompt or a replay fallback.
            decision (str): Effective decision label.
        """
        self._call(self._source.permission, prompt, decision)

    def conversation_ended(self) -> None:
        """Display that the conversation has ended."""
        self._call(self._source.conversation_ended)
