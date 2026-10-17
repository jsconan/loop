"""Validate and collect normalized backend events without presentation or persistence."""

from .backend.errors import BackendResponseError
from .models import (
    AnswerCompleted,
    AnswerDelta,
    ReasoningCompleted,
    ReasoningDelta,
    Response,
    ResponseCompleted,
    ResponseEvent,
    ToolCall,
    ToolCallCompleted,
)


class ResponseCollector:
    """Collect one response using its explicit terminal event as authoritative content.

    Initialize an empty collector for one event stream. Incremental events mark partial output
    and collect tool calls; only the terminal event supplies final text and metadata. Unknown
    events before completion are ignored for compatibility with backend extensions.
    """

    _completion: ResponseCompleted | None
    _tool_calls: list[ToolCall]
    _response_observed: bool

    def __init__(self) -> None:
        self._completion = None
        self._tool_calls = []
        self._response_observed = False

    def add(self, event: ResponseEvent) -> None:
        """Validate and collect one normalized event.

        Args:
            event (ResponseEvent): Next event in backend order.

        Raises:
            BackendResponseError: If an event follows the explicit response completion.
        """
        if self._completion is not None:
            reason = (
                "more than one response completion"
                if isinstance(event, ResponseCompleted)
                else "an event after response completion"
            )
            raise BackendResponseError(
                f"Backend emitted {reason}.",
                provider="backend",
                operation="collect_response",
                response_started=True,
            )
        if isinstance(
            event,
            (ReasoningDelta, AnswerDelta, ReasoningCompleted, AnswerCompleted, ToolCallCompleted),
        ):
            self._response_observed = True
        if isinstance(event, ToolCallCompleted):
            self._tool_calls.append(event.call)
        elif isinstance(event, ResponseCompleted):
            self._completion = event.model_copy()

    def finish(self) -> Response:
        """Return the completed response after the event source has been exhausted.

        Returns:
            Response: Authoritative final content, ordered tool calls, and terminal metadata.

        Raises:
            BackendResponseError: If no explicit response completion was collected.
        """
        if self._completion is None:
            raise BackendResponseError(
                "Backend response ended without an explicit completion.",
                provider="backend",
                operation="collect_response",
                response_started=self._response_observed,
            )
        completion = self._completion
        return Response(
            answer=completion.answer,
            reasoning=completion.reasoning,
            tool_calls=tuple(self._tool_calls),
            items=completion.items,
            usage=completion.usage,
            model=completion.model,
            structured_output=completion.structured_output,
        )
