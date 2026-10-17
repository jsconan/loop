"""Present accepted response events independently of response collection."""

from ..models import AnswerCompleted, AnswerDelta, ReasoningCompleted, ReasoningDelta, ResponseEvent
from .interaction import InteractionOutput


class ResponsePresenter:
    """Present one response without duplicating streamed text on completion.

    Args:
        output (InteractionOutput): Output surface receiving accepted events. The caller owns
            its response context and validates events before presenting them.
    """

    _output: InteractionOutput
    _reasoning_started: bool
    _answer_started: bool

    def __init__(self, output: InteractionOutput) -> None:
        self._output = output
        self._reasoning_started = False
        self._answer_started = False

    def present(self, event: ResponseEvent) -> None:
        """Render one accepted event, leaving terminal metadata and tools undisplayed.

        Args:
            event (ResponseEvent): Validated event in response order.
        """
        if isinstance(event, ReasoningDelta):
            self._output.reasoning_delta(event.text)
            self._reasoning_started = True
        elif isinstance(event, AnswerDelta):
            self._output.answer_delta(event.text)
            self._answer_started = True
        elif isinstance(event, ReasoningCompleted):
            if not self._reasoning_started:
                self._output.reasoning(event.text)
        elif isinstance(event, AnswerCompleted):
            if not self._answer_started:
                self._output.answer(event.text)
