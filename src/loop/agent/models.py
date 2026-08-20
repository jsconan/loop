"""Define results produced by an agent run."""

from typing import Literal

from pydantic import BaseModel

from ..models import Response

type AgentRunStopReason = Literal["completed", "cancelled", "max_turns"]


class AgentRunResult(BaseModel):
    """Describe how one agent run ended.

    Args:
        final_response (Response | None): Last completed model response, or ``None`` when no
            response completed.
        turns (int): Number of completed model turns in the run.
        stop_reason (AgentRunStopReason): Reason control returned to the caller.
    """

    final_response: Response | None
    turns: int
    stop_reason: AgentRunStopReason
