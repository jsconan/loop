"""Define results produced by an agent run."""

from pydantic import BaseModel

from ..models import AgentRunStopReason, Response, RunMetrics


class AgentRunResult(BaseModel):
    """Describe how one agent run ended.

    Args:
        final_response (Response | None): Last completed model response, or ``None`` when no
            response completed.
        turns (int): Number of completed model turns in the run.
        stop_reason (AgentRunStopReason): Reason control returned to the caller.
        metrics (RunMetrics | None): Persisted run statistics after completion.
    """

    final_response: Response | None
    turns: int
    stop_reason: AgentRunStopReason
    metrics: RunMetrics | None = None
