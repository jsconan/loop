"""Tests for agent run result models."""

from loop import AgentRunResult, AgentRunStopReason, Response
from loop.agent import AgentRunRecoveryRequired


def test_agent_run_result_records_completion_state():
    """Run results retain the final response, turn count, and stop reason."""
    response = Response(answer="done", reasoning="")

    result = AgentRunResult(final_response=response, turns=2, stop_reason="completed")

    assert result.final_response is response
    assert result.turns == 2
    assert result.stop_reason is AgentRunStopReason.COMPLETED
    assert result.stop_reason == "completed"
    assert result.model_dump(mode="json")["stop_reason"] == "completed"


def test_agent_run_recovery_required_marks_a_durable_run_for_later_recovery():
    """Recoverable agent failures have a distinct control-flow signal."""
    failure = AgentRunRecoveryRequired()

    assert isinstance(failure, Exception)
