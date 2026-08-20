"""Tests for agent run result models."""

from loop import AgentRunResult, Response


def test_agent_run_result_records_completion_state():
    """Run results retain the final response, turn count, and stop reason."""
    response = Response(answer="done", reasoning="")

    result = AgentRunResult(final_response=response, turns=2, stop_reason="completed")

    assert result.final_response is response
    assert result.turns == 2
    assert result.stop_reason == "completed"
