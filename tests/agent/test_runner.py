"""Tests for bounded single-agent execution."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from loop import Agent, AgentRunner, InstructionsManager, Interaction, Response, ToolCall


def agent_runner(*, responses, max_turns=25):
    """Build a runner with isolated execution collaborators."""
    backend = Mock()
    agent = Agent(
        "Assistant",
        backend,
        InstructionsManager(),
        Mock(),
        Mock(),
    )
    session_manager = Mock()
    interaction = MagicMock(spec=Interaction)
    runner = AgentRunner(
        agent,
        session_manager,
        Mock(),
        Mock(),
        interaction,
        lambda: Path.cwd(),
        max_turns=max_turns,
    )
    runner.query = Mock(side_effect=responses)
    return runner, session_manager, interaction


def test_runner_returns_the_first_final_response():
    """A response without tool calls completes one logical agent run."""
    response = Response(answer="done", reasoning="")
    runner, sessions, _ = agent_runner(responses=[response])

    result = runner.run()

    assert runner.agent.name == "Assistant"
    assert runner.max_turns == 25
    assert result.final_response is response
    assert result.turns == 1
    assert result.stop_reason == "completed"
    sessions.add_response.assert_called_once_with(response)


def test_runner_cancels_when_response_recovery_is_exhausted():
    """An unavailable response returns cancellation without committing model output."""
    runner, sessions, _ = agent_runner(responses=[None])

    result = runner.run()

    assert result.final_response is None
    assert result.turns == 0
    assert result.stop_reason == "cancelled"
    sessions.add_response.assert_not_called()


def test_runner_stops_repeated_tool_calls_at_the_safety_limit():
    """Repeated tool requests persist their last result and stop before another model call."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, interaction = agent_runner(responses=[response, response], max_turns=2)
    runner.handle_tool_calls = Mock(return_value=True)

    result = runner.run()

    assert result.final_response is response
    assert result.turns == 2
    assert result.stop_reason == "max_turns"
    assert sessions.add_response.call_count == 2
    assert runner.handle_tool_calls.call_count == 2
    interaction.warning.assert_called_once_with(
        "Agent stopped after reaching the 2-turn safety limit."
    )


def test_runner_rejects_a_non_positive_turn_limit():
    """Runner construction rejects limits that could never execute an agent."""
    with pytest.raises(ValueError, match="greater than zero"):
        agent_runner(responses=[], max_turns=0)
