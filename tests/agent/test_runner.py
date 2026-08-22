"""Tests for bounded single-agent execution."""

from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from loop import (
    Agent,
    AgentRunner,
    InstructionsManager,
    Interaction,
    Response,
    ResponseMetrics,
    Session,
    ToolCall,
    ToolExecutionMetrics,
    Usage,
)


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
    session_manager.messages = []
    session_manager.session = Session()
    session_manager.model = None
    session_manager.tokens = 0
    session_manager.context_window = None
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
    sessions.record_run.assert_called_once()
    assert result.metrics is sessions.record_run.call_args.args[2]


def test_runner_cancels_when_response_recovery_is_exhausted():
    """An unavailable response returns cancellation without committing model output."""
    runner, sessions, _ = agent_runner(responses=[None])

    result = runner.run()

    assert result.final_response is None
    assert result.turns == 0
    assert result.stop_reason == "cancelled"
    sessions.add_response.assert_not_called()
    sessions.record_run.assert_called_once()


def test_runner_stops_repeated_tool_calls_at_the_safety_limit():
    """Repeated tool requests persist their last result and stop before another model call."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, interaction = agent_runner(responses=[response, response], max_turns=2)
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.1, succeeded=True)
    runner.handle_tool_calls = Mock(return_value=(execution,))

    result = runner.run()

    assert result.final_response is response
    assert result.turns == 2
    assert result.stop_reason == "max_turns"
    assert sessions.add_response.call_count == 2
    assert runner.handle_tool_calls.call_count == 2
    sessions.record_run.assert_called_once()
    interaction.warning.assert_called_once_with(
        "Agent stopped after reaching the 2-turn safety limit."
    )


def test_runner_does_not_present_partial_usage_as_an_exact_aggregate():
    """A missing per-call token field keeps the corresponding run aggregate unknown."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    first = Response(
        answer="",
        reasoning="",
        tool_calls=(call,),
        usage=Usage(input_tokens=10, output_tokens=4),
        metrics=ResponseMetrics(duration_seconds=1),
    )
    final = Response(
        answer="done",
        reasoning="",
        usage=Usage(input_tokens=12),
        metrics=ResponseMetrics(duration_seconds=2),
    )
    runner, sessions, _ = agent_runner(responses=[first, final])
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.5, succeeded=True)
    runner.handle_tool_calls = Mock(side_effect=[(execution,), ()])

    runner.run()

    metrics = sessions.record_run.call_args.args[2]
    assert metrics.usage.input_tokens == 22
    assert metrics.usage.output_tokens is None
    assert metrics.model_duration_seconds == 3
    assert metrics.tool_duration_seconds == 0.5
    assert metrics.active_duration_seconds == 3.5


def test_runner_rejects_a_negative_turn_limit():
    """Runner construction rejects negative turn limits."""
    with pytest.raises(ValueError, match="non-negative"):
        agent_runner(responses=[], max_turns=-1)


def test_runner_continues_after_max_turns_when_user_affirms():
    """When the safety limit is reached and the user confirms, another round starts."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response_with_tools = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    response_without_tools = Response(answer="finished", reasoning="")
    interaction = MagicMock(spec=Interaction)
    interaction.confirm.return_value = True
    backend = Mock()
    agent = Agent("Assistant", backend, InstructionsManager(), Mock(), Mock())
    session_manager = Mock()
    session_manager.messages = []
    session_manager.session = Session()
    session_manager.model = None
    session_manager.tokens = 0
    session_manager.context_window = None
    runner = AgentRunner(
        agent,
        session_manager,
        Mock(),
        Mock(),
        interaction,
        lambda: Path.cwd(),
        max_turns=1,
    )
    runner.query = Mock(side_effect=[response_with_tools, response_without_tools])
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.5, succeeded=True)
    runner.handle_tool_calls = Mock(side_effect=[(execution,), ()])
    result = runner.run()
    assert result.final_response is response_without_tools
    assert result.turns == 1
    assert result.stop_reason == "completed"
    assert interaction.confirm.call_count == 1
