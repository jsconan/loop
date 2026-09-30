"""Tests for bounded single-agent execution."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock

import pytest

from loop import (
    BackendConnectionError,
    BackendRepetitionError,
    InstructionsManager,
    Interaction,
    PendingToolCall,
    Response,
    ResponseMetrics,
    Session,
    SessionExecutionConflictError,
    SessionRecoveryState,
    ToolCall,
    ToolExecutionMetrics,
    Usage,
)
from loop.agent import Agent, AgentRunner
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.utils import PathHolder


def agent_runner(
    *, responses=None, max_turns=25, backend=None, options=None, instructions_manager=None
):
    """Build a runner with isolated execution collaborators."""
    backend = backend or Mock()
    agent = Agent("Assistant", tools=Mock())
    instructions = instructions_manager or InstructionsManager()
    session_manager = Mock()
    session_manager.messages = []
    session_manager.session = Session()
    session_manager.model = None
    session_manager.tokens = 0
    session_manager.context_window = None
    session_manager.execution.side_effect = nullcontext
    interaction = MagicMock(spec=Interaction)
    options = options or {
        "stream": False,
        "debug": False,
        "max_turns": max_turns,
        "prompt": True,
    }
    runner = AgentRunner(
        agent,
        backend,
        instructions,
        Mock(),
        session_manager,
        Mock(),
        Mock(),
        interaction,
        PathHolder(Path.cwd()),
        stream=options["stream"],
        debug=options["debug"],
        max_turns=options["max_turns"],
        prompt_on_recoverable_error=options["prompt"],
    )
    if responses is not None:
        runner.query = Mock(side_effect=responses)
    return runner, session_manager, interaction


def test_runner_returns_the_first_final_response():
    """A response without tool calls completes one logical agent run."""
    response = Response(answer="done", reasoning="")
    backend = Mock()
    runner, sessions, _ = agent_runner(responses=[response], backend=backend)

    result = runner.run()

    assert runner.agent.name == "Assistant"
    assert runner.backend is backend
    assert runner.max_turns == 25
    assert result.final_response is response
    assert result.turns == 1
    assert result.stop_reason == "completed"
    sessions.add_response.assert_called_once_with(response)
    sessions.record_run.assert_called_once()
    assert result.metrics is sessions.record_run.call_args.args[2]


def test_runner_rejects_owned_sessions_before_tool_execution():
    """A conflicting execution lease prevents model and side-effecting tool work."""
    call = ToolCall(call_id="call", name="mutate", arguments="{}")
    runner, sessions, _ = agent_runner(
        responses=[Response(answer="", reasoning="", tool_calls=(call,), items=(call,))]
    )
    sessions.execution.side_effect = SessionExecutionConflictError("already executing")

    with pytest.raises(SessionExecutionConflictError, match="already executing"):
        runner.run()

    runner.query.assert_not_called()
    runner.agent.tools.call_with_timing.assert_not_called()


def test_runner_reconfigures_subsequent_runs():
    """The runner validates and owns live execution preferences."""
    runner, _, _ = agent_runner(responses=[Response(answer="done", reasoning="")])
    backend = Mock()

    runner.backend = backend
    runner.debug = True
    runner.stream = True
    runner.max_turns = 0
    runner.prompt_on_recoverable_error = False

    assert runner.backend is backend
    assert runner.debug is True
    assert runner.stream is True
    assert runner.max_turns == 0
    assert runner.prompt_on_recoverable_error is False
    with pytest.raises(ValueError, match="non-negative"):
        runner.max_turns = -1


def test_runner_retries_one_detected_generation_loop_without_prompting():
    """A stopped looping response is retried once with recovery guidance."""
    error = BackendRepetitionError(
        "looping", provider="test", operation="stream_response", response_started=True
    )
    final = Response(answer="done", reasoning="")
    runner, _, interaction = agent_runner(responses=[error])
    runner._query = Mock(return_value=final)

    result = runner.run()

    assert result.final_response is final
    assert result.turns == 1
    assert runner.query.call_count == 1
    assert runner.query.call_count == 1
    assert runner._query.call_args.args == ("repetition",)
    interaction.confirm.assert_not_called()
    interaction.warning.assert_called_once_with(
        "The model response was stopped after repetitive output. Retrying once..."
    )


def test_runner_replaces_agent():
    """Assigning a replacement agent makes it available through the public property."""
    runner, _, _ = agent_runner(responses=[])
    replacement = Agent("Replacement", tools=Mock())

    runner.agent = replacement

    assert runner.agent is replacement


def test_runner_records_run_and_tool_execution_traces():
    """Configured telemetry covers run boundaries and exact tool requests and responses."""
    response = Response(
        answer="",
        reasoning="",
        tool_calls=(ToolCall(call_id="call", name="demo", arguments='{"value":1}'),),
    )
    final = Response(answer="done", reasoning="")
    runner, _, _ = agent_runner(responses=[response, final])
    runner.agent.tools.call_with_timing.return_value = ('{"ok":true}', 0.2)
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        result = runner.run()
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert result.stop_reason == "completed"
    assert [record.event_name for record in adapter.records] == [
        "agent.run.started",
        "tool.request",
        "tool.response",
        "agent.run.completed",
    ]
    started, request, response_record, completed = adapter.records
    assert started.span_id == completed.span_id
    assert request.span_id == response_record.span_id
    assert request.parent_span_id == started.span_id
    assert request.span_id != started.span_id


def test_runner_recovers_a_model_boundary_without_adding_user_input():
    """Recovery requeries existing context and completes the interrupted run."""
    response = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[response])

    sessions.recovery_state = SessionRecoveryState(action="query_model")
    interaction.confirm.return_value = True

    recovery = runner.recover_session()

    assert recovery.result.final_response is response
    assert recovery.result.turns == 1
    assert len(recovery.result.metrics.model_calls) == recovery.result.turns
    assert recovery.pending is False
    sessions.add_response.assert_called_once_with(response)


def test_runner_recovers_unstarted_tools_before_requerying():
    """Definitely unstarted calls execute once with their stable call identifier."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    runner.agent.tools.call_with_timing.return_value = (
        '{"ok": true, "result": "done"}',
        0.5,
    )

    sessions.recovery_state = SessionRecoveryState(
        action="execute_tools",
        pending_calls=(PendingToolCall(call=call, status="not_started"),),
    )
    interaction.confirm.return_value = True

    runner.recover_session()

    sessions.add_tool_call_event.assert_called_once_with("call")
    runner.agent.tools.call_with_timing.assert_called_once_with(
        "echo",
        "{}",
        interaction=interaction,
        instructions_manager=runner.instructions_manager,
        permission_manager=runner.permission_manager,
        call_id="call",
        execution_started=ANY,
    )
    runner.agent.tools.call_with_timing.call_args.kwargs["execution_started"]()
    sessions.record_tool_execution_started.assert_called_once_with("call")
    assert sessions.add_tool_call.call_args.kwargs["succeeded"] is True


def test_runner_skips_completed_calls_in_a_partially_recovered_batch():
    """Mixed recovery executes only calls whose durable results are still absent."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    runner.agent.tools.call_with_timing.return_value = ('{"ok": true}', 0.1)
    completed = ToolCall(call_id="completed", name="read", arguments="{}")
    pending = ToolCall(call_id="pending", name="write", arguments="{}")

    sessions.recovery_state = SessionRecoveryState(
        action="execute_tools",
        pending_calls=(
            PendingToolCall(call=completed, status="result_available"),
            PendingToolCall(call=pending, status="not_started"),
        ),
    )
    interaction.confirm.return_value = True

    runner.recover_session()

    callback = runner.agent.tools.call_with_timing.call_args.kwargs["execution_started"]
    callback()
    sessions.record_tool_execution_started.assert_called_once_with("pending")


def test_runner_resolves_uncertain_tools_without_retry_by_default():
    """An uncertain side effect becomes an explicit model-visible error unless retry is approved."""
    final = Response(answer="reconciled", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    interaction.confirm.side_effect = [True, False]
    call = ToolCall(call_id="call", name="mutate", arguments="{}")

    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    runner.recover_session()

    runner.agent.tools.call_with_timing.assert_not_called()
    assert "outcome is unknown" in sessions.add_tool_call.call_args.kwargs["output"]
    assert interaction.confirm.call_args_list[-1].args == (
        "Tool 'mutate' may already have run. Retry it?",
    )


def test_runner_retries_an_uncertain_tool_only_after_explicit_approval():
    """Explicit approval retries an uncertain call without duplicating its presentation event."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    interaction.confirm.side_effect = [True, True]
    runner.agent.tools.call_with_timing.return_value = ('{"ok": false}', 0.1)
    call = ToolCall(call_id="call", name="mutate", arguments="{}")

    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    runner.recover_session()

    sessions.add_tool_call_event.assert_not_called()
    callback = runner.agent.tools.call_with_timing.call_args.kwargs["execution_started"]
    callback()
    sessions.record_tool_execution_started.assert_called_once_with("call")


def test_runner_finalizes_a_completed_response_without_another_model_call():
    """A persisted final assistant response only receives missing run bookkeeping."""
    runner, sessions, interaction = agent_runner(responses=[])

    sessions.recovery_state = SessionRecoveryState(action="finalize_run")
    interaction.confirm.return_value = True

    recovery = runner.recover_session()

    runner.query.assert_not_called()
    assert recovery.result.stop_reason == "completed"
    sessions.record_run.assert_called_once()


def test_runner_cancels_when_response_recovery_is_exhausted():
    """An unavailable response returns cancellation without committing model output."""
    runner, sessions, _ = agent_runner(responses=[None])

    result = runner.run()

    assert result.final_response is None
    assert result.turns == 0
    assert result.stop_reason == "cancelled"
    sessions.add_response.assert_not_called()
    sessions.record_run.assert_called_once()


def test_runner_reports_and_cancels_on_a_request_preparation_failure():
    """A request that cannot be prepared, such as an oversized instruction chain, is reported."""
    runner, sessions, interaction = agent_runner(
        instructions_manager=InstructionsManager(project_instructions="required", max_bytes=1)
    )

    result = runner.run()

    assert result.final_response is None
    assert result.stop_reason == "cancelled"
    sessions.add_response.assert_not_called()
    problem = interaction.report.call_args.args[0]
    assert problem.code == "agent.query_failed"
    assert "instructions exceed the configured budget" in problem.detail


def test_runner_does_not_reclassify_an_unexpected_value_error():
    """A ValueError outside known preparation boundaries remains visible to the caller."""
    runner, _, interaction = agent_runner(responses=[ValueError("malformed response")])

    with pytest.raises(ValueError, match="malformed response"):
        runner.run()

    interaction.report.assert_not_called()


def test_runner_counts_a_retried_model_request_as_one_completed_turn():
    """Recoverable request retries do not count as separate completed agent turns."""
    error = BackendConnectionError("offline", provider="test", operation="respond")
    final = Response(answer="done", reasoning="")
    runner, _, interaction = agent_runner(responses=[error, final])
    interaction.confirm.return_value = True

    result = runner.run()

    assert result.turns == 1
    assert len(result.metrics.model_calls) == 1
    assert runner.query.call_count == 2
    interaction.confirm.assert_called_once_with("Retry the complete response?", default=False)


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
    """One approved safety window preserves the run's total completed-turn count."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response_with_tools = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    response_without_tools = Response(answer="finished", reasoning="")
    interaction = MagicMock(spec=Interaction)
    interaction.confirm.return_value = True
    backend = Mock()
    agent = Agent("Assistant", tools=Mock())
    instructions = InstructionsManager()
    session_manager = Mock()
    session_manager.messages = []
    session_manager.session = Session()
    session_manager.model = None
    session_manager.tokens = 0
    session_manager.context_window = None
    session_manager.execution.side_effect = nullcontext
    runner = AgentRunner(
        agent,
        backend,
        instructions,
        Mock(),
        session_manager,
        Mock(),
        Mock(),
        interaction,
        PathHolder(Path.cwd()),
        max_turns=1,
    )
    runner.query = Mock(side_effect=[response_with_tools, response_without_tools])
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.5, succeeded=True)
    runner.handle_tool_calls = Mock(side_effect=[(execution,), ()])
    result = runner.run()
    assert result.final_response is response_without_tools
    assert result.turns == 2
    assert result.stop_reason == "completed"
    assert interaction.confirm.call_count == 1
    assert len(result.metrics.model_calls) == result.turns


def test_runner_counts_turns_across_multiple_safety_confirmations():
    """Repeated approvals reset only the safety interval, not total run accounting."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response_with_tools = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    final = Response(answer="finished", reasoning="")
    runner, sessions, interaction = agent_runner(
        responses=[response_with_tools, response_with_tools, response_with_tools, final],
        max_turns=1,
    )
    interaction.confirm.return_value = True
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.1, succeeded=True)
    runner.handle_tool_calls = Mock(side_effect=[(execution,), (execution,), (execution,), ()])
    adapter = MemoryTelemetryAdapter()
    telemetry = Telemetry(adapter, flush_seconds=0.01)
    set_telemetry(telemetry)

    try:
        result = runner.run()
        assert telemetry.close(1)
    finally:
        set_telemetry(None)

    assert result.turns == 4
    assert len(result.metrics.model_calls) == 4
    assert sessions.add_response.call_count == 4
    assert interaction.confirm.call_count == 3
    assert all(
        call.args == ("Agent has reached the 1-turn safety limit. Do you want to continue?",)
        for call in interaction.confirm.call_args_list
    )
    completed = next(
        record for record in adapter.records if record.event_name == "agent.run.completed"
    )
    assert completed.attributes["turns"] == result.turns


def test_runner_declines_after_a_confirmed_safety_window():
    """Declining a later safety prompt reports turns from every completed window."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, interaction = agent_runner(
        responses=[response, response, response, response], max_turns=2
    )
    interaction.confirm.side_effect = [True, False]
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.1, succeeded=True)
    runner.handle_tool_calls = Mock(return_value=(execution,))

    result = runner.run()

    assert result.turns == 4
    assert result.stop_reason == "max_turns"
    assert len(result.metrics.model_calls) == 4
    assert sessions.add_response.call_count == 4
    assert interaction.confirm.call_count == 2
