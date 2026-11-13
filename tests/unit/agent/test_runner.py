"""Tests for bounded single-agent execution."""

import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread, get_ident
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, Mock

import pytest
from openai.types.responses import ResponseFunctionToolCall

from loop import (
    BUILTIN_TOOLS,
    AnswerDelta,
    ApprovalChoice,
    BackendConnectionError,
    BackendError,
    BackendNotFoundError,
    BackendRepetitionError,
    ConsoleInteraction,
    ContextReference,
    InstructionsManager,
    Interaction,
    Message,
    ModelSelectionError,
    OpenAIBackend,
    OwnedToolTask,
    PendingToolCall,
    PermissionManager,
    Problem,
    ProblemException,
    Response,
    ResponseCompleted,
    ResponseMetrics,
    RunControl,
    RunInterrupted,
    RunStopRequested,
    RuntimeEnvironment,
    Session,
    SessionExecutionConflictError,
    SessionManager,
    SessionRecoveryState,
    SQLiteSessionStore,
    ToolCall,
    ToolCallCompleted,
    ToolExecutionMetrics,
    ToolRegistration,
    ToolRegistry,
    ToolResult,
    Usage,
    edit_text_file,
    read_text_file,
)
from loop.agent import Agent, AgentRunner, AgentRunRecoveryRequired, AgentRunResult
from loop.agent import runner as agent_runner_module
from loop.conversation.interaction import ApprovalInteraction, MailboxInteraction
from loop.execution import CommandExecutionService
from loop.execution import jobs as command_jobs
from loop.execution.sandbox import CommandProcessResult, SandboxOutcome
from loop.execution.sandbox.linux.backend import LinuxNativeBackend
from loop.session.models import RunCompletedEvent, RunStartedEvent
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.tooling import (
    FileObservations,
    Tool,
    ToolContext,
    ToolExecutionAdmissionRejected,
    ToolRuntimeSettings,
)
from loop.tooling.models import ToolExecutionInterrupted, ToolExecutionNotStarted
from loop.tools.system import control_command_terminal, run_command
from loop.utils import PathHolder, VirtualPath, await_value_async, content_digest, utc_now
from tests.scheduling import deliver_callbacks

from ..execution.sandbox.helpers import logical_private_name


def agent_runner(
    *,
    responses=None,
    max_turns=25,
    backend=None,
    options=None,
    instructions_manager=None,
    run_control=None,
):
    """Build a runner with isolated execution collaborators."""
    backend = backend or Mock()
    if isinstance(backend, Mock):
        backend.prepare_context.side_effect = list
    agent = Agent("Assistant", tools=Mock())
    instructions = instructions_manager or InstructionsManager()
    session_manager = Mock()
    model_selection = Mock()
    model_selection.record_response_async = AsyncMock()
    model_selection.synchronize_session_async = AsyncMock()
    for async_name, sync_name in (
        ("add_response_async", "add_response"),
        ("add_tool_call_async", "add_tool_call"),
        ("add_tool_call_event_async", "add_tool_call_event"),
        ("record_tool_execution_started_async", "record_tool_execution_started"),
        ("record_run_async", "record_run"),
    ):
        setattr(
            session_manager, async_name, AsyncMock(side_effect=getattr(session_manager, sync_name))
        )
    session_manager.messages = []
    session_manager.model_context = []
    session_manager.model_context_async = AsyncMock(return_value=[])
    session_manager.session = Session()
    run = SimpleNamespace(run_id="run-id", started_at=utc_now())

    def begin_run():
        """Add the durable run identity expected by failure and recovery tests."""
        if session_manager.session.active_run_id is None:
            session_manager.session.events.append(
                RunStartedEvent(
                    id="run-event",
                    created_at=run.started_at,
                    run_id=run.run_id,
                    started_at=run.started_at,
                )
            )
        return run

    session_manager.begin_run.side_effect = begin_run
    session_manager.begin_run_async = AsyncMock(side_effect=begin_run)
    session_manager.retry_pending = False
    session_manager.record_run_failure = Mock(
        side_effect=session_manager.session.record_run_failure
    )
    session_manager.record_run_failure_async = AsyncMock(
        side_effect=session_manager.session.record_run_failure
    )
    session_manager.begin_run_retry = Mock(side_effect=session_manager.session.begin_run_retry)
    session_manager.begin_run_retry_async = AsyncMock(
        side_effect=session_manager.session.begin_run_retry
    )
    session_manager.model = None
    session_manager.tokens = 0
    session_manager.context_window = None
    session_manager.recovery_state = None
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
        model_selection,
        Mock(),
        interaction,
        PathHolder(Path.cwd()),
        stream=options["stream"],
        debug=options["debug"],
        max_turns=options["max_turns"],
        prompt_on_recoverable_error=options["prompt"],
        run_control=run_control,
    )
    if responses is not None:
        runner.query = Mock(side_effect=responses)
    return runner, session_manager, interaction


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("superseded", [False, True])
def test_yolo_recovery_preserves_unknown_tool_outcome_without_prompt(asynchronous, superseded):
    """No-prompt recovery leaves a possibly executed tool unresolved for an explicit action."""
    runner, sessions, interaction = agent_runner()
    runner.permission_manager.yolo = True
    call = ToolCall(call_id="uncertain", name="mutate", arguments="{}")
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
        superseded=superseded,
    )
    if asynchronous:
        result = asyncio.run(runner.recover_session_async())
    else:
        result = runner.recover_session()
    assert result.pending is True
    runner.agent.tools.call_with_timing.assert_not_called()
    interaction.confirm.assert_not_called()
    runner.permission_manager.confirmation_decision.assert_called_once_with(certain=False)
    assert "may already have run" in interaction.warning.call_args.args[0]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_yolo_retries_a_recoverable_model_request_at_most_once(asynchronous):
    """Known no-effect model failures receive one automatic retry and then need recovery."""
    error = BackendConnectionError("temporary", provider="test", operation="query")
    runner, sessions, interaction = agent_runner(responses=[error, error])
    sessions.begin_run()
    runner.permission_manager.yolo = True
    if asynchronous:
        prompt = MagicMock(spec=ConsoleInteraction)
        prompt.confirm_async = AsyncMock(return_value=True)
        runner._interaction = prompt
        runner.query_async = AsyncMock(side_effect=[error, error])
        with pytest.raises(AgentRunRecoveryRequired):
            asyncio.run(runner._query_with_recovery_async())
        assert runner.query_async.await_count == 2
        prompt.confirm_async.assert_awaited_once()
        sessions.record_run_failure_async.assert_awaited_once()
    else:
        interaction.confirm.return_value = True
        with pytest.raises(AgentRunRecoveryRequired):
            runner._query_with_recovery()
        assert runner.query.call_count == 2
        interaction.confirm.assert_called_once()
        sessions.record_run_failure.assert_called_once()
    runner.permission_manager.confirmation_decision.assert_called_once_with(certain=False)


def test_runner_confirm_async_requires_native_async_interaction():
    """Async agent confirmation never calls a synchronous prompt on the owner loop."""
    runner, _, interaction = agent_runner()

    with pytest.raises(ProblemException) as captured:
        asyncio.run(runner._confirm_async("Continue?", default=False))

    assert captured.value.problem.code == "interaction.async_required"
    interaction.confirm.assert_not_called()


def test_runner_confirm_async_uses_native_async_interaction():
    """Async agent confirmation awaits the interaction's native async method."""
    runner, _, _ = agent_runner()
    interaction = ConsoleInteraction()
    interaction.confirm = Mock()
    interaction.confirm_async = AsyncMock(return_value=True)
    runner._interaction = interaction

    assert asyncio.run(runner._confirm_async("Continue?", default=True)) is True

    interaction.confirm_async.assert_awaited_once_with("Continue?", default=True)
    interaction.confirm.assert_not_called()


def file_runner(tmp_path, backend, *, registry=None, sessions=None, run_control=None):
    """Build a real file-tool runtime with deterministic model and interaction boundaries."""
    if isinstance(backend, Mock):
        backend.prepare_context.side_effect = list
    interaction = MagicMock(spec=Interaction)
    interaction.response_context.side_effect = nullcontext
    interaction.prompt.return_value = ApprovalChoice.ONCE
    sessions = sessions or SessionManager(interaction=interaction)
    instructions = InstructionsManager()
    instructions.set_runtime_environment(
        RuntimeEnvironment(
            working_directory=tmp_path, temporary_directory=sessions.temporary_directory
        )
    )
    permissions = PermissionManager(tmp_path, interaction=interaction)
    permissions.bind_temporary_directory(sessions.temporary_directory)
    if registry is not None:
        registry.bind_session_manager(sessions)
    agent = Agent("Assistant", tools=registry or ToolRegistry(BUILTIN_TOOLS))
    selection = Mock()
    selection.effective = "local-model"
    selection.record_response_async = AsyncMock()
    selection.synchronize_session_async = AsyncMock()
    return AgentRunner(
        agent,
        backend,
        instructions,
        permissions,
        sessions,
        selection,
        Mock(),
        interaction,
        PathHolder(tmp_path),
        run_control=run_control,
    )


def model_calls(*calls):
    """Emit one complete deterministic model response containing the requested calls."""
    return [*(ToolCallCompleted(call=call) for call in calls), ResponseCompleted(items=calls)]


@pytest.mark.parametrize("interrupted", [False, True])
def test_foreign_owned_result_keeps_effectful_tool_in_recovery(tmp_path, monkeypatch, interrupted):
    """An unclaimable result never becomes a completed receipt after a tool effect."""
    effects = []

    async def foreign():
        """Remain available to the dispatcher that owns this handle."""
        return "foreign result"

    handle = OwnedToolTask(foreign())
    handle.reserve()

    async def interrupted_effect():
        """Interrupt the runner after the effect, then return foreign-owned work."""
        effects.append("entered")
        try:
            asyncio.get_running_loop().stop()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return handle

    def unavailable_effect():
        """Return foreign work after the effect when the runner is unavailable."""
        effects.append("entered")
        return handle

    effect = interrupted_effect if interrupted else unavailable_effect
    registry = ToolRegistry([effect])
    if not interrupted:
        monkeypatch.setattr(registry, "_sync_runner", Mock(side_effect=RuntimeError("unavailable")))
    call = ToolCall(call_id="effect-call", name=effect.__name__, arguments="{}")
    model = Mock()
    model.get_response.return_value = model_calls(call)
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("perform the action")
    try:
        result = runner.run()
        assert result.stop_reason == "recovery_required"
        assert runner.session_manager.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in runner.session_manager.messages)
        assert effects == ["entered"]
        assert model.get_response.call_count == 1
        with asyncio.Runner() as owner:
            task = handle.start(owner.get_loop())
            assert owner.run(await_value_async(task)) == "foreign result"
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_pending_foreign_work_keeps_entered_tool_in_recovery(tmp_path, async_mode, nested):
    """A pending foreign task cannot become a completed receipt or automatic replay."""
    effects = []

    async def foreign():
        """Perform an effect only when the original owner runs its loop."""
        effects.append("foreign")
        return "foreign result"

    with asyncio.Runner() as owner:
        handle = OwnedToolTask(foreign())
        task = handle.start(owner.get_loop())
        outer = owner.get_loop().create_future()
        outer.set_result(task)

        def effect():
            """Perform an effect and return work attached to the other loop."""
            effects.append("tool")
            return outer if nested else handle

        registry = ToolRegistry([effect])
        call = ToolCall(call_id="effect-call", name="effect", arguments="{}")
        model = Mock()
        model.get_response.return_value = model_calls(call)

        async def response(*args, **kwargs):
            """Emit the same model request through the async boundary."""
            for event in model_calls(call):
                yield event

        model.get_response_async.side_effect = response
        runner = file_runner(tmp_path, model, registry=registry)
        runner._compaction.compact_if_needed_async = AsyncMock()
        runner.session_manager.add_user_message("perform the action")
        try:
            result = asyncio.run(runner.run_async()) if async_mode else runner.run()
            assert result.stop_reason == "recovery_required"
            assert runner.session_manager.recovery_pending
            assert not any(isinstance(item, ToolResult) for item in runner.session_manager.messages)
            assert effects == ["tool"]
            assert not task.done()
            assert owner.run(await_value_async(task)) == "foreign result"
        finally:
            registry.close()
    assert effects == ["tool", "foreign"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_started_borrowed_coroutine_keeps_entered_tool_in_recovery(tmp_path, async_mode):
    """A coroutine already owned by another task cannot become a completed receipt."""
    effects = []

    async def foreign():
        """Wait for the original owner before producing its effect."""
        await ready.wait()
        effects.append("foreign")

    with asyncio.Runner() as owner:
        ready = asyncio.Event()
        coroutine = foreign()
        task = owner.get_loop().create_task(coroutine)
        owner.run(asyncio.sleep(0))

        def effect():
            """Return work that belongs to the other task."""
            effects.append("tool")
            return coroutine

        registry = ToolRegistry([effect])
        call = ToolCall(call_id="effect-call", name="effect", arguments="{}")
        model = Mock()
        model.get_response.return_value = model_calls(call)

        async def response(*args, **kwargs):
            """Emit the same model request through the async boundary."""
            for event in model_calls(call):
                yield event

        model.get_response_async.side_effect = response
        runner = file_runner(tmp_path, model, registry=registry)
        runner._compaction.compact_if_needed_async = AsyncMock()
        runner.session_manager.add_user_message("perform the action")
        try:
            result = asyncio.run(runner.run_async()) if async_mode else runner.run()
            assert result.stop_reason == "recovery_required"
            assert runner.session_manager.recovery_pending
            assert not any(isinstance(item, ToolResult) for item in runner.session_manager.messages)
            assert effects == ["tool"]
            assert not task.done()
            ready.set()
            owner.run(await_value_async(task))
        finally:
            registry.close()
    assert effects == ["tool", "foreign"]


def test_unhosted_borrowed_future_keeps_entered_tool_in_recovery(tmp_path, monkeypatch):
    """A failed runner cannot publish a result while borrowed work remains unsettled."""
    effects = []
    with asyncio.Runner() as owner:
        future = owner.get_loop().create_future()
        future.add_done_callback(lambda _future: effects.append("foreign"))

        def effect():
            """Perform an effect and return an unresolved borrowed future."""
            effects.append("tool")
            return future

        registry = ToolRegistry([effect])
        monkeypatch.setattr(registry, "_sync_runner", Mock(side_effect=RuntimeError("unavailable")))
        call = ToolCall(call_id="effect-call", name="effect", arguments="{}")
        model = Mock()
        model.get_response.return_value = model_calls(call)
        runner = file_runner(tmp_path, model, registry=registry)
        runner.session_manager.add_user_message("perform the action")
        try:
            result = runner.run()
            assert result.stop_reason == "recovery_required"
            assert runner.session_manager.recovery_pending
            assert not any(isinstance(item, ToolResult) for item in runner.session_manager.messages)
            assert effects == ["tool"]
            assert not future.done()
            future.set_result("foreign result")
            assert owner.run(await_value_async(future)) == "foreign result"
        finally:
            registry.close()
    assert effects == ["tool", "foreign"]


@pytest.mark.parametrize("sync_tool", [False, True])
def test_async_run_preserves_unknown_outcome_for_foreign_owned_work(tmp_path, sync_tool):
    """Async dispatch retains recovery after an effect and an unclaimable result."""
    effects = []

    async def foreign():
        """Remain available to the dispatcher that reserved this work."""
        return "foreign"

    handle = OwnedToolTask(foreign())
    handle.reserve()

    def sync_effect():
        """Produce an effect before returning someone else's work."""
        effects.append("entered")
        return handle

    async def async_effect():
        """Produce an effect before returning someone else's work."""
        effects.append("entered")
        return handle

    effect = sync_effect if sync_tool else async_effect
    registry = ToolRegistry([effect])
    call = ToolCall(call_id="effect-call", name=effect.__name__, arguments="{}")

    async def response(*args, **kwargs):
        """Emit the model request once."""
        for event in model_calls(call):
            yield event

    model = Mock()
    model.get_response_async.side_effect = response
    runner = file_runner(tmp_path, model, registry=registry)
    runner._compaction.compact_if_needed_async = AsyncMock()
    runner.session_manager.add_user_message("perform the action")
    try:
        result = asyncio.run(runner.run_async())
        assert result.stop_reason == "recovery_required"
        assert runner.session_manager.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in runner.session_manager.messages)
        assert effects == ["entered"]
        assert model.get_response_async.call_count == 1
        with asyncio.Runner() as owner:
            assert owner.run(await_value_async(handle.start(owner.get_loop()))) == "foreign"
    finally:
        registry.close()


def test_renamed_terminal_tool_executes_input_without_saving_it(tmp_path):
    """Registered aliases retain argument privacy while the original bytes reach one dispatch."""
    secret = "private-input-marker"
    call = ToolCall(
        call_id="terminal",
        name="renamed_terminal",
        arguments=json.dumps({"handle": "a" * 32, "action": "write", "content": secret}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    service = Mock()
    service.has_active_jobs.return_value = False
    service.control_command_terminal.return_value = {"terminal": True}
    registry = ToolRegistry(
        [ToolRegistration(control_command_terminal, name="renamed_terminal")],
        execution_service=service,
    )
    assert registry.view().private_argument_names == frozenset({"renamed_terminal"})
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Send terminal input")

    result = runner.run()

    assert result.turns == 2
    assert service.control_command_terminal.call_args.args[2] == "write"
    assert service.control_command_terminal.call_args.args[3] == secret
    assert secret not in runner.session_manager.session.serialize()
    assert (
        next(
            item.arguments
            for item in runner.session_manager.messages
            if isinstance(item, ToolCall) and item.name == "renamed_terminal"
        )
        == "{}"
    )
    runner.interaction.tool_call.assert_called_once_with("renamed_terminal", "{}")


def test_runner_uses_existing_executable_grant_in_one_tool_call(tmp_path, monkeypatch):
    """A full grant cache does not force an extra model response or approval for a managed tool."""
    tools = tmp_path.parent / "fixture-tools"
    tools.mkdir()
    executable = tools / "builder"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    native = Mock()
    native.system_read_roots = ()
    native.installed_tool_roots.side_effect = lambda path: ((path.parent,), ())
    native.managed_tool_root.return_value = True
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(native, Mock())
    lookup = ToolContext(None, "resolve_executable")
    references = [
        service.resolve_executable(lookup, "builder", str(tmp_path))["grant_reference"]
        for _ in range(128)
    ]
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps({"command": "builder", "executable_grant": references[0]}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=service,
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Run the builder")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"] and output["result"]["boundary"] == "sandbox"
    assert result.turns == model.get_response.call_count == 2
    assert native.run.call_count == 1
    runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize("spelling", ["None", "nOnE", "null", "NULL"])
def test_runner_normalizes_optional_capability_null_spellings_in_one_call(tmp_path, spelling):
    """A malformed optional null does not require another model response or expand access."""
    native = Mock()
    native.system_read_roots = ()
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    host = Mock()
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps(
            {
                "command": "printf ready",
                "read_only": True,
                "read_roots": spelling,
                "write_roots": spelling,
                "executable_grant": spelling,
                "egress_origins": spelling,
            }
        ),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(native, host),
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Print ready")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"] and output["result"]["boundary"] == "sandbox"
    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    request = native.run.call_args.args[0]
    assert request.read_roots == request.write_roots == request.egress_origins == ()
    assert request.network is False
    runner.interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


def test_runner_executes_reused_provider_id_in_distinct_responses(tmp_path, monkeypatch):
    """Each response starts its own job even when the provider repeats an identifier."""
    native = Mock()
    native.system_read_roots = ()
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    service = CommandExecutionService(native, Mock())
    sessions = SessionManager()
    registry = ToolRegistry([run_command], execution_service=service, session_manager=sessions)
    calls = [
        ToolCall(
            call_id="reused",
            name="run_command",
            arguments=json.dumps(
                {
                    "command": command,
                    "wait_seconds": 1,
                }
            ),
        )
        for command in ("printf first", "printf second")
    ]
    model = Mock()
    model.get_response.side_effect = [
        model_calls(calls[0]),
        model_calls(calls[1]),
        [ResponseCompleted(answer="done")],
    ]
    runner = file_runner(tmp_path, model, registry=registry, sessions=sessions)
    runner.permission_manager.bind_temporary_directory(sessions.temporary_directory)
    runner.instructions_manager.set_runtime_environment(
        RuntimeEnvironment(
            working_directory=tmp_path, temporary_directory=sessions.temporary_directory
        )
    )

    class Worker:
        """Run each retained worker deterministically in the owning test thread."""

        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            """Settle the synthetic worker immediately."""
            self.target()

        def join(self, timeout):
            """Leave an already settled worker unchanged."""

    monkeypatch.setattr(command_jobs.threading, "Thread", Worker)
    runner.session_manager.add_user_message("Run both commands")
    outcome = runner.run()
    results = [
        json.loads(item.output) for item in sessions.messages if isinstance(item, ToolResult)
    ]
    assert outcome.turns == 3
    assert len(results) == 2 and all(result["ok"] for result in results)
    assert [result["result"]["exit_code"] for result in results] == [0, 0]
    assert [call.args[0].source for call in native.run.call_args_list] == [
        "printf first",
        "printf second",
    ]


@pytest.mark.parametrize("exit_code", [126, 127])
def test_runner_reports_executable_failure_without_replaying_or_extra_tool_calls(
    tmp_path, exit_code
):
    """A completed failure returns its uncertainty in the original tool result."""
    native = Mock()
    native.system_read_roots = ()
    native.run.return_value = CommandProcessResult(
        SandboxOutcome.COMPLETED,
        exit_code=exit_code,
        stderr="fixture command unavailable",
    )
    host = Mock()
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps({"command": "exit 127", "read_only": True}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(native, host),
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Run the project task")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["problem"]["code"] == "process.nonzero_exit"
    metadata = output["problem"]["metadata"]
    assert metadata["exit_code"] == exit_code
    assert metadata["sandbox_diagnostics"]["possible_effects"] is True
    assert "may already have had effects" in metadata["executable_hint"]
    assert "host_offer" not in metadata
    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    runner.interaction.prompt.assert_not_called()
    host.run_host_command.assert_not_called()


@pytest.mark.parametrize("command", ["builder", "cd child; builder"])
def test_runner_discovers_managed_executable_without_extra_turns(tmp_path, monkeypatch, command):
    """A managed executable on an absolute PATH needs one call before or after cd."""
    tools = tmp_path.parent / "fixture-tools"
    tools.mkdir()
    executable = tools / "builder"
    executable.write_text("tool", encoding="utf-8")
    executable.chmod(0o755)
    (tmp_path / "child").mkdir()
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    native = Mock()
    native.system_read_roots = ()
    native.installed_tool_roots.side_effect = lambda path: ((path.parent,), ())
    native.managed_tool_root.return_value = True
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    call = ToolCall(
        call_id="command", name="run_command", arguments=json.dumps({"command": command})
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(native, Mock()),
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Run the project task")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"], output
    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    assert native.run.call_args.args[0].automatic_tool_reads == (tools,)
    runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize(
    "selection",
    [
        "direct",
        "identified-binary",
        "identified-file",
        "indirect",
        "interpreter",
        "argument",
        "name",
        "nested",
        "identified",
        "extra",
        "first-chain",
        "wrapped-chain",
        "forwarded",
        "forwarded-binary",
        "forwarded-path",
        "forwarded-path-chain",
        "literal-forward",
        "literal-fixed",
        "literal-fixed-chain",
        "literal-options",
        "literal-prefix-chain",
        "literal-chain",
        "forwarded-chain",
        "quoted",
    ],
)
def test_runner_executes_scoped_executable_in_the_original_call(
    shebang_root, monkeypatch, selection
):
    """Scoped aliases and off-PATH interpreters need one tool call and no extra model response."""
    tmp_path = shebang_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lookup = workspace / "lookup"
    lookup.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    links = tmp_path / "links"
    links.mkdir()
    executable = installation / "actual"
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o755)
    alias = links / "alias"
    alias.symlink_to(installation, target_is_directory=True)
    (lookup / "fixture-task").symlink_to(
        executable if selection == "identified-file" else alias / executable.name
    )
    launcher = workspace / ("launcher with space" if selection == "quoted" else "launcher")
    launcher.write_text(
        "#!/bin/sh\nexec ./forwarder ./lookup/fixture-task\n"
        if selection == "literal-fixed-chain"
        else "#!/bin/sh\nexec ./lookup/fixture-task --fixed\n"
        if selection == "literal-fixed"
        else '#!/bin/sh\nexec ./forwarder ./lookup/fixture-task "$@"\n'
        if selection == "literal-prefix-chain"
        else '#!/bin/sh\nexec ./lookup/fixture-task --fixture "$@"\n'
        if selection == "literal-options"
        else '#!/bin/sh\nexec ./lookup/fixture-task "$@"\n'
        if selection == "literal-forward"
        else '#!/bin/sh\nexec ./forwarder "$@"\n'
        if selection == "literal-chain"
        else '#!/bin/sh\nexec "$@"\n'
        if selection in {"forwarded-chain", "forwarded-path-chain"}
        else '#!/bin/sh\nexec \t"$1"\n'
        if selection == "forwarded"
        else '#!/bin/sh\nexec "$1"\n'
        if selection in {"forwarded-binary", "forwarded-path"}
        else f"#!{lookup / 'fixture-task'}\n"
        if selection
        in {"interpreter", "identified", "extra", "first-chain", "wrapped-chain", "quoted"}
        else "fixture-task",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    if selection in {
        "forwarded-chain",
        "forwarded-path-chain",
        "literal-chain",
        "literal-prefix-chain",
        "literal-fixed-chain",
    }:
        forwarder = workspace / "forwarder"
        forwarder.write_text(
            '#!/bin/sh\nexec ./lookup/fixture-task "$@"\n'
            if selection == "literal-chain"
            else '#!/bin/sh\nexec "$@"\n',
            encoding="utf-8",
        )
        forwarder.chmod(0o755)
    if selection == "literal-fixed-chain":
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        other = unrelated / "actual"
        other.write_text("unrelated tool", encoding="utf-8")
        other.chmod(0o755)
        (lookup / "unrelated").symlink_to(other)
    if selection in {"argument", "name", "nested", "forwarded"}:
        task = workspace / "task"
        interpreter = lookup / "fixture-task"
        if selection == "nested":
            nested = workspace / "nested"
            nested.write_text(f"#!{interpreter}\n", encoding="utf-8")
            nested.chmod(0o755)
            interpreter = nested
        task.write_text(f"#!{interpreter}\n", encoding="utf-8")
        task.chmod(0o755)
    empty = workspace / "empty-path"
    empty.mkdir()
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(workspace), str(lookup)))
        if selection == "forwarded-path-chain"
        else str(
            empty
            if selection
            in {
                "interpreter",
                "argument",
                "name",
                "nested",
                "identified",
                "extra",
                "first-chain",
                "wrapped-chain",
                "forwarded",
                "forwarded-binary",
                "forwarded-chain",
                "literal-forward",
                "literal-fixed",
                "literal-fixed-chain",
                "literal-options",
                "literal-prefix-chain",
                "literal-chain",
                "quoted",
            }
            else lookup
        ),
    )
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    native = Mock()
    native.system_read_roots = (
        (Path("/bin"), Path("/usr/bin"))
        if selection
        in {
            "forwarded",
            "forwarded-binary",
            "forwarded-path",
            "forwarded-path-chain",
            "forwarded-chain",
            "literal-forward",
            "literal-fixed",
            "literal-fixed-chain",
            "literal-options",
            "literal-prefix-chain",
            "literal-chain",
        }
        else ()
    )
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps(
            {
                "command": (
                    "./lookup/fixture-task"
                    if selection in {"direct", "identified-binary", "identified-file"}
                    else "'./launcher with space' && :"
                    if selection == "quoted"
                    else "./launcher && :"
                    if selection == "first-chain"
                    else "exec ./launcher; :"
                    if selection == "wrapped-chain"
                    else "./launcher task"
                    if selection == "name"
                    else "./launcher ./task"
                    if selection in {"argument", "nested", "forwarded"}
                    else "./launcher ./forwarder ./lookup/fixture-task"
                    if selection == "forwarded-chain"
                    else "./launcher ./lookup/fixture-task"
                    if selection == "forwarded-binary"
                    else "./launcher fixture-task"
                    if selection == "forwarded-path"
                    else "./launcher forwarder fixture-task"
                    if selection == "forwarded-path-chain"
                    else "./launcher ./lookup/unrelated"
                    if selection == "literal-fixed-chain"
                    else "./launcher ignored-argument"
                    if selection == "literal-fixed"
                    else "./launcher"
                ),
                "read_only": True,
            }
            | (
                {}
                if selection
                in {
                    "identified",
                    "identified-binary",
                    "identified-file",
                    "first-chain",
                    "wrapped-chain",
                    "quoted",
                    "forwarded",
                    "forwarded-binary",
                    "forwarded-path",
                    "forwarded-path-chain",
                    "forwarded-chain",
                    "literal-forward",
                    "literal-fixed",
                    "literal-fixed-chain",
                    "literal-options",
                    "literal-prefix-chain",
                    "literal-chain",
                }
                else {"read_roots": [str(extra if selection == "extra" else installation)]}
            )
        ),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(native, Mock()),
    )
    runner = file_runner(workspace, model, registry=registry)
    runner.interaction.prompt.return_value = "approve"
    runner.session_manager.add_user_message("Run the project task")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"], output
    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    assert runner.interaction.prompt.call_count == 1
    assert native.run.call_args.args[0].read_aliases == (
        () if selection == "identified-file" else (alias,)
    )
    assert native.run.call_args.args[0].read_roots == (
        (extra, installation) if selection == "extra" else (installation,)
    )


@pytest.mark.parametrize("selection", ["project", "direct", "script"])
def test_runner_prepares_selected_runtime_without_an_extra_turn(
    shebang_root, monkeypatch, selection
):
    """One tool call combines a selected runtime scope and execution."""
    tmp_path = shebang_root
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = workspace / ".venv"
    commands = environment / "bin"
    commands.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    interpreter = runtime / "engine"
    interpreter.write_text("fixture", encoding="utf-8")
    interpreter.chmod(0o755)
    (commands / "python").symlink_to(interpreter)
    (environment / "pyvenv.cfg").write_text(
        f"home = {runtime}\nimplementation = CPython\nversion_info = 3.14.2\n", encoding="utf-8"
    )
    (workspace / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    script = workspace / "fixture-task"
    script.write_text(f"#!{commands / 'python'}\n", encoding="utf-8")
    script.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    executable = tools / "uv"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    native = Mock()
    native.system_read_roots = ()
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    monkeypatch.setattr(
        type(native),
        "automatic_tool_read_roots",
        lambda _self, _workspace, _roots, identities: (
            (runtime,) if any(identity[1] == commands / "python" for identity in identities) else ()
        ),
        raising=False,
    )
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps(
            {
                "command": (
                    "uv run --no-sync pytest"
                    if selection == "project"
                    else "./.venv/bin/python -c pass"
                    if selection == "direct"
                    else "./fixture-task"
                ),
                "read_only": True,
            }
        ),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(native, Mock()),
    )
    runner = file_runner(workspace, model, registry=registry)
    runner.interaction.prompt.return_value = "approve"
    runner.session_manager.add_user_message("Run the project task")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"], output
    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    assert runner.interaction.prompt.call_count == 1
    assert native.run.call_args.args[0].read_roots == (runtime,)


@pytest.mark.parametrize("selection", ["direct", "forwarded", "safe_options"])
def test_runner_prepares_env_interpreter_in_the_original_call(tmp_path, monkeypatch, selection):
    """An env interpreter or shell wrapper needs no model-facing discovery workaround."""
    forwarded = selection != "direct"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    installation = tmp_path / "installation"
    installation.mkdir()
    executable = installation / "fixture-runtime"
    executable.write_text("fixture", encoding="utf-8")
    executable.chmod(0o755)
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(installation, target_is_directory=True)
    tools = workspace / "tools"
    tools.mkdir()
    (tools / "fixture-engine").symlink_to(alias / executable.name)
    task = workspace / "task"
    task.write_text("#!/usr/bin/env fixture-engine\n", encoding="utf-8")
    task.chmod(0o755)
    if forwarded:
        (tools / "fixture-shell").symlink_to("/bin/sh")
        launcher = workspace / "launcher"
        header = (
            "#!/usr/bin/env -S fixture-shell -e -u\n"
            if selection == "safe_options"
            else "#!/usr/bin/env fixture-shell\n"
        )
        launcher.write_text(f'{header}exec "$1"\n', encoding="utf-8")
        launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    native = Mock()
    native.system_read_roots = (Path("/bin"), Path("/usr/bin"))
    native.run.return_value = CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps({"command": "./launcher ./task" if forwarded else "./task"}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(workspace),
        execution_service=CommandExecutionService(native, Mock()),
    )
    runner = file_runner(workspace, model, registry=registry)
    runner.interaction.prompt.return_value = "approve"
    runner.session_manager.add_user_message("Run the task")

    result = runner.run()

    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    assert runner.interaction.prompt.call_count == 1
    assert native.run.call_args.args[0].read_roots == (installation,)


def test_runner_waits_for_one_configured_command_without_an_extra_turn(tmp_path, monkeypatch):
    """A longer configured budget completes in one call with a controlled clock."""
    native = Mock()
    native.system_read_roots = (Path("/bin"), Path("/usr/bin"))
    clock = [100.0]

    def complete_after_virtual_runtime(request, **_kwargs):
        """Complete one simulated command after advancing beyond the old default."""
        assert request.deadline == 235
        clock[0] += 65
        return CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0)

    native.run.side_effect = complete_after_virtual_runtime
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps({"command": "printf ready", "read_only": True}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=CommandExecutionService(native, Mock()),
        settings=ToolRuntimeSettings(command_timeout=75, command_hard_timeout=75),
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Run the task")

    monkeypatch.setattr(
        "loop.execution.coordinator.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    result = runner.run()

    assert result.turns == model.get_response.call_count == 2
    assert len(result.metrics.tools) == native.run.call_count == 1
    assert native.run.call_args.args[0].deadline == 235
    assert clock[0] == 165
    runner.interaction.prompt.assert_not_called()


def test_runner_ignores_unidentified_runtime_metadata_in_the_original_call(tmp_path, monkeypatch):
    """An unrelated tool completes without an extra model response or approval."""
    tools = tmp_path / "bin"
    tools.mkdir()
    executable = tools / "fixture-processor"
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o755)
    (tmp_path / "pyvenv.cfg").symlink_to(tmp_path / "unrelated")
    home = tmp_path.parent / "fixture-home"
    home.mkdir()
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.setattr(
        "loop.execution.facade.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.is_protected_external_read",
        lambda path: logical_private_name(path, tmp_path.parent),
    )
    monkeypatch.setattr(
        "loop.execution.sandbox.linux.paths.pwd.getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(home)),
    )
    monkeypatch.setattr("loop.execution.sandbox.linux.backend.runtime_read_roots", lambda: ())
    monkeypatch.setattr(LinuxNativeBackend, "file_creation_mask", 0o077)
    native = LinuxNativeBackend(tmp_path / "management.json")
    native.capability_failure = Mock(return_value=None)
    native.run = Mock(return_value=CommandProcessResult(SandboxOutcome.COMPLETED, exit_code=0))
    service = CommandExecutionService(native, Mock())
    call = ToolCall(
        call_id="command",
        name="run_command",
        arguments=json.dumps({"command": executable.name, "read_only": True}),
    )
    model = Mock()
    model.get_response.side_effect = [model_calls(call), [ResponseCompleted(answer="done")]]
    registry = ToolRegistry(
        [run_command],
        permission_manager=PermissionManager(tmp_path),
        execution_service=service,
    )
    runner = file_runner(tmp_path, model, registry=registry)
    runner.session_manager.add_user_message("Run the project processor")

    result = runner.run()

    output = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "command"
    )
    assert output["ok"], output
    assert output["result"]["boundary"] == "sandbox"
    assert result.turns == model.get_response.call_count == 2
    assert native.run.call_count == 1
    runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize(
    "tool,partial,same_batch",
    [
        ("write_text_file", False, False),
        ("edit_text_file", True, False),
        ("edit_text_file", False, True),
    ],
)
def test_runner_rejects_stale_edits_and_delivers_reread_recovery(
    tmp_path, tool, partial, same_batch
):
    """User saves during slow inference survive until the model sees and reconciles them."""
    target = tmp_path / "file.txt"
    target.write_text("anchor\nuser: old\n", encoding="utf-8")
    read = ToolCall(
        call_id="read",
        name="read_text_file",
        arguments=json.dumps(
            {
                "path": "$WORKSPACE_ROOT/file.txt",
                **({"max_lines": 1} if partial else {}),
            }
        ),
    )
    reread = ToolCall(
        call_id="reread",
        name="read_text_file",
        arguments=json.dumps(
            {
                "path": "$WORKSPACE_ROOT/file.txt",
            }
        ),
    )
    arguments = {"path": "$WORKSPACE_ROOT/file.txt"}
    arguments.update(
        {"content": "model\nuser: old\n"}
        if tool == "write_text_file"
        else {
            "old_content": "anchor",
            "new_content": "model",
        }
    )
    stale = ToolCall(call_id="stale", name=tool, arguments=json.dumps(arguments))
    fresh = ToolCall(
        call_id="fresh",
        name=tool,
        arguments=json.dumps(
            {
                **arguments,
                **({"content": "model\nuser: new\n"} if tool == "write_text_file" else {}),
            }
        ),
    )
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    turn = 0

    def generate(**request):
        """Save external changes after request capture, without timing-dependent sleeps."""
        nonlocal turn
        turn += 1
        if turn == 1:
            return model_calls(read)
        if turn == 2:
            before = target.stat()
            target.write_text("anchor\nuser: new\n", encoding="utf-8")
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            return model_calls(*((reread, stale) if same_batch else (stale,)))
        outputs = {
            item.call_id: json.loads(item.output)
            for item in request["input"]
            if isinstance(item, ToolResult)
        }
        if turn == 3:
            assert outputs["stale"]["problem"]["code"] == "filesystem.stale_context"
            assert outputs["stale"]["problem"]["metadata"]["reread_required"] is True
            assert target.read_text(encoding="utf-8") == "anchor\nuser: new\n"
            runner.interaction.prompt.assert_not_called()
            return model_calls(fresh if same_batch else reread)
        if turn == 4 and not same_batch:
            assert outputs["reread"]["result"]["content"] == "anchor\nuser: new\n"
            return model_calls(fresh)
        assert outputs["fresh"]["ok"] is True
        return [ResponseCompleted(answer="done")]

    backend.get_response.side_effect = generate
    result = runner.run()

    assert result.stop_reason == "completed"
    assert target.read_text(encoding="utf-8") == "model\nuser: new\n"
    assert [entry.succeeded for entry in result.metrics.tools if entry.name == tool] == [
        False,
        True,
    ]


@pytest.mark.parametrize("current", [b"user replaced the text", b"anchor and anchor", b"\xff"])
def test_stale_context_precedes_edit_anchor_validation(tmp_path, current):
    """Anchor changes and non-text revisions explain divergence before any edit approval."""
    target = tmp_path / "file.txt"
    target.write_text("anchor", encoding="utf-8")
    read = ToolCall(call_id="read", name="read_text_file", arguments='{"path":"file.txt"}')
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments='{"path":"file.txt","old_content":"anchor","new_content":"model"}',
    )
    backend = Mock()

    def generate(**request):
        """Save a conflicting source after the model begins preparing its edit."""
        if backend.get_response.call_count == 1:
            return model_calls(read)
        if backend.get_response.call_count == 2:
            target.write_bytes(current)
            return model_calls(edit)
        return [ResponseCompleted(answer="done")]

    backend.get_response.side_effect = generate
    runner = file_runner(tmp_path, backend)
    runner.run()
    result = next(
        json.loads(item.output)
        for item in runner.session_manager.messages
        if isinstance(item, ToolResult) and item.call_id == "edit"
    )
    assert result["problem"]["code"] == "filesystem.stale_context"
    assert result["problem"]["metadata"]["reread_required"] is True
    assert ("diff" in result["problem"]["metadata"]) is (current != b"\xff")
    assert target.read_bytes() == current
    runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("targeted", [False, True])
def test_snapshot_visibility_uses_the_backend_prepared_reference(
    tmp_path, monkeypatch, attached, targeted
):
    """Redacted sources reject whole rewrites while independently visible exact edits remain valid."""
    secret = "abcdefghijklmnop"
    content = f"{secret}\nkeep\n"
    target = tmp_path / "file.txt"
    target.write_text(content, encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "update the file",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content=content,
                size_bytes=len(content),
                included_bytes=len(content),
                truncated=False,
            ),
        )
        if attached
        else (),
    )
    sdk = Mock()
    sdk.responses.create.side_effect = [
        SimpleNamespace(
            status="completed",
            model="local-model",
            usage=None,
            output_text="",
            output=[
                ResponseFunctionToolCall(
                    id="fc_write",
                    call_id="write",
                    name="edit_text_file" if targeted else "write_text_file",
                    type="function_call",
                    status="completed",
                    arguments=json.dumps(
                        {"path": "file.txt", "old_content": "keep", "new_content": "changed"}
                        if targeted
                        else {"path": "file.txt", "content": "<redacted:secret>\nchanged\n"}
                    ),
                )
            ],
        ),
        SimpleNamespace(
            status="completed", model="local-model", usage=None, output_text="done", output=[]
        ),
    ]
    if targeted:
        sdk.responses.create.side_effect = [
            SimpleNamespace(
                status="completed",
                model="local-model",
                usage=None,
                output_text="",
                output=[
                    ResponseFunctionToolCall(
                        id="fc_range",
                        call_id="range",
                        name="read_text_file",
                        type="function_call",
                        status="completed",
                        arguments='{"path":"file.txt","start_line":2,"max_lines":1}',
                    )
                ],
            ),
            *sdk.responses.create.side_effect,
        ]
    if not attached:
        sdk.responses.create.side_effect = [
            SimpleNamespace(
                status="completed",
                model="local-model",
                usage=None,
                output_text="",
                output=[
                    ResponseFunctionToolCall(
                        id="fc_read",
                        call_id="read",
                        name="read_text_file",
                        type="function_call",
                        status="completed",
                        arguments='{"path":"file.txt"}',
                    )
                ],
            ),
            *sdk.responses.create.side_effect,
        ]
    monkeypatch.setattr("loop.backend.openai.OpenAI", lambda **kwargs: sdk)
    backend = OpenAIBackend(default_model="local-model", api_key=secret, file_input_mode="text")
    runner = file_runner(tmp_path, backend, sessions=sessions)
    runner.run()
    assert secret not in repr(
        [call.kwargs["input"] for call in sdk.responses.create.call_args_list]
    )
    assert target.read_text(encoding="utf-8") == (
        content.replace("keep", "changed") if targeted else content
    )
    result = next(
        json.loads(item.output)
        for item in sessions.messages
        if isinstance(item, ToolResult) and item.call_id == "write"
    )
    if targeted:
        assert result["ok"] is True
        runner.interaction.prompt.assert_called_once()
    else:
        assert result["problem"]["code"] == "filesystem.unread_file"
        runner.interaction.prompt.assert_not_called()


def test_shared_registry_does_not_share_file_observations_between_sessions(tmp_path):
    """Only the context sent to each session can authorize that session's file edits."""
    target = tmp_path / "file.txt"
    target.write_text("source", encoding="utf-8")
    registry = ToolRegistry(BUILTIN_TOOLS)
    read = ToolCall(call_id="read", name="read_text_file", arguments='{"path":"file.txt"}')
    edit = ToolCall(
        call_id="edit",
        name="edit_text_file",
        arguments=json.dumps(
            {
                "path": "file.txt",
                "old_content": "source",
                "new_content": "changed",
            }
        ),
    )
    first_backend = Mock()
    first_backend.get_response.side_effect = [model_calls(read), [ResponseCompleted(answer="read")]]
    first = file_runner(tmp_path, first_backend, registry=registry)
    first.run()

    second_backend = Mock()
    second_backend.get_response.side_effect = [
        model_calls(edit),
        [ResponseCompleted(answer="blocked")],
    ]
    second = file_runner(tmp_path, second_backend, registry=registry)
    second.run()
    denied = next(item for item in second.session_manager.messages if isinstance(item, ToolResult))
    assert json.loads(denied.output)["problem"]["code"] == "filesystem.unread_file"
    assert target.read_text(encoding="utf-8") == "source"

    first_backend.get_response.side_effect = [
        model_calls(edit),
        [ResponseCompleted(answer="edited")],
    ]
    first.run()
    assert target.read_text(encoding="utf-8") == "changed"


def test_shared_registry_does_not_share_deletion_receipts_between_sessions(tmp_path):
    """A deletion warning in one session cannot refresh another session's original generation."""
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    reference = ContextReference(
        kind="file",
        path="file.txt",
        content="old",
        size_bytes=3,
        included_bytes=3,
        truncated=False,
    )
    first_session, second_session = SessionManager(), SessionManager()
    for session in (first_session, second_session):
        session.add_user_message("edit", context=(reference,))
    write = ToolCall(
        call_id="write",
        name="write_text_file",
        arguments='{"path":"file.txt","content":"model"}',
    )
    registry = ToolRegistry(BUILTIN_TOOLS)
    first_backend, second_backend = Mock(), Mock()

    def generate_first(**request):
        """Delete the file while only the first session is generating a write."""
        if first_backend.get_response.call_count == 1:
            target.unlink()
            return model_calls(write)
        return [ResponseCompleted(answer="reconsider")]

    first_backend.get_response.side_effect = generate_first
    second_backend.get_response.side_effect = [
        model_calls(write),
        [ResponseCompleted(answer="reconsider")],
    ]
    first = file_runner(tmp_path, first_backend, registry=registry, sessions=first_session)
    second = file_runner(tmp_path, second_backend, registry=registry, sessions=second_session)
    first.run()
    second.run()
    for runner in (first, second):
        result = next(
            json.loads(item.output)
            for item in runner.session_manager.messages
            if isinstance(item, ToolResult)
        )
        assert result["problem"]["code"] == "filesystem.stale_context"
        assert result["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
        runner.interaction.prompt.assert_not_called()
    assert not target.exists()


def test_registered_file_aliases_preserve_read_and_successor_observations(tmp_path):
    """Renaming file tools retains snapshot validation and deterministic own-edit updates."""
    target = tmp_path / "file.txt"
    target.write_text("original", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(read_text_file, name="view")
    registry.register(edit_text_file, name="revise")
    read = ToolCall(call_id="view", name="view", arguments='{"path":"file.txt"}')
    first_edit = ToolCall(
        call_id="first",
        name="revise",
        arguments=json.dumps(
            {
                "path": "file.txt",
                "old_content": "original",
                "new_content": "first change",
            }
        ),
    )
    second_edit = ToolCall(
        call_id="second",
        name="revise",
        arguments=json.dumps(
            {
                "path": "file.txt",
                "old_content": "first change",
                "new_content": "final change",
            }
        ),
    )
    backend = Mock()
    backend.get_response.side_effect = [
        model_calls(read),
        model_calls(first_edit),
        model_calls(second_edit),
        [ResponseCompleted(answer="done")],
    ]
    runner = file_runner(tmp_path, backend, registry=registry)
    result = runner.run()

    assert all(entry.succeeded for entry in result.metrics.tools)
    assert target.read_text(encoding="utf-8") == "final change"


def test_resumed_attached_snapshots_retain_their_original_file_version(tmp_path):
    """Session hydration reuses stored reference content instead of blessing newer disk bytes."""
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "edit the attachment",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    resumed = SessionManager(session=sessions.session.id, session_store=sessions.store)
    backend = Mock()
    edit = ToolCall(
        call_id="edit", name="write_text_file", arguments='{"path":"file.txt","content":"model"}'
    )

    def generate(**request):
        """Modify disk after the persisted attachment has been supplied for inference."""
        target.write_text("new", encoding="utf-8")
        return model_calls(edit)

    backend.get_response.side_effect = lambda **request: (
        generate(**request)
        if backend.get_response.call_count == 1
        else [ResponseCompleted(answer="blocked")]
    )
    runner = file_runner(tmp_path, backend, sessions=resumed)
    runner.run()
    output = next(item.output for item in resumed.messages if isinstance(item, ToolResult))
    assert json.loads(output)["problem"]["code"] == "filesystem.stale_context"
    assert target.read_text(encoding="utf-8") == "new"


def test_recovery_does_not_use_a_same_batch_reread_for_an_unexecuted_edit(tmp_path):
    """Restarting between a reread and edit preserves the original generation's stale baseline."""
    target = tmp_path / "file.txt"
    target.write_text("user update", encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "edit",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    reread = ToolCall(call_id="reread", name="read_text_file", arguments='{"path":"file.txt"}')
    edit = ToolCall(
        call_id="pending",
        name="write_text_file",
        arguments='{"path":"file.txt","content":"stale model content"}',
    )
    sessions.add_response(
        Response(answer="", reasoning="", tool_calls=(reread, edit), items=(reread, edit))
    )
    backend = Mock()
    backend.get_response.return_value = [ResponseCompleted(answer="done")]
    runner = file_runner(tmp_path, backend, sessions=sessions)
    output, _ = runner.agent.tools.call_with_timing(
        reread.name,
        reread.arguments,
        instructions_manager=runner.instructions_manager,
        permission_manager=runner.permission_manager,
    )
    sessions.add_tool_call(
        call_id=reread.call_id, output=output, working_directory=str(tmp_path), active_skills=()
    )
    runner.interaction.confirm.return_value = True

    assert runner.recover_session().pending is False
    result = next(
        item
        for item in sessions.messages
        if isinstance(item, ToolResult) and item.call_id == edit.call_id
    )
    assert json.loads(result.output)["problem"]["code"] == "filesystem.stale_context"
    assert target.read_text(encoding="utf-8") == "user update"
    runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize("prior_generation", [False, True])
def test_persisted_recovery_preserves_deletion_receipt_delivery_boundaries(
    tmp_path, prior_generation
):
    """Restarting a batch uses absence only when its warning preceded the recovered generation."""
    sessions = SessionManager()
    sessions.add_user_message(
        "recover if appropriate",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    backend = Mock()
    backend.get_response.return_value = [ResponseCompleted(answer="done")]
    original = file_runner(tmp_path, backend, sessions=sessions)
    observations = FileObservations(
        sessions.model_context, original.instructions_manager.virtual_paths
    )
    rejected, pending = (
        ToolCall(
            call_id=identifier,
            name="write_text_file",
            arguments=json.dumps({"path": "file.txt", "content": identifier}),
        )
        for identifier in ("rejected", "pending")
    )
    calls = (rejected,) if prior_generation else (rejected, pending)
    sessions.add_response(Response(answer="", reasoning="", tool_calls=calls, items=calls))
    output, _ = original.agent.tools.call_with_timing(
        rejected.name,
        rejected.arguments,
        instructions_manager=original.instructions_manager,
        permission_manager=original.permission_manager,
        file_observations=observations,
    )
    assert json.loads(output)["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
    sessions.add_tool_call(
        call_id=rejected.call_id, output=output, working_directory=str(tmp_path), active_skills=()
    )
    if prior_generation:
        sessions.add_response(
            Response(answer="", reasoning="", tool_calls=(pending,), items=(pending,))
        )
    resumed = SessionManager(session=sessions.session.id, session_store=sessions.store)
    runner = file_runner(tmp_path, backend, sessions=resumed)
    runner.interaction.confirm.return_value = True
    assert runner.recover_session().pending is False
    result = next(
        json.loads(item.output)
        for item in resumed.messages
        if isinstance(item, ToolResult) and item.call_id == pending.call_id
    )
    target = tmp_path / "file.txt"
    if prior_generation:
        assert result["ok"] is True
        assert target.read_text(encoding="utf-8") == "pending"
        runner.interaction.prompt.assert_called_once()
    else:
        assert result["problem"]["code"] == "filesystem.stale_context"
        assert not target.exists()
        runner.interaction.prompt.assert_not_called()


@pytest.mark.parametrize("recovering", [False, True])
def test_new_file_creation_does_not_require_a_prior_read(tmp_path, recovering):
    """A genuinely absent file can be created in a fresh or recovered batch without reading it."""
    target = tmp_path / "new.txt"
    assert not target.exists()
    sessions = SessionManager()
    sessions.add_user_message("Create new.txt containing new content")
    write = ToolCall(
        call_id="create",
        name="write_text_file",
        arguments='{"path":"new.txt","content":"new content"}',
    )
    backend = Mock()
    final = [ResponseCompleted(answer="done")]
    if recovering:
        sessions.add_response(
            Response(answer="", reasoning="", tool_calls=(write,), items=(write,))
        )
        backend.get_response.return_value = final
    else:
        backend.get_response.side_effect = [model_calls(write), final]
    runner = file_runner(tmp_path, backend, sessions=sessions)
    if recovering:
        runner.interaction.confirm.return_value = True
        assert runner.recover_session().pending is False
    else:
        runner.run()
    results = [item for item in sessions.messages if isinstance(item, ToolResult)]
    assert len(results) == 1
    assert json.loads(results[0].output)["ok"] is True
    assert target.read_text(encoding="utf-8") == "new content"
    assert not any(
        isinstance(item, ToolCall) and item.name == "read_text_file" for item in sessions.messages
    )


def test_user_deletion_during_generation_cannot_be_recreated_by_a_stale_write(tmp_path):
    """A file removed after being supplied to the model remains removed after a stale write."""
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "edit",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    write = ToolCall(
        call_id="write", name="write_text_file", arguments='{"path":"file.txt","content":"model"}'
    )
    backend = Mock()

    def generate(**request):
        """Delete the source after the generation input has been captured."""
        target.unlink()
        return model_calls(write)

    backend.get_response.side_effect = lambda **request: (
        generate(**request)
        if backend.get_response.call_count == 1
        else [ResponseCompleted(answer="done")]
    )
    runner = file_runner(tmp_path, backend, sessions=sessions)
    runner.run()
    output = next(item.output for item in sessions.messages if isinstance(item, ToolResult))
    assert json.loads(output)["problem"]["code"] == "filesystem.stale_context"
    assert not target.exists()


@pytest.mark.parametrize("outcome", ["created", "competing", "denied", "race"])
def test_deletion_warning_allows_reconsidered_creation_with_normal_authorization(tmp_path, outcome):
    """Delivered deletion warnings allow new creation while preserving consent and competing files."""
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "recover the file if appropriate",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    calls = tuple(
        ToolCall(
            call_id=identifier,
            name="write_text_file",
            arguments=json.dumps({"path": "file.txt", "content": identifier}),
        )
        for identifier in ("stale", "same_batch", "reconsidered")
    )
    backend = Mock()
    runner = file_runner(tmp_path, backend, sessions=sessions)

    def approve_creation(*args, **kwargs):
        """Insert a competing file after inspection but before an approved creation."""
        target.write_text("external", encoding="utf-8")
        return ApprovalChoice.ONCE

    def generate(**request):
        """Mutate disk during inference and inspect warnings before choosing a fresh operation."""
        if backend.get_response.call_count == 1:
            target.unlink()
            return model_calls(*calls[:2])
        results = {
            item.call_id: json.loads(item.output)
            for item in request["input"]
            if isinstance(item, ToolResult)
        }
        if backend.get_response.call_count == 2:
            for identifier in ("stale", "same_batch"):
                assert results[identifier]["problem"]["code"] == "filesystem.stale_context"
                assert results[identifier]["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
                assert results[identifier]["problem"]["metadata"] == {
                    "reread_required": False,
                    "regeneration_required": True,
                }
            assert not target.exists()
            runner.interaction.prompt.assert_not_called()
            if outcome == "competing":
                target.write_text("external", encoding="utf-8")
            elif outcome == "denied":
                runner.interaction.prompt.return_value = ApprovalChoice.DENY
            elif outcome == "race":
                runner.interaction.prompt.side_effect = approve_creation
            return model_calls(calls[2])
        result = results["reconsidered"]
        if outcome == "created":
            assert result["ok"] is True
            assert target.read_text(encoding="utf-8") == "reconsidered"
        else:
            assert result["ok"] is False
            assert (
                result["problem"]["code"]
                == {
                    "competing": "filesystem.stale_context",
                    "denied": "tool.denied",
                    "race": "filesystem.write_failed",
                }[outcome]
            )
            if outcome == "denied":
                assert not target.exists()
            else:
                assert target.read_text(encoding="utf-8") == "external"
        return [ResponseCompleted(answer="done")]

    backend.get_response.side_effect = generate
    runner.run()
    assert not any(
        isinstance(item, ToolCall) and item.name == "read_text_file" for item in sessions.messages
    )
    if outcome == "competing":
        runner.interaction.prompt.assert_not_called()
    else:
        runner.interaction.prompt.assert_called_once()


@pytest.mark.parametrize("removed_by_model", [False, True])
def test_explicit_absence_or_own_deletion_allows_intentional_recreation(tmp_path, removed_by_model):
    """Only a delivered absence report or completed own deletion permits recreating a source."""
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    sessions = SessionManager()
    sessions.add_user_message(
        "recreate",
        context=(
            ContextReference(
                kind="file",
                path="file.txt",
                content="old",
                size_bytes=3,
                included_bytes=3,
                truncated=False,
            ),
        ),
    )
    recreate = ToolCall(
        call_id="recreate",
        name="write_text_file",
        arguments='{"path":"file.txt","content":"intentional"}',
    )
    backend = Mock()
    if removed_by_model:
        remove = ToolCall(call_id="remove", name="delete_path", arguments='{"path":"file.txt"}')
        backend.get_response.side_effect = [
            model_calls(remove, recreate),
            [ResponseCompleted(answer="done")],
        ]
    else:
        read = ToolCall(call_id="absence", name="read_text_file", arguments='{"path":"file.txt"}')

        def generate(**request):
            """Remove the source while a stale write is generated, then reconsider it."""
            if backend.get_response.call_count == 1:
                target.unlink()
                return model_calls(
                    ToolCall(
                        call_id="stale",
                        name="write_text_file",
                        arguments='{"path":"file.txt","content":"stale"}',
                    )
                )
            return (
                model_calls(read)
                if backend.get_response.call_count == 2
                else model_calls(recreate)
                if backend.get_response.call_count == 3
                else [ResponseCompleted(answer="done")]
            )

        backend.get_response.side_effect = generate
    runner = file_runner(tmp_path, backend, sessions=sessions)
    runner.run()
    assert target.read_text(encoding="utf-8") == "intentional"
    results = {
        item.call_id: json.loads(item.output)
        for item in sessions.messages
        if isinstance(item, ToolResult)
    }
    assert results["recreate"]["ok"] is True
    if removed_by_model:
        assert results["remove"]["file_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
    else:
        assert results["stale"]["problem"]["code"] == "filesystem.stale_context"
        assert results["absence"]["read_versions"] == {"$WORKSPACE_ROOT/file.txt": None}
        assert results["absence"]["problem"]["code"] == "filesystem.path_missing"


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


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_run_recovery_confirmation_preserves_pending_state(asynchronous):
    """A truthy non-boolean cannot authorize recovery or issue a model request."""
    runner, sessions, interaction = agent_runner()
    sessions.recovery_state = SessionRecoveryState(action="query_model")
    interaction.confirm.return_value = object()
    runner._confirm_async = AsyncMock(return_value=object())
    runner._recover = Mock()
    runner._recover_async = AsyncMock()

    if asynchronous:
        result = asyncio.run(runner.recover_session_async())
        runner._confirm_async.assert_awaited_once()
    else:
        result = runner.recover_session()
        interaction.confirm.assert_called_once()

    assert result.pending is True
    runner._recover.assert_not_called()
    runner._recover_async.assert_not_awaited()
    runner.backend.get_response.assert_not_called()


@pytest.mark.parametrize("name", ["echo", "run_command"])
def test_runner_recovers_unstarted_tools_before_requerying(name):
    """Definitely unstarted calls execute once with their stable call identifier."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    call = ToolCall(call_id="call", name=name, arguments="{}")
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
        name,
        "{}",
        interaction=interaction,
        instructions_manager=runner.instructions_manager,
        permission_manager=runner.permission_manager,
        call_id="call",
        cancellation_event=None,
        execution_started=ANY,
        file_observations=ANY,
    )
    runner.agent.tools.call_with_timing.call_args.kwargs["execution_started"]()
    sessions.record_tool_execution_started.assert_called_once_with("call")
    assert sessions.add_tool_call.call_args.kwargs["succeeded"] is True


def test_runner_records_synchronous_interrupted_tool_receipt():
    """A known interrupted tool result is durably recorded before Ctrl+C propagates."""
    runner, sessions, _ = agent_runner(responses=[])
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    output = '{"ok":true,"result":"confirmed"}'
    interruption = ToolExecutionInterrupted(output, 0.25)
    interruption.add_note("Tool cleanup also stopped with KeyboardInterrupt.")
    runner.agent.tools.call_with_timing.side_effect = interruption

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.handle_tool_calls(response)

    sessions.add_tool_call.assert_called_once()
    recorded = sessions.add_tool_call.call_args.kwargs
    assert recorded["call_id"] == "call"
    assert recorded["output"] == output
    assert recorded["succeeded"] is True
    assert caught.value.metrics.name == "effect"
    assert caught.value.metrics.duration_seconds == 0.25
    assert caught.value.__notes__ == ["Tool cleanup also stopped with KeyboardInterrupt."]


def test_cancelled_queued_tool_records_definitely_unstarted_result():
    """A tool canceled before worker entry receives a durable non-replay receipt."""
    runner, sessions, _ = agent_runner(responses=[])
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=ToolExecutionNotStarted())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.handle_tool_calls_async(response))

    sessions.add_tool_call.assert_called_once()
    recorded = sessions.add_tool_call.call_args.kwargs
    assert recorded["call_id"] == "call"
    assert recorded["succeeded"] is False
    assert "cancelled before execution" in recorded["output"]


def test_cancelled_queued_retry_preserves_previous_unknown_outcome(tmp_path):
    """Canceling an unstarted retry cannot erase the earlier uncertain invocation."""
    runner, sessions, _ = agent_runner(responses=[])
    permissions = PermissionManager(tmp_path)
    runner._permission_manager = permissions
    source = MagicMock(spec=ConsoleInteraction)
    source.confirm_async = AsyncMock(return_value=True)
    runner._interaction = ApprovalInteraction.for_permissions(source, permissions)
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=ToolExecutionNotStarted())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.recover_session_async(confirmed=True))

    sessions.add_tool_call.assert_not_called()
    runner.agent.tools.call_with_timing_async.assert_awaited_once()


def test_runner_preserves_interrupted_metrics_in_an_active_run():
    """A completed interrupted tool retains metrics while the run stops."""
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, _ = agent_runner(responses=[response])
    runner.agent.tools.call_with_timing.side_effect = ToolExecutionInterrupted(
        '{"ok":true,"result":"confirmed"}', 0.25
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run()
    assert caught.value.metrics.name == "effect"
    assert sessions.add_tool_call.call_args.kwargs["succeeded"] is True
    assert runner.query.call_count == 1


def test_uninterrupted_tool_publication_propagates_keyboard_interrupt():
    """A first interrupt during ordinary result publication is not replaced."""
    runner, sessions, _ = agent_runner(responses=[])
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner.agent.tools.call_with_timing.return_value = ('{"ok":true}', 0.25)
    sessions.add_tool_call.side_effect = KeyboardInterrupt("publication interrupted")

    with pytest.raises(KeyboardInterrupt, match="publication interrupted") as caught:
        runner.handle_tool_calls(response)
    assert not isinstance(caught.value, ToolExecutionInterrupted)
    sessions.add_tool_call.assert_called_once()


def test_repeated_interrupt_keeps_receipt_without_replaying_publication():
    """A second interrupt preserves the receipt without retrying an uncertain save."""
    runner, sessions, _ = agent_runner(responses=[])
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    output = '{"ok":true,"result":"confirmed"}'
    runner.agent.tools.call_with_timing.side_effect = ToolExecutionInterrupted(output, 0.25)
    sessions.add_tool_call.side_effect = KeyboardInterrupt

    with pytest.raises(ToolExecutionInterrupted) as caught:
        runner.handle_tool_calls(response)

    assert caught.value.output == output
    assert caught.value.__notes__ == [
        "Tool result publication may be incomplete; inspect recovery before retrying."
    ]
    sessions.add_tool_call.assert_called_once()


def test_sync_interrupted_final_receipt_is_persisted_without_replay(tmp_path):
    """A confirmed final result is saved before a synchronous interrupt escapes."""
    effects = []

    def interrupt():
        """Interrupt the event loop while the tool is running."""
        raise KeyboardInterrupt

    async def child():
        """Complete an effect while settling cancellation."""
        try:
            asyncio.get_running_loop().call_soon(interrupt)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            effects.append("committed")
            return "confirmed"

    async def calculate() -> str:
        """Return the owned child's complete tool outcome."""
        return await OwnedToolTask(child(), final_result=True)

    registry = ToolRegistry([calculate])
    runner = file_runner(tmp_path, Mock(), registry=registry)
    sessions = runner.session_manager
    sessions.add_user_message("run the synthetic tool")
    sessions.begin_run()
    call = ToolCall(call_id="effect", name="calculate", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    sessions.add_response(response)

    try:
        with pytest.raises(KeyboardInterrupt):
            runner.handle_tool_calls(response)
        results = [item for item in sessions.messages if isinstance(item, ToolResult)]
        assert len(results) == 1
        assert results[0].call_id == "effect"
        assert json.loads(results[0].output)["result"] == "confirmed"
        assert effects == ["committed"]
        assert sessions.recovery_state.action == "query_model"
        assert [pending.status for pending in sessions.recovery_state.pending_calls] == [
            "result_available"
        ]
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("unknown_count", [1, 2])
def test_yolo_new_message_continues_without_replaying_unknown_tool(
    tmp_path, async_mode, unknown_count
):
    """A new user instruction sees the unknown outcome and can issue a fresh tool call."""
    sessions = SessionManager()
    sessions.add_user_message("perform the old action")
    sessions.begin_run()
    old = tuple(
        ToolCall(call_id=f"old-{index}", name="old_action", arguments="{}")
        for index in range(unknown_count)
    )
    sessions.add_response(Response(answer="", reasoning="", tool_calls=old, items=old))
    for call in old:
        sessions.record_tool_execution_started(call.call_id)
    sessions.add_user_message("inspect the result instead")
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    registry = Mock(spec=ToolRegistry)
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    registry = runner.agent.tools
    registry.call_with_timing.return_value = ('{"ok":true}', 0.1)
    registry.call_with_timing_async = AsyncMock(return_value=('{"ok":true}', 0.1))
    runner.permission_manager.set_yolo(True)
    fresh = ToolCall(call_id="new", name="inspect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(fresh,), items=(fresh,))
    final = Response(
        answer="done", reasoning="", items=(Message(role="assistant", content="done"),)
    )
    responses = iter((response, final))

    def query_after_warning(*_args, **_kwargs):
        """Require the nonblocking user notice before any new model request."""
        runner.interaction.warning.assert_called_once_with(
            "A previous tool may already have run; its outcome is unknown. "
            "A new action may repeat its effects. The old call will not be replayed."
        )
        return next(responses)

    if async_mode:
        runner.query_async = AsyncMock(side_effect=query_after_warning)
        result = asyncio.run(runner.recover_session_async(confirmed=True))
        registry.call_with_timing_async.assert_awaited_once()
        assert registry.call_with_timing_async.await_args.args[0] == "inspect"
    else:
        runner.query = Mock(side_effect=query_after_warning)
        result = runner.recover_session(confirmed=True)
        registry.call_with_timing.assert_called_once()
        assert registry.call_with_timing.call_args.args[0] == "inspect"
    assert result.pending is False
    outputs = [item.output for item in restored.messages if isinstance(item, ToolResult)]
    assert len(outputs) == unknown_count + 1
    assert all("external outcome is unknown" in output for output in outputs[:unknown_count])
    assert all("may repeat its effects" in output for output in outputs[:unknown_count])
    runner.interaction.confirm.assert_not_called()
    assert restored.recovery_pending is False


@pytest.mark.parametrize("async_mode", [False, True])
def test_recovery_uses_saved_correction_without_replaying_unstarted_tools(tmp_path, async_mode):
    """A restart after saving input keeps the old tool cancelled before the resumed query."""
    sessions = SessionManager()
    sessions.add_user_message("delete files")
    sessions.begin_run()
    call = ToolCall(call_id="old", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.add_user_message("only list files")
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    backend = Mock()
    runner = file_runner(tmp_path, backend, sessions=restored)
    final = Response(
        answer="listed", reasoning="", items=(Message(role="assistant", content="listed"),)
    )
    if async_mode:
        runner.query_async = AsyncMock(return_value=final)
        result = asyncio.run(runner.recover_session_async(confirmed=True))
    else:
        runner.query = Mock(return_value=final)
        result = runner.recover_session(confirmed=True)

    assert result.result.stop_reason == "completed"
    backend.get_response.assert_not_called()
    outputs = [item.output for item in restored.messages if isinstance(item, ToolResult)]
    assert len(outputs) == 1
    assert "cancelled before execution" in outputs[0]
    assert restored.recovery_pending is False


@pytest.mark.parametrize("async_mode", [False, True])
def test_steering_after_pause_prevents_an_approved_tool_from_starting(async_mode):
    """A correction published while paused is checked again after approval and before effects."""
    control = RunControl()
    call = ToolCall(call_id="old", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, _interaction = agent_runner(responses=[response], run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="execute_tools", pending_calls=(PendingToolCall(call=call, status="not_started"),)
    )

    def approve(**kwargs):
        """Publish accepted input before releasing the same paused execution owner."""
        assert control.pause()
        assert control.request_steering(lambda: True)
        assert control.continue_run()
        return kwargs["execution_started"]

    def execute(*_args, **kwargs):
        """Reach the synchronous guard immediately before dispatching tool effects."""
        approve(**kwargs)()
        pytest.fail("Superseded tool must not execute")

    async def execute_async(*_args, **kwargs):
        """Reach the asynchronous guard at the same approved execution boundary."""
        await approve(**kwargs)()
        pytest.fail("Superseded tool must not execute")

    if async_mode:
        runner.query_async = AsyncMock(return_value=response)
        runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=execute_async)
        result = asyncio.run(runner.run_async())
    else:
        runner.agent.tools.call_with_timing.side_effect = execute
        result = runner.run()

    assert result.stop_reason == "steered"
    sessions.record_tool_execution_started.assert_not_called()
    assert "cancelled before execution" in sessions.add_tool_call.call_args.kwargs["output"]
    assert control.active is False
    assert control.stopped is False


def test_runner_skips_completed_calls_in_a_partially_recovered_batch():
    """Mixed recovery executes only calls whose durable results are still absent."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    runner.agent.tools.call_with_timing.return_value = ('{"ok": true}', 0.1)
    completed = ToolCall(call_id="completed", name="run_command", arguments="{}")
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

    runner.agent.tools.call_with_timing.assert_called_once()
    assert runner.agent.tools.call_with_timing.call_args.args[0] == "write"
    callback = runner.agent.tools.call_with_timing.call_args.kwargs["execution_started"]
    callback()
    sessions.record_tool_execution_started.assert_called_once_with("pending")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("answer", ["truthy", "future"])
def test_unknown_tool_is_not_replayed_for_invalid_confirmation(tmp_path, async_mode, answer):
    """A malformed confirmation cannot authorize replay or resolve the unknown outcome."""
    effects = []

    def effect():
        """Record an incorrectly retried tool."""
        effects.append("replayed")

    sessions = SessionManager()
    sessions.add_user_message("perform the action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    registry = ToolRegistry([effect])
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    foreign_loop = asyncio.new_event_loop()
    future = foreign_loop.create_future()
    response = future if answer == "future" else object()
    runner.interaction.confirm.return_value = response
    runner.interaction.confirm_async = AsyncMock(return_value=response)
    runner.interaction.prompt_async = AsyncMock()
    try:
        if async_mode:
            result = asyncio.run(runner.recover_session_async(confirmed=True))
        else:
            result = runner.recover_session(confirmed=True)
        assert result.pending
        assert result.result.stop_reason == "recovery_required"
        assert restored.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in restored.messages)
        assert effects == []
        assert not future.done()
    finally:
        registry.close()
        future.cancel()
        foreign_loop.close()


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


@pytest.mark.parametrize("name", ["mutate", "run_command"])
def test_runner_retries_an_uncertain_tool_only_after_explicit_approval(name):
    """Explicit approval retries an uncertain call without duplicating its presentation event."""
    final = Response(answer="done", reasoning="")
    runner, sessions, interaction = agent_runner(responses=[final])
    interaction.confirm.side_effect = [True, True]

    def execute(*_args, **kwargs):
        """Record execution and admission before returning the retried tool's result."""
        kwargs["execution_started"]()
        assert kwargs["execution_admission"]()
        return '{"ok": false}', 0.1

    runner.agent.tools.call_with_timing.side_effect = execute
    call = ToolCall(call_id="call", name=name, arguments="{}")

    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    runner.recover_session()

    runner.agent.tools.call_with_timing.assert_called_once()
    assert runner.agent.tools.call_with_timing.call_args.args[0] == name
    assert interaction.confirm.call_args_list[-1].args == (
        f"Tool '{name}' may already have run. Retry it?",
    )
    sessions.add_tool_call_event.assert_not_called()
    sessions.record_tool_execution_started.assert_called_once_with("call")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("change_at", ["confirmation", "dispatch", "persistence"])
@pytest.mark.parametrize("disable_again", [False, True])
def test_uncertain_retry_rejects_changed_yolo_generation(
    tmp_path, monkeypatch, async_mode, change_at, disable_again
):
    """A changed automatic-decision generation cannot admit an uncertain tool retry."""
    runner, sessions, _ = agent_runner(responses=[])
    permissions = PermissionManager(tmp_path)
    runner._permission_manager = permissions
    source = MagicMock(spec=ConsoleInteraction)
    source.confirm.return_value = True
    source.confirm_async = AsyncMock(return_value=True)
    runner._interaction = ApprovalInteraction.for_permissions(source, permissions)
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    effects = []

    def change():
        """Change authority, optionally returning it to the original visible state."""
        permissions.set_yolo(True)
        if disable_again:
            permissions.set_yolo(False)

    if change_at == "persistence":
        sessions.record_tool_execution_started.side_effect = lambda _call_id: change()

    if change_at == "confirmation":
        snapshot = permissions.recovery_retry_generation

        def snapshot_then_change():
            """Change authority after capturing the generation before confirmation."""
            generation = snapshot()
            change()
            return generation

        monkeypatch.setattr(permissions, "recovery_retry_generation", snapshot_then_change)

    def dispatch(*_args, **kwargs):
        """Change authority after approval and before the execution boundary."""
        if change_at == "dispatch":
            change()
        kwargs["execution_started"]()
        if not kwargs["execution_admission"]():
            raise ToolExecutionAdmissionRejected
        effects.append("executed")
        return '{"ok":true}', 0.0

    async def dispatch_async(*_args, **kwargs):
        """Exercise the asynchronous execution boundary after approval."""
        if change_at == "dispatch":
            change()
        await kwargs["execution_started"]()
        if not kwargs["execution_admission"]():
            raise ToolExecutionAdmissionRejected
        effects.append("executed")
        return '{"ok":true}', 0.0

    runner.agent.tools.call_with_timing.side_effect = dispatch
    runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=dispatch_async)
    if async_mode:
        result = asyncio.run(runner.recover_session_async(confirmed=True))
    else:
        result = runner.recover_session(confirmed=True)

    assert result.result.stop_reason == "recovery_required"
    assert result.pending is True
    assert effects == []
    sessions.add_tool_call.assert_not_called()
    runner.query.assert_not_called()
    if change_at == "confirmation" and not disable_again:
        source.confirm.assert_not_called()
        source.confirm_async.assert_not_awaited()


@pytest.mark.parametrize("async_mode", [False, True])
def test_uncertain_retry_policy_change_keeps_real_tool_unexecuted(
    tmp_path, monkeypatch, async_mode
):
    """A policy change during durable admission leaves a real tool outcome unresolved."""
    effects = []

    def effect():
        """Record the synthetic side effect."""
        effects.append(True)

    sessions = SessionManager()
    sessions.add_user_message("perform the action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    runner = file_runner(tmp_path, Mock(), registry=ToolRegistry([effect]), sessions=restored)
    runner.interaction.confirm.return_value = True
    if async_mode:
        interaction = MagicMock(spec=ConsoleInteraction)
        interaction.confirm_async = AsyncMock(return_value=True)
        runner._interaction = interaction
        original_async = restored.record_tool_execution_started_async

        async def record_async(call_id):
            """Change policy while the async durable boundary is being recorded."""
            await original_async(call_id)
            runner.permission_manager.set_yolo(True)

        monkeypatch.setattr(restored, "record_tool_execution_started_async", record_async)
        result = asyncio.run(runner.recover_session_async(confirmed=True))
    else:
        original = restored.record_tool_execution_started

        def record(call_id):
            """Change policy while the durable boundary is being recorded."""
            original(call_id)
            runner.permission_manager.set_yolo(True)

        monkeypatch.setattr(restored, "record_tool_execution_started", record)
        result = runner.recover_session(confirmed=True)

    assert result.pending and result.result.stop_reason == "recovery_required"
    assert restored.recovery_pending
    assert not any(isinstance(item, ToolResult) for item in restored.messages)
    assert effects == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_uncertain_retry_rechecks_policy_at_real_tool_dispatch(tmp_path, monkeypatch, async_mode):
    """A policy transition after durable start prevents the tool callable from running."""
    effects = []

    def effect():
        """Record a synthetic side effect."""
        effects.append(True)

    sessions = SessionManager()
    sessions.add_user_message("perform the action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    registry = ToolRegistry([effect])
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    runner.interaction.confirm.return_value = True
    if async_mode:
        interaction = MagicMock(spec=ConsoleInteraction)
        interaction.confirm_async = AsyncMock(return_value=True)
        runner._interaction = interaction
        original_async = Tool.execute_async

        async def change_before_async_dispatch(tool, *args, **kwargs):
            """Change policy after the durable callback but before tool dispatch."""
            runner.permission_manager.set_yolo(True)
            return await original_async(tool, *args, **kwargs)

        monkeypatch.setattr(Tool, "execute_async", change_before_async_dispatch)
        result = asyncio.run(runner.recover_session_async(confirmed=True))
    else:
        original = Tool.execute

        def change_before_dispatch(tool, *args, **kwargs):
            """Change policy after the durable callback but before tool dispatch."""
            runner.permission_manager.set_yolo(True)
            return original(tool, *args, **kwargs)

        monkeypatch.setattr(Tool, "execute", change_before_dispatch)
        result = runner.recover_session(confirmed=True)

    assert result.pending and result.result.stop_reason == "recovery_required"
    assert restored.recovery_pending
    assert not any(isinstance(item, ToolResult) for item in restored.messages)
    assert effects == []
    registry.close()


def test_uncertain_retry_rechecks_policy_in_async_tool_worker(tmp_path, monkeypatch):
    """A worker held after durable start must recheck policy before its tool starts."""
    effects = []
    release = Event()

    def effect():
        """Record a synthetic side effect."""
        effects.append(True)

    sessions = SessionManager()
    sessions.add_user_message("perform the action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    registry = ToolRegistry([effect])
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async = AsyncMock(return_value=True)
    runner._interaction = interaction
    original = Tool.execute_async

    async def exercise():
        """Hold the actual invocation worker while changing YOLO."""
        ready = asyncio.Event()
        owner = asyncio.get_running_loop()

        async def delayed_tool(tool, *args, **kwargs):
            """Pause the registry's admitted worker before its tool callable."""
            dispatch = kwargs["sync_dispatch"]

            async def delayed_dispatch(invoke):
                """Run the normal worker with a deterministic pre-invocation barrier."""

                def held():
                    """Wait on the worker until the owner changes policy."""
                    owner.call_soon_threadsafe(ready.set)
                    if not release.wait(5):
                        raise TimeoutError("Tool worker barrier was not released.")
                    return invoke()

                return await dispatch(held)

            kwargs["sync_dispatch"] = delayed_dispatch
            return await original(tool, *args, **kwargs)

        monkeypatch.setattr(Tool, "execute_async", delayed_tool)
        task = asyncio.create_task(runner.recover_session_async(confirmed=True))
        try:
            await asyncio.wait_for(ready.wait(), 5)
            runner.permission_manager.set_yolo(True)
        finally:
            release.set()
        return await task

    result = asyncio.run(exercise())
    assert result.pending and result.result.stop_reason == "recovery_required"
    assert restored.recovery_pending
    assert not any(isinstance(item, ToolResult) for item in restored.messages)
    assert effects == []
    registry.close()


@pytest.mark.parametrize("failure", ["saturated", "closed"])
@pytest.mark.parametrize("activate_yolo", [False, True])
def test_uncertain_retry_without_worker_admission_preserves_unknown(
    tmp_path, monkeypatch, failure, activate_yolo
):
    """Worker refusal after durable intent cannot settle an earlier unknown outcome."""
    release = Event()
    effects = []

    def effect():
        """Record a synthetic effect."""
        effects.append(True)

    def occupy():
        """Retain a registry worker until the test releases it."""
        assert release.wait(5), "worker was not released"

    registry = ToolRegistry([effect, occupy])
    sessions = SessionManager()
    sessions.add_user_message("perform the original action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    backend = Mock()
    runner = file_runner(tmp_path, backend, registry=registry, sessions=restored)
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async = AsyncMock(return_value=True)
    runner._interaction = interaction
    original = restored.record_tool_execution_started_async

    async def exercise():
        """Fill or close the actual registry after intent is recorded."""
        loop = asyncio.get_running_loop()
        queued = asyncio.Event()
        count = 0
        tasks = []

        def queued_one():
            """Signal after each blocker has reached the worker submission boundary."""
            nonlocal count
            count += 1
            if count == 8:
                queued.set()

        async def blocker_started():
            """Observe submission after this callback returns to the registry."""
            loop.call_soon(queued_one)

        async def record_and_block(call_id):
            """Arrange worker refusal after durable intent and before final admission."""
            await original(call_id)
            if failure == "closed":
                registry.close()
            else:
                tasks.extend(
                    asyncio.create_task(
                        registry.call_with_timing_async(
                            "occupy", "{}", execution_started=blocker_started
                        )
                    )
                    for _ in range(8)
                )
                await asyncio.wait_for(queued.wait(), 5)
            if activate_yolo:
                runner.permission_manager.set_yolo(True)

        monkeypatch.setattr(restored, "record_tool_execution_started_async", record_and_block)
        try:
            return await asyncio.wait_for(runner.recover_session_async(confirmed=True), 10)
        finally:
            release.set()
            await asyncio.gather(*tasks)

    try:
        result = asyncio.run(exercise())
        assert result.pending and result.result.stop_reason == "recovery_required"
        assert restored.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in restored.messages)
        assert json.loads(interaction.tool_result.call_args.args[0])["problem"]["code"] == (
            "tool.dispatch_busy"
        )
        backend.get_response.assert_not_called()
        assert effects == []
    finally:
        release.set()
        registry.close()


@pytest.mark.parametrize("activate_yolo", [False, True])
def test_uncertain_coroutine_retry_with_closed_runner_preserves_unknown(
    tmp_path, monkeypatch, caplog, activate_yolo
):
    """Coroutine scheduling failure after durable intent leaves the prior outcome unknown."""
    effects = []

    async def effect():
        """Record a synthetic effect."""
        effects.append(True)

    registry = ToolRegistry([effect])
    sessions = SessionManager()
    sessions.add_user_message("perform the original action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    backend = Mock()
    runner = file_runner(tmp_path, backend, registry=registry, sessions=restored)
    runner.interaction.confirm.return_value = True
    original = restored.record_tool_execution_started

    def record_and_close(call_id):
        """Close the initialized coroutine runner after execution intent is recorded."""
        original(call_id)
        registry.close()
        if activate_yolo:
            runner.permission_manager.set_yolo(True)

    monkeypatch.setattr(restored, "record_tool_execution_started", record_and_close)
    try:
        result = runner.recover_session(confirmed=True)
        assert result.pending and result.result.stop_reason == "recovery_required"
        assert restored.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in restored.messages)
        shown = json.loads(runner.interaction.tool_result.call_args.args[0])
        assert shown["problem"]["code"] == "tool.execution_failed"
        assert len(caplog.records) == 1
        assert caplog.records[0].__dict__["error.type"] == "tool.execution_failed"
        assert caplog.records[0].__dict__["exception.type"] == "builtins.RuntimeError"
        backend.get_response.assert_not_called()
        assert effects == []
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
def test_admitted_uncertain_retry_publishes_the_new_result(tmp_path, async_mode):
    """An explicitly approved and admitted retry executes once and proceeds normally."""
    effects = []

    def effect():
        """Record a synthetic effect."""
        effects.append(True)
        return "completed"

    registry = ToolRegistry([effect])
    sessions = SessionManager()
    sessions.add_user_message("perform the original action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    runner.interaction.confirm.return_value = True
    final = Response(
        answer="done", reasoning="", items=(Message(role="assistant", content="done"),)
    )
    try:
        if async_mode:
            interaction = MagicMock(spec=ConsoleInteraction)
            interaction.confirm_async = AsyncMock(return_value=True)
            runner._interaction = interaction
            runner.query_async = AsyncMock(return_value=final)
            result = asyncio.run(runner.recover_session_async(confirmed=True))
            runner.query_async.assert_awaited_once()
        else:
            runner.query = Mock(return_value=final)
            result = runner.recover_session(confirmed=True)
            runner.query.assert_called_once()
        assert not result.pending and result.result.stop_reason == "completed"
        assert result.result.turns == 1
        assert effects == [True]
        output = next(item.output for item in restored.messages if isinstance(item, ToolResult))
        assert json.loads(output) == {"ok": True, "result": "completed"}
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
def test_uncertain_retry_preserves_policy_check_failures(tmp_path, monkeypatch, async_mode):
    """An unavailable policy check cannot settle or execute an unknown tool call."""
    effects = []

    def effect():
        """Record a synthetic side effect."""
        effects.append(True)

    sessions = SessionManager()
    sessions.add_user_message("perform the action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    registry = ToolRegistry([effect])
    runner = file_runner(tmp_path, Mock(), registry=registry, sessions=restored)
    runner.interaction.confirm.return_value = True
    check = Mock(side_effect=[True, OSError("policy unavailable")])
    monkeypatch.setattr(runner.permission_manager, "recovery_retry_current", check)
    if async_mode:
        interaction = MagicMock(spec=ConsoleInteraction)
        interaction.confirm_async = AsyncMock(return_value=True)
        runner._interaction = interaction
    try:
        with pytest.raises(OSError, match="policy unavailable"):
            if async_mode:
                asyncio.run(runner.recover_session_async(confirmed=True))
            else:
                runner.recover_session(confirmed=True)
        assert restored.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in restored.messages)
        assert effects == []
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("activate_yolo", [False, True])
@pytest.mark.parametrize("failure", ["unavailable", "invalid", "planning", "denied"])
def test_unstarted_retry_leaves_earlier_unknown_outcome_pending(
    tmp_path, monkeypatch, async_mode, activate_yolo, failure
):
    """A failed retry cannot replace an earlier unknown outcome or trigger a model query."""
    effects = []

    def effect(number: int) -> int:
        """Record a synthetic effect."""
        effects.append(number)
        return number

    def fail_plan(_arguments):
        """Refuse synthetic planning before invocation."""
        raise ValueError("synthetic planning failure")

    registration = ToolRegistration(
        effect, operation_planner=fail_plan if failure == "planning" else None
    )
    registry = ToolRegistry([registration])
    trace = Mock()
    monkeypatch.setattr(agent_runner_module, "telemetry_trace_event", trace)
    sessions = SessionManager()
    sessions.add_user_message("perform the original action")
    sessions.begin_run()
    call = ToolCall(
        call_id="unknown",
        name="unavailable" if failure == "unavailable" else "effect",
        arguments="{}" if failure == "invalid" else '{"number": 1}',
    )
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    backend = Mock()
    runner = file_runner(tmp_path, backend, registry=registry, sessions=restored)
    runner.interaction.confirm.return_value = True
    if async_mode:
        interaction = MagicMock(spec=ConsoleInteraction)
        interaction.confirm_async = AsyncMock(return_value=True)
        runner._interaction = interaction
    if activate_yolo:
        runner.interaction.tool_call.side_effect = lambda *_args: (
            runner.permission_manager.set_yolo(True)
        )
    if failure == "denied":
        denial = SimpleNamespace(reason="synthetic policy denial")
        monkeypatch.setattr(
            runner.permission_manager, "check_boundaries", Mock(return_value=denial)
        )
        monkeypatch.setattr(
            runner.permission_manager, "check_boundaries_async", AsyncMock(return_value=denial)
        )
    try:
        if async_mode:
            result = asyncio.run(runner.recover_session_async(confirmed=True))
        else:
            result = runner.recover_session(confirmed=True)
        assert result.pending and result.result.stop_reason == "recovery_required"
        assert restored.recovery_pending
        assert not any(isinstance(item, ToolResult) for item in restored.messages)
        assert effects == []
        backend.get_response.assert_not_called()
        shown = json.loads(runner.interaction.tool_result.call_args.args[0])
        trace.assert_any_call(
            "tool.response",
            payload={
                "call_id": "unknown",
                "output": runner.interaction.tool_result.call_args.args[0],
            },
            duration_seconds=0,
        )
        assert (
            shown["problem"]["code"]
            == {
                "unavailable": "tool.unknown",
                "invalid": "tool.invalid_arguments",
                "planning": "tool.planning_failed",
                "denied": "tool.denied",
            }[failure]
        )
        assert (
            "earlier tool outcome is still unknown"
            in runner.interaction.warning.call_args_list[0].args[0]
        )
    finally:
        registry.close()


@pytest.mark.parametrize("async_mode", [False, True])
def test_yolo_activation_during_unsuperseded_recovery_stays_pending(
    tmp_path, monkeypatch, async_mode
):
    """A live YOLO change cannot resume an old request after an unknown outcome."""
    sessions = SessionManager()
    sessions.add_user_message("perform the original action")
    sessions.begin_run()
    call = ToolCall(call_id="unknown", name="effect", arguments="{}")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.record_tool_execution_started(call.call_id)
    restored = SessionManager(session=Session.deserialize(sessions.session.serialize()))
    backend = Mock()
    runner = file_runner(tmp_path, backend, sessions=restored)
    snapshot = runner.permission_manager.recovery_retry_generation

    def enable_then_snapshot():
        """Activate YOLO after the initial recovery check."""
        runner.permission_manager.set_yolo(True)
        return snapshot()

    monkeypatch.setattr(
        runner.permission_manager, "recovery_retry_generation", enable_then_snapshot
    )
    if async_mode:
        interaction = MagicMock(spec=ConsoleInteraction)
        interaction.confirm_async = AsyncMock(return_value=True)
        runner._interaction = interaction
        result = asyncio.run(runner.recover_session_async(confirmed=True))
        interaction.confirm_async.assert_not_awaited()
    else:
        result = runner.recover_session(confirmed=True)
        runner.interaction.confirm.assert_not_called()

    assert result.pending and result.result.stop_reason == "recovery_required"
    assert restored.recovery_pending
    assert not any(isinstance(item, ToolResult) for item in restored.messages)
    backend.get_response.assert_not_called()


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
    session_manager.model_context = []
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


def test_stopped_runner_does_not_query_or_begin_recovery():
    """A cooperative stop prevents new model work and recovery approvals."""
    control = RunControl()
    control.stop()
    runner, sessions, interaction = agent_runner(run_control=control)

    with pytest.raises(KeyboardInterrupt):
        runner.run()
    with pytest.raises(KeyboardInterrupt):
        runner.recover_session()

    runner.backend.get_response.assert_not_called()
    sessions.record_run.assert_not_called()
    interaction.confirm.assert_not_called()


@pytest.mark.parametrize("streaming", [False, True])
def test_runner_queries_backend_asynchronously(tmp_path, streaming):
    """Async queries collect completed backend events and preserve response timing semantics."""
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    runner.stream = streaming
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager

    async def events():
        """Yield normalized response events in backend order."""
        yield AnswerDelta(text="final")
        yield ResponseCompleted(answer="final")

    backend.get_response_async.return_value = events()

    async def query():
        """Run the query while holding the session's async-task execution ownership."""
        with sessions.execution():
            return await runner.query_async()

    response = asyncio.run(query())

    assert response.answer == "final"
    assert response.metrics is not None
    assert (response.metrics.time_to_first_chunk_seconds is not None) is streaming
    backend.get_response_async.assert_called_once()
    backend.get_response.assert_not_called()
    runner.compaction.compact_if_needed_async.assert_awaited_once()


def test_runner_async_recovery_waits_with_async_confirmation_and_control(tmp_path):
    """Retry-after recovery uses native async confirmation and the cancellable wait API."""
    runner = file_runner(tmp_path, Mock())
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async = AsyncMock(return_value=True)
    runner._interaction = interaction
    runner._run_control = Mock(spec=RunControl)
    runner._run_control.wait_async = AsyncMock()
    completed = Response(answer="complete", reasoning="")
    runner.query_async = AsyncMock(
        side_effect=[
            BackendConnectionError(
                "private failure",
                provider="test",
                operation="query",
                retry_after=2.5,
                response_started=True,
            ),
            completed,
        ]
    )

    result = asyncio.run(runner._query_with_recovery_async())

    assert result is completed
    interaction.confirm_async.assert_awaited_once_with(
        "Partial output was discarded. Retry the complete response after at least 2.5 seconds?",
        default=False,
    )
    interaction.confirm.assert_not_called()
    runner._run_control.wait_async.assert_awaited_once_with(2.5)
    assert runner.query_async.await_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_invalid_backend_retry_confirmation_does_not_repeat_request(asynchronous):
    """Malformed approval leaves the failed run pending without another model request."""
    runner, sessions, interaction = agent_runner()
    sessions.begin_run()
    error = BackendConnectionError("temporary", provider="test", operation="query")
    runner.query = Mock(side_effect=error)
    runner.query_async = AsyncMock(side_effect=error)
    interaction.confirm.return_value = object()
    runner._confirm_async = AsyncMock(return_value=object())

    with pytest.raises(AgentRunRecoveryRequired):
        if asynchronous:
            asyncio.run(runner._query_with_recovery_async())
        else:
            runner._query_with_recovery()

    if asynchronous:
        runner.query_async.assert_awaited_once()
        sessions.record_run_failure_async.assert_awaited_once()
    else:
        runner.query.assert_called_once()
        sessions.record_run_failure.assert_called_once()


def test_runner_async_recovery_waits_without_a_run_control(tmp_path, monkeypatch):
    """Retry-after recovery yields through asyncio when no scoped control is configured."""
    runner = file_runner(tmp_path, Mock())
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async.return_value = True
    runner._interaction = interaction
    completed = Response(answer="complete", reasoning="")
    runner.query_async = AsyncMock(
        side_effect=[
            BackendConnectionError(
                "temporary failure",
                provider="test",
                operation="query",
                retry_after=3,
            ),
            completed,
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("loop.agent.runner.asyncio.sleep", sleep)

    result = asyncio.run(runner._query_with_recovery_async())

    assert result is completed
    interaction.confirm_async.assert_awaited_once_with(
        "Retry the complete response after at least 3 seconds?", default=False
    )
    interaction.confirm.assert_not_called()
    sleep.assert_awaited_once_with(3)


@pytest.mark.parametrize(
    "error,prompt_enabled",
    [
        (BackendError("failed", provider="test", operation="query"), True),
        (
            BackendConnectionError("failed", provider="test", operation="query"),
            False,
        ),
    ],
)
def test_runner_async_recovery_does_not_prompt_when_retry_is_unavailable(
    tmp_path, error, prompt_enabled
):
    """Non-recoverable errors and disabled prompting stop without retrying."""
    runner = file_runner(tmp_path, Mock())
    runner.prompt_on_recoverable_error = prompt_enabled
    interaction = MagicMock(spec=Interaction)
    runner._interaction = interaction
    runner.query_async = AsyncMock(side_effect=error)

    result = asyncio.run(runner._query_with_recovery_async())

    assert result is None
    interaction.confirm.assert_not_called()
    runner.query_async.assert_awaited_once_with()


@pytest.mark.parametrize(
    "retry_after,response_started",
    [(None, False), (0, False)],
)
def test_runner_async_recovery_skips_zero_or_missing_retry_delays(
    tmp_path, retry_after, response_started
):
    """A retry without a positive provider delay does not sleep or alter its prompt."""
    runner = file_runner(tmp_path, Mock())
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async.return_value = True
    runner._interaction = interaction
    runner._run_control = None
    completed = Response(answer="complete", reasoning="")
    runner.query_async = AsyncMock(
        side_effect=[
            BackendConnectionError(
                "temporary failure",
                provider="test",
                operation="query",
                retry_after=retry_after,
                response_started=response_started,
            ),
            completed,
        ]
    )

    assert asyncio.run(runner._query_with_recovery_async()) is completed
    interaction.confirm_async.assert_awaited_once_with(
        "Retry the complete response?", default=False
    )
    interaction.confirm.assert_not_called()


def test_runner_async_recovery_retries_repetitive_output_once(tmp_path):
    """Repetition recovery performs one async model retry without a user prompt."""
    runner = file_runner(tmp_path, Mock())
    completed = Response(answer="complete", reasoning="")
    runner._query_async = AsyncMock(
        side_effect=[
            BackendRepetitionError("repetition", provider="test", operation="query"),
            completed,
        ]
    )

    result = asyncio.run(runner._query_with_recovery_async())

    assert result is completed
    assert runner._query_async.await_args_list[0].args == ()
    assert runner._query_async.await_args_list[1].args == ("repetition",)
    runner.interaction.confirm.assert_not_called()


def test_runner_async_recovery_preserves_a_recoverable_repetition_failure(tmp_path):
    """A declined retry after repetition recovery leaves the run available to resume."""
    runner = file_runner(tmp_path, Mock())
    interaction = MagicMock(spec=ConsoleInteraction)
    interaction.confirm_async.return_value = False
    runner._interaction = interaction
    asyncio.run(runner.session_manager.begin_run_async())
    runner._query_async = AsyncMock(
        side_effect=[
            BackendRepetitionError("repetition", provider="test", operation="query"),
            BackendRepetitionError("repetition again", provider="test", operation="query"),
        ]
    )
    with pytest.raises(AgentRunRecoveryRequired):
        asyncio.run(runner._query_with_recovery_async())
    interaction.confirm_async.assert_awaited_once_with(
        "Retry the complete response?", default=False
    )
    interaction.confirm.assert_not_called()


@pytest.mark.parametrize("fallback_selected", [True, False])
def test_runner_async_recovery_handles_unavailable_models(tmp_path, fallback_selected):
    """An unavailable model uses async fallback selection and retries only after selection."""
    runner = file_runner(tmp_path, Mock())
    runner._model_selection.select_fallback_async = AsyncMock(return_value=fallback_selected)
    error = BackendNotFoundError("missing", provider="test", operation="query")
    completed = Response(answer="complete", reasoning="")
    runner.query_async = AsyncMock(side_effect=[error, completed] if fallback_selected else error)

    result = asyncio.run(runner._query_with_recovery_async())

    assert result is (completed if fallback_selected else None)
    runner._model_selection.select_fallback_async.assert_awaited_once_with(runner.interaction)
    assert runner.query_async.await_count == (2 if fallback_selected else 1)


def test_runner_async_recovery_reports_preparation_errors(tmp_path):
    """Request preparation errors are reported and do not prompt for a retry."""
    runner = file_runner(tmp_path, Mock())
    runner.query_async = AsyncMock(side_effect=ModelSelectionError("no model"))

    assert asyncio.run(runner._query_with_recovery_async()) is None
    runner.interaction.report.assert_called_once()
    runner.interaction.confirm.assert_not_called()


def test_runner_run_async_uses_async_query_and_completes():
    """The async run owner collects one response and records its run metrics."""
    runner, sessions, _interaction = agent_runner()
    response = Response(answer="done", reasoning="")
    runner.query_async = AsyncMock(return_value=response)
    runner.query = Mock(side_effect=AssertionError("sync query used"))
    runner.handle_tool_calls_async = AsyncMock(return_value=())

    result = asyncio.run(runner.run_async())

    assert result.final_response is response
    assert result.turns == 1
    assert result.stop_reason == "completed"
    runner.query_async.assert_awaited_once_with()
    runner.query.assert_not_called()
    runner.handle_tool_calls_async.assert_awaited_once_with(response)
    runner.model_selection.record_response_async.assert_awaited_once_with(response.model)
    sessions.add_response.assert_called_once_with(response)
    sessions.record_run.assert_called_once()


def test_runner_sync_entry_points_reject_a_running_event_loop():
    """Sync orchestration entry points reject before touching their dependencies in a loop."""
    runner, sessions, interaction = agent_runner()
    runner._query = Mock(side_effect=AssertionError("sync query reached its backend"))
    methods = (
        ("run", ()),
        ("recover_session", ()),
        ("handle_tool_calls", (Response(answer="", reasoning=""),)),
        ("query", ()),
    )

    async def scenario():
        """Use the matching native async API whenever the caller owns an event loop."""
        for method_name, arguments in methods:
            with pytest.raises(RuntimeError, match="await AgentRunner\\."):
                getattr(runner, method_name)(*arguments)

    asyncio.run(scenario())

    sessions.execution.assert_not_called()
    sessions.begin_run.assert_not_called()
    interaction.confirm.assert_not_called()
    runner._query.assert_not_called()


@pytest.mark.parametrize("async_mode", [False, True])
def test_agent_run_interactions_include_the_owned_session_and_run(async_mode):
    """Both native run paths attach a session and fresh run identity to UI requests."""
    runner, sessions, _interaction = agent_runner()
    source = MagicMock(spec=Interaction)
    requests = []

    def dispatch(request):
        """Capture a marshalled interaction request and invoke its source operation."""
        requests.append(request)
        return request.function(*request.args, **request.kwargs)

    interaction = MailboxInteraction(source)
    interaction.bind(dispatch)
    runner._interaction = interaction
    expected = AgentRunResult(final_response=None, turns=0, stop_reason="completed")

    if async_mode:

        async def boundary(**_kwargs):
            """Emit one interaction from the async run body."""
            interaction.info("async run")
            return expected

        runner._run_from_boundary_async = AsyncMock(side_effect=boundary)
        assert asyncio.run(runner.run_async()) is expected
    else:

        def boundary(**_kwargs):
            """Emit one interaction from the sync run body."""
            interaction.info("sync run")
            return expected

        runner._run_from_boundary = Mock(side_effect=boundary)
        assert runner.run() is expected

    assert len(requests) == 1
    assert requests[0].session_id == sessions.session.id
    assert requests[0].run_id
    assert requests[0].operation_id is None


@pytest.mark.parametrize("async_mode", [False, True])
def test_tool_interactions_include_the_model_call_operation_id(async_mode):
    """Tool UI requests preserve the model call identity alongside session and run IDs."""
    runner, sessions, _interaction = agent_runner()
    source = MagicMock(spec=Interaction)
    requests = []

    def dispatch(request):
        """Capture a marshalled request and invoke the source output operation."""
        requests.append(request)
        return request.function(*request.args, **request.kwargs)

    interaction = MailboxInteraction(source)
    interaction.bind(dispatch)
    runner._interaction = interaction
    tool_call = ToolCall(call_id="call-identity", name="echo", arguments="{}")
    runner.agent.tools.call_with_timing.return_value = ('{"ok":true}', 0.01)
    runner.agent.tools.call_with_timing_async = AsyncMock(return_value=('{"ok":true}', 0.01))

    if async_mode:
        asyncio.run(
            runner.handle_tool_calls_async(
                Response(answer="", reasoning="", tool_calls=(tool_call,))
            )
        )
    else:
        runner.handle_tool_calls(Response(answer="", reasoning="", tool_calls=(tool_call,)))

    tool_request = next(request for request in requests if request.function == source.tool_call)
    assert tool_request.session_id == sessions.session.id
    assert tool_request.run_id
    assert tool_request.operation_id == tool_call.call_id


@pytest.mark.parametrize("async_mode", [False, True])
def test_uncertain_tool_retry_prompt_carries_the_original_operation_id(async_mode):
    """Explicit recovery still confirms uncertain tools against their original operation."""
    final = Response(answer="reconciled", reasoning="")
    runner, sessions, _interaction = agent_runner(responses=[final])
    source = MagicMock(spec=ConsoleInteraction if async_mode else Interaction)
    requests = []

    def dispatch(request):
        """Capture a sync recovery prompt and invoke its source operation."""
        requests.append(request)
        return request.function(*request.args, **request.kwargs)

    async def dispatch_async(request):
        """Capture an async recovery prompt and await its source operation."""
        requests.append(request)
        return await request.function(*request.args, **request.kwargs)

    if async_mode:
        source.confirm_async = AsyncMock(return_value=False)
        runner.query_async = AsyncMock(return_value=final)
    else:
        source.confirm.return_value = False

    interaction = MailboxInteraction(source)
    interaction.bind(dispatch, dispatch_async)
    runner._interaction = interaction
    call = ToolCall(call_id="uncertain-call", name="mutate", arguments="{}")
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )

    if async_mode:
        asyncio.run(runner.recover_session_async(confirmed=True))
    else:
        runner.recover_session(confirmed=True)
        source.confirm.assert_called_once()

    if async_mode:
        source.confirm_async.assert_awaited_once()
    retry_request = next(request for request in requests if "Retry it?" in request.args[0])
    assert retry_request.session_id == sessions.session.id
    assert retry_request.run_id
    assert retry_request.operation_id == call.call_id


@pytest.mark.parametrize(
    ("continue_run", "expected_reason", "expected_turns"),
    [(False, "max_turns", 1), (True, "completed", 2), (object(), "max_turns", 1)],
)
def test_runner_run_async_applies_turn_limit_confirmation(
    continue_run, expected_reason, expected_turns
):
    """Async runs preserve the turn safety prompt and its continuation decision."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    first = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    final = Response(answer="done", reasoning="")
    runner, _sessions, interaction = agent_runner(max_turns=1)
    runner.query_async = AsyncMock(side_effect=[first, final])
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.1, succeeded=True)
    runner.handle_tool_calls_async = AsyncMock(side_effect=[(execution,), ()])
    runner._confirm_async = AsyncMock(return_value=continue_run)

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == expected_reason
    assert result.turns == expected_turns
    assert runner._confirm_async.await_count == 1
    if continue_run is True:
        interaction.warning.assert_not_called()
    else:
        interaction.warning.assert_called_once_with(
            "Agent stopped after reaching the 1-turn safety limit."
        )


def test_runner_run_async_continues_below_turn_safety_limit():
    """Async runs continue without prompting while they remain within the turn limit."""
    call = ToolCall(call_id="call", name="echo", arguments="{}")
    first = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    final = Response(answer="done", reasoning="")
    runner, _sessions, interaction = agent_runner(max_turns=2)
    runner.query_async = AsyncMock(side_effect=[first, final])
    execution = ToolExecutionMetrics(name="echo", duration_seconds=0.1, succeeded=True)
    runner.handle_tool_calls_async = AsyncMock(side_effect=[(execution,), ()])
    runner._confirm_async = AsyncMock()

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == "completed"
    assert result.turns == 2
    runner._confirm_async.assert_not_awaited()
    interaction.warning.assert_not_called()


def test_runner_run_async_records_interruption_as_cancelled():
    """A cooperative async interruption settles pending calls and records a cancelled run."""
    runner, sessions, interaction = agent_runner()
    runner.query_async = AsyncMock(side_effect=RunInterrupted())

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == "cancelled"
    interaction.info.assert_called_once_with("Run interrupted; the conversation remains open.")
    sessions.record_run.assert_called_once()


def test_runner_run_async_stops_when_async_recovery_declines():
    """Declining the asynchronous retry records a cancelled run without another request."""
    runner, sessions, _interaction = agent_runner()
    runner.query_async = AsyncMock(return_value=None)

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == "cancelled"
    assert result.final_response is None
    assert result.turns == 0
    sessions.add_response.assert_not_called()
    sessions.record_run.assert_called_once()


@pytest.mark.parametrize("during_tool", [False, True])
def test_async_steering_finishes_current_work_and_resolves_unstarted_calls(during_tool):
    """Steering preserves a complete response and settles every tool that did not start."""
    control = RunControl()
    first = ToolCall(call_id="first", name="effect", arguments="{}")
    second = ToolCall(call_id="second", name="effect", arguments="{}")
    response = Response(
        answer="",
        reasoning="",
        tool_calls=(first, second),
        items=(first, second),
    )
    runner, sessions, _interaction = agent_runner(run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(
            PendingToolCall(
                call=first, status="result_available" if during_tool else "not_started"
            ),
            PendingToolCall(call=second, status="not_started"),
        ),
    )

    async def query():
        """Finish generation and optionally request steering after the first tool settles."""
        assert control.active
        if not during_tool:
            assert control.request_steering(lambda: True)
        return response

    async def execute(call, **_kwargs):
        """Complete the first invocation, then publish steering before the next call starts."""
        assert call is first
        assert control.request_steering(lambda: True)
        return ToolExecutionMetrics(name=call.name, duration_seconds=0.1, succeeded=True)

    runner.query_async = AsyncMock(side_effect=query)
    runner._execute_tool_call_async = AsyncMock(side_effect=execute)

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == "steered"
    assert result.final_response is response
    sessions.add_response.assert_called_once_with(response)
    if during_tool:
        runner._execute_tool_call_async.assert_awaited_once()
        assert result.metrics.tools == (
            ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True),
        )
        assert sessions.add_tool_call.call_count == 1
        assert sessions.add_tool_call.call_args.kwargs["call_id"] == "second"
    else:
        runner._execute_tool_call_async.assert_not_awaited()
        assert sessions.add_tool_call.call_count == 2
        assert {call.kwargs["call_id"] for call in sessions.add_tool_call.call_args_list} == {
            "first",
            "second",
        }
    skipped_ids = {"second"} if during_tool else {"first", "second"}
    skipped = (
        call
        for call in sessions.add_tool_call.call_args_list
        if call.kwargs["call_id"] in skipped_ids
    )
    assert all(
        json.loads(call.kwargs["output"])["problem"]["code"] == "tool.execution_interrupted"
        for call in skipped
    )
    assert control.steering_pending is False


@pytest.mark.parametrize("during_tool", [False, True])
def test_sync_steering_finishes_current_work_and_resolves_unstarted_calls(during_tool):
    """The synchronous compatibility runner honors the same steering boundaries."""
    control = RunControl()
    first = ToolCall(call_id="first", name="effect", arguments="{}")
    second = ToolCall(call_id="second", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(first, second), items=(first, second))
    runner, sessions, _interaction = agent_runner(run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(
            PendingToolCall(
                call=first, status="result_available" if during_tool else "not_started"
            ),
            PendingToolCall(call=second, status="not_started"),
        ),
    )

    def query():
        """Complete model generation and optionally stop before the first tool."""
        if not during_tool:
            assert control.request_steering(lambda: True)
        return response

    def execute(call, **_kwargs):
        """Finish the first tool before steering prevents the second tool from starting."""
        assert call is first
        assert control.request_steering(lambda: True)
        return ToolExecutionMetrics(name=call.name, duration_seconds=0.1, succeeded=True)

    runner.query = Mock(side_effect=query)
    runner._execute_tool_call = Mock(side_effect=execute)

    result = runner.run()

    assert result.stop_reason == "steered"
    assert result.final_response is response
    sessions.add_response.assert_called_once_with(response)
    if during_tool:
        runner._execute_tool_call.assert_called_once()
        assert result.metrics.tools == (
            ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True),
        )
        assert sessions.add_tool_call.call_count == 1
        assert sessions.add_tool_call.call_args.kwargs["call_id"] == "second"
    else:
        runner._execute_tool_call.assert_not_called()
        assert sessions.add_tool_call.call_count == 2
        assert {call.kwargs["call_id"] for call in sessions.add_tool_call.call_args_list} == {
            "first",
            "second",
        }
    assert control.steering_pending is False


@pytest.mark.parametrize("async_mode", [False, True])
def test_withdrawal_after_steering_boundary_settles_remaining_tools(async_mode):
    """A recalled request cannot reverse a decision to skip the remaining tool batch."""
    control = RunControl()
    first = ToolCall(call_id="first", name="effect", arguments="{}")
    second = ToolCall(call_id="second", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(first, second), items=(first, second))
    runner, sessions, _ = agent_runner(run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=second, status="not_started"),),
    )

    def execute(*_args, **_kwargs):
        """Finish the first tool and request steering before the next one."""
        assert control.request_steering(lambda: True, key=1)
        return '{"ok": true}', 0.1

    async def execute_async(*args, **kwargs):
        """Finish the first tool through the native asynchronous boundary."""
        return execute(*args, **kwargs)

    if async_mode:
        runner.query_async = AsyncMock(return_value=response)
        runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=execute_async)
        handle = runner.handle_tool_calls_async

        async def withdraw_after_batch(value):
            """Withdraw the request after the runner has skipped the second tool."""
            result = await handle(value)
            control.withdraw_steering(1)
            return result

        runner.handle_tool_calls_async = withdraw_after_batch
        result = asyncio.run(runner.run_async())
        assert runner.agent.tools.call_with_timing_async.await_count == 1
    else:
        runner.query = Mock(return_value=response)
        runner.agent.tools.call_with_timing = Mock(side_effect=execute)
        handle = runner.handle_tool_calls

        def withdraw_after_batch(value):
            """Withdraw the request after the runner has skipped the second tool."""
            result = handle(value)
            control.withdraw_steering(1)
            return result

        runner.handle_tool_calls = withdraw_after_batch
        result = runner.run()
        assert runner.agent.tools.call_with_timing.call_count == 1

    assert result.stop_reason == "steered"
    assert len(result.metrics.tools) == 1
    calls = sessions.add_tool_call.call_args_list
    assert [call.kwargs["call_id"] for call in calls] == ["first", "second"]
    assert "cancelled before execution" in calls[1].kwargs["output"]


@pytest.mark.parametrize("async_mode", [False, True])
def test_withdrawal_after_steering_checkpoint_keeps_its_stop_reason(async_mode):
    """A committed checkpoint remains steered when its input is immediately withdrawn."""
    control = RunControl()
    call = ToolCall(call_id="old", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    runner, sessions, _ = agent_runner(run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="execute_tools", pending_calls=(PendingToolCall(call=call, status="not_started"),)
    )
    commit = control.commit_steering

    def commit_then_withdraw():
        """Withdraw after the checkpoint commits but before it raises its interruption."""
        result = commit()
        if result:
            control.withdraw_steering(1)
        return result

    control.commit_steering = commit_then_withdraw

    def execute(*_args, **kwargs):
        """Request steering after approval, then enter the tool execution guard."""
        assert control.request_steering(lambda: True, key=1)
        kwargs["execution_started"]()
        pytest.fail("Steering must prevent the tool from starting")

    async def execute_async(*_args, **kwargs):
        """Enter the asynchronous guard after publishing the steering request."""
        assert control.request_steering(lambda: True, key=1)
        await kwargs["execution_started"]()
        pytest.fail("Steering must prevent the tool from starting")

    if async_mode:
        runner.query_async = AsyncMock(return_value=response)
        runner.agent.tools.call_with_timing_async = AsyncMock(side_effect=execute_async)
        result = asyncio.run(runner.run_async())
    else:
        runner.query = Mock(return_value=response)
        runner.agent.tools.call_with_timing = Mock(side_effect=execute)
        result = runner.run()

    assert result.stop_reason == "steered"
    sessions.record_tool_execution_started.assert_not_called()
    assert "cancelled before execution" in sessions.add_tool_call.call_args.kwargs["output"]


def test_runner_run_async_resolves_recovery_before_querying():
    """A supplied recovery boundary settles its pending tool plan before the next query."""
    runner, _sessions, _interaction = agent_runner()
    recovery = SessionRecoveryState(action="query_model")
    response = Response(answer="done", reasoning="")
    runner._recover_tools_async = AsyncMock()
    runner.query_async = AsyncMock(return_value=response)
    runner.handle_tool_calls_async = AsyncMock(return_value=())

    result = asyncio.run(runner._run_from_boundary_async(recovery=recovery))

    assert result.final_response is response
    runner._recover_tools_async.assert_awaited_once_with(recovery, ANY)


@pytest.mark.parametrize(
    ("status", "approval", "expected_execution", "expected_present", "expected_skip"),
    [
        ("result_available", None, False, None, False),
        ("outcome_unknown", False, False, None, True),
        ("outcome_unknown", True, True, False, False),
        ("not_started", None, True, True, False),
    ],
)
@pytest.mark.parametrize("name", ["effect", "run_command"])
def test_async_recovery_preserves_unknown_outcome_and_pending_boundaries(
    monkeypatch, status, approval, expected_execution, expected_present, expected_skip, name
):
    """Async recovery retries only after approval and preserves already available results."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name=name, arguments="{}")
    recovery = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status=status),),
    )
    observations = Mock(spec=FileObservations)
    monkeypatch.setattr(FileObservations, "for_recovery", Mock(return_value=observations))
    runner._confirm_async = AsyncMock(return_value=approval)
    runner._persist_interrupted_tool_result_async = AsyncMock()
    runner._execute_tool_calls_async = AsyncMock(return_value=())
    tools = []

    asyncio.run(runner._recover_tools_async(recovery, tools))

    assert runner._execute_tool_calls_async.await_count == int(expected_execution)
    if expected_execution:
        assert runner._execute_tool_calls_async.await_args.kwargs["present"] is expected_present
    assert runner._persist_interrupted_tool_result_async.await_count == int(expected_skip)
    assert runner._confirm_async.await_count == int(status == "outcome_unknown")


def test_async_recovery_resolves_observation_paths_in_a_worker(tmp_path, monkeypatch):
    """Async recovery reconstructs file observations away from the event-loop thread."""
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    reference = ContextReference(
        kind="file",
        path="source.txt",
        content="snapshot",
        size_bytes=8,
        included_bytes=8,
        truncated=False,
        version=content_digest("snapshot"),
    )
    runner._session_manager.model_context_async = AsyncMock(
        return_value=[Message(role="user", content="Review", context=(reference,))]
    )

    async def run_blocking(function):
        """Use an injected application worker for synchronous observation recovery."""
        return await asyncio.to_thread(function)

    runner._async_blocking_runner = run_blocking
    owner_thread = get_ident()
    resolution_threads = []
    original_resolve = Path.resolve

    def track_resolve(path, strict=False):
        resolution_threads.append(get_ident())
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", track_resolve)

    asyncio.run(
        runner._recover_tools_async(SessionRecoveryState(action="resolve_uncertain_tools"), [])
    )

    assert resolution_threads
    assert owner_thread not in resolution_threads


def test_async_tool_results_update_file_observations_in_a_worker(tmp_path, monkeypatch):
    """Async result observation canonicalizes file paths outside the owner thread."""
    runner, _sessions, _interaction = agent_runner()
    runner._file_observations = FileObservations((), VirtualPath(tmp_path))
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    output = json.dumps(
        {
            "ok": True,
            "result": "done",
            "file_versions": {"$WORKSPACE_ROOT/source.txt": content_digest("snapshot")},
        }
    )
    runner.agent.tools.call_with_timing_async = AsyncMock(return_value=(output, 0.1))
    owner_thread = get_ident()
    resolution_threads = []
    original_resolve = Path.resolve

    def track_resolve(path, strict=False):
        resolution_threads.append(get_ident())
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", track_resolve)

    result = asyncio.run(
        runner._execute_tool_call_async(
            call,
            instructions=runner._instructions_manager,
            present=False,
        )
    )

    assert result.succeeded
    assert resolution_threads
    assert owner_thread not in resolution_threads


@pytest.mark.parametrize(
    ("error", "persist_result"),
    [
        (
            ProblemException(Problem(code="command.dispatch_busy", title="Busy", detail="Busy")),
            True,
        ),
        (RuntimeError("The bounded worker queue is full."), True),
        (ProblemException(Problem(code="other", title="Failure", detail="Failure")), False),
        (RuntimeError("The worker is unavailable."), False),
    ],
)
def test_async_tool_result_survives_observation_worker_capacity_errors(error, persist_result):
    """A busy ephemeral observation worker cannot discard an executed tool result."""
    runner, sessions, _interaction = agent_runner()
    runner._file_observations = Mock(spec=FileObservations)
    runner.agent.tools.call_with_timing_async = AsyncMock(return_value=('{"ok":true}', 0.1))

    async def run_blocking(_operation):
        """Raise the configured worker outcome."""
        raise error

    runner._async_blocking_runner = run_blocking
    call = ToolCall(call_id="call", name="effect", arguments="{}")

    if persist_result:
        result = asyncio.run(
            runner._execute_tool_call_async(
                call,
                instructions=runner.instructions_manager,
                present=False,
            )
        )
        assert result.succeeded
        sessions.add_tool_call_async.assert_awaited_once()
    else:
        with pytest.raises(type(error)):
            asyncio.run(
                runner._execute_tool_call_async(
                    call,
                    instructions=runner.instructions_manager,
                    present=False,
                )
            )
        sessions.add_tool_call_async.assert_not_awaited()


@pytest.mark.parametrize(
    ("present", "observe", "output", "succeeded"),
    [
        (False, False, '{"ok":true}', True),
        (True, True, "not-json", False),
        (True, False, '{"ok":false}', False),
    ],
)
def test_async_tool_execution_preserves_boundary_records_and_outcomes(
    present, observe, output, succeeded
):
    """Async tool execution keeps start records, observations, and result metrics intact."""
    runner, sessions, interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    observations = Mock(spec=FileObservations) if observe else None
    runner._file_observations = observations
    runner._run_control = RunControl()
    runner.agent.tools.call_with_timing_async = AsyncMock(return_value=(output, 0.25))

    result = asyncio.run(
        runner._execute_tool_call_async(
            call,
            instructions=runner.instructions_manager,
            present=present,
        )
    )

    kwargs = runner.agent.tools.call_with_timing_async.await_args.kwargs
    assert kwargs["cancellation_event"] is runner._run_control.cancellation_event
    asyncio.run(kwargs["execution_started"]())
    assert result.succeeded is succeeded
    assert result.duration_seconds == 0.25
    if present:
        sessions.add_tool_call_event.assert_called_once_with(call.call_id)
    else:
        sessions.add_tool_call_event.assert_not_called()
    sessions.record_tool_execution_started.assert_called_once_with(call.call_id)
    sessions.add_tool_call.assert_called_once()
    if observe:
        observations.observe_result.assert_called_once_with(call, output, include_reads=False)
    else:
        assert interaction.tool_call.call_count == 1


@pytest.mark.parametrize("track_interrupted_tools", [False, True])
def test_async_tool_batch_retains_settled_metrics_when_interrupted(track_interrupted_tools):
    """An interrupted async batch keeps metrics for tools that already settled."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    settled = ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True)
    runner._execute_tool_call_async = AsyncMock(side_effect=[settled, RunInterrupted()])
    runner._interrupted_tools = [] if track_interrupted_tools else None

    with pytest.raises(RunInterrupted):
        asyncio.run(runner._execute_tool_calls_async((call, call), present=True))

    if track_interrupted_tools:
        assert runner._interrupted_tools == [settled]


@pytest.mark.parametrize("track_interrupted_tools", [False, True])
def test_async_tool_batch_retains_settled_metrics_when_task_is_cancelled(track_interrupted_tools):
    """Task cancellation does not discard metrics for previously settled tools."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    settled = ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True)
    runner._execute_tool_call_async = AsyncMock(side_effect=[settled, asyncio.CancelledError()])
    runner._interrupted_tools = [] if track_interrupted_tools else None

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._execute_tool_calls_async((call, call), present=True))

    if track_interrupted_tools:
        assert runner._interrupted_tools == [settled]


def test_async_tool_batch_returns_all_settled_metrics():
    """An uninterrupted asynchronous tool batch returns results in request order."""
    runner, _sessions, _interaction = agent_runner()
    calls = (
        ToolCall(call_id="first", name="effect", arguments="{}"),
        ToolCall(call_id="second", name="effect", arguments="{}"),
    )
    first = ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True)
    second = ToolExecutionMetrics(name="effect", duration_seconds=0.2, succeeded=False)
    runner._execute_tool_call_async = AsyncMock(side_effect=[first, second])

    result = asyncio.run(runner._execute_tool_calls_async(calls, present=True))

    assert result == (first, second)
    assert runner._execute_tool_call_async.await_count == 2


def test_handle_tool_calls_async_skips_empty_responses_and_owns_nonempty_execution():
    """Async tool dispatch skips empty responses and leases nonempty calls."""
    runner, sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    runner._execute_tool_calls_async = AsyncMock(return_value=())

    assert asyncio.run(runner.handle_tool_calls_async(Response(answer="done", reasoning=""))) == ()
    runner._execute_tool_calls_async.assert_not_awaited()

    assert (
        asyncio.run(
            runner.handle_tool_calls_async(
                Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
            )
        )
        == ()
    )
    runner._execute_tool_calls_async.assert_awaited_once_with((call,), present=True)
    assert sessions.execution.call_count == 1


@pytest.mark.parametrize(
    ("has_recovery", "approval", "expected_pending"),
    [(False, None, False), (True, False, True), (True, True, False)],
)
def test_recover_session_async_preserves_approval_gate(has_recovery, approval, expected_pending):
    """Async session recovery asks first and runs only after explicit approval."""
    runner, sessions, _interaction = agent_runner()
    state = SessionRecoveryState(action="finalize_run") if has_recovery else None
    sessions.recovery_state = state
    recovered = AgentRunResult(final_response=None, turns=0, stop_reason="completed")
    runner._confirm_async = AsyncMock(return_value=approval)
    runner._recover_async = AsyncMock(return_value=recovered)

    result = asyncio.run(runner.recover_session_async())

    assert result.pending is expected_pending
    if has_recovery:
        runner._confirm_async.assert_awaited_once_with(
            "This session stopped during an agent run. Recover and continue it?",
            default=False,
        )
    else:
        runner._confirm_async.assert_not_awaited()
    if has_recovery and approval:
        runner._recover_async.assert_awaited_once_with(state)
        assert result.result is recovered
    else:
        runner._recover_async.assert_not_awaited()


def test_recover_session_async_skips_only_the_run_level_prompt_when_explicitly_confirmed():
    """An explicit recovery command bypasses the general prompt but keeps normal recovery."""
    runner, sessions, _interaction = agent_runner()
    sessions.recovery_state = SessionRecoveryState(action="query_model")
    recovered = AgentRunResult(final_response=None, turns=0, stop_reason="completed")
    runner._recover_async = AsyncMock(return_value=recovered)
    runner._confirm_async = AsyncMock()

    result = asyncio.run(runner.recover_session_async(confirmed=True))

    assert result.pending is False
    assert result.result is recovered
    runner._confirm_async.assert_not_awaited()
    runner._recover_async.assert_awaited_once_with(sessions.recovery_state)


@pytest.mark.parametrize(("active_run", "retry_pending"), [(False, False), (True, True)])
def test_recover_session_starts_only_the_required_durable_run_transition(
    active_run,
    retry_pending,
):
    """Sync recovery starts a run after cold stops and records authorized retries."""
    runner, sessions, interaction = agent_runner()
    state = SessionRecoveryState(action="query_model")
    sessions.recovery_state = state
    sessions.retry_pending = retry_pending
    recovered = AgentRunResult(final_response=None, turns=0, stop_reason="completed")
    runner._recover = Mock(return_value=recovered)
    interaction.confirm.return_value = True
    if active_run:
        sessions.begin_run()
        sessions.session.record_run_failure()
    sessions.begin_run.reset_mock()

    result = runner.recover_session()

    assert result.pending is False
    assert result.result is recovered
    if active_run:
        sessions.begin_run.assert_not_called()
    else:
        sessions.begin_run.assert_called_once_with()
    if retry_pending:
        sessions.begin_run_retry.assert_called_once_with()
    else:
        sessions.begin_run_retry.assert_not_called()
    runner._recover.assert_called_once_with(state)


@pytest.mark.parametrize(("active_run", "retry_pending"), [(False, False), (True, True)])
def test_recover_session_async_starts_only_the_required_durable_run_transition(
    active_run,
    retry_pending,
):
    """Async recovery starts cold runs and persists retry authorization before resuming."""
    runner, sessions, _interaction = agent_runner()
    state = SessionRecoveryState(action="query_model")
    sessions.recovery_state = state
    sessions.retry_pending = retry_pending
    recovered = AgentRunResult(final_response=None, turns=0, stop_reason="completed")
    runner._recover_async = AsyncMock(return_value=recovered)
    if active_run:
        sessions.begin_run()
        sessions.session.record_run_failure()
    sessions.begin_run.reset_mock()

    result = asyncio.run(runner.recover_session_async(confirmed=True))

    assert result.pending is False
    assert result.result is recovered
    if active_run:
        sessions.begin_run.assert_not_called()
    else:
        sessions.begin_run_async.assert_awaited_once_with()
    if retry_pending:
        sessions.begin_run_retry_async.assert_awaited_once_with()
    else:
        sessions.begin_run_retry_async.assert_not_awaited()
    runner._recover_async.assert_awaited_once_with(state)


def test_recoverable_async_backend_failure_leaves_an_uncompleted_run():
    """Declining a recoverable backend retry leaves the run available to ``/retry``."""
    runner, sessions, interaction = agent_runner()
    runner.query_async = AsyncMock(
        side_effect=BackendConnectionError("private failure", provider="test", operation="query")
    )
    runner._confirm_async = AsyncMock(return_value=False)

    result = asyncio.run(runner.run_async())

    assert result.stop_reason == "recovery_required"
    assert result.metrics is not None
    interaction.report.assert_called_once()
    sessions.record_run_async.assert_not_awaited()
    sessions.record_run_failure_async.assert_awaited_once()


def test_runner_cold_stop_returns_metrics_without_completing_the_run():
    """A synchronous cold stop preserves the run boundary without cancellation."""
    runner, sessions, interaction = agent_runner(run_control=RunControl())
    runner._query_with_recovery = Mock(side_effect=RunStopRequested)

    result = runner.run()

    assert result.stop_reason == "stopped"
    assert result.metrics is not None
    sessions.record_run.assert_not_called()
    interaction.info.assert_any_call("Run stopped; its durable state is ready to continue.")


def test_runner_does_not_cancel_tools_without_a_recovery_state():
    """Cancellation before durable run state exists has no tool calls to settle."""
    runner, _sessions, _interaction = agent_runner()
    runner.session_manager.recovery_state = None

    runner._cancel_pending_tools()

    runner.session_manager.add_tool_call.assert_not_called()


def test_recover_async_finalizes_or_reenters_query_boundary():
    """Async recovery either finalizes a completed run or reenters model orchestration."""
    runner, sessions, _interaction = agent_runner()
    completed = SessionRecoveryState(action="finalize_run")

    finalized = asyncio.run(runner._recover_async(completed))

    assert finalized.stop_reason == "completed"
    sessions.record_run.assert_called_once()

    recovery = SessionRecoveryState(action="query_model")
    result = object()
    runner._run_from_boundary_async = AsyncMock(return_value=result)
    assert asyncio.run(runner._recover_async(recovery)) is result
    runner._run_from_boundary_async.assert_awaited_once_with(recovery=recovery)


@pytest.mark.parametrize("storage_failure", [False, True])
def test_runner_async_query_closes_backend_stream_when_cancelled(
    tmp_path, monkeypatch, storage_failure
):
    """Close cancelled streams and propagate setup errors without waiting for stream startup."""
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    waiting = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        """Pause after partial output and signal when cancellation closes the stream."""
        try:
            yield AnswerDelta(text="partial")
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    backend.get_response_async.return_value = events()

    async def query():
        """Cancel a query at a deterministic point in its event stream."""
        async with asyncio.TaskGroup() as tasks:
            task = tasks.create_task(runner.query_async())
            await waiting.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    if storage_failure:
        error = ValueError("Session working storage recovery failed: namespace unavailable.")
        monkeypatch.setattr(sessions, "execution", Mock(side_effect=error))
        with pytest.raises(ExceptionGroup) as captured:
            asyncio.run(query())
        assert captured.value.exceptions == (error,)
        assert not waiting.is_set()
        assert not closed.is_set()
        runner.interaction.answer_delta.assert_not_called()
    else:
        asyncio.run(query())
        assert closed.is_set()
        runner.interaction.answer_delta.assert_called_once_with("partial")
    assert sessions.messages == []


@pytest.mark.parametrize("mode", ["interrupt", "stop", "external"])
def test_controlled_async_run_cancels_stream_without_persisting_partial_response(tmp_path, mode):
    """Control cancellation closes the stream; external task cancellation still propagates."""
    control = RunControl()
    backend = Mock()
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.stream = True
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    sessions.add_user_message("question")
    waiting = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        """Pause after a delta so each cancellation path reaches an active stream."""
        try:
            yield AnswerDelta(text="partial")
            waiting.set()
            await asyncio.Event().wait()
            yield ResponseCompleted(answer="discarded")
        finally:
            closed.set()

    backend.get_response_async.return_value = events()

    async def scenario():
        """Cancel the owned run or its task while model streaming is suspended."""
        task = asyncio.create_task(runner.run_async())
        await asyncio.wait_for(waiting.wait(), timeout=1)
        if mode == "interrupt":
            assert control.interrupt() is True
        elif mode == "stop":
            control.stop()
        else:
            task.cancel()
        if mode == "external":
            with pytest.raises(asyncio.CancelledError):
                await task
            return None
        return await asyncio.wait_for(task, timeout=1)

    result = asyncio.run(scenario())

    assert closed.is_set()
    assert [item.content for item in sessions.messages] == ["question"]
    runner.interaction.answer_delta.assert_called_once_with("partial")
    if mode == "external":
        assert result is None
        assert control.cancelled is False
        assert not any(isinstance(event, RunCompletedEvent) for event in sessions.session.events)
    else:
        assert result is not None
        assert result.stop_reason == "cancelled"
        assert result.final_response is None
        assert result.turns == 0
        assert control.cancelled is True
        assert [
            event.stop_reason
            for event in sessions.session.events
            if isinstance(event, RunCompletedEvent)
        ] == ["cancelled"]
        if mode == "interrupt":
            runner.interaction.info.assert_called_once_with(
                "Run interrupted; the conversation remains open."
            )


async def wait_for_worker_event(event, worker):
    """Wait for a checkpoint or report a worker that settled before reaching it."""
    waiter = asyncio.create_task(event.wait())
    try:
        await asyncio.wait((waiter, worker), return_when=asyncio.FIRST_COMPLETED)
        if worker.done():
            await worker
            raise AssertionError("Worker finished before the expected checkpoint.")
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["checkpoint", "failure", "finished", "cancelled"])
def test_wait_for_worker_event_observes_early_settlement(outcome):
    """Checkpoint waits report early worker failures and release their own waiter."""

    async def scenario():
        """Exercise each worker outcome without timers or external scheduling."""
        event = asyncio.Event()

        async def work():
            """Reach the selected checkpoint or settle before it."""
            if outcome == "failure":
                raise RuntimeError("worker setup failed")
            if outcome == "finished":
                return
            if outcome == "cancelled":
                raise asyncio.CancelledError
            event.set()
            await asyncio.Event().wait()

        worker = asyncio.create_task(work())
        try:
            if outcome == "failure":
                with pytest.raises(RuntimeError, match="worker setup failed"):
                    await wait_for_worker_event(event, worker)
            elif outcome == "finished":
                with pytest.raises(AssertionError, match="before the expected checkpoint"):
                    await wait_for_worker_event(event, worker)
            elif outcome == "cancelled":
                with pytest.raises(asyncio.CancelledError):
                    await wait_for_worker_event(event, worker)
            else:
                await wait_for_worker_event(event, worker)
                assert event.is_set()
                assert not worker.done()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "previous_batch"),
    [
        ("interrupt", False),
        ("interrupt", True),
        ("stop", False),
        ("stop", True),
        ("external", False),
        ("standalone", False),
    ],
)
@pytest.mark.parametrize("successful", [False, True])
def test_async_cancellation_persists_a_settled_sync_tool_receipt(
    tmp_path, mode, successful, previous_batch
):
    """Interrupted effects retain exact receipts and ordered metrics across completed batches."""
    control = RunControl()
    backend = Mock()
    registry = ToolRegistry()
    started = asyncio.Event()
    release = Event()
    owner_loop = None
    receipt = (
        {"receipt": "confirmed-receipt-123"}
        if successful
        else Problem(code="effect.failed", title="Effect failed", detail="confirmed-receipt-123")
    )
    ran = []

    def save_receipt():
        """Finish a controlled effect after the run receives cancellation."""
        owner_loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test synchronization was not released"
        (tmp_path / "receipt.txt").write_text("confirmed-receipt-123", encoding="utf-8")
        return receipt

    def later_effect():
        """Record any forbidden work after the interrupted effect."""
        ran.append(True)
        return {"ok": True}

    def first_effect():
        """Complete an earlier invocation in the interrupted batch."""
        return "first receipt"

    def previous_effect():
        """Complete an invocation in a previous model-response batch."""
        return "previous receipt"

    registry.register(first_effect)
    registry.register(previous_effect)
    registry.register(save_receipt)
    registry.register(later_effect)
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    sessions = SessionManager(session_store=store, workspace_id="workspace")
    runner = file_runner(
        tmp_path, backend, registry=registry, sessions=sessions, run_control=control
    )
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions.add_user_message("perform effects")
    calls = (
        ToolCall(call_id="first", name="first_effect", arguments="{}"),
        ToolCall(call_id="effect", name="save_receipt", arguments="{}"),
        ToolCall(call_id="later", name="later_effect", arguments="{}"),
    )

    async def events(batch):
        """Return a complete model response requesting two effects in order."""
        for event in model_calls(*batch):
            yield event

    previous = (ToolCall(call_id="previous", name="previous_effect", arguments="{}"),)
    if mode != "standalone":
        backend.get_response_async.side_effect = (
            [events(previous), events(calls)] if previous_batch else [events(calls)]
        )
    publishing, publish = asyncio.Event(), asyncio.Event()
    record_result = sessions.add_tool_call_async

    async def hold_publication(**kwargs):
        """Hold result persistence so repeated cancellation reaches the publication boundary."""
        if kwargs["call_id"] == "effect":
            publishing.set()
            await publish.wait()
        return await record_result(**kwargs)

    sessions.add_tool_call_async = hold_publication

    async def scenario():
        """Cancel a worker, cancel again during publication, and settle the durable outcome."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        if mode == "standalone":
            await sessions.begin_run_async()
            response = Response(answer="", reasoning="", items=calls, tool_calls=calls)
            await sessions.add_response_async(response)
            task = asyncio.create_task(runner.handle_tool_calls_async(response))
        else:
            task = asyncio.create_task(runner.run_async())
        try:
            await wait_for_worker_event(started, task)
            if mode == "interrupt":
                assert control.interrupt()
            elif mode == "stop":
                control.stop()
            else:
                task.cancel()
            await deliver_callbacks()
            release.set()
            await wait_for_worker_event(publishing, task)
            task.cancel()
            publish.set()
            if mode in {"external", "standalone"}:
                with pytest.raises(asyncio.CancelledError):
                    await task
                return None
            return await task
        finally:
            release.set()
            publish.set()
            await asyncio.gather(task, return_exceptions=True)

    try:
        result = asyncio.run(scenario())
    finally:
        registry.close()
    assert (tmp_path / "receipt.txt").read_text(encoding="utf-8") == "confirmed-receipt-123"
    saved = store.load(sessions.session.id)
    outcomes = [item for item in saved.messages if isinstance(item, ToolResult)]
    expected_output = (
        {"ok": True, "result": receipt}
        if successful
        else {"ok": False, "problem": receipt.model_dump(mode="json")}
    )
    completed_ids = (["previous"] if previous_batch else []) + ["first", "effect"]
    assert [item.call_id for item in outcomes[: len(completed_ids)]] == completed_ids
    assert (
        json.loads(next(item.output for item in outcomes if item.call_id == "effect"))
        == expected_output
    )
    assert (
        sum(
            event.type == "tool_execution_completed" and event.call_id == "effect"
            for event in saved.events
        )
        == 1
    )
    assert ran == []
    if result is not None:
        assert result.stop_reason == "cancelled"
        expected_names = (["previous_effect"] if previous_batch else []) + [
            "first_effect",
            "save_receipt",
        ]
        assert [metric.name for metric in result.metrics.tools] == expected_names
        assert result.metrics.tools[-1].succeeded is successful
        assert all(metric.duration_seconds >= 0 for metric in result.metrics.tools)
        assert "cancelled before execution" in outcomes[-1].output
    else:
        assert saved.recovery_state().action == "execute_tools"


@pytest.mark.parametrize("mode", ["interrupt", "stop", "external"])
@pytest.mark.parametrize(
    ("kind", "after_return"),
    [
        ("completed_task", False),
        ("completed_future", False),
        ("pending_task", False),
        ("pending_task", True),
        ("recovering_task", False),
        ("recovering_task", True),
        ("pending_future", False),
        ("pending_future", True),
    ],
)
def test_cancellation_settles_returned_owner_loop_work_before_the_run_ends(
    tmp_path, monkeypatch, mode, kind, after_return
):
    """Returned tasks and futures settle under repeated cancellation and retain known receipts."""
    control = RunControl()
    registry = ToolRegistry()
    backend = Mock()
    worker_ready, awaiting_result, effect_started, cancelling = (asyncio.Event() for _ in range(4))
    release_worker = Event()
    finish_cleanup = asyncio.Event()
    owned_results, cancelled, later = [], [], []
    owner_loop = None
    receipt = "owned-receipt"
    shield = asyncio.shield

    def observe_shield(value):
        """Expose when the returned future is being awaited by its public dispatch owner."""
        if owned_results and value is owned_results[0]:
            awaiting_result.set()
        return shield(value)

    monkeypatch.setattr(asyncio, "shield", observe_shield)

    async def effect():
        """Finish immediately or settle cancellation under an explicitly released cleanup gate."""
        effect_started.set()
        if kind != "completed_task":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                cancelling.set()
                await finish_cleanup.wait()
                if kind != "recovering_task":
                    raise
        (tmp_path / "effect.txt").write_text(receipt, encoding="utf-8")
        return receipt

    async def create_result():
        """Construct a task or future on the invocation's actual owning event loop."""
        handle = None
        if kind.endswith("future"):
            result = asyncio.get_running_loop().create_future()
            if kind == "completed_future":
                result.set_result(receipt)
        else:
            handle = OwnedToolTask(effect())
            result = handle.start(asyncio.get_running_loop())
            await effect_started.wait()
            if kind == "completed_task":
                await result
        owned_results.append(result)
        return handle if handle is not None else result

    def returned_work():
        """Return already-scheduled owner-loop work from a synchronous tool."""
        result = asyncio.run_coroutine_threadsafe(create_result(), owner_loop).result()
        owner_loop.call_soon_threadsafe(worker_ready.set)
        assert release_worker.wait(timeout=5), "test synchronization was not released"
        return result

    def later_effect():
        """Record any forbidden invocation after cancellation."""
        later.append(True)
        return "unexpected"

    registry.register(returned_work)
    registry.register(later_effect)
    store = SQLiteSessionStore(tmp_path / "sessions.db", workspace_id="workspace")
    sessions = SessionManager(session_store=store, workspace_id="workspace")
    runner = file_runner(
        tmp_path, backend, registry=registry, sessions=sessions, run_control=control
    )
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions.add_user_message("perform effects")
    calls = (
        ToolCall(call_id="owned", name="returned_work", arguments="{}"),
        ToolCall(call_id="later", name="later_effect", arguments="{}"),
    )

    async def events():
        """Request the owned invocation followed by a tool that must never run."""
        for event in model_calls(*calls):
            yield event

    backend.get_response_async.return_value = events()

    async def scenario():
        """Cancel before or after the worker returns, then wait for its returned work to settle."""
        nonlocal owner_loop
        owner_loop = asyncio.get_running_loop()
        task = asyncio.create_task(runner.run_async())
        try:
            await worker_ready.wait()
            if after_return:
                release_worker.set()
                await awaiting_result.wait()
            if mode == "interrupt":
                assert control.interrupt()
            elif mode == "stop":
                control.stop()
            else:
                task.cancel()
            await deliver_callbacks()
            release_worker.set()
            if kind in {"pending_task", "recovering_task"}:
                await asyncio.wait_for(cancelling.wait(), timeout=1)
                assert not task.done()
                task.cancel()
                await deliver_callbacks()
                assert not owned_results[0].done()
                finish_cleanup.set()
            elif kind == "pending_future":
                assert not owned_results[0].done()
                owned_results[0].cancel()
            if mode == "external":
                with pytest.raises(asyncio.CancelledError):
                    await task
                return None
            return await task
        finally:
            release_worker.set()
            finish_cleanup.set()
            for result in owned_results:
                if not result.done():
                    result.cancel()
            await asyncio.gather(task, *owned_results, return_exceptions=True)

    try:
        result = asyncio.run(scenario())
    finally:
        registry.close()
    assert owned_results[0].done()
    assert cancelled == ([True] if kind in {"pending_task", "recovering_task"} else [])
    assert later == []
    saved = store.load(sessions.session.id)
    output = next(
        (
            item.output
            for item in saved.messages
            if isinstance(item, ToolResult) and item.call_id == "owned"
        ),
        None,
    )
    known_result = kind in {"completed_task", "completed_future", "recovering_task"}
    if known_result:
        assert json.loads(output) == {"ok": True, "result": receipt}
        assert (
            sum(
                event.type == "tool_execution_completed" and event.call_id == "owned"
                for event in saved.events
            )
            == 1
        )
    elif mode == "external":
        assert output is None
        assert saved.recovery_state().action == "resolve_uncertain_tools"
    else:
        assert "external outcome is unknown" in output
    if kind.endswith("task"):
        assert (tmp_path / "effect.txt").exists() is known_result
    if result is not None:
        assert [metric.name for metric in result.metrics.tools] == (
            ["returned_work"] if known_result else []
        )
        assert result.stop_reason == "cancelled"


def test_async_interruption_preserves_completed_unknown_and_unstarted_tool_outcomes(tmp_path):
    """Cancellation keeps completed effects, marks active outcomes unknown, and skips later calls."""
    control = RunControl()
    backend = Mock()
    completed = ToolCall(call_id="completed", name="completed_effect", arguments="{}")
    active = ToolCall(call_id="active", name="active_effect", arguments="{}")
    unstarted = ToolCall(call_id="unstarted", name="unstarted_effect", arguments="{}")
    calls = (completed, active, unstarted)
    registry = ToolRegistry()
    started = asyncio.Event()
    never_ran = []

    async def completed_effect():
        """Complete one tool effect before the active call begins."""
        return {"ok": True}

    async def active_effect():
        """Wait until the controller interrupts this in-flight operation."""
        started.set()
        await asyncio.Event().wait()

    async def unstarted_effect():
        """Record an error if cancellation incorrectly starts a later tool."""
        never_ran.append(True)
        return {"ok": True}

    registry.register(completed_effect)
    registry.register(active_effect)
    registry.register(unstarted_effect)
    runner = file_runner(tmp_path, backend, registry=registry, run_control=control)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    sessions.add_user_message("perform these effects")

    async def events():
        """Return one completed model response that requests the ordered tool calls."""
        for call in calls:
            yield ToolCallCompleted(call=call)
        yield ResponseCompleted(items=calls)

    backend.get_response_async.return_value = events()

    async def scenario():
        """Cancel while the second tool is running, before the third may launch."""
        task = asyncio.create_task(runner.run_async())
        await asyncio.wait_for(started.wait(), timeout=1)
        assert control.interrupt() is True
        return await asyncio.wait_for(task, timeout=1)

    result = asyncio.run(scenario())
    tool_results = {
        item.call_id: json.loads(item.output)
        for item in sessions.messages
        if isinstance(item, ToolResult)
    }

    assert result.stop_reason == "cancelled"
    assert never_ran == []
    assert tool_results["completed"]["ok"] is True
    assert "external outcome is unknown" in tool_results["active"]["problem"]["detail"]
    assert "cancelled before execution" in tool_results["unstarted"]["problem"]["detail"]


def test_runner_async_query_accepts_iterators_without_close(tmp_path):
    """Async queries accept backend iterators that do not expose a close method."""
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager

    class EventStream:
        """Yield normalized response events without an async close operation."""

        def __init__(self):
            self._events = iter([ResponseCompleted(answer="final")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    backend.get_response_async.return_value = EventStream()

    async def query():
        """Run the query while holding the session's async-task execution ownership."""
        with sessions.execution():
            return await runner.query_async()

    assert asyncio.run(query()).answer == "final"


def test_runner_checks_cancellation_after_compaction(tmp_path):
    """Cancelling during compaction prevents the following model request."""
    control = RunControl()
    backend = Mock()
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.compaction.compact_if_needed.side_effect = lambda _snapshot: control.stop()

    with pytest.raises(KeyboardInterrupt):
        runner.query()

    backend.get_response.assert_not_called()


def test_cancelled_response_closes_its_source_and_preserves_recovery(tmp_path):
    """A stopped partial response closes transport and never enters canonical session history."""
    control = RunControl()
    closed = Mock()

    def events():
        """Request cancellation between a displayed delta and terminal completion."""
        try:
            yield AnswerDelta(text="partial")
            control.stop()
            yield ResponseCompleted(answer="discarded")
        finally:
            closed()

    backend = Mock()
    backend.get_response.return_value = events()
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.session_manager.add_user_message("question")

    with pytest.raises(KeyboardInterrupt):
        runner.run()

    closed.assert_called_once_with()
    assert [item.content for item in runner.session_manager.messages] == ["question"]
    assert runner.session_manager.recovery_state.action == "query_model"
    with runner.session_manager.execution():
        assert runner.session_manager.session.id


def test_cancellation_after_authorization_prevents_the_tool_execution_boundary():
    """A stopped approved operation cannot record a start or invoke its function."""
    control = RunControl()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    runner, sessions, _interaction = agent_runner(
        responses=[Response(answer="", reasoning="", tool_calls=(call,))],
        run_control=control,
    )
    effects = []

    def dispatch(_name, _arguments, **kwargs):
        """Stop after planning and approval but before the registry starts execution."""
        control.stop()
        kwargs["execution_started"]()
        effects.append("executed")
        return '{"ok":true}', 0

    runner.agent.tools.call_with_timing.side_effect = dispatch

    with pytest.raises(KeyboardInterrupt):
        runner.run()

    assert effects == []
    sessions.record_tool_execution_started.assert_not_called()
    sessions.add_tool_call.assert_not_called()


def test_controlled_retry_wait_is_cancellable(caplog):
    """Recoverable failures retain their diagnostic while retry waits use the stop signal."""
    control = Mock(spec=RunControl)
    control.execution.side_effect = nullcontext
    control.wait.side_effect = KeyboardInterrupt
    runner, _sessions, interaction = agent_runner(
        responses=[
            BackendConnectionError(
                "private failure", provider="test", operation="query", retry_after=2.5
            )
        ],
        run_control=control,
    )
    interaction.confirm.return_value = True

    with pytest.raises(KeyboardInterrupt):
        runner.run()

    control.wait.assert_called_once_with(2.5)
    assert [getattr(record, "error.type", None) for record in caplog.records] == [
        "backend.request_failed"
    ]
    assert "private failure" not in caplog.text


def test_interrupted_stream_is_closed_cancelled_and_ready_for_another_run(tmp_path):
    """A scoped interruption discards partial output and leaves a clean durable continuation."""
    control = RunControl()
    closed = Mock()

    def events():
        """Interrupt after a displayed delta, before authoritative response completion."""
        try:
            yield AnswerDelta(text="partial")
            assert control.interrupt() is True
            yield ResponseCompleted(answer="discarded")
        finally:
            closed()

    backend = Mock()
    backend.get_response.side_effect = [events(), [ResponseCompleted(answer="corrected")]]
    runner = file_runner(tmp_path, backend, run_control=control)
    sessions = runner.session_manager
    sessions.add_user_message("question")

    result = runner.run()

    closed.assert_called_once_with()
    assert result.stop_reason == "cancelled"
    assert result.turns == 0
    assert result.final_response is None
    assert sessions.recovery_state is None
    assert [item.content for item in sessions.messages] == ["question"]
    assert control.active is False
    assert control.stopped is False
    sessions.add_user_message("correction")
    assert runner.run().stop_reason == "completed"
    assert backend.get_response.call_count == 2


def test_synchronous_pause_resumes_the_same_stream_at_a_safe_checkpoint(tmp_path):
    """A synchronous pause parks the existing stream and resumes it without cancellation."""
    control = RunControl()
    closed = Mock()
    pause_requested = Event()

    def events():
        """Pause after partial output, then complete after the control resumes."""
        try:
            yield AnswerDelta(text="partial")
            assert control.pause() is True
            pause_requested.set()
            yield ResponseCompleted(answer="continued")
        finally:
            closed()

    backend = Mock()
    backend.get_response.side_effect = [events()]
    runner = file_runner(tmp_path, backend, run_control=control)
    sessions = runner.session_manager
    sessions.add_user_message("question")

    def continue_paused_run():
        """Resume only after the stream parks, with a bounded failure path."""
        assert pause_requested.wait(timeout=5), "test stream did not pause"
        control.continue_run()

    resume = Thread(target=continue_paused_run)
    resume.start()

    result = runner.run()
    resume.join(timeout=5)
    assert not resume.is_alive()

    closed.assert_called_once_with()
    assert result.stop_reason == "completed"
    assert result.turns == 1
    assert sessions.session.active_run_id is None
    assert sessions.recovery_state is None
    assert any(isinstance(event, RunCompletedEvent) for event in sessions.session.events)
    assert control.paused is False


def test_async_pause_preserves_the_stream_and_run_owner_until_continue(tmp_path):
    """An async pause leaves an active stream open until the same owner can continue."""
    control = RunControl()
    backend = Mock()
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        """Yield a final event only after the test has observed a live paused stream."""
        try:
            yield AnswerDelta(text="partial")
            started.set()
            await release.wait()
            yield ResponseCompleted(answer="continued")
        finally:
            closed.set()

    backend.get_response_async.return_value = events()
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    sessions.add_user_message("question")

    async def scenario():
        """Pause without cancelling the owner, then continue the same stream."""
        task = asyncio.create_task(runner.run_async())
        await started.wait()
        assert control.pause() is True
        assert not task.done()
        assert not closed.is_set()
        release.set()
        assert control.continue_run() is True
        return await task

    result = asyncio.run(scenario())

    assert closed.is_set()
    assert result.stop_reason == "completed"
    assert sessions.session.active_run_id is None
    assert sessions.recovery_state is None
    assert any(isinstance(event, RunCompletedEvent) for event in sessions.session.events)
    assert control.paused is False


def test_async_cold_stop_preserves_the_durable_run_and_does_not_cancel_stream(tmp_path):
    """A cold stop settles after the active stream without cancelling its process boundary."""
    control = RunControl()
    backend = Mock()
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        """Pause the stream until the cold stop has been requested."""
        try:
            yield AnswerDelta(text="partial")
            started.set()
            await release.wait()
            yield ResponseCompleted(answer="discarded")
        finally:
            closed.set()

    backend.get_response_async.return_value = events()
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    sessions.add_user_message("question")

    async def scenario():
        """Request a cold stop and allow the active response operation to settle."""
        task = asyncio.create_task(runner.run_async())
        await started.wait()
        assert control.request_stop_run() is True
        assert control.cancellation_event.is_set() is False
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert closed.is_set()
    assert result.stop_reason == "stopped"
    assert sessions.session.active_run_id is not None
    assert sessions.recovery_state.action == "query_model"
    assert not any(isinstance(event, RunCompletedEvent) for event in sessions.session.events)
    assert control.cancellation_event.is_set() is False


def test_interruption_preserves_settled_tool_results_and_cancels_remaining_calls(
    tmp_path, monkeypatch
):
    """A completed effect is saved once; later calls receive cancellation results and never run."""
    control = RunControl()
    target = tmp_path / "file.txt"
    target.write_text("original", encoding="utf-8")
    calls = tuple(
        ToolCall(
            call_id=name, name="read_text_file", arguments='{"path":"$WORKSPACE_ROOT/file.txt"}'
        )
        for name in ("first", "second")
    )
    backend = Mock()
    backend.get_response.return_value = model_calls(*calls)
    runner = file_runner(tmp_path, backend, run_control=control)
    runner.session_manager.add_user_message("read twice")
    dispatch = runner.agent.tools.call_with_timing

    def interrupt_after_tool(*args, **kwargs):
        """Request interruption after a tool settles but before the next invocation."""
        result = dispatch(*args, **kwargs)
        control.interrupt()
        return result

    wrapped = Mock(side_effect=interrupt_after_tool)
    monkeypatch.setattr(runner.agent.tools, "call_with_timing", wrapped)

    result = runner.run()

    assert result.stop_reason == "cancelled"
    assert result.turns == 1
    assert len(result.metrics.model_calls) == 1
    assert len(result.metrics.tools) == 1
    assert result.metrics.tools[0].succeeded is True
    wrapped.assert_called_once()
    results = [item for item in runner.session_manager.messages if isinstance(item, ToolResult)]
    assert [item.call_id for item in results] == ["first", "second"]
    assert json.loads(results[0].output)["ok"] is True
    assert "cancelled before execution" in results[1].output
    assert runner.session_manager.recovery_state is None
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("standalone", [False, True])
def test_run_interruption_after_authorization_never_starts_the_tool(standalone):
    """Interruption after approval preserves the pre-execution guard for run and direct calls."""

    control = RunControl()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    response = Response(answer="", reasoning="", tool_calls=(call,))
    runner, sessions, _interaction = agent_runner(responses=[response], run_control=control)
    sessions.recovery_state = SessionRecoveryState(
        action="resolve_uncertain_tools",
        pending_calls=(PendingToolCall(call=call, status="outcome_unknown"),),
    )
    effects = []

    def dispatch(_name, _arguments, **kwargs):
        """Interrupt after authorization and before the durable execution-start callback."""
        control.interrupt()
        kwargs["execution_started"]()
        effects.append("executed")
        return '{"ok":true}', 0

    runner.agent.tools.call_with_timing.side_effect = dispatch
    if standalone:
        with control.execution(), pytest.raises(RunInterrupted):
            runner.handle_tool_calls(response)
        sessions.add_tool_call.assert_not_called()
    else:
        assert runner.run().stop_reason == "cancelled"
        assert "outcome is unknown" in sessions.add_tool_call.call_args.kwargs["output"]
    assert effects == []
    sessions.record_tool_execution_started.assert_not_called()


def test_interrupting_an_empty_library_run_needs_no_recovery_plan(tmp_path):
    """A direct runner can be interrupted before any conversation item exists."""
    control = RunControl()

    def events():
        """Request interruption before the first authoritative response event."""
        control.interrupt()
        yield ResponseCompleted(answer="discarded")

    backend = Mock()
    backend.get_response.return_value = events()
    runner = file_runner(tmp_path, backend, run_control=control)

    assert runner.run().stop_reason == "cancelled"
    assert runner.session_manager.messages == []
    assert runner.session_manager.recovery_state is None


def test_interrupting_approved_recovery_never_retries_an_uncertain_effect(tmp_path):
    """A scoped interrupt can revoke recovery after approval while preserving unknown outcomes."""
    control = RunControl()
    backend = Mock()
    runner = file_runner(tmp_path, backend, run_control=control)
    sessions = runner.session_manager
    call = ToolCall(call_id="uncertain", name="read_text_file", arguments="{}")
    sessions.add_user_message("original")
    sessions.add_response(Response(answer="", reasoning="", tool_calls=(call,), items=(call,)))
    sessions.add_tool_call_event(call.call_id)
    approvals = []

    def approve(message, **_kwargs):
        """Accept recovery, then interrupt after explicit retry approval but before execution."""
        approvals.append(message)
        if len(approvals) == 2:
            assert control.interrupt() is True
        return True

    runner.interaction.confirm.side_effect = approve

    recovery = runner.recover_session()

    assert recovery.pending is False
    assert recovery.result.stop_reason == "cancelled"
    assert len(approvals) == 2
    backend.get_response.assert_not_called()
    results = [item for item in sessions.messages if isinstance(item, ToolResult)]
    assert len(results) == 1
    assert "outcome is unknown" in results[0].output
    assert sessions.recovery_state is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_model_queries_prevent_session_selection(tmp_path, asynchronous):
    """Direct model requests retain their session binding even inside idle input ownership."""
    backend = Mock()
    runner = file_runner(tmp_path, backend)
    runner.compaction.compact_if_needed_async = AsyncMock(return_value=False)
    sessions = runner.session_manager
    original = sessions.session
    directory = sessions.temporary_directory.get()

    def check_selection():
        """Attempt a session change while the backend owns a model request."""
        with pytest.raises(SessionExecutionConflictError):
            sessions.new_session()
        assert sessions.session is original
        assert sessions.temporary_directory.get() == directory

    def response(**_kwargs):
        """Return a complete synchronous response after checking model ownership."""
        check_selection()
        return [ResponseCompleted(answer="complete")]

    async def events():
        """Return a complete asynchronous response after checking model ownership."""
        check_selection()
        yield ResponseCompleted(answer="complete")

    backend.get_response.side_effect = response
    backend.get_response_async.return_value = events()

    async def query():
        """Exercise nested model ownership from the asynchronous idle input context."""
        with sessions.execution(allow_session_selection=True):
            return await runner.query_async()

    if asynchronous:
        result = asyncio.run(query())
    else:
        with sessions.execution(allow_session_selection=True):
            result = runner.query()
    assert result.answer == "complete"
