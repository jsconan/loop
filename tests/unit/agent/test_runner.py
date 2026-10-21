"""Tests for bounded single-agent execution."""

import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path
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
    ModelSelectionError,
    OpenAIBackend,
    PendingToolCall,
    PermissionManager,
    Response,
    ResponseCompleted,
    ResponseMetrics,
    RunControl,
    RunInterrupted,
    RuntimeEnvironment,
    Session,
    SessionExecutionConflictError,
    SessionManager,
    SessionRecoveryState,
    ToolCall,
    ToolCallCompleted,
    ToolExecutionMetrics,
    ToolRegistry,
    ToolResult,
    Usage,
    edit_text_file,
    read_text_file,
)
from loop.agent import Agent, AgentRunner, AgentRunResult
from loop.session.models import RunCompletedEvent
from loop.telemetry import MemoryTelemetryAdapter, Telemetry, set_telemetry
from loop.tooling import FileObservations
from loop.utils import PathHolder


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
    session_manager.session = Session()
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
        RuntimeEnvironment(working_directory=tmp_path, temporary_directory=tmp_path.parent)
    )
    permissions = PermissionManager(tmp_path, interaction=interaction)
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
                "path": "/workspace/file.txt",
                **({"max_lines": 1} if partial else {}),
            }
        ),
    )
    reread = ToolCall(
        call_id="reread",
        name="read_text_file",
        arguments=json.dumps(
            {
                "path": "/workspace/file.txt",
            }
        ),
    )
    arguments = {"path": "/workspace/file.txt"}
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
        assert result["read_versions"] == {"/workspace/file.txt": None}
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
    assert json.loads(output)["read_versions"] == {"/workspace/file.txt": None}
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
                assert results[identifier]["read_versions"] == {"/workspace/file.txt": None}
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
        assert results["remove"]["file_versions"] == {"/workspace/file.txt": None}
    else:
        assert results["stale"]["problem"]["code"] == "filesystem.stale_context"
        assert results["absence"]["read_versions"] == {"/workspace/file.txt": None}
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
        cancellation_event=None,
        execution_started=ANY,
        file_observations=ANY,
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


def test_runner_async_recovery_waits_without_a_run_control(tmp_path, monkeypatch):
    """Retry-after recovery yields through asyncio when no scoped control is configured."""
    runner = file_runner(tmp_path, Mock())
    interaction = MagicMock(spec=Interaction)
    interaction.confirm.return_value = True
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
    interaction.confirm.assert_called_once_with(
        "Retry the complete response after at least 3 seconds?", default=False
    )
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
    interaction = MagicMock(spec=Interaction)
    interaction.confirm.return_value = True
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
    interaction.confirm.assert_called_once_with("Retry the complete response?", default=False)


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


def test_runner_async_recovery_returns_none_when_repetition_retry_also_fails(tmp_path):
    """A repeated failure after repetition recovery follows normal retry policy."""
    runner = file_runner(tmp_path, Mock())
    runner._query_async = AsyncMock(
        side_effect=[
            BackendRepetitionError("repetition", provider="test", operation="query"),
            BackendRepetitionError("repetition again", provider="test", operation="query"),
        ]
    )
    runner.interaction.confirm.return_value = False

    assert asyncio.run(runner._query_with_recovery_async()) is None
    runner.interaction.confirm.assert_called_once_with(
        "Retry the complete response?", default=False
    )


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


@pytest.mark.parametrize(
    ("continue_run", "expected_reason", "expected_turns"),
    [(False, "max_turns", 1), (True, "completed", 2)],
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
    if continue_run:
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
def test_async_recovery_preserves_unknown_outcome_and_pending_boundaries(
    monkeypatch, status, approval, expected_execution, expected_present, expected_skip
):
    """Async recovery retries only after approval and preserves already available results."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
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


@pytest.mark.parametrize("interrupted_tools", [None, []])
def test_async_tool_batch_retains_settled_metrics_when_interrupted(interrupted_tools):
    """An interrupted async batch keeps metrics for tools that already settled."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    settled = ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True)
    runner._execute_tool_call_async = AsyncMock(side_effect=[settled, RunInterrupted()])
    runner._interrupted_tools = interrupted_tools

    with pytest.raises(RunInterrupted):
        asyncio.run(runner._execute_tool_calls_async((call, call), present=True))

    if interrupted_tools is not None:
        assert runner._interrupted_tools == [settled]


@pytest.mark.parametrize("interrupted_tools", [None, []])
def test_async_tool_batch_retains_settled_metrics_when_task_is_cancelled(interrupted_tools):
    """Task cancellation does not discard metrics for previously settled tools."""
    runner, _sessions, _interaction = agent_runner()
    call = ToolCall(call_id="call", name="effect", arguments="{}")
    settled = ToolExecutionMetrics(name="effect", duration_seconds=0.1, succeeded=True)
    runner._execute_tool_call_async = AsyncMock(side_effect=[settled, asyncio.CancelledError()])
    runner._interrupted_tools = interrupted_tools

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._execute_tool_calls_async((call, call), present=True))

    if interrupted_tools is not None:
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


def test_runner_async_query_closes_backend_stream_when_cancelled(tmp_path):
    """Cancelling async query closes its backend stream and retains no partial response."""
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
        with sessions.execution():
            task = asyncio.create_task(runner.query_async())
            try:
                await waiting.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(query())

    assert closed.is_set()
    assert sessions.messages == []
    runner.interaction.answer_delta.assert_called_once_with("partial")


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


def test_interruption_preserves_settled_tool_results_and_cancels_remaining_calls(
    tmp_path, monkeypatch
):
    """A completed effect is saved once; later calls receive cancellation results and never run."""
    control = RunControl()
    target = tmp_path / "file.txt"
    target.write_text("original", encoding="utf-8")
    calls = tuple(
        ToolCall(call_id=name, name="read_text_file", arguments='{"path":"/workspace/file.txt"}')
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
