"""Exercise durable model-window assembly with the real agent and session services."""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from loop import Message, Reasoning, ResponseCompleted, Session, ToolRegistry
from loop.backend import BackendConnectionError
from loop.configuration import ApplicationSettings, ContextSettings
from loop.errors import ProblemException
from loop.instructions import InstructionsManager
from loop.loop import Loop
from loop.models import (
    CompactionContextItem,
    CompactionResult,
    DeveloperInput,
    ReasoningConfiguration,
    ToolCall,
    ToolResult,
)
from loop.session import Compaction, InstructionSnapshot
from loop.tools import activate_skill, manage_skills
from loop.tools.system import poll_command_job, run_command
from loop.utils import sha256_digest, utc_now
from tests.backend_doubles import model_backend


def _backend(layout="suffix", route="test-route", model="model"):
    """Record the agent's provider-facing arguments and return a settled response."""
    result = ResponseCompleted(items=(Message(role="assistant", content="Recorded."),), model=model)
    backend = model_backend(
        default_model=model, route=route, instruction_layout=lambda model: layout
    )
    backend.get_response.side_effect = lambda **kwargs: iter((result,))

    async def respond(**kwargs):
        """Yield the same response through the asynchronous provider boundary."""
        yield result

    backend.get_response_async = Mock(side_effect=respond)
    return backend


def _portable(text):
    """Build a portable checkpoint with the backend's compaction result."""
    return CompactionResult(items=(Message(role="user", content=text),))


def _turn(loop, text):
    """Persist a normal user input and commit the response after dispatch."""
    runner = loop.agent_runner
    runner.session_manager.add_user_message(text)
    response = runner.query()
    runner.session_manager.add_response(response)


async def _turn_async(loop, text):
    """Persist and run one turn through asynchronous hydration and writing."""
    runner = loop.agent_runner
    await runner.session_manager.add_user_message_async(text)
    response = await runner.query_async()
    await runner.session_manager.add_response_async(response)


def test_inline_layout_keeps_environment_out_of_input_replay(tmp_path):
    """Keep non-time facts in a separate field and apply disclosure edits on the next turn."""
    backend = _backend(layout="inline")
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        context_settings=ContextSettings(timezone="UTC", show_country=True, country="LU"),
    )
    _turn(loop, "First")
    first = backend.get_response.call_args.kwargs
    assert "country: LU" in first["instructions"]
    assert "current_date:" not in first["instructions"]
    assert all("<environment_context>" not in item.content for item in first["input"])

    loop.configure_context(
        ApplicationSettings(
            context=ContextSettings(timezone="Asia/Tokyo", show_country=True, country="LU")
        )
    )
    _turn(loop, "Use the new time zone")
    second = backend.get_response.call_args.kwargs
    assert second["instructions"] == first["instructions"]
    assert second["input"][: len(first["input"])] == first["input"]

    loop.configure_context(ApplicationSettings(context=ContextSettings(timezone="Asia/Tokyo")))
    _turn(loop, "Limit disclosure")
    third = backend.get_response.call_args.kwargs
    assert "country:" not in third["instructions"]
    assert third["input"][: len(second["input"])] == second["input"]
    assert "<environment_context>" not in loop.session.serialize()


def test_inline_environment_is_omitted_when_it_exceeds_the_budget(tmp_path):
    """Drop the complete block when new project instructions consume its available space."""
    backend = _backend(layout="inline")
    probe = Loop.create_default(backend, working_directory=tmp_path)
    budget = len(probe.instructions.encode()) + 512
    manager = InstructionsManager.discover(tmp_path, max_bytes=budget)
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        instructions_manager=manager,
    )
    _turn(loop, "Start")
    assert "<environment_context>" in backend.get_response.call_args.kwargs["instructions"]

    (tmp_path / "AGENTS.md").write_text("x" * 450, encoding="utf-8")
    _turn(loop, "Continue")

    assert backend.get_response.call_args.kwargs["instructions"] is None


def test_instruction_edits_and_removal_append_in_replay_order(tmp_path):
    """Changed project rules replace their prior section without rewriting input history."""
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("Old rules.", encoding="utf-8")
    backend = _backend()
    loop = Loop.create_default(backend, working_directory=tmp_path)
    _turn(loop, "First prompt")
    first = backend.get_response.call_args.kwargs["input"]
    assert "Old rules." in first[0].content
    instructions.write_text("New rules.", encoding="utf-8")
    _turn(loop, "Next prompt")
    second = backend.get_response.call_args.kwargs["input"]
    assert second[: len(first)] == first
    assert second[-2].content == "Next prompt"
    assert "<instruction_revision>" in second[-1].content
    assert "New rules." in second[-1].content
    assert str(instructions) not in second[-1].content
    loop.instructions_manager.invalidate()
    _turn(loop, "Same rules")
    assert len([record for record in loop.session.dispatches if record.revision]) == 1
    instructions.unlink()
    _turn(loop, "Rules removed")
    updated = backend.get_response.call_args.kwargs["input"]
    assert '"deactivate": ["agents:' in updated[-1].content
    restored = Session.deserialize(loop.session.serialize())
    assert restored.model_context(window=restored.window)[:-1] == updated


def test_uncertain_dispatch_reuses_committed_positions_after_reopen(tmp_path):
    """A lost transport result cannot insert a different revision into the same request."""
    source = tmp_path / "AGENTS.md"
    source.write_text("First rules.", encoding="utf-8")
    backend = _backend()
    loop = Loop.create_default(backend, working_directory=tmp_path)
    _turn(loop, "First")
    source.write_text("Committed revision.", encoding="utf-8")
    loop.agent_runner.session_manager.add_user_message("Ambiguous delivery")
    backend.get_response.side_effect = BackendConnectionError(
        "connection lost", provider="test", operation="generation"
    )
    with pytest.raises(BackendConnectionError, match="connection lost"):
        loop.agent_runner.query()
    sent = backend.get_response.call_args.kwargs["input"]
    assert "Committed revision." in sent[-1].content
    original = loop.session.dispatches[-1]
    restored = Session.deserialize(loop.session.serialize())
    source.write_text("Later revision.", encoding="utf-8")
    replay_backend = _backend()
    resumed = Loop.create_default(replay_backend, working_directory=tmp_path, session=restored)
    response = resumed.agent_runner.query()
    assert replay_backend.get_response.call_args.kwargs["input"] == sent
    assert resumed.session.dispatches[-1] == original
    resumed.agent_runner.session_manager.add_response(response)
    _turn(resumed, "New logical dispatch")
    after = replay_backend.get_response.call_args.kwargs["input"]
    assert "Later revision." in after[-1].content
    assert after[: len(sent)] == sent


@pytest.mark.parametrize("lower_threshold", [False, True])
def test_live_effort_change_preserves_the_admitted_retry(tmp_path, monkeypatch, lower_threshold):
    """A retry keeps its instructions and environment while applying the current effort."""
    now = datetime(2026, 11, 24, 23, 59, tzinfo=UTC)
    monkeypatch.setattr("loop.agent.context.utc_now", lambda: now)
    source = tmp_path / "AGENTS.md"
    source.write_text("Initial rules.", encoding="utf-8")
    backend = _backend()
    backend.supports_reasoning_updates = lambda model: True
    backend.get_context_window.return_value = 32768
    backend.compact.return_value = _portable("Older conversation.")
    loop = Loop.create_default(
        backend,
        working_directory=tmp_path,
        context_settings=ContextSettings(timezone="UTC"),
        reasoning_effort="low",
    )
    for index in range(7):
        _turn(loop, f"Earlier request {index}")
    loop.session.tokens = 10_000
    source.write_text("Admitted rules.", encoding="utf-8")
    runner = loop.agent_runner
    runner.session_manager.add_user_message("Interrupted request")
    respond = backend.get_response.side_effect
    backend.get_response.side_effect = BackendConnectionError(
        "connection lost", provider="test", operation="generation"
    )
    with pytest.raises(BackendConnectionError, match="connection lost"):
        runner.query()
    sent = backend.get_response.call_args.kwargs
    assert "Admitted rules." in sent["input"][-1].content
    assert "current_date: 2026-11-24" in sent["instructions"]

    now = datetime(2026, 11, 25, 0, 1, tzinfo=UTC)
    source.write_text("Later rules.", encoding="utf-8")
    assert loop.instructions_manager.refresh()
    settings = ApplicationSettings.model_validate(
        {
            "app": {
                "reasoning_effort": "high",
                "compaction_threshold": 0.2 if lower_threshold else 0.8,
            }
        }
    )
    loop.apply_runtime_settings("app.reasoning_effort", settings)
    loop.apply_runtime_settings("app.compaction_threshold", settings)
    backend.get_response.side_effect = respond
    runner.session_manager.add_response(runner.query())

    backend.compact.assert_not_called()
    retried = backend.get_response.call_args.kwargs
    assert retried["input"] == [*sent["input"], ReasoningConfiguration("high")]
    assert retried["instructions"] == sent["instructions"]

    _turn(loop, "New request")
    next_request = backend.get_response.call_args.kwargs
    assert "current_date: 2026-11-25" in next_request["instructions"]
    if lower_threshold:
        backend.compact.assert_called_once()
        assert "Later rules." in next_request["input"][0].content
    else:
        backend.compact.assert_not_called()
        assert "Later rules." in next_request["input"][-1].content


def test_same_model_preference_reset_follows_new_backend_default_on_resume(tmp_path):
    """Resetting selection to the same effective model preserves the window and follows defaults."""
    backend = _backend(model="first-model")
    loop = Loop.create_default(backend, model="first-model", working_directory=tmp_path)
    _turn(loop, "Start with an explicit selection")
    window_id = loop.session.window.id

    loop.apply_runtime_settings("app.model", ApplicationSettings())
    _turn(loop, "Use the current default")
    assert loop.session.window.id == window_id

    backend = _backend(model="new-default")
    resumed = Loop.create_default(
        backend,
        working_directory=tmp_path,
        session=Session.deserialize(loop.session.serialize()),
    )
    _turn(resumed, "Continue with the new default")
    assert backend.get_response.call_args.kwargs["model"] == "new-default"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_preparing_a_request_keeps_the_last_provider_assignment(tmp_path, asynchronous):
    """The requested model and the provider's last reported assignment stay distinct."""
    backend = _backend(model="provider-version")
    loop = Loop.create_default(backend, model="selected-alias", working_directory=tmp_path)
    runner = loop.agent_runner
    runner.session_manager.add_user_message("First")
    if asynchronous:
        asyncio.run(runner.run_async())
    else:
        runner.run()
    assert loop.model == "selected-alias"
    assert loop.session.model == "provider-version"

    runner.session_manager.add_user_message("Second")
    if asynchronous:
        asyncio.run(runner.query_async())
        request = backend.get_response_async.call_args.kwargs
    else:
        runner.query()
        request = backend.get_response.call_args.kwargs
    assert request["model"] == "selected-alias"
    persisted = Session.deserialize(loop.session.serialize())
    assert persisted.model == "provider-version"
    assert persisted.window.model == "selected-alias"

    replacement = _backend(model="provider-version")
    replacement.get_context_window.return_value = 32768
    resumed = Loop.create_default(replacement, session=persisted, working_directory=tmp_path)
    assert resumed.model == "selected-alias"
    assert resumed.session.model == "provider-version"
    assert resumed.session.context_window == 32768


@pytest.mark.parametrize("reported_model", [None, "origin-version"])
def test_backend_switch_keeps_the_in_flight_response_assignment(tmp_path, reported_model):
    """A response uses the model and capacity of the deployment that produced it."""
    origin = _backend(layout="leading", route="origin", model="origin-model")
    origin.get_context_window.return_value = 2000
    origin.get_context_window_async.return_value = 2000
    replacement = _backend(layout="leading", route="replacement", model="next-model")
    replacement.get_context_window_async.return_value = 8192
    loop = Loop.create_default(origin, working_directory=tmp_path)

    async def respond(**kwargs):
        """Apply a backend change as the old request streams its response."""
        await loop.replace_backend_async(replacement)
        yield ResponseCompleted(
            items=(Message(role="assistant", content="Completed by the original backend."),),
            model=reported_model,
        )

    origin.get_response_async.side_effect = respond

    async def scenario():
        """Finish the admitted response and dispatch the next request on the replacement."""
        runner = loop.agent_runner
        await runner.session_manager.add_user_message_async("First request")
        await runner.run_async()
        assert loop.session.model == (reported_model or "origin-model")
        assert loop.session.context_window == 2000
        assert (
            runner.session_manager.store.load(loop.session.id).assignment == loop.session.assignment
        )
        await runner.session_manager.add_user_message_async("Next request")
        await runner.run_async()
        assert replacement.get_response_async.call_args.kwargs["model"] == "next-model"
        assert loop.session.context_window == 8192

    asyncio.run(scenario())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completed_response_is_saved_when_capacity_lookup_fails(tmp_path, caplog, asynchronous):
    """Lost optional metadata leaves the completed response available in the session."""
    backend = _backend()
    loop = Loop.create_default(backend, working_directory=tmp_path)
    metadata = backend.get_context_window_async if asynchronous else backend.get_context_window
    metadata.side_effect = BackendConnectionError(
        "catalog unavailable", provider="test", operation="get_context_window"
    )
    runner = loop.agent_runner
    runner.session_manager.add_user_message("Finish the task")

    if asynchronous:
        asyncio.run(runner.run_async())
        backend.get_response_async.assert_called_once()
    else:
        runner.run()
        backend.get_response.assert_called_once()

    assert loop.session.messages[-1] == Message(role="assistant", content="Recorded.")
    assert loop.session.model == "model"
    assert loop.session.context_window is None
    assert "Model context capacity is unavailable" in caplog.text


def test_saved_name_file_and_four_timings_wait_for_rebuild(tmp_path):
    """Deferred values remain pinned across mixed live updates, fresh clients, and resume."""
    backend = _backend(layout="leading")
    tools = ToolRegistry([run_command, poll_command_job])
    loop = Loop.create_default(backend, working_directory=tmp_path, tool_registry=tools)
    settings = ApplicationSettings.model_validate(
        {
            "app": {"agent_name": "First identity"},
            "backend": {"file_input_mode": "text"},
        }
    )
    loop.configure_context(settings)
    _turn(loop, "First")
    first = backend.get_response.call_args.kwargs
    assert "First identity" in first["instructions"]
    before = loop.session.window
    updated = ApplicationSettings.model_validate(
        {
            "app": {"agent_name": "Pending identity", "temperature": 0.6},
            "backend": {"file_input_mode": "native"},
            "tools": {
                "command_job_runtime_limit_seconds": 42,
                "command_default_initial_wait_seconds": 2,
                "command_default_poll_wait_seconds": 3,
                "command_max_wait_seconds": 4,
            },
        }
    )
    loop.apply_runtime_settings("app.temperature", updated)
    loop.configure_context(updated)
    assert "effective value is 'text'" in loop.configuration_status("backend.file_input_mode")
    for key in type(updated.tools).model_fields:
        if key.startswith("command_"):
            assert loop.configuration_status(f"tools.{key}") is not None
    replacement = _backend(layout="leading")
    loop.replace_backend(replacement)
    loop.configure_context(updated)
    _turn(loop, "Second")
    second = replacement.get_response.call_args.kwargs
    assert second["instructions"] == first["instructions"]
    assert second["tools"] == first["tools"]
    assert second["hyperparameters"].temperature == 0.6
    assert loop.session.window == before
    restored = Session.deserialize(loop.session.serialize())
    next_backend = _backend(layout="leading")
    resumed = Loop.create_default(
        next_backend,
        session=restored,
        working_directory=tmp_path,
        tool_registry=ToolRegistry([run_command, poll_command_job]),
    )
    resumed.configure_context(updated)
    _turn(resumed, "After resume")
    assert resumed.session.window == before
    assert next_backend.get_response.call_args.kwargs["tools"] == first["tools"]
    next_backend.compact.return_value = _portable("Old history")
    assert resumed.agent_runner.compaction.compact()
    _turn(resumed, "After rebuild")
    after = resumed.session.window
    assert after.settings.agent_name == "Pending identity"
    assert after.settings.file_input_mode == "native"
    assert after.settings.command_job_runtime_limit_seconds == 42
    assert after.settings.command_default_initial_wait_seconds == 2
    assert after.settings.command_default_poll_wait_seconds == 3
    assert after.settings.command_max_wait_seconds == 4
    assert resumed.configuration_status("app.agent_name") is None
    assert resumed.tool_registry.settings.command_max_wait_seconds == 4
    next_backend.compact.assert_not_called()
    latest = next_backend.get_response.call_args.kwargs
    assert "Pending identity" in latest["instructions"]
    assert latest["tools"] != first["tools"]


def test_live_reasoning_replays_updates_and_restores_effort_after_rebuild(tmp_path):
    """Append effort and default-reset items while preserving the top-level setting."""
    backend = _backend()
    backend.supports_reasoning_updates = lambda model: True
    loop = Loop.create_default(backend, working_directory=tmp_path, reasoning_effort="low")
    _turn(loop, "First")
    loop.apply_runtime_settings(
        "app.reasoning_effort",
        ApplicationSettings.model_validate({"app": {"reasoning_effort": "high"}}),
    )
    _turn(loop, "Second")
    request = backend.get_response.call_args.kwargs
    assert request["hyperparameters"].reasoning_effort == "low"
    assert request["input"][-1] == ReasoningConfiguration("high")
    loop.apply_runtime_settings("app.reasoning_effort", ApplicationSettings())
    _turn(loop, "Reset effort")
    request = backend.get_response.call_args.kwargs
    reasoning = [item for item in request["input"] if isinstance(item, ReasoningConfiguration)]
    assert reasoning == [ReasoningConfiguration("high"), ReasoningConfiguration(None)]
    backend.compact.return_value = _portable("Old history")
    backend.estimate_generation = Mock(return_value=1000)
    assert loop.agent_runner.compaction.compact()
    measured = backend.estimate_generation.call_args.args[0]
    assert [item for item in measured if isinstance(item, ReasoningConfiguration)] == [
        ReasoningConfiguration(None)
    ]
    _turn(loop, "After compaction")
    request = backend.get_response.call_args.kwargs
    reasoning = [item for item in request["input"] if isinstance(item, ReasoningConfiguration)]
    assert request["hyperparameters"].reasoning_effort == "low"
    assert reasoning == [ReasoningConfiguration(None)]


def test_unavailable_pinned_route_requires_explicit_switch_and_projects_semantic_pairs(tmp_path):
    """Resume blocks another deployment until selected and then drops its foreign continuation."""
    backend = _backend()
    loop = Loop.create_default(backend, working_directory=tmp_path)
    _turn(loop, "First")
    owner = loop.agent_runner.session_manager
    owner.add_user_message("Tool work")
    owner.add_message(Reasoning(content="Old reasoning", encrypted_content="opaque"))
    owner.add_message(ToolCall(call_id="foreign", name="test", arguments="{}", id="foreign-id"))
    owner.add_message(ToolResult(call_id="foreign", output="completed work"))
    owner.add_message(ToolResult(call_id="no matched call", output="orphan"))
    new_backend = _backend(route="another-route")
    other = Loop.create_default(
        new_backend,
        working_directory=tmp_path,
        session=Session.deserialize(loop.session.serialize()),
    )
    with pytest.raises(ProblemException, match="saved model route is unavailable"):
        other.agent_runner.query()
    new_backend.get_response.assert_not_called()
    other.select_model("model")
    other.agent_runner.query()
    outgoing = new_backend.get_response.call_args.kwargs["input"]
    assert not any(isinstance(item, Reasoning) for item in outgoing)
    call = next(item for item in outgoing if isinstance(item, ToolCall))
    result = next(item for item in outgoing if isinstance(item, ToolResult))
    assert call.call_id == result.call_id != "foreign"
    assert call.id is None
    assert sum(isinstance(item, ToolResult) for item in outgoing) == 1
    assert other.session.window.route == "another-route"
    new_backend.compact.assert_not_called()
    assert not other.session.compactions

    owner = other.agent_runner.session_manager
    owner.add_message(ToolCall(call_id="current", name="test", arguments="{}", id="current-id"))
    owner.add_message(ToolResult(call_id="current", output="new completed work"))
    other.agent_runner.query()
    replay = new_backend.get_response.call_args.kwargs["input"]
    assert replay[-2].call_id == replay[-1].call_id == "current"
    assert replay[-2].id == "current-id"


@pytest.mark.parametrize("native", [False, True])
def test_backend_switch_replays_locally_without_summarizing_after_a_checkpoint(tmp_path, native):
    """A route switch keeps portable text or expands native history without another compaction."""
    previous = _backend()
    loop = Loop.create_default(previous, working_directory=tmp_path)
    _turn(loop, "Initial request")
    session = loop.session
    owner = loop.agent_runner.session_manager
    item = (
        CompactionContextItem(
            route=previous.route,
            model="model",
            data={"type": "compaction", "encrypted_content": "private provider state"},
        )
        if native
        else Message(role="user", content="Portable checkpoint")
    )
    checkpoint = Compaction(
        id="saved-checkpoint",
        model="model",
        created_at=utc_now(),
        boundary=len(session.messages),
        context=(item,),
        input_tokens_after=30_000,
        instructions=InstructionSnapshot(
            working_directory=str(tmp_path),
            content=None,
            digest=sha256_digest(""),
        ),
    )
    window = session.window.model_copy(update={"id": "compacted", "checkpoint_id": checkpoint.id})
    owner.commit_model_window(session.id, session.window.id, None, window, checkpoint=checkpoint)
    owner.add_message(Reasoning(content="", encrypted_content="recent opaque reasoning"))
    other = _backend(route="next-route")
    other.get_context_window.return_value = 32768
    loop.replace_backend(other)

    _turn(loop, "Continue")

    sent = other.get_response.call_args.kwargs["input"]
    assert not any(isinstance(entry, (CompactionContextItem, Reasoning)) for entry in sent)
    assert sent[1].content == ("Initial request" if native else "Portable checkpoint")
    assert session.compactions == [checkpoint]
    restored = Session.deserialize(session.serialize())
    assert restored.window.checkpoint_id == (None if native else checkpoint.id)
    assert not any(isinstance(entry, Reasoning) for entry in restored.model_context())
    assert session.tokens == 0
    _turn(loop, "Next request without a usage report")
    other.compact.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_compaction_uses_the_same_route_and_replay_checks_as_generation(tmp_path, asynchronous):
    """History reaches a replacement route only after authorization and portable projection."""
    origin = _backend()
    origin.get_context_window.return_value = 32768
    loop = Loop.create_default(origin, working_directory=tmp_path)
    _turn(loop, "Saved user task")
    for index in range(6):
        _turn(loop, f"Saved turn {index}")
    checkpoint = CompactionContextItem(
        route=origin.route,
        model="model",
        data={"type": "compaction", "encrypted_content": "opaque checkpoint"},
    )
    origin.compact.return_value = CompactionResult(items=(checkpoint,))
    assert loop.agent_runner.compaction.compact()
    _turn(loop, "Continue on the same route")
    bound = origin.get_response.call_args.kwargs["input"]
    assert checkpoint in bound
    loop.agent_runner.session_manager.add_message(
        Reasoning(content="Private", encrypted_content="opaque")
    )

    replacement = _backend(route="other")
    replacement.compact.return_value = _portable("Portable summary")
    replacement.compact_async = AsyncMock(return_value=_portable("Portable summary"))
    resumed = Loop.create_default(
        replacement,
        working_directory=tmp_path,
        session=Session.deserialize(loop.session.serialize()),
    )
    compaction = resumed.agent_runner.compaction
    transport = replacement.compact_async if asynchronous else replacement.compact

    def compact():
        """Exercise the compactor's public boundary."""
        return asyncio.run(compaction.compact_async()) if asynchronous else compaction.compact()

    with pytest.raises(ProblemException, match="saved model route is unavailable"):
        compact()
    transport.assert_not_called()

    settings = ApplicationSettings.model_validate({"app": {"model": "model"}})
    if asynchronous:
        asyncio.run(resumed.apply_runtime_settings_async("app.model", settings))
    else:
        resumed.apply_runtime_settings("app.model", settings)
    assert compact()
    sent = transport.call_args.args[0]
    assert sent[0] == Message(role="assistant", content="Recorded.")
    assert all(isinstance(item, Message) for item in sent)
    assert resumed.session.compactions[-1].context == _portable("Portable summary").items
    assert resumed.session.window.route == replacement.route


def test_slash_skill_state_persists_before_next_revision(tmp_path):
    """Activation and deactivation persist as state before model-only replay is updated."""
    skill = tmp_path / ".agents" / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill.\n---\nFollow the demo procedure.\n",
        encoding="utf-8",
    )
    backend = _backend()
    tools = ToolRegistry([activate_skill, manage_skills])
    loop = Loop.create_default(backend, working_directory=tmp_path, tool_registry=tools)
    _turn(loop, "Initial turn")
    assert loop.command_manager.handle_user_command("/use demo")
    assert Session.deserialize(loop.session.serialize()).active_skills[0][0] == "demo"
    assert all(record.revision is None for record in loop.session.dispatches)
    _turn(loop, "Use the skill")
    assert (
        "Follow the demo procedure." in backend.get_response.call_args.kwargs["input"][-1].content
    )
    assert loop.command_manager.handle_user_command("/skills deactivate demo")
    assert Session.deserialize(loop.session.serialize()).active_skills == []
    assert loop.command_manager.handle_user_command("/skills activate demo")
    assert loop.command_manager.handle_user_command("/skills deactivate-all")
    assert Session.deserialize(loop.session.serialize()).active_skills == []
    _turn(loop, "Continue without it")
    assert (
        '"deactivate": ["active_skill:'
        in backend.get_response.call_args.kwargs["input"][-1].content
    )


def test_retries_reuse_the_same_ephemeral_environment_sample(tmp_path, monkeypatch):
    """A retry uses the same policy-prepared date until a distinct request is admitted."""
    moments = Mock(
        side_effect=[
            datetime(2026, 11, 22, 23, 59, tzinfo=UTC),
            datetime(2026, 11, 23, 0, 1, tzinfo=UTC),
        ]
    )
    monkeypatch.setattr("loop.agent.context.utc_now", moments)
    backend = _backend()
    response = ResponseCompleted(items=(Message(role="assistant", content="Done."),))
    backend.get_response.side_effect = [
        BackendConnectionError("transport lost", provider="test", operation="generation"),
        iter((response,)),
        iter((response,)),
    ]
    loop = Loop.create_default(
        backend, working_directory=tmp_path, context_settings=ContextSettings(timezone="UTC")
    )
    runner = loop.agent_runner
    runner.session_manager.add_user_message("Current task")
    with pytest.raises(BackendConnectionError):
        runner.query()
    first = backend.get_response.call_args.kwargs
    runner.session_manager.add_response(runner.query())
    second = backend.get_response.call_args.kwargs
    assert first["instructions"] == second["instructions"]
    moments.assert_called_once()
    _turn(loop, "Different request")
    assert "current_date: 2026-11-23" in backend.get_response.call_args.kwargs["instructions"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_leading_source_change_updates_the_field_without_rebuilding(tmp_path, asynchronous):
    """Leading routes use current instructions while retaining the same window and history."""
    source = tmp_path / "AGENTS.md"
    source.write_text("Initial rules.", encoding="utf-8")
    backend = _backend(layout="leading")
    loop = Loop.create_default(backend, working_directory=tmp_path)
    _turn(loop, "Initial request")
    old = loop.session.window
    source.write_text("Updated rules.", encoding="utf-8")
    method = backend.get_response_async if asynchronous else backend.get_response
    if asynchronous:
        asyncio.run(_turn_async(loop, "New"))
    else:
        _turn(loop, "New")
    request = method.call_args.kwargs
    assert "Updated rules." in request["instructions"]
    assert "Initial rules." not in request["instructions"]
    assert request["input"] == loop.session.messages[:-1]
    assert loop.session.window == old
    assert loop.session.dispatches[-1].revision is not None
    assert not loop.session.compactions


def test_request_validation_checks_withdrawn_disclosure(tmp_path):
    """An admitted turn cannot send environment facts after its policy changed."""
    loop = Loop.create_default(_backend(), working_directory=tmp_path)
    context = loop.agent_runner.request_context
    request = context.prepare(None)
    loop.configure_context(ApplicationSettings(context=ContextSettings(mode="off")))
    with pytest.raises(ProblemException, match="request changed before dispatch"):
        context.verify(request)
    assert not loop.session.dispatches


def test_async_dispatch_rechecks_disclosure_after_persisting(tmp_path, monkeypatch):
    """A policy edit during the session write prevents sending withdrawn facts."""

    async def scenario():
        """Withdraw optional facts in the interval between admission and transport."""
        backend = _backend()
        loop = Loop.create_default(
            backend,
            working_directory=tmp_path,
            context_settings=ContextSettings(show_country=True, country="LU"),
        )
        runner = loop.agent_runner
        manager = runner.session_manager
        await manager.add_user_message_async("Pending message")
        commit = manager.commit_model_window_async

        async def commit_then_withdraw(*args, **kwargs):
            """Simulate an edit after a real durable dispatch write."""
            await commit(*args, **kwargs)
            loop.configure_context(ApplicationSettings(context=ContextSettings(mode="off")))

        monkeypatch.setattr(manager, "commit_model_window_async", commit_then_withdraw)
        with pytest.raises(ProblemException, match="request changed before dispatch"):
            await runner.query_async()
        backend.get_response_async.assert_not_called()
        assert loop.session.pending_dispatch == loop.session.dispatches[-1].id

        await runner.query_async()
        assert backend.get_response_async.call_args.kwargs["instructions"] is None
        assert len(loop.session.dispatches) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("compaction", [False, True], ids=["generation", "compaction"])
def test_async_instruction_discovery_rejects_a_superseded_model_target(
    tmp_path, monkeypatch, compaction
):
    """An asynchronous discovery result for a previous model cannot seed a window."""

    async def scenario():
        """Select another model while collecting the previous target's instructions."""
        backend = _backend()
        loop = Loop.create_default(backend, working_directory=tmp_path)
        prepare = loop.instructions_manager.prepare_async

        async def prepare_then_switch(*args, **kwargs):
            """Apply the new selection at the suspension point."""
            snapshot = await prepare(*args, **kwargs)
            await loop.apply_runtime_settings_async(
                "app.model", ApplicationSettings.model_validate({"app": {"model": "next-model"}})
            )
            return snapshot

        monkeypatch.setattr(loop.instructions_manager, "prepare_async", prepare_then_switch)
        with pytest.raises(ProblemException, match="request changed before dispatch"):
            if compaction:
                await loop.agent_runner.compaction.compact_async()
            else:
                await loop.agent_runner.query_async()
        backend.get_response_async.assert_not_called()
        backend.compact_async.assert_not_called()
        assert not loop.session.dispatches

    asyncio.run(scenario())


def test_first_dispatch_rejects_settings_changed_during_preparation(tmp_path):
    """A window is pinned only after its candidate settings survive admission."""
    loop = Loop.create_default(_backend(), working_directory=tmp_path)
    context = loop.agent_runner.request_context
    request = context.prepare(None)
    loop.configure_context(ApplicationSettings.model_validate({"app": {"agent_name": "Changed"}}))
    with pytest.raises(ProblemException, match="request changed before dispatch"):
        context.verify(request)
    assert loop.session.window is None


def test_inline_recovery_suffix_is_request_local(tmp_path):
    """Transient recovery guidance expires without a durable model input item."""
    loop = Loop.create_default(
        _backend(layout="inline"),
        working_directory=tmp_path,
        context_settings=ContextSettings(mode="off"),
    )
    context = loop.agent_runner.request_context
    request = context.prepare("repetition")
    followup = context.prepare(None)
    assert "previous generation was stopped" in request.input[-1].content
    assert all("previous generation was stopped" not in item.content for item in followup.input)
    assert not loop.session.dispatches


def test_async_uncertain_request_replays_without_second_commit(tmp_path):
    """An asynchronous retry restores the input record that preceded uncertain transport."""

    async def scenario():
        """Drive two dispatches separated by a transport error and source edit."""
        source = tmp_path / "AGENTS.md"
        source.write_text("Pinned rules.", encoding="utf-8")
        backend = _backend()
        respond = backend.get_response_async.side_effect
        backend.get_response_async.side_effect = BackendConnectionError(
            "transport lost", provider="test", operation="generation"
        )
        loop = Loop.create_default(backend, working_directory=tmp_path)
        runner = loop.agent_runner
        await runner.session_manager.add_user_message_async("Pending")
        with pytest.raises(BackendConnectionError):
            await runner.query_async()
        before = backend.get_response_async.call_args.kwargs["input"]
        persisted = loop.session.dispatches[-1]
        source.write_text("Future rules.", encoding="utf-8")
        backend.get_response_async.side_effect = respond
        await runner.query_async()
        assert loop.session.dispatches == [persisted]
        assert backend.get_response_async.call_args.kwargs["input"] == before

    asyncio.run(scenario())


def test_compatible_legacy_checkpoint_gets_a_fresh_baseline(tmp_path):
    """A portable legacy checkpoint remains eligible at its original frontier."""
    older = Message(role="user", content="previous task")
    later = Message(role="user", content="next task")
    item = Message(role="user", content="Summary")
    checkpoint = Compaction(
        id="old",
        created_at=utc_now(),
        boundary=1,
        model="model",
        context=(item,),
        instructions=InstructionSnapshot(
            working_directory=str(tmp_path),
            content=None,
            digest=sha256_digest(""),
        ),
    )
    backend = _backend()
    session = Session(messages=[older, later], compactions=[checkpoint], model="model")
    loop = Loop.create_default(backend, working_directory=tmp_path, session=session)
    loop.agent_runner.query()
    request = backend.get_response.call_args.kwargs["input"]
    assert isinstance(request[0], DeveloperInput)
    assert "Opening messages and checkpoints" not in request[0].content
    assert request[1:] == [item, later]
    assert loop.session.window.checkpoint_id == "old"
    assert loop.session.messages == [older, later]


def test_legacy_session_replays_without_unattributed_provider_state(tmp_path):
    """The first compatible dispatch retains legacy text and complete tool pairs."""
    history = [
        Message(role="user", content="Previous task"),
        Reasoning(content="Prior reasoning", encrypted_content="Opaque state", id="rs_old"),
        ToolCall(call_id="old-call", name="lookup", arguments="{}", id="fc_old"),
        ToolResult(call_id="old-call", output="Completed work"),
        Message(role="assistant", content="Previous answer"),
    ]
    legacy = json.loads(Session(messages=history, model="model").serialize())
    legacy["version"] = 16
    for field in ("windows", "dispatches", "pending_dispatch"):
        del legacy[field]

    backend = _backend(layout="leading")
    session = Session.deserialize(json.dumps(legacy))
    loop = Loop.create_default(backend, working_directory=tmp_path, session=session)
    _turn(loop, "Continue on this gateway")

    sent = backend.get_response.call_args.kwargs["input"]
    assert not any(isinstance(item, Reasoning) for item in sent)
    previous, call, result, answer, follow_up = sent
    assert previous == history[0]
    assert answer == history[-1]
    assert follow_up == Message(role="user", content="Continue on this gateway")
    assert isinstance(call, ToolCall) and call.id is None and call.name == "lookup"
    assert call.call_id != "old-call"
    assert result == ToolResult(call_id=call.call_id, output="Completed work")
    assert loop.session.messages[: len(history)] == history


def test_skill_activation_budget_includes_the_pinned_agent_instructions(tmp_path):
    """A skill that exceeds the complete instruction budget is rejected before persistence."""
    skill = tmp_path / ".agents" / "skills" / "too-large"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: too-large\ndescription: A large skill.\n---\n" + "x" * 400,
        encoding="utf-8",
    )
    backend = _backend()
    probe = Loop.create_default(backend, working_directory=tmp_path)
    manager = InstructionsManager.discover(
        tmp_path, max_bytes=len(probe.instructions.encode()) + 40
    )
    registry = ToolRegistry([activate_skill])
    loop = Loop.create_default(
        backend, working_directory=tmp_path, tool_registry=registry, instructions_manager=manager
    )
    _turn(loop, "Before activation")
    result = manager.activate_skill("too-large")
    assert result.code == "skill.instruction_budget_exceeded"
    assert not manager.active_skill_identities
    _turn(loop, "The baseline still fits")
    assert not any(item.revision for item in loop.session.dispatches)


def test_failed_bound_instruction_refresh_preserves_the_committed_snapshot(tmp_path):
    """An instruction edit that exceeds the pinned budget never replaces the prior sources."""
    source = tmp_path / "AGENTS.md"
    source.write_text("Short rules.", encoding="utf-8")
    baseline = Loop.create_default(_backend(), working_directory=tmp_path)
    manager = InstructionsManager.discover(tmp_path, max_bytes=len(baseline.instructions.encode()))
    backend = _backend()
    loop = Loop.create_default(backend, working_directory=tmp_path, instructions_manager=manager)
    _turn(loop, "First")
    current = loop.session.window
    source.write_text("More instructions. " * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        _turn(loop, "Second")
    assert loop.session.window == current
    assert "Short rules." in manager.instructions
    backend.get_response.assert_called_once()
