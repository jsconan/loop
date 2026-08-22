"""Tests for normalized response handling and conversation orchestration."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from prompt_toolkit.document import Document

from loop import (
    BUILTIN_TOOLS,
    AgentRunResult,
    AnswerCompleted,
    AnswerDelta,
    BackendAuthenticationError,
    BackendConnectionError,
    BackendNotFoundError,
    BackendServerError,
    CompactionContextItem,
    CompactionResult,
    ContextReference,
    InstructionsManager,
    Interaction,
    Loop,
    MentionManager,
    Message,
    ModelInfo,
    PermissionConfiguration,
    PermissionManager,
    Response,
    ResponseCompleted,
    Session,
    SessionManager,
    Skill,
    SkillManager,
    SQLiteSessionStore,
    ToolCall,
    ToolCallCompleted,
    ToolRegistry,
    ToolResult,
    Usage,
    manage_skills,
    tool,
)


def function_call() -> ToolCall:
    """Build a completed local function-tool call."""
    return ToolCall(call_id="call_123", name="get_current_datetime", arguments="{}", id="fc_123")


def loop_backend(**attributes):
    """Build a minimal backend satisfying the loop contract."""
    defaults = {
        "default_model": "default-model",
        "get_context_window": lambda _model: None,
    }
    return SimpleNamespace(**(defaults | attributes))


def output_interaction() -> MagicMock:
    """Build an interaction mock with a no-op response scope."""
    interaction = MagicMock(spec=Interaction)
    interaction.response_context.return_value = nullcontext()
    interaction.confirm.return_value = True
    return interaction


def test_loop_exposes_its_configured_state(tmp_path):
    """Loop accessors expose configured dependencies and mutable state."""
    backend = loop_backend()
    interaction = MagicMock(spec=Interaction)
    loop = Loop(
        backend=backend,
        agent_name="Reviewer",
        model="requested-model",
        debug=True,
        interaction=interaction,
        working_directory=tmp_path,
    )

    assert loop.backend is backend
    assert loop.agent.backend is backend
    assert loop.agent.name == "Reviewer"
    assert loop.agent_runner.agent is loop.agent
    assert loop.messages == []
    assert loop.debug is True
    assert loop.stream is False
    assert loop.interaction is interaction
    assert loop.working_directory == tmp_path.resolve()
    assert loop.instructions is None
    assert loop.instructions_manager is not None
    assert loop.permission_manager.configuration_path == tmp_path / ".loop" / "permissions.yaml"
    assert loop.tool_registry.names == []
    assert loop.session == Session(model="requested-model")
    assert loop.model == "requested-model"

    loop.debug = False
    assert loop.debug is False


def test_loop_rejects_an_invalid_compaction_threshold(tmp_path):
    """Loop construction validates automatic compaction policy at its composition boundary."""
    with pytest.raises(ValueError, match="between zero and one"):
        Loop(
            backend=loop_backend(),
            working_directory=tmp_path,
            compaction_threshold=1,
        )


def test_model_command_selects_models_and_reports_backend_catalog_failures(tmp_path):
    """Model commands update loop state and normalize model-list failures as command feedback."""
    interaction = MagicMock(spec=Interaction)
    backend = Mock(default_model="default-model")
    backend.get_context_window.return_value = 8192
    backend.get_models.return_value = [ModelInfo(id="selected-model")]
    loop = Loop(backend=backend, interaction=interaction, working_directory=tmp_path)

    loop.select_model("direct-model")
    assert loop.model == "direct-model"
    assert loop.session.context_window == 8192

    interaction.prompt.side_effect = ["/model selected-model", False]
    loop.run()
    assert loop.model == "selected-model"
    assert interaction.info.call_args.args[0] == "Using model: selected-model"

    backend.get_models.side_effect = BackendConnectionError(
        "offline",
        provider="test",
        operation="list_models",
    )
    interaction.prompt.side_effect = ["/model unavailable", False]
    loop.run()
    assert "Could not list available models: offline" in interaction.warning.call_args.args[0]


def test_run_supplies_registered_dynamic_completion_capabilities(tmp_path):
    """Interactive input receives adapters for current commands, files, skills, and tools."""
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.return_value = False
    skill = Skill("review", "Review code.", tmp_path / "skills" / "review" / "SKILL.md")
    instructions = InstructionsManager(
        skill_manager=SkillManager([skill]), working_directory=tmp_path
    )
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    store.save(Session(id="older-session", name="Alpha session", name_source="user"))
    store.save(Session(id="newer-session", name="Zebra session", name_source="user"))
    session_manager = SessionManager(interaction=interaction, session_store=store)
    registry = ToolRegistry()

    @tool
    def inspect() -> str:
        """Inspect the project."""
        return "done"

    registry.register(inspect)
    loop = Loop(
        backend=loop_backend(),
        tool_registry=registry,
        instructions_manager=instructions,
        interaction=interaction,
        session_manager=session_manager,
        working_directory=tmp_path,
    )

    loop.run()

    completer = interaction.prompt.call_args.kwargs["completer"]

    def values(text):
        return [item.text for item in completer.get_completions(Document(text), Mock())]

    assert values("/he") == ["/help"]
    assert values("$rev") == ["$review"]
    assert values("/permissions add allow ins") == ["inspect"]
    resume = list(completer.get_completions(Document("/resume "), Mock()))
    assert [item.text for item in resume] == ["newer-session", "older-session"]
    assert [item.display_text for item in resume] == ["Zebra session", "Alpha session"]


def test_resume_command_loads_a_persisted_session_id(tmp_path):
    """Submitting a persisted session ID resumes its history and selected model."""
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.side_effect = ["/resume internal-id", False]
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    selected = Session(
        id="internal-id",
        name="Alpha session",
        name_source="user",
        model="session-model",
    )
    store.save(selected)
    sessions = SessionManager(interaction=interaction, session_store=store)
    loop = Loop(
        backend=loop_backend(),
        interaction=interaction,
        session_manager=sessions,
        working_directory=tmp_path,
    )

    loop.run()

    assert sessions.session.id == "internal-id"
    assert loop.model == "session-model"
    assert interaction.prompt.call_args_list[0].args == ()


def test_resumed_missing_model_uses_existing_query_fallback(tmp_path):
    """A resumed model is tried first and replaced through normal query recovery."""
    interaction = output_interaction()
    interaction.prompt.side_effect = ["/resume internal-id", "hello", False]
    interaction.confirm.return_value = True
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    store.save(Session(id="internal-id", model="missing"))
    sessions = SessionManager(interaction=interaction, session_store=store)
    backend = Mock(default_model="default-model")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="replacement")]
    backend.get_response.side_effect = [
        BackendNotFoundError(
            "model missing", provider="openai", operation="create_response", status_code=404
        ),
        [ResponseCompleted(model="replacement")],
    ]
    loop = Loop(
        backend=backend,
        interaction=interaction,
        session_manager=sessions,
        working_directory=tmp_path,
    )

    loop.run()

    assert [request.kwargs["model"] for request in backend.get_response.call_args_list] == [
        "missing",
        "replacement",
    ]
    assert loop.model == "replacement"


def test_resume_command_reports_an_unknown_session_id(tmp_path):
    """An unknown session ID is rejected through normal command feedback."""
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.side_effect = ["/resume missing-id", False]
    sessions = SessionManager(interaction=interaction)
    loop = Loop(
        backend=loop_backend(),
        interaction=interaction,
        session_manager=sessions,
        working_directory=tmp_path,
    )

    loop.run()

    assert "Session 'missing-id' was not found" in interaction.warning.call_args.args[0]


def test_run_resolves_file_context_and_activates_mentioned_skills_before_query(tmp_path):
    """Mentioned files are attached and mentioned skill instructions enter the first request."""
    source = tmp_path / "my app.py"
    source.write_text("print('hello')\n", encoding="utf-8")
    location = tmp_path / "skills" / "review" / "SKILL.md"
    location.parent.mkdir(parents=True)
    location.write_text(
        "---\nname: review\ndescription: Review code.\n---\nFollow review instructions.\n",
        encoding="utf-8",
    )
    instructions = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review code.", location)]),
        working_directory=tmp_path,
    )
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.return_value = [ResponseCompleted()]
    interaction = output_interaction()
    interaction.prompt.side_effect = ['Use $review on @"my app.py"', False]

    loop = Loop(
        backend=backend,
        interaction=interaction,
        instructions_manager=instructions,
        working_directory=tmp_path,
    )
    loop.run()

    message = backend.get_response.call_args.kwargs["input"][0]
    assert message == Message(
        role="user",
        content='Use $review on @"my app.py"',
        context=(
            ContextReference(
                kind="file",
                path="my app.py",
                content="print('hello')\n",
                size_bytes=15,
                included_bytes=15,
                truncated=False,
            ),
        ),
    )
    assert "Follow review instructions." in backend.get_response.call_args.kwargs["instructions"]
    assert loop.session.active_skills == [("review", str(location))]


def test_run_reports_invalid_mentions_without_mutating_or_querying(tmp_path):
    """Mention resolution failures return to input without storing a partial user turn."""
    path = tmp_path / "binary.bin"
    path.write_bytes(b"bad\0data")
    backend = Mock(default_model="model")
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.side_effect = ["Read @binary.bin", False]
    loop = Loop(backend=backend, interaction=interaction, working_directory=tmp_path)

    loop.run()

    assert loop.messages == []
    backend.get_response.assert_not_called()
    interaction.error.assert_called_once_with("Content appears to be binary.")


def test_run_retries_an_exhausted_recoverable_failure(tmp_path):
    """Interactive approval retries the unchanged turn after automatic recovery is exhausted."""
    error = BackendServerError(
        "temporarily unavailable",
        provider="openai",
        operation="create_response",
        status_code=503,
        request_id="request-1",
        retry_after=2.5,
    )
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.side_effect = [error, [ResponseCompleted(answer="done")]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]

    with patch("loop.agent.runner.sleep") as sleep:
        Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    assert backend.get_response.call_count == 2
    interaction.error.assert_called_once_with(
        "temporarily unavailable (HTTP 503, request request-1)"
    )
    interaction.confirm.assert_called_once_with(
        "Retry the complete response after at least 2.5 seconds?", default=False
    )
    interaction.info.assert_any_call("Retrying in 2.5 seconds...")
    sleep.assert_called_once_with(2.5)
    assert [
        item for item in backend.get_response.call_args.kwargs["input"] if item.role == "user"
    ] == [Message(role="user", content="hello")]


def test_run_describes_partial_output_before_retrying(tmp_path):
    """A failed streamed attempt warns that retrying replaces its discarded partial output."""
    error = BackendConnectionError(
        "stream ended",
        provider="openai",
        operation="stream_response",
        response_started=True,
    )
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.side_effect = [error, [ResponseCompleted()]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    interaction.confirm.assert_called_once_with(
        "Partial output was discarded. Retry the complete response?", default=False
    )


def test_run_declines_or_disables_recoverable_retries(tmp_path):
    """Declined and disabled recovery report retained usage without assistant output."""
    for enabled in (True, False):
        error = BackendConnectionError("offline", provider="openai", operation="create_response")
        backend = Mock(default_model="model")
        backend.get_context_window.return_value = None
        backend.get_response.side_effect = error
        interaction = output_interaction()
        interaction.confirm.return_value = False
        interaction.prompt.side_effect = ["hello", False]

        loop = Loop(
            backend=backend,
            interaction=interaction,
            working_directory=tmp_path,
            prompt_on_recoverable_error=enabled,
        )
        loop.run()

        assert loop.messages == [Message(role="user", content="hello")]
        assert interaction.confirm.call_count == int(enabled)
        interaction.run_metrics.assert_called_once()
        assert interaction.run_metrics.call_args.args[0].active_duration_seconds == 0


def test_run_rejects_a_runner_result_without_metrics(tmp_path):
    """The loop rejects an incomplete runner result before presenting statistics."""
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello"]
    loop = Loop(
        backend=loop_backend(get_response=Mock(return_value=[])),
        interaction=interaction,
        working_directory=tmp_path,
    )
    loop.agent_runner.run = Mock(
        return_value=AgentRunResult(final_response=None, turns=0, stop_reason="cancelled")
    )

    with pytest.raises(TypeError, match="completion metrics"):
        loop.run()


def test_run_does_not_offer_to_retry_permanent_failures(tmp_path):
    """Permanent backend failures are reported once and leave the application running."""
    error = BackendAuthenticationError(
        "invalid key", provider="openai", operation="create_response", status_code=401
    )
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.side_effect = error
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    interaction.error.assert_called_once_with("invalid key (HTTP 401)")
    interaction.confirm.assert_not_called()


def test_run_selects_an_available_model_after_not_found(tmp_path):
    """A missing model delegates replacement selection to the interaction."""
    error = BackendNotFoundError(
        "model missing", provider="openai", operation="create_response", status_code=404
    )
    models = [ModelInfo(id="first"), ModelInfo(id="second")]
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = models
    backend.get_response.side_effect = [error, [ResponseCompleted(model="second")]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", "second", False]

    loop = Loop(backend=backend, interaction=interaction, working_directory=tmp_path)
    loop.run()

    assert loop.model == "second"
    assert loop.session.model == "second"
    assert backend.get_response.call_args_list[1].kwargs["model"] == "second"
    assert interaction.prompt.call_args_list[1].args == (
        "Select a replacement model, or enter 'q' to stop: ",
    )
    assert interaction.prompt.call_args_list[1].kwargs == {
        "choices": {"first": "first", "second": "second"},
    }
    interaction.info.assert_any_call("Using model: second")


def test_run_re_prompts_when_user_selects_same_failed_model(tmp_path):
    """The fallback selector re-prompts when the user selects the already-failed model."""
    error = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="second"), ModelInfo(id="missing")]
    backend.get_response.side_effect = [error, [ResponseCompleted()], [ResponseCompleted()]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", "missing", "second", False]
    # First call returns False (re-prompt), second call returns True (accept)
    interaction.confirm.side_effect = [False, True]

    loop = Loop(
        backend=backend, interaction=interaction, model="missing", working_directory=tmp_path
    )
    loop.run()

    assert loop.model == "second"
    interaction.warning.assert_any_call(
        "Model 'missing' was already unavailable; the same failure is "
        "likely to re-occur unless the backend is updated."
    )
    interaction.confirm.assert_called()


def test_run_accepts_same_failed_model_on_confirm(tmp_path):
    """Selecting the failed model and confirming accepts it, breaking immediately."""
    error = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="second"), ModelInfo(id="missing")]
    backend.get_response.side_effect = [error, [ResponseCompleted()]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", "missing", False]
    # confirm returns True to accept the same failed model
    interaction.confirm.return_value = True

    loop = Loop(
        backend=backend, interaction=interaction, model="missing", working_directory=tmp_path
    )
    loop.run()

    assert loop.model == "missing"
    assert loop.session.model == "missing"
    interaction.warning.assert_called_once_with(
        "Model 'missing' was already unavailable; the same failure is "
        "likely to re-occur unless the backend is updated."
    )


@pytest.mark.parametrize(
    "models", [[], BackendConnectionError("offline", provider="openai", operation="list_models")]
)
def test_run_stops_model_fallback_when_discovery_fails(tmp_path, models):
    """Unavailable or empty model discovery returns safely to the conversation prompt."""
    missing = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_response.side_effect = missing
    if isinstance(models, Exception):
        backend.get_models.side_effect = models
    else:
        backend.get_models.return_value = models
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    if isinstance(models, Exception):
        interaction.error.assert_any_call("Could not list available models: offline")
    else:
        interaction.warning.assert_called_once_with("The backend reported no available models.")


def test_run_can_stop_model_selection(tmp_path):
    """Declining the sole discovered model abandons only the failed response turn."""
    missing = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="replacement")]
    backend.get_response.side_effect = missing
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]
    interaction.confirm.return_value = False

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    assert backend.get_response.call_count == 1
    interaction.confirm.assert_called_once_with(
        "Only model 'replacement' is available. Use this model?", default=True
    )


def test_run_accepts_the_only_available_model_after_not_found(tmp_path):
    """Approving the sole discovered model retries the response with that model."""
    missing = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="replacement")]
    backend.get_response.side_effect = [missing, [ResponseCompleted(model="replacement")]]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]
    interaction.confirm.return_value = True

    loop = Loop(backend=backend, interaction=interaction, working_directory=tmp_path)
    loop.run()

    assert loop.model == "replacement"
    assert backend.get_response.call_args_list[1].kwargs["model"] == "replacement"
    interaction.confirm.assert_called_once_with(
        "Only model 'replacement' is available. Use this model?", default=True
    )


def test_run_can_exit_multi_model_fallback_selection(tmp_path):
    """Exiting a multi-model fallback selector abandons only the failed response turn."""
    missing = BackendNotFoundError("missing", provider="openai", operation="create_response")
    backend = Mock(default_model="missing")
    backend.get_context_window.return_value = None
    backend.get_models.return_value = [ModelInfo(id="first"), ModelInfo(id="second")]
    backend.get_response.side_effect = missing
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False, False]

    Loop(backend=backend, interaction=interaction, working_directory=tmp_path).run()

    assert backend.get_response.call_count == 1
    interaction.confirm.assert_not_called()


def test_loop_uses_an_injected_mention_registry(tmp_path):
    """Library callers can replace all default mention semantics and completion."""
    mentions = Mock(spec=MentionManager)
    mentions.completion_adapters = ()
    mentions.resolve.return_value = ()
    backend = MagicMock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.return_value = [ResponseCompleted()]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["Custom !reference", False]

    Loop(
        backend=backend,
        interaction=interaction,
        mention_manager=mentions,
        working_directory=tmp_path,
    ).run()

    mentions.resolve.assert_called_once_with("Custom !reference")


def test_loop_passes_custom_instruction_fallbacks_to_discovery(tmp_path):
    """Loop discovery exposes configured fallback instruction filenames to library callers."""
    (tmp_path / "CUSTOM.md").write_text("custom instructions", encoding="utf-8")

    loop = Loop(
        backend=loop_backend(),
        working_directory=tmp_path,
        agents_filenames=("AGENTS.md", "CUSTOM.md"),
    )

    assert loop.instructions == "custom instructions"


def test_loop_uses_an_injected_permission_manager(tmp_path):
    """An explicit permission manager replaces local policy discovery."""
    permissions = PermissionManager(configuration=PermissionConfiguration())

    loop = Loop(
        backend=loop_backend(),
        permission_manager=permissions,
        working_directory=tmp_path,
    )

    assert loop.permission_manager is permissions


def test_loops_share_local_conversation_context(tmp_path):
    """Injected context keeps history while each loop applies its backend metadata."""
    session = Session(
        messages=[Message(role="user", content="hello")], tokens=12, model="served-model"
    )
    first = Loop(backend=loop_backend(), session=session, working_directory=tmp_path)
    second_backend = Mock(default_model="other-model")
    second_backend.get_response.return_value = []
    second = Loop(backend=second_backend, session=session, working_directory=tmp_path, stream=True)

    assert first.session is second.session is session
    assert second.messages == [Message(role="user", content="hello")]
    assert second.session.tokens == 12
    assert second.session.model == "other-model"
    response = second.agent_runner.query()
    assert response.answer == ""
    assert response.reasoning == ""
    assert response.metrics.duration_seconds >= 0
    second_backend.get_response.assert_called_once_with(
        input=session.messages,
        instructions=None,
        stream=True,
        model="other-model",
        tools=[],
    )


def test_new_session_is_not_persisted_until_its_first_completed_query(tmp_path):
    """A fresh session creates storage only after a query result forms a complete snapshot."""
    backend = loop_backend(
        get_response=Mock(
            return_value=[
                ResponseCompleted(
                    items=(Message(role="assistant", content="answer"),),
                    usage=Usage(total_tokens=9),
                    model="served-model",
                )
            ]
        )
    )
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]
    store = SQLiteSessionStore(tmp_path / ".loop" / "sessions.db")
    session_manager = SessionManager(interaction=interaction, session_store=store)
    loop = Loop(
        backend=backend,
        working_directory=tmp_path,
        interaction=interaction,
        session_manager=session_manager,
    )

    assert not (tmp_path / ".loop").exists()

    loop.run()

    session_info = store.list()[0]
    assert store.load(session_info.id) == loop.session
    assert loop.session.messages == [
        Message(role="user", content="hello"),
        Message(role="assistant", content="answer"),
    ]
    assert loop.session.tokens == 9
    assert loop.session.model == "served-model"


def test_run_does_not_generate_a_name_for_an_already_named_session(tmp_path):
    """Completed queries retain a non-provisional session name at the loop boundary."""
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]
    generator = Mock()
    loop = Loop(
        backend=loop_backend(get_response=Mock(return_value=[ResponseCompleted()])),
        working_directory=tmp_path,
        interaction=interaction,
        session=Session(name="My session", name_source="user"),
        session_name_generator=generator,
    )

    loop.run()

    generator.generate.assert_not_called()
    assert loop.session.name == "My session"


def test_loop_without_a_session_store_never_creates_session_files(tmp_path):
    """A caller that omits persistence keeps completed queries entirely in memory."""
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]
    loop = Loop(
        backend=loop_backend(get_response=Mock(return_value=[ResponseCompleted()])),
        working_directory=tmp_path,
        interaction=interaction,
    )

    loop.run()

    assert not (tmp_path / ".loop").exists()


def test_loop_delegates_a_session_identifier_to_an_injected_manager():
    """A persisted identifier is resolved through the injected session manager."""
    stored = Session(messages=[Message(role="user", content="saved")])
    manager = Mock(spec=SessionManager)
    manager.session = stored
    manager.interaction = MagicMock(spec=Interaction)

    loop = Loop(backend=loop_backend(), session="session-id", session_manager=manager)

    assert loop.session is stored
    manager.load_session.assert_called_once_with("session-id")


def test_loop_loads_a_persisted_session_identifier(tmp_path):
    """The constructor resumes persisted state under the attached backend model."""
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    stored = Session(
        messages=[Message(role="user", content="saved")], tokens=4, model="saved-model"
    )
    session_id = store.save(stored)

    manager = SessionManager(session_store=store)
    loop = Loop(backend=loop_backend(), session=session_id, session_manager=manager)

    assert loop.session.messages == stored.messages
    assert loop.session.tokens == stored.tokens
    assert loop.session.model == "default-model"


def test_loop_uses_an_injected_manager_session_without_reloading_it():
    """An injected manager keeps its active session when no replacement is requested."""
    session = Session(messages=[Message(role="user", content="saved")])
    manager = SessionManager(session=session)

    loop = Loop(backend=loop_backend(), session_manager=manager)

    assert loop.session is session


def test_loop_prefers_an_explicit_interaction_over_the_manager_interaction():
    """An explicit interaction controls loop I/O when a manager is also supplied."""
    manager = SessionManager(interaction=output_interaction())
    interaction = MagicMock(spec=Interaction)

    loop = Loop(backend=loop_backend(), interaction=interaction, session_manager=manager)

    assert loop.interaction is interaction


def test_loop_loads_project_instructions_for_its_normalized_working_directory(tmp_path):
    """A loop exposes instructions discovered for its normalized working directory."""
    (tmp_path / "AGENTS.md").write_text("project rules", encoding="utf-8")

    loop = Loop(backend=loop_backend(), working_directory=str(tmp_path))

    assert loop.instructions == "project rules"


def test_query_refreshes_instructions_and_explicit_working_directory(tmp_path):
    """Each query prepares current sources and an explicit directory change updates scope."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "AGENTS.md").write_text("First rules.", encoding="utf-8")
    (second / "AGENTS.md").write_text("Second rules.", encoding="utf-8")
    backend = Mock(default_model="default-model")
    backend.get_response.return_value = []
    loop = Loop(
        backend=backend,
        tool_registry=ToolRegistry(BUILTIN_TOOLS),
        working_directory=first,
    )

    loop.set_working_directory(second)
    loop.agent_runner.query()

    assert loop.working_directory == second.resolve()
    assert backend.get_response.call_args.kwargs["instructions"] == "Second rules."


def test_query_does_not_request_model_metadata_or_tokenization(tmp_path):
    """A query persists model context capacity without hidden tokenization calls."""
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = 128000
    backend.get_response.return_value = []
    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    loop = Loop(
        backend=backend,
        tool_registry=ToolRegistry(BUILTIN_TOOLS),
        working_directory=tmp_path,
    )

    loop.agent_runner.query()

    assert backend.get_context_window.call_args_list == [call("model"), call("model")]
    backend.count_tokens.assert_not_called()
    assert loop.session.context_window == 128000


def test_shared_backend_receives_each_loops_agent_scoped_tools(tmp_path):
    """Loops sharing one backend expose only their own tool definitions on each request."""
    backend = Mock(default_model="model")
    backend.get_context_window.return_value = None
    backend.get_response.return_value = []

    @tool
    def first_tool() -> str:
        """Run the first capability."""
        return "first"

    @tool
    def second_tool() -> str:
        """Run the second capability."""
        return "second"

    first_registry = ToolRegistry([first_tool])
    second_registry = ToolRegistry([second_tool])
    first = Loop(backend=backend, tool_registry=first_registry, working_directory=tmp_path)
    second = Loop(backend=backend, tool_registry=second_registry, working_directory=tmp_path)

    first.agent_runner.query()
    second.agent_runner.query()

    assert first.tool_registry is first_registry
    assert second.tool_registry is second_registry
    assert [request.kwargs["tools"][0].name for request in backend.get_response.call_args_list] == [
        "first_tool",
        "second_tool",
    ]


def test_loop_aligns_injected_session_capacity_with_backend(tmp_path):
    """Loop construction replaces stale persisted capacity with backend runtime metadata."""
    session = Session(model="old-model", context_window=32768)
    manager = SessionManager(session=session)
    backend = loop_backend(
        default_model="active-model",
        get_context_window=Mock(return_value=128000),
    )

    Loop(backend=backend, session_manager=manager, working_directory=tmp_path)

    backend.get_context_window.assert_called_once_with("active-model")
    assert session.model == "active-model"
    assert session.context_window == 128000


def test_query_compacts_above_threshold_and_sends_only_latest_working_context(tmp_path):
    """Threshold compaction persists full history while replacing the next provider input."""
    messages = [
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="new question"),
    ]
    session = Session(messages=messages, tokens=85)
    compacted = CompactionContextItem(
        provider="openai",
        data={"type": "compaction", "encrypted_content": "opaque"},
    )
    backend = loop_backend(
        get_context_window=Mock(return_value=100),
        compact=Mock(
            return_value=CompactionResult(
                items=(compacted,),
                usage=Usage(input_tokens=85, output_tokens=20, total_tokens=105),
                context_tokens=20,
            )
        ),
        get_response=Mock(return_value=[]),
    )
    interaction = MagicMock(spec=Interaction)
    loop = Loop(
        backend=backend,
        session=session,
        interaction=interaction,
        working_directory=tmp_path,
    )

    loop.agent_runner.query()

    assert session.messages == messages
    assert len(session.compactions) == 1
    assert session.compactions[0].boundary == 3
    assert session.compactions[0].instructions.working_directory == str(tmp_path)
    assert session.tokens == 20
    assert session.context_window == 100
    backend.compact.assert_called_once_with(messages, instructions=None, model="default-model")
    assert backend.get_response.call_args.kwargs["input"] == [compacted]
    assert [call.args[0] for call in interaction.info.call_args_list] == [
        "Compacting session context...",
        "Compacted session context from 85 to 20 tokens.",
    ]


def test_loop_rejects_a_missing_explicit_working_directory(tmp_path):
    """Explicit directory changes reject missing targets without altering loop state."""
    loop = Loop(backend=loop_backend(), working_directory=tmp_path)

    with pytest.raises(NotADirectoryError, match="does not exist"):
        loop.set_working_directory(tmp_path / "missing")

    assert loop.working_directory == tmp_path.resolve()


def test_loop_restores_only_skills_with_the_same_discovered_identity(tmp_path):
    """Session restoration reloads valid active identities and ignores stale locations."""
    (tmp_path / ".git").mkdir()
    location = tmp_path / ".agents" / "skills" / "review" / "SKILL.md"
    location.parent.mkdir(parents=True)
    location.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nRestored body.\n",
        encoding="utf-8",
    )
    session = Session(
        instruction_working_directory=str(tmp_path),
        active_skills=[
            ("missing", str(tmp_path / "missing" / "SKILL.md")),
            ("review", str(location.resolve())),
        ],
    )

    loop = Loop(backend=loop_backend(), session=session)

    assert loop.working_directory == tmp_path.resolve()
    assert "Restored body." in loop.instructions


def test_skill_activation_updates_instructions_for_the_immediate_requery(tmp_path):
    """A skill tool result stays compact while its body enters the next backend request."""
    location = tmp_path / "review" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nFollow review instructions.",
        encoding="utf-8",
    )
    registry = ToolRegistry([manage_skills])
    backend = Mock(default_model="model")
    backend.get_response.return_value = []
    manager = SkillManager([Skill("review", "Review code.", location)])
    instructions_manager = InstructionsManager(skill_manager=manager)
    loop = Loop(
        backend=backend,
        tool_registry=registry,
        interaction=output_interaction(),
        working_directory=tmp_path,
        instructions_manager=instructions_manager,
    )
    call = ToolCall(
        call_id="skill-call",
        name="manage_skills",
        arguments='{"action":"activate","name":"review"}',
    )
    response = Response(
        answer="",
        reasoning="",
        tool_calls=(call,),
        items=(call,),
    )
    loop.session.add_message(response)

    assert len(loop.agent_runner.handle_tool_calls(response)) == 1
    result = loop.messages[-1]
    loop.agent_runner.query()

    assert isinstance(result, ToolResult)
    assert "Follow review instructions." not in result.output
    assert "Follow review instructions." in backend.get_response.call_args.kwargs["instructions"]


def test_skill_activation_is_persisted_with_its_tool_result(tmp_path):
    """A successful skill mutation is durable without requiring another backend query."""
    location = tmp_path / "review" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nReview.", encoding="utf-8"
    )
    registry = ToolRegistry([manage_skills])
    backend = Mock(default_model="model")
    store = SQLiteSessionStore(tmp_path / "sessions.db")
    sessions = SessionManager(session_store=store)
    manager = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review code.", location)])
    )
    interaction = output_interaction()
    loop = Loop(
        backend=backend,
        tool_registry=registry,
        instructions_manager=manager,
        session_manager=sessions,
        interaction=interaction,
        working_directory=tmp_path,
    )
    call = ToolCall(
        call_id="skill-call",
        name="manage_skills",
        arguments='{"action":"activate","name":"review"}',
    )

    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    sessions.add_response(response)
    loop.agent_runner.handle_tool_calls(response)

    restored = store.load(loop.session.id)
    assert restored.active_skills == [("review", str(location))]


def test_skill_deactivation_updates_instructions_for_the_immediate_requery(tmp_path):
    """Deactivation removes a skill body from the next backend request."""
    location = tmp_path / "review" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: review\ndescription: Review code.\n---\n\nFollow review instructions.",
        encoding="utf-8",
    )
    registry = ToolRegistry([manage_skills])
    backend = Mock(default_model="model")
    backend.get_response.return_value = []
    manager = SkillManager([Skill("review", "Review code.", location)])
    instructions_manager = InstructionsManager(skill_manager=manager)
    instructions_manager.activate_skill("review")
    loop = Loop(
        backend=backend,
        tool_registry=registry,
        interaction=output_interaction(),
        working_directory=tmp_path,
        instructions_manager=instructions_manager,
    )
    call = ToolCall(
        call_id="skill-call",
        name="manage_skills",
        arguments='{"action":"deactivate","name":"review"}',
    )
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))
    loop.session.add_message(response)

    assert len(loop.agent_runner.handle_tool_calls(response)) == 1
    result = loop.messages[-1]
    loop.agent_runner.query()

    assert isinstance(result, ToolResult)
    assert json.loads(result.output)["instructions_updated"] is True
    updated_instructions = backend.get_response.call_args.kwargs["instructions"]
    assert "Follow review instructions." not in updated_instructions


def test_run_requeries_after_a_tool_call_and_records_local_items(tmp_path):
    """The runner records a tool result, requeries, reports usage, and exits."""
    registry = ToolRegistry()

    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    registry.register(echo)
    call = ToolCall(call_id="call", name="echo", arguments='{"text":"done"}', id="fc")
    backend = Mock(default_model="requested-model")
    backend.get_context_window.return_value = 1000
    backend.get_response.side_effect = [
        [
            ToolCallCompleted(call=call),
            ResponseCompleted(items=(call,), usage=Usage(total_tokens=10)),
        ],
        [
            AnswerDelta(text="done"),
            AnswerCompleted(text="done"),
            ResponseCompleted(
                items=(Message(role="assistant", content="done"),),
                usage=Usage(total_tokens=12),
                answer="done",
            ),
        ],
    ]
    interaction = output_interaction()
    interaction.prompt.side_effect = ["hello", False]

    loop = Loop(
        backend=backend,
        interaction=interaction,
        tool_registry=registry,
        working_directory=tmp_path,
    )
    loop.run()

    second_input = backend.get_response.call_args_list[1].kwargs["input"]
    assert second_input[:3] == [
        Message(role="user", content="hello"),
        call,
        ToolResult(call_id="call", output="done"),
    ]
    assert second_input[-1] == Message(role="assistant", content="done")
    interaction.answer_delta.assert_called_once_with("done", start=True)
    assert interaction.response_context.call_count == 2
    interaction.run_metrics.assert_called_once()
    metrics = interaction.run_metrics.call_args.args[0]
    assert metrics.model == "requested-model"
    assert metrics.context_tokens == 12
    assert metrics.context_window == 1000
    assert len(metrics.model_calls) == 2
    assert [event.type for event in loop.session.events] == [
        "conversation_item",
        "conversation_item",
        "permission",
        "conversation_item",
        "conversation_item",
        "run_completed",
    ]
    visible_items = [
        loop.messages[event.item_index]
        for event in loop.session.events
        if event.type == "conversation_item"
    ]
    assert [type(item) for item in visible_items] == [Message, ToolCall, ToolResult, Message]
    interaction.conversation_ended.assert_called_once_with()


def test_run_keeps_handled_commands_out_of_model_history():
    """The runner skips every command consumed by its command manager."""
    backend = Mock(default_model="model")
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.side_effect = ["/help", "/missing", False]

    loop = Loop(backend=backend, interaction=interaction)
    loop.run()

    assert loop.messages == []
    backend.get_response.assert_not_called()


@pytest.mark.parametrize("command", ["/exit", "/quit"])
def test_run_exit_commands_end_the_conversation(command):
    """Predefined slash exit commands terminate without a backend request."""
    backend = Mock(default_model="model")
    interaction = MagicMock(spec=Interaction)
    interaction.prompt.return_value = command

    loop = Loop(backend=backend, interaction=interaction)
    loop.run()

    assert loop.messages == []
    backend.get_response.assert_not_called()
    interaction.conversation_ended.assert_called_once_with()


def test_handle_tool_calls_delegates_session_updates():
    """Tool results and instruction state are delegated to the session manager."""
    registry = Mock()
    registry.call_with_timing.return_value = ("tool result", 0.25)
    backend = loop_backend(get_response=Mock(return_value=[]))
    session_manager = Mock(spec=SessionManager)
    session_manager.interaction = MagicMock(spec=Interaction)
    session_manager.session = Session()
    loop = Loop(backend=backend, tool_registry=registry, session_manager=session_manager)
    call = function_call()
    response = Response(answer="", reasoning="", tool_calls=(call,), items=(call,))

    assert len(loop.agent_runner.handle_tool_calls(response)) == 1

    session_manager.add_tool_call.assert_called_once_with(
        call_id="call_123",
        output="tool result",
        working_directory=str(loop.working_directory),
        active_skills=[],
    )
    registry.call_with_timing.assert_called_once_with(
        call.name,
        call.arguments,
        interaction=loop.interaction,
        instructions_manager=loop.instructions_manager,
        permission_manager=loop.permission_manager,
    )
    assert loop.permission_manager.recorder is session_manager
    assert loop.agent_runner.handle_tool_calls(Response(answer="", reasoning="")) == ()


def test_query_selects_only_the_event_production_mode():
    """Both loop modes forward identical history with only the stream flag differing."""
    backend = loop_backend(get_response=Mock(return_value=[]))
    session = Session(messages=[Message(role="user", content="hello")])

    Loop(backend=backend, session=session).agent_runner.query()
    Loop(backend=backend, session=session, stream=True).agent_runner.query()

    assert backend.get_response.call_args_list[0].kwargs["stream"] is False
    assert backend.get_response.call_args_list[1].kwargs["stream"] is True


def test_query_delegates_instruction_state_to_the_session_manager():
    """Queries delegate their prepared instruction state to the session manager."""
    backend = loop_backend(get_response=Mock(return_value=[]))
    session_manager = Mock(spec=SessionManager)
    session_manager.interaction = MagicMock(spec=Interaction)
    session_manager.session = Session()
    session_manager.messages = []
    session_manager.model_context = []
    session_manager.context_window = None
    loop = Loop(backend=backend, session_manager=session_manager)

    loop.agent_runner.query()

    session_manager.update_instruction_state.assert_called_once_with(
        working_directory=str(loop.working_directory),
        active_skills=[],
    )


def test_query_prefers_the_explicit_model_over_response_metadata():
    """Request selection stays independent of a model reported by an earlier response."""
    backend = loop_backend(get_response=Mock(return_value=[]))
    session = Session(model="served-model")

    Loop(backend=backend, model="requested-model", session=session).agent_runner.query()

    assert backend.get_response.call_args.kwargs["model"] == "requested-model"


def test_query_rejects_missing_model_selection():
    """A query fails clearly when neither the loop nor backend selects a model."""
    backend = loop_backend(default_model=None, get_response=Mock())

    with pytest.raises(ValueError, match="No model was selected"):
        Loop(backend=backend).agent_runner.query()

    backend.get_response.assert_not_called()
