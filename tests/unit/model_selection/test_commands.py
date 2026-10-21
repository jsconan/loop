"""Tests for model discovery and selection commands."""

import asyncio
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from loop import (
    AsyncInteractionInput,
    BackendAuthenticationError,
    BackendConnectionError,
    CommandManager,
    Interaction,
    ModelInfo,
    ModelSelection,
    OpenAIBackend,
    SessionManager,
)
from loop.model_selection.commands import ModelCommands


def model_selection_mock():
    """Build a selection double with native async operations and sync completion lookup."""
    selection = Mock(spec=ModelSelection)
    selection.available = Mock(return_value=[])
    selection.available_async = AsyncMock(return_value=[])
    selection.select_async = AsyncMock()
    selection.select_fallback_async = AsyncMock(return_value=False)
    return selection


def test_model_commands_list_show_and_select_available_models():
    """Commands list optional context data and select available models."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.selected = None
    models = [
        ModelInfo(id="model-a", context_window="10"),
        ModelInfo(id="model-b", context_window="20"),
        ModelInfo(id="model-c"),
    ]
    selection.available.return_value = models
    selection.available_async.return_value = models
    provider = ModelCommands(selection)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(provider)

    manager.call("models")
    manager.call("model")
    manager.call("model", "model-b")

    interaction.table.assert_called_once_with(
        [
            SimpleNamespace(id="model-a", context_window="10 tokens"),
            SimpleNamespace(id="model-b", context_window="20 tokens"),
            SimpleNamespace(id="model-c", context_window="Not reported"),
        ],
        title="Available models:",
        columns=("id", "context_window"),
    )
    assert interaction.info.call_args_list[0].args[0] == "Using the backend default model."
    selection.select_async.assert_awaited_once_with("model-b")
    assert manager.commands[-1].completion.provider == "models"
    completion = provider.get_completion_providers()[0]
    assert [(value.value, value.description) for value in completion.provider()] == [
        ("model-a", "10 tokens"),
        ("model-b", "20 tokens"),
        ("model-c", "Not reported"),
    ]


def test_check_and_models_commands_handle_an_async_backend_paginator():
    """Both catalog commands consume the OpenAI SDK's asynchronous model paginator."""
    interaction = Mock(spec=Interaction)
    sdk = Mock()

    async def model_page():
        """Yield a model as an async SDK page does."""
        yield SimpleNamespace(
            id="model-a",
            model_extra={"max_model_len": 65536},
        )

    sdk.models.list.return_value = model_page()
    backend = OpenAIBackend(default_model="model-a", context_window=65536)
    selection = ModelSelection(backend, SessionManager())
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    async def dispatch_catalog_commands():
        """Run both commands through normal async dispatch in one event loop."""
        await manager.call_async("check")
        await manager.call_async("models")

    try:
        with patch("loop.backend.openai.AsyncOpenAI", return_value=sdk):
            asyncio.run(dispatch_catalog_commands())
    finally:
        manager.close()

    sdk.models.list.assert_called_once_with(timeout=2.0)
    interaction.info.assert_called_once_with("Backend is available. Model 'model-a' is available.")
    interaction.table.assert_called_once_with(
        [SimpleNamespace(id="model-a", context_window="65,536 tokens")],
        title="Available models:",
        columns=("id", "context_window"),
    )


def test_async_dispatch_runs_native_model_commands_on_the_owner():
    """Async model commands use async selection APIs on the active event-loop thread."""
    interaction = Mock(spec=Interaction)
    interaction.worker_calls_routed = True
    selection = model_selection_mock()
    observed_calls = []

    async def available_async():
        """Record the thread that runs the native asynchronous catalog method."""
        observed_calls.append(("available", get_ident()))
        return [ModelInfo(id="model-a")]

    async def select_async(model):
        """Record the thread that applies the async model selection."""
        observed_calls.append(("select", get_ident()))

    selection.available_async.side_effect = available_async
    selection.select_async.side_effect = select_async
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    async def dispatch():
        """Dispatch model listing from an async owner and return its thread identity."""
        owner_thread = get_ident()
        await manager.call_async("models")
        await manager.call_async("model", "model-a")
        return owner_thread

    try:
        owner_thread = asyncio.run(dispatch())
    finally:
        manager.close()

    assert observed_calls == [
        ("available", owner_thread),
        ("available", owner_thread),
        ("select", owner_thread),
    ]
    selection.available.assert_not_called()
    selection.select.assert_not_called()
    interaction.table.assert_called_once()


def test_async_dispatch_reports_model_preference_write_failure():
    """A failed durable selection is reported without escaping to the app error handler."""
    interaction = Mock(spec=Interaction)
    interaction.worker_calls_routed = True
    session_manager = SessionManager()
    selection = ModelSelection(
        SimpleNamespace(
            default_model="model-a",
            get_context_window=Mock(return_value=2048),
            get_context_window_async=AsyncMock(return_value=2048),
            get_models=Mock(return_value=[ModelInfo(id="model-a")]),
            get_models_async=AsyncMock(return_value=[ModelInfo(id="model-a")]),
        ),
        session_manager,
        on_select=Mock(side_effect=OSError("disk full")),
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    try:
        asyncio.run(manager.call_async("model", "model-a"))
    finally:
        manager.close()

    problem = interaction.report.call_args.args[0]
    assert problem.code == "model_selection.preference_save_failed"
    assert "could not be saved" in problem.detail
    assert selection.selected is None
    assert session_manager.model == "model-a"


def test_check_command_reports_a_healthy_effective_model():
    """The check command confirms that the active model remains available."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.effective = "model-a"
    selection.available_async.return_value = [ModelInfo(id="model-a")]
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    interaction.info.assert_called_once_with("Backend is available. Model 'model-a' is available.")
    selection.select_fallback_async.assert_not_awaited()


def test_check_command_gently_reports_an_unreachable_backend():
    """The check command presents failed probes as diagnostic warnings rather than errors."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.available_async.side_effect = BackendConnectionError(
        "offline",
        provider="test",
        operation="list_models",
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    interaction.warning.assert_called_once_with("The backend is not reachable.")
    interaction.report.assert_not_called()
    selection.select_fallback_async.assert_not_awaited()


def test_check_command_offers_existing_fallback_for_a_missing_model():
    """The check command delegates absent model recovery to the established fallback flow."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.available_async.return_value = [ModelInfo(id="replacement")]
    selection.effective = "missing"
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    assert interaction.warning.call_args.args[0] == (
        "Backend is available, but model 'missing' is not available."
    )
    fallback_interaction = selection.select_fallback_async.await_args.args[0]
    assert isinstance(fallback_interaction, AsyncInteractionInput)


def test_check_command_reports_fallback_preference_write_failure():
    """A failed fallback preference write is reported without stopping the app."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.available_async.return_value = [ModelInfo(id="replacement")]
    selection.effective = "missing"
    selection.select_fallback_async.side_effect = OSError("disk full")
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    problem = interaction.report.call_args.args[0]
    assert problem.code == "model_selection.preference_save_failed"


def test_check_command_reports_fallback_backend_failure():
    """Fallback backend failures use the established sanitized backend reporting."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.available_async.return_value = [ModelInfo(id="replacement")]
    selection.effective = "missing"
    selection.select_fallback_async.side_effect = BackendAuthenticationError(
        "invalid credentials",
        provider="test",
        operation="context_window",
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    problem = interaction.report.call_args.args[0]
    assert problem.code == "backend.unavailable"
    assert problem.detail == "The backend is not reachable."


def test_check_command_offers_a_model_when_none_is_configured():
    """The check command delegates a singleton model catalog to the shared choice prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = False
    selection = ModelSelection(
        SimpleNamespace(
            default_model=None,
            get_context_window=Mock(),
            get_models=Mock(return_value=[ModelInfo(id="replacement")]),
            get_context_window_async=AsyncMock(return_value=2048),
            get_models_async=AsyncMock(return_value=[ModelInfo(id="replacement")]),
        ),
        SessionManager(),
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    interaction.warning.assert_called_once_with("Backend is available, but no model is selected.")
    interaction.prompt.assert_called_once_with(
        "Select a replacement model:", choices={"replacement": "replacement"}
    )


def test_model_command_reports_an_explicit_selection():
    """Model inspection distinguishes an explicit selection from the backend default."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.selected = "selected"
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("model")

    interaction.info.assert_called_once_with("Selected model: selected")


def test_model_commands_report_empty_invalid_and_failed_catalogs():
    """Catalog absence, invalid choices, and backend failures become command feedback."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.selected = None
    selection.available_async.return_value = []
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("models")
    manager.call("model", "missing")
    interaction.info.assert_called_once_with("No models available.")
    interaction.warning.assert_called_once_with("Model 'missing' is not available.")
    interaction.report.assert_not_called()

    selection.available_async.side_effect = BackendConnectionError(
        "offline",
        provider="test",
        operation="list_models",
    )
    manager.call("models")
    interaction.report.assert_not_called()
    assert interaction.warning.call_count == 2
    interaction.warning.assert_called_with("The backend is not reachable.")


def test_model_commands_report_nonrecoverable_catalog_failures_as_errors():
    """Model commands preserve true backend failures as structured error reports."""
    interaction = Mock(spec=Interaction)
    selection = model_selection_mock()
    selection.available_async.side_effect = BackendAuthenticationError(
        "invalid credentials",
        provider="test",
        operation="list_models",
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("models")

    problem = interaction.report.call_args.args[0]
    assert problem.detail == "The backend is not reachable."
    assert problem.severity == "error"
    interaction.warning.assert_not_called()
