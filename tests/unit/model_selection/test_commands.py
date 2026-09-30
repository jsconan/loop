"""Tests for model discovery and selection commands."""

from types import SimpleNamespace
from unittest.mock import Mock

from loop import (
    BackendAuthenticationError,
    BackendConnectionError,
    CommandManager,
    Interaction,
    ModelInfo,
    ModelSelection,
    SessionManager,
)
from loop.model_selection.commands import ModelCommands


def test_model_commands_list_show_and_select_available_models():
    """Commands present the catalog and update a valid active selection."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.selected = None
    selection.available.return_value = [
        ModelInfo(id="model-a", context_window="10"),
        ModelInfo(id="model-b", context_window="20"),
    ]
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
        ],
        title="Available models:",
        columns=("id", "context_window"),
    )
    assert interaction.info.call_args_list[0].args[0] == "Using the backend default model."
    selection.select.assert_called_once_with("model-b")
    assert manager.commands[-1].completion.provider == "models"
    completion = provider.get_completion_providers()[0]
    assert [(value.value, value.description) for value in completion.provider()] == [
        ("model-a", "10 tokens"),
        ("model-b", "20 tokens"),
    ]


def test_check_command_reports_a_healthy_effective_model():
    """The check command confirms that the active model remains available."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.effective = "model-a"
    selection.available.return_value = [ModelInfo(id="model-a")]
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    interaction.info.assert_called_once_with("Backend is available. Model 'model-a' is available.")
    selection.select_fallback.assert_not_called()


def test_check_command_gently_reports_an_unreachable_backend():
    """The check command presents failed probes as diagnostic warnings rather than errors."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.available.side_effect = BackendConnectionError(
        "offline",
        provider="test",
        operation="list_models",
    )
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    interaction.warning.assert_called_once_with("The backend is not reachable.")
    interaction.report.assert_not_called()
    selection.select_fallback.assert_not_called()


def test_check_command_offers_existing_fallback_for_a_missing_model():
    """The check command delegates absent model recovery to the established fallback flow."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.available.return_value = [ModelInfo(id="replacement")]
    selection.effective = "missing"
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("check")

    assert interaction.warning.call_args.args[0] == (
        "Backend is available, but model 'missing' is not available."
    )
    selection.select_fallback.assert_called_once_with(interaction)


def test_check_command_offers_a_model_when_none_is_configured():
    """The check command delegates a singleton model catalog to the shared choice prompt."""
    interaction = Mock(spec=Interaction)
    interaction.prompt.return_value = False
    selection = ModelSelection(
        SimpleNamespace(
            default_model=None,
            get_context_window=Mock(),
            get_models=Mock(return_value=[ModelInfo(id="replacement")]),
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
    selection = Mock(spec=ModelSelection)
    selection.selected = "selected"
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("model")

    interaction.info.assert_called_once_with("Selected model: selected")


def test_model_commands_report_empty_invalid_and_failed_catalogs():
    """Catalog absence, invalid choices, and backend failures become command feedback."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.selected = None
    selection.available.return_value = []
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("models")
    manager.call("model", "missing")
    interaction.info.assert_called_once_with("No models available.")
    interaction.warning.assert_called_once_with("Model 'missing' is not available.")
    interaction.report.assert_not_called()

    selection.available.side_effect = BackendConnectionError(
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
    selection = Mock(spec=ModelSelection)
    selection.available.side_effect = BackendAuthenticationError(
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
