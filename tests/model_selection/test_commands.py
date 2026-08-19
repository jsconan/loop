"""Tests for model discovery and selection commands."""

from unittest.mock import Mock

from loop import BackendConnectionError, CommandManager, Interaction, ModelInfo, ModelSelection
from loop.model_selection.commands import ModelCommands


def test_model_commands_list_show_and_select_available_models():
    """Commands present the catalog and update a valid active selection."""
    interaction = Mock(spec=Interaction)
    selection = Mock(spec=ModelSelection)
    selection.selected = None
    selection.available.return_value = [ModelInfo(id="model-a"), ModelInfo(id="model-b")]
    manager = CommandManager(interaction=interaction)
    manager.register_provider(ModelCommands(selection))

    manager.call("models")
    manager.call("model")
    manager.call("model", "model-b")

    interaction.table.assert_called_once_with(
        [ModelInfo(id="model-a"), ModelInfo(id="model-b")],
        title="Available models:",
        columns=("id", "context_window"),
    )
    assert interaction.info.call_args_list[0].args[0] == "Using the backend default model."
    selection.select.assert_called_once_with("model-b")
    assert manager.commands[-1].completion.provider == "models"


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
    assert "Model 'missing' is not available" in interaction.warning.call_args.args[0]

    selection.available.side_effect = BackendConnectionError(
        "offline",
        provider="test",
        operation="list_models",
    )
    manager.call("models")
    assert "Could not list available models: offline" in interaction.warning.call_args.args[0]
