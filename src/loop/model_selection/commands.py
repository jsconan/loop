"""Expose active model selection through user commands."""

from typing import Annotated

from pydantic import Field

from ..backend import BackendError
from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion
from ..models import ModelInfo
from .selection import ModelSelection


class ModelCommands:
    """Expose one model selection through interactive commands.

    Args:
        model_selection (ModelSelection): Active conversation model selection to inspect or change.
    """

    _model_selection: ModelSelection

    def __init__(self, model_selection: ModelSelection) -> None:
        self._model_selection = model_selection

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return model discovery and selection command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Model discovery and selection commands.
        """
        return (
            CommandRegistration(self.models, name="models"),
            CommandRegistration(
                self.model,
                name="model",
                completion=CommandCompletion(provider="models"),
            ),
        )

    def models(self, context: CommandContext) -> None:
        """List all models available from the backend."""
        models = self._available()
        if not models:
            context.interaction.info("No models available.")
            return
        context.interaction.table(
            models,
            title="Available models:",
            columns=("id", "context_window"),
        )

    def model(
        self,
        context: CommandContext,
        name: Annotated[
            str | None,
            Field(description="Exact backend model ID, or omit it to show the selection."),
        ] = None,
    ) -> None:
        """Show or select the model used for subsequent requests."""
        if name is None:
            selected = self._model_selection.selected
            context.interaction.info(
                f"Selected model: {selected}" if selected else "Using the backend default model."
            )
            return
        if name not in {model.id for model in self._available()}:
            raise CommandArgumentError(f"Model '{name}' is not available.")
        self._model_selection.select(name)
        context.interaction.info(f"Using model: {name}")

    def _available(self) -> list[ModelInfo]:
        """Return available models with backend failures normalized for command dispatch."""
        try:
            return self._model_selection.available()
        except BackendError as error:
            raise CommandArgumentError(f"Could not list available models: {error}") from error
