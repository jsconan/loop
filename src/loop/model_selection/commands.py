"""Expose active model selection through user commands."""

from types import SimpleNamespace
from typing import Annotated

from pydantic import Field

from ..backend import BackendError
from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionProviderRegistration, CompletionValue
from ..errors import Problem
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
            CommandRegistration(self.backend, name="backend"),
            CommandRegistration(self.models, name="models"),
            CommandRegistration(
                self.model,
                name="model",
                completion=CommandCompletion(provider="models"),
            ),
        )

    def get_completion_providers(self) -> tuple[CompletionProviderRegistration, ...]:
        """Return dynamic model completion sources.

        Returns:
            tuple[CompletionProviderRegistration, ...]: Named model completion source.
        """
        return (CompletionProviderRegistration("models", self._model_values),)

    def _model_values(self) -> tuple[CompletionValue, ...]:
        """Return models currently available from the backend."""
        return tuple(
            CompletionValue(model.id, f"{model.context_window:,} tokens")
            for model in self._available()
        )

    def backend(self, context: CommandContext) -> None:
        """Check backend reachability and the effective model's availability."""
        try:
            models = self._available()
        except BackendError:
            context.interaction.warning("The backend is not reachable.")
            return
        try:
            model = self._model_selection.effective
        except ValueError:
            context.interaction.warning("Backend is available, but no model is selected.")
            self._model_selection.select_fallback(context.interaction)
            return
        if model in {available_model.id for available_model in models}:
            context.interaction.info(f"Backend is available. Model '{model}' is available.")
            return
        context.interaction.warning(f"Backend is available, but model '{model}' is not available.")
        self._model_selection.select_fallback(context.interaction)

    def models(self, context: CommandContext) -> None:
        """List all models available from the backend."""
        try:
            values = self._model_values()
        except BackendError as error:
            self._report_unavailable(context, error)
            return
        if not values:
            context.interaction.info("No models available.")
            return
        context.interaction.table(
            [SimpleNamespace(id=value.value, context_window=value.description) for value in values],
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
        try:
            available = self._available()
        except BackendError as error:
            self._report_unavailable(context, error)
            return
        if name not in {model.id for model in available}:
            raise CommandArgumentError(f"Model '{name}' is not available.")
        self._model_selection.select(name)
        context.interaction.info(f"Using model: {name}")

    def _available(self) -> list[ModelInfo]:
        """Return models currently available from the backend."""
        return self._model_selection.available()

    @staticmethod
    def _report_unavailable(context: CommandContext, error: BackendError) -> None:
        """Report a model-catalog failure without exposing provider error text."""
        context.interaction.report(
            Problem.from_exception(
                error,
                code="backend.unavailable",
                title="Backend unavailable",
                detail="The backend is not reachable.",
                retryable=error.recoverable,
                operation="list_models",
            )
        )
