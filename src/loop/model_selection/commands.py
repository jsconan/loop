"""Expose active model selection through user commands."""

import logging
from types import SimpleNamespace
from typing import Annotated

from pydantic import Field

from ..backend import BackendError
from ..commands import CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionProviderRegistration, CompletionValue
from ..errors import Problem, log_problem
from ..models import ModelInfo
from .selection import ModelSelection, ModelSelectionError

_LOGGER = logging.getLogger(__name__)


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
            CommandRegistration(self.check, name="check"),
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
            CompletionValue(
                model.id,
                f"{model.context_window:,} tokens"
                if model.context_window is not None
                else "Not reported",
            )
            for model in self._available()
        )

    async def check(self, context: CommandContext) -> None:
        """Check backend reachability and the effective model's availability.

        Args:
            context (CommandContext): Interaction used to report status and select a fallback.
        """
        try:
            models = await self._model_selection.available_async()
        except BackendError as error:
            self._log_unavailable(error)
            context.interaction.warning("The backend is not reachable.")
            return
        try:
            model = self._model_selection.effective
        except ModelSelectionError:
            context.interaction.warning("Backend is available, but no model is selected.")
            await self._select_fallback(context)
            return
        if model in {available_model.id for available_model in models}:
            context.interaction.info(f"Backend is available. Model '{model}' is available.")
            return
        context.interaction.warning(f"Backend is available, but model '{model}' is not available.")
        await self._select_fallback(context)

    async def models(self, context: CommandContext) -> None:
        """List all models available from the backend.

        Args:
            context (CommandContext): Interaction used to display the model catalog.
        """
        try:
            models = await self._model_selection.available_async()
        except BackendError as error:
            self._report_unavailable(context, error)
            return
        if not models:
            context.interaction.info("No models available.")
            return
        context.interaction.table(
            [
                SimpleNamespace(
                    id=model.id,
                    context_window=(
                        f"{model.context_window:,} tokens"
                        if model.context_window is not None
                        else "Not reported"
                    ),
                )
                for model in models
            ],
            title="Available models:",
            columns=("id", "context_window"),
        )

    async def model(
        self,
        context: CommandContext,
        name: Annotated[
            str | None,
            Field(description="Exact backend model ID, or omit it to show the selection."),
        ] = None,
    ) -> None:
        """Show or select the model used for subsequent requests.

        Args:
            context (CommandContext): Interaction used to display model selection status.
            name (str | None): Exact backend model ID, or ``None`` to show the selection.
        """
        if name is None:
            selected = self._model_selection.selected
            context.interaction.info(
                f"Selected model: {selected}" if selected else "Using the backend default model."
            )
            return
        try:
            available = await self._model_selection.available_async()
        except BackendError as error:
            self._report_unavailable(context, error)
            return
        if name not in {model.id for model in available}:
            context.interaction.warning(f"Model '{name}' is not available.")
            return
        try:
            await self._model_selection.select_async(name)
        except (BackendError, OSError) as error:
            self._report_selection_failure(context, error)
            return
        context.interaction.info(f"Using model: {name}")

    def _available(self) -> list[ModelInfo]:
        """Return models currently available from the backend."""
        return self._model_selection.available()

    async def _select_fallback(self, context: CommandContext) -> None:
        """Select a replacement model and report expected persistence failures."""
        try:
            await self._model_selection.select_fallback_async(context.interaction)
        except (BackendError, OSError) as error:
            self._report_selection_failure(context, error)

    @classmethod
    def _report_selection_failure(
        cls,
        context: CommandContext,
        error: BackendError | OSError,
    ) -> None:
        """Present model lookup or preference persistence failures without stopping the app."""
        if isinstance(error, BackendError):
            cls._report_unavailable(context, error)
            return
        problem = Problem.from_exception(
            error,
            code="model_selection.preference_save_failed",
            title="Model preference could not be saved",
            detail=(
                "The model preference could not be saved. Check configuration access and try again."
            ),
            operation="select_model",
        )
        log_problem(_LOGGER, problem, error)
        context.interaction.report(problem)

    @staticmethod
    def _report_unavailable(context: CommandContext, error: BackendError) -> None:
        """Report a model-catalog failure without exposing provider error text."""
        problem = Problem.from_exception(
            error,
            code="backend.unavailable",
            title="Backend unavailable",
            detail="The backend is not reachable.",
            severity="warning" if error.recoverable else "error",
            retryable=error.recoverable,
            operation="list_models",
        )
        log_problem(_LOGGER, problem, error)
        if error.recoverable:
            context.interaction.warning(problem.detail)
        else:
            context.interaction.report(problem)

    @staticmethod
    def _log_unavailable(error: BackendError) -> None:
        """Record a sanitized model-catalog failure."""
        problem = Problem.from_exception(
            error,
            code="backend.unavailable",
            title="Backend unavailable",
            detail="The backend is not reachable.",
            severity="warning" if error.recoverable else "error",
            retryable=error.recoverable,
            operation="list_models",
        )
        log_problem(_LOGGER, problem, error)
