"""Manage model selection for one active conversation session."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING

from .. import constants
from ..backend import BackendError
from ..interaction import Interaction
from ..models import ModelAssignment, ModelInfo

if TYPE_CHECKING:
    from ..backend import Backend
    from ..session import SessionManager


class ModelSelection:
    """Own model selection and durable last-used assignment reconciliation.

    Args:
        backend (Backend): Backend supplying the default model, catalog, and context metadata.
        session_manager (SessionManager): Active session that records successful assignments.
        selected (str | None): Explicit model selection, or ``None`` to use the backend default.
    """

    _backend: Backend
    _session_manager: SessionManager
    _selected: str | None
    _available_models: list[ModelInfo] | None
    _available_models_expires_at: float

    def __init__(
        self,
        backend: Backend,
        session_manager: SessionManager,
        selected: str | None = None,
    ) -> None:
        self._backend = backend
        self._session_manager = session_manager
        self._selected = selected if selected is not None else session_manager.model
        self._available_models = None
        self._available_models_expires_at = 0.0
        if self._selected is not None or backend.default_model is not None:
            self.synchronize_session()

    @property
    def selected(self) -> str | None:
        """Return the explicitly selected model.

        Returns:
            str | None: Explicit model identifier, or ``None`` when using the backend default.
        """
        return self._selected

    @property
    def effective(self) -> str:
        """Return the explicit or backend-default model.

        Returns:
            str: Model identifier to use for backend operations.

        Raises:
            ValueError: If neither this selection nor the backend provides a model.
        """
        model = self._selected or self._backend.default_model
        if not model:
            raise ValueError("No model was selected and the backend has no default model.")
        return model

    @property
    def assignment(self) -> ModelAssignment:
        """Return the backend-resolved assignment for the next operation.

        Returns:
            ModelAssignment: Exact model and known context capacity for the operation.

        Raises:
            ValueError: If neither this selection nor the backend provides a model.
        """
        return self._get_assignment(self.effective)

    def _get_assignment(self, model: str | None) -> ModelAssignment:
        """Return the backend-resolved assignment for a model."""
        return ModelAssignment(
            model=model,
            context_window=self._get_context_window(model),
        )

    def _get_context_window(self, model: str | None) -> int | None:
        """Return a positive context window for a model, otherwise ``None``."""
        context_window = self._backend.get_context_window(model)
        if (
            isinstance(context_window, int)
            and not isinstance(context_window, bool)
            and context_window > 0
        ):
            return context_window
        return None

    def available(self) -> list[ModelInfo]:
        """Return models currently available from the backend.

        Returns:
            list[ModelInfo]: Available backend models, reusing a recent non-empty catalog.

        Raises:
            BackendError: If the backend cannot list its models.
        """
        if self._available_models is not None and monotonic() < self._available_models_expires_at:
            return self._available_models
        models = self._backend.get_models()
        if models:
            self._available_models = models
            self._available_models_expires_at = monotonic() + constants.MODEL_CATALOG_CACHE_SECONDS
        return models

    def select(self, model: str) -> None:
        """Select a model and synchronize its active-session metadata.

        Args:
            model (str): Exact backend model identifier to select.
        """
        self._selected = model
        self.synchronize_session()

    def restore(self, model: str | None) -> None:
        """Restore a persisted selection and synchronize available model metadata.

        Args:
            model (str | None): Persisted model, or ``None`` to use the backend default.
        """
        self._selected = model
        if model is not None or self._backend.default_model is not None:
            self.synchronize_session()
        else:
            self._session_manager.assignment = None

    def record_response(self, model: str | None) -> ModelAssignment:
        """Record the model actually used by a completed response.

        Args:
            model (str | None): Provider-reported model identifier, when available.

        Returns:
            ModelAssignment: Durable assignment for the completed response.
        """
        if model is None:
            assignment = self.assignment
        else:
            self._selected = model
            assignment = self._get_assignment(model)
        return self.record_assignment(assignment)

    def record_assignment(self, assignment: ModelAssignment | None = None) -> ModelAssignment:
        """Persist one assignment that successfully completed an operation.

        Args:
            assignment (ModelAssignment | None): Completed assignment, or ``None`` to resolve the
                current selection.

        Returns:
            ModelAssignment: Persisted assignment.

        Raises:
            ValueError: If no effective model is configured.
        """
        assignment = assignment or self.assignment
        self._session_manager.assignment = assignment
        return assignment

    def synchronize_session(self) -> None:
        """Persist the effective model and its normalized context window in the active session.

        Raises:
            ValueError: If neither the selection nor backend defines a model.
        """
        self._session_manager.assignment = self.assignment

    def select_fallback(self, interaction: Interaction) -> bool:
        """Let the user replace a model rejected by the backend.

        Args:
            interaction (Interaction): Service used to show recovery choices and receive input.

        Returns:
            bool: ``True`` when a replacement model was selected, otherwise ``False``.
        """
        try:
            models = self.available()
        except BackendError as error:
            interaction.error(f"Could not list available models: {error}")
            return False
        if not models:
            interaction.warning("The backend reported no available models.")
            return False
        failing_model = self.selected
        while True:
            selection = interaction.prompt(
                "Select a replacement model:",
                choices={model.id: model.id for model in models},
            )
            if selection is False:
                return False
            if selection != failing_model:
                break
            interaction.warning(
                f"Model '{selection}' was already unavailable; the same "
                "failure is likely to re-occur unless the backend is updated."
            )
            if interaction.confirm(
                "Continue with this model, or select a different one?", default=True
            ):
                break
        self.select(selection)
        interaction.info(f"Using model: {selection}")
        return True
