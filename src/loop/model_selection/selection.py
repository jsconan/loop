"""Manage model selection for one active conversation session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import ModelInfo

if TYPE_CHECKING:
    from ..backend import Backend
    from ..session import SessionManager


class ModelSelection:
    """Keep backend model selection and active-session metadata consistent.

    Args:
        backend (Backend): Backend supplying the default model, catalog, and context metadata.
        session_manager (SessionManager): Active session whose effective model metadata is updated.
        selected (str | None): Explicit model selection, or ``None`` to use the backend default.
    """

    _backend: Backend
    _session_manager: SessionManager
    _selected: str | None

    def __init__(
        self,
        backend: Backend,
        session_manager: SessionManager,
        selected: str | None = None,
    ) -> None:
        self._backend = backend
        self._session_manager = session_manager
        self._selected = selected
        if self._selected is not None or backend.default_model is not None:
            self.synchronize_session()
        else:
            self._clear_session()

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

    def available(self) -> list[ModelInfo]:
        """Return models currently available from the backend.

        Returns:
            list[ModelInfo]: Available backend models.

        Raises:
            BackendError: If the backend cannot list its models.
        """
        return self._backend.get_models()

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
            self._clear_session()

    def synchronize_session(self) -> None:
        """Persist the effective model and its normalized context window in the active session.

        Raises:
            ValueError: If no effective model is configured.
        """
        model = self.effective
        context_window = self._backend.get_context_window(model)
        self._session_manager.model = model
        self._session_manager.context_window = (
            context_window
            if isinstance(context_window, int) and not isinstance(context_window, bool)
            else None
        )

    def _clear_session(self) -> None:
        """Clear effective model metadata when no model can be resolved."""
        self._session_manager.model = None
        self._session_manager.context_window = None
