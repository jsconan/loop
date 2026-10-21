"""Manage model selection for one active conversation session."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import TYPE_CHECKING, Any

from .. import constants
from ..backend import BackendError
from ..errors import Problem, log_problem
from ..interaction import Interaction, require_async_interaction
from ..models import ModelAssignment, ModelInfo
from ..utils import run_in_worker

if TYPE_CHECKING:
    from ..backend import Backend
    from ..session import SessionManager

_LOGGER = logging.getLogger(__name__)


class ModelSelectionError(ValueError):
    """Identify a model-selection failure at the request-preparation boundary."""


class ModelSelection:
    """Own model selection and durable last-used assignment reconciliation.

    Args:
        backend (Backend): Backend supplying the default model, catalog, and context metadata.
        session_manager (SessionManager): Active session that records successful assignments.
        selected (str | None): Explicit model selection, or ``None`` to use the backend default.
        on_select (Callable[[str], None] | None): Durable preference writer invoked before an
            explicit selection takes effect. Defaults to no durable writer.
        async_blocking_runner (Callable[..., Awaitable[Any]] | None): Optional application-owned
            worker for the synchronous preference writer. Defaults to a shared bounded process
            worker.
    """

    _backend: Backend
    _session_manager: SessionManager
    _selected: str | None
    _on_select: Callable[[str], None] | None
    _async_blocking_runner: Callable[..., Awaitable[Any]] | None
    _available_models: list[ModelInfo] | None
    _available_models_expires_at: float

    def __init__(
        self,
        backend: Backend,
        session_manager: SessionManager,
        selected: str | None = None,
        on_select: Callable[[str], None] | None = None,
        async_blocking_runner: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._backend = backend
        self._session_manager = session_manager
        self._selected = selected if selected is not None else session_manager.model
        self._on_select = on_select
        self._async_blocking_runner = async_blocking_runner
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
    def backend(self) -> Backend:
        """Return the backend supplying model information.

        Returns:
            Backend: Active model backend.
        """
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        """Replace the model backend and discard its cached catalog.

        Args:
            backend (Backend): Fully configured replacement backend.
        """
        self._backend = backend
        self._available_models = None
        self._available_models_expires_at = 0.0

    @property
    def effective(self) -> str:
        """Return the explicit or backend-default model.

        Returns:
            str: Model identifier to use for backend operations.

        Raises:
            ModelSelectionError: If neither this selection nor the backend provides a model.
        """
        model = self._selected or self._backend.default_model
        if not model:
            raise ModelSelectionError("No model was selected and the backend has no default model.")
        return model

    @property
    def assignment(self) -> ModelAssignment:
        """Return the backend-resolved assignment for the next operation.

        Returns:
            ModelAssignment: Exact model and known context capacity for the operation.

        Raises:
            ModelSelectionError: If neither this selection nor the backend provides a model.
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
        return self._normalize_context_window(self._backend.get_context_window(model))

    async def _get_assignment_async(self, model: str | None) -> ModelAssignment:
        """Resolve model metadata through the backend's async API."""
        return ModelAssignment(
            model=model,
            context_window=await self._get_context_window_async(model),
        )

    async def _get_context_window_async(self, model: str | None) -> int | None:
        """Return async context metadata normalized to a positive integer."""
        context_window = await self._backend.get_context_window_async(model)
        return self._normalize_context_window(context_window)

    @staticmethod
    def _normalize_context_window(context_window: int | None) -> int | None:
        """Return a positive context window, otherwise ``None``."""
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
            asyncio.CancelledError: If the asynchronous catalog request is cancelled.
        """
        if self._available_models is not None and monotonic() < self._available_models_expires_at:
            return self._available_models
        models = self._backend.get_models()
        if models:
            self._available_models = models
            self._available_models_expires_at = monotonic() + constants.MODEL_CATALOG_CACHE_SECONDS
        return models

    async def available_async(self) -> list[ModelInfo]:
        """Return models using the backend's asynchronous catalog operation.

        Returns:
            list[ModelInfo]: Available models, reusing a recent non-empty catalog.

        Raises:
            BackendError: If the backend cannot list its models.
        """
        if self._available_models is not None and monotonic() < self._available_models_expires_at:
            return self._available_models
        models = await self._backend.get_models_async()
        if models:
            self._available_models = models
            self._available_models_expires_at = monotonic() + constants.MODEL_CATALOG_CACHE_SECONDS
        return models

    def select(self, model: str) -> None:
        """Select a model and synchronize its active-session metadata.

        Args:
            model (str): Exact backend model identifier to select.
        """
        if self._on_select is not None:
            self._on_select(model)
        self._selected = model
        self.synchronize_session()

    async def select_async(self, model: str) -> None:
        """Select a model and asynchronously refresh its active-session context metadata.

        Args:
            model (str): Exact backend model identifier to select.

        Raises:
            BackendError: If context metadata cannot be retrieved from the backend.
            OSError: If the configured durable preference writer fails.
            asyncio.CancelledError: If the asynchronous context lookup is cancelled.
        """
        cancellation = await self._persist_selection_async(model)
        self._selected = model
        if cancellation is not None:
            raise cancellation
        self._session_manager.assignment = await self._get_assignment_async(model)

    async def _persist_selection_async(self, model: str) -> asyncio.CancelledError | None:
        """Run the synchronous durable preference writer off-loop and settle cancellation."""
        if self._on_select is None:
            return None
        pending = asyncio.ensure_future(
            self._async_blocking_runner(self._on_select, model)
            if self._async_blocking_runner is not None
            else run_in_worker(self._on_select, model)
        )
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError as cancellation:
            failure = None
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except BaseException as error:  # noqa: BLE001  # settle the owned worker
                    failure = error
                    break
            if pending.cancelled():
                failure = asyncio.CancelledError()
            elif failure is None:
                failure = pending.exception()
            if failure is not None:
                cancellation.add_note(
                    f"Model preference writer also stopped with {type(failure).__name__}."
                )
                raise
            return cancellation
        return None

    def restore(self, model: str | None) -> None:
        """Restore a session model and synchronize its available metadata.

        Args:
            model (str | None): Session model to restore, or ``None`` to use the backend default.
        """
        self._selected = model
        if model is not None or self._backend.default_model is not None:
            self.synchronize_session()
        else:
            self._session_manager.assignment = None

    async def restore_async(self, model: str | None) -> None:
        """Restore a session model and resolve its metadata through async backend calls.

        Args:
            model (str | None): Session model to restore, or ``None`` to use the backend default.

        Raises:
            BackendError: If the backend cannot retrieve model context metadata.
            asyncio.CancelledError: If the backend metadata request is cancelled.
        """
        self._selected = model
        if model is not None or self._backend.default_model is not None:
            self._session_manager.assignment = await self._get_assignment_async(self.effective)
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

    async def record_response_async(self, model: str | None) -> ModelAssignment:
        """Record a completed response using async context metadata lookup.

        Args:
            model (str | None): Provider-reported model identifier, when available.

        Returns:
            ModelAssignment: Durable assignment for the completed response.

        Raises:
            ModelSelectionError: If no model is available when ``model`` is ``None``.
            BackendError: If the backend cannot retrieve model context metadata.
            asyncio.CancelledError: If the backend metadata request is cancelled.
        """
        if model is None:
            assignment = await self._get_assignment_async(self.effective)
        else:
            self._selected = model
            assignment = await self._get_assignment_async(model)
        return self.record_assignment(assignment)

    def record_assignment(self, assignment: ModelAssignment | None = None) -> ModelAssignment:
        """Persist one assignment that successfully completed an operation.

        Args:
            assignment (ModelAssignment | None): Completed assignment, or ``None`` to resolve the
                current selection.

        Returns:
            ModelAssignment: Persisted assignment.

        Raises:
            ModelSelectionError: If no effective model is configured.
        """
        assignment = assignment or self.assignment
        self._session_manager.assignment = assignment
        return assignment

    def synchronize_session(self) -> None:
        """Persist the effective model and its normalized context window in the active session.

        Raises:
            ModelSelectionError: If neither the selection nor backend defines a model.
        """
        self._session_manager.assignment = self.assignment

    async def synchronize_session_async(self) -> None:
        """Synchronize the active session using native asynchronous model metadata.

        Returns:
            None: The active session assignment is synchronized.

        Raises:
            ModelSelectionError: If neither the selection nor backend defines a model.
            BackendError: If context metadata cannot be retrieved from the backend.
            asyncio.CancelledError: If the backend metadata request is cancelled.
        """
        self._session_manager.assignment = await self._get_assignment_async(self.effective)

    def select_fallback(self, interaction: Interaction) -> bool:
        """Let the user replace a model rejected by the backend.

        Args:
            interaction (Interaction): Service used to show recovery choices and receive input.

        Returns:
            bool: ``True`` when a replacement model was selected, otherwise ``False``.

        Raises:
            BackendError: If selecting the replacement cannot retrieve its context metadata.
            OSError: If the configured durable preference writer fails.
            asyncio.CancelledError: If model lookup or user input is cancelled.
        """
        try:
            models = self.available()
        except BackendError as error:
            problem = Problem.from_exception(
                error,
                code="backend.model_listing_failed",
                title="Could not list available models",
                detail="The backend is not reachable.",
                retryable=error.recoverable,
                operation="list_models",
            )
            log_problem(_LOGGER, problem, error)
            interaction.report(problem)
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

    async def select_fallback_async(self, interaction: Interaction) -> bool:
        """Asynchronously replace a model rejected by the backend.

        Args:
            interaction (Interaction): Service used to show recovery choices and receive input.

        Returns:
            bool: ``True`` when a replacement model was selected, otherwise ``False``.

        Raises:
            ProblemException: If the interaction does not provide asynchronous input.
            BackendError: If the backend cannot retrieve the replacement model's context metadata.
            OSError: If persisting the replacement model preference fails.
            asyncio.CancelledError: If model lookup or user input is cancelled.
        """
        async_interaction = require_async_interaction(interaction, "model selection")
        try:
            models = await self.available_async()
        except BackendError as error:
            problem = Problem.from_exception(
                error,
                code="backend.model_listing_failed",
                title="Could not list available models",
                detail="The backend is not reachable.",
                retryable=error.recoverable,
                operation="list_models",
            )
            log_problem(_LOGGER, problem, error)
            interaction.report(problem)
            return False
        if not models:
            interaction.warning("The backend reported no available models.")
            return False
        failing_model = self.selected
        while True:
            prompt_kwargs = {"choices": {model.id: model.id for model in models}}
            selection = await async_interaction.prompt_async(
                "Select a replacement model:",
                **prompt_kwargs,
            )
            if selection is False:
                return False
            if selection != failing_model:
                break
            interaction.warning(
                f"Model '{selection}' was already unavailable; the same "
                "failure is likely to re-occur unless the backend is updated."
            )
            prompt = "Continue with this model, or select a different one?"
            confirmed = await async_interaction.confirm_async(prompt, default=True)
            if confirmed:
                break
        await self.select_async(selection)
        interaction.info(f"Using model: {selection}")
        return True
