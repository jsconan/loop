"""Coordinate context compaction for one active conversation session."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from .. import constants
from ..backend import Backend
from ..instructions import PreparedInstructions
from ..interaction import Interaction
from ..model_selection import ModelSelection
from ..models import CompactionResult
from ..session import SessionManager
from ..telemetry import telemetry_activity, telemetry_error
from ..utils import PathHolder, PathReference, run_in_worker_async


class ContextCompaction:
    """Compact active model context according to a configurable utilization policy.

    Args:
        backend (Backend): Backend used to produce replacement context.
        session_manager (SessionManager): Active session context and checkpoint persistence.
        model_selection (ModelSelection): Active model and context-window selection.
        instructions (Callable[[], PreparedInstructions]): Provider that prepares exact
            model-ready instructions for an explicit agent.
        instructions_async (Callable[[], Awaitable[PreparedInstructions]] | None): Native async
            provider used by ``compact_async``. If omitted, the synchronous provider runs through
            bounded worker admission.
        interaction (Interaction): Service used to report compaction progress and outcomes.
        working_directory (str | Path | PathReference): Current working-directory reference.
        threshold (float): Context-window utilization that triggers automatic compaction.
            Defaults to ``0.8``.

    Raises:
        ValueError: If ``threshold`` is not strictly between zero and one.
    """

    _backend: Backend
    _session_manager: SessionManager
    _model_selection: ModelSelection
    _instructions: Callable[[], PreparedInstructions]
    _instructions_async: Callable[[], Awaitable[PreparedInstructions]] | None
    _interaction: Interaction
    _working_directory: PathReference
    _threshold: float

    def __init__(
        self,
        backend: Backend,
        session_manager: SessionManager,
        model_selection: ModelSelection,
        instructions: Callable[[], PreparedInstructions],
        interaction: Interaction,
        working_directory: str | Path | PathReference,
        *,
        threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
        instructions_async: Callable[[], Awaitable[PreparedInstructions]] | None = None,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Compaction threshold must be between zero and one.")
        self._backend = backend
        self._session_manager = session_manager
        self._model_selection = model_selection
        self._instructions = instructions
        self._instructions_async = instructions_async
        self._interaction = interaction
        self._working_directory = PathHolder.from_value(working_directory)
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Return the automatic compaction utilization threshold.

        Returns:
            float: Utilization ratio that triggers compaction.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        """Set the automatic compaction utilization threshold.

        Args:
            threshold (float): Utilization ratio strictly between zero and one.

        Raises:
            ValueError: If the threshold is outside the supported range.
        """
        if not 0 < threshold < 1:
            raise ValueError("Compaction threshold must be between zero and one.")
        self._threshold = threshold

    @property
    def backend(self) -> Backend:
        """Return the backend used for context compaction.

        Returns:
            Backend: Active compaction backend.
        """
        return self._backend

    @backend.setter
    def backend(self, backend: Backend) -> None:
        """Replace the backend used for future compaction.

        Args:
            backend (Backend): Fully configured replacement backend.
        """
        self._backend = backend

    def can_compact(self) -> bool:
        """Return whether complete history advanced beyond the latest checkpoint.

        Returns:
            bool: ``True`` when at least one uncompacted history item exists.
        """
        session = self._session_manager.session
        boundary = session.compactions[-1].boundary if session.compactions else 0
        return boundary < len(session.messages)

    def needed(self) -> bool:
        """Return whether current usage reached the automatic compaction threshold.

        Returns:
            bool: ``True`` when capacity is known, usage reached the threshold, and new history
                can be compacted.
        """
        context_window = self._session_manager.context_window
        return (
            context_window is not None
            and self._session_manager.tokens >= context_window * self._threshold
            and self.can_compact()
        )

    def compact_if_needed(self, instructions: PreparedInstructions | None = None) -> bool:
        """Compact the active context when its utilization policy is satisfied.

        Args:
            instructions (PreparedInstructions | None): Exact instructions prepared for the
                surrounding model turn, or ``None`` to request a fresh explicit snapshot.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            ValueError: If compaction is required but no effective model is configured.
        """
        return self.compact(instructions=instructions) if self.needed() else False

    async def compact_if_needed_async(
        self,
        instructions: PreparedInstructions | None = None,
    ) -> bool:
        """Compact asynchronously when the active context reaches its utilization threshold.

        Args:
            instructions (PreparedInstructions | None): Exact instructions prepared for the
                surrounding model turn, or ``None`` to request a fresh explicit snapshot.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            BackendError: If the provider request or response fails.
            ValueError: If compaction is required but no effective model is configured.
            RuntimeError: Bounded worker admission is full or unavailable when no async provider
                was configured.
            asyncio.CancelledError: The task is cancelled after instruction preparation settles.
        """
        if not self.needed():
            return False
        return await self.compact_async(instructions=instructions)

    def compact(
        self,
        *,
        force: bool = True,
        instructions: PreparedInstructions | None = None,
    ) -> bool:
        """Compact the active context and persist its replacement checkpoint.

        Args:
            force (bool): Whether to compact regardless of the utilization threshold. Defaults to
                ``True`` for explicit calls.
            instructions (PreparedInstructions | None): Exact prepared instructions to preserve,
                or ``None`` to request a fresh explicit snapshot.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            ValueError: If no effective model is configured.
        """
        if not self._can_start(force):
            return False

        snapshot = instructions or self._instructions()
        model = self._model_selection.effective
        self._start(model)
        try:
            result = self._backend.compact(
                self._session_manager.model_context,
                instructions=snapshot.content,
                model=model,
            )
        except NotImplementedError:
            return self._unsupported()
        return self._finish(result, snapshot, model)

    async def compact_async(
        self,
        *,
        force: bool = True,
        instructions: PreparedInstructions | None = None,
    ) -> bool:
        """Asynchronously compact the active context and persist its replacement checkpoint.

        Args:
            force (bool): Whether to compact regardless of the utilization threshold. Defaults to
                ``True`` for explicit calls.
            instructions (PreparedInstructions | None): Exact prepared instructions to preserve,
                or ``None`` to request a fresh explicit snapshot.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            BackendError: If the provider request or response fails.
            ValueError: If no effective model is configured.
            RuntimeError: Bounded worker admission is full or unavailable when no async provider
                was configured.
            asyncio.CancelledError: If backend compaction or session persistence is cancelled.
        """
        if not self._can_start(force):
            return False

        if instructions is not None:
            snapshot = instructions
        elif self._instructions_async is not None:
            snapshot = await self._instructions_async()
        else:
            snapshot = await run_in_worker_async(self._instructions)
        model = self._model_selection.effective
        await self._start_async(model)
        try:
            result = await self._backend.compact_async(
                self._session_manager.model_context,
                instructions=snapshot.content,
                model=model,
            )
        except NotImplementedError:
            return self._unsupported()
        return await self._finish_async(result, snapshot, model)

    def _can_start(self, force: bool) -> bool:
        """Return whether compaction should proceed, reporting an explicit empty request."""
        if not self.can_compact():
            if force:
                self._interaction.warning("There is no new session context to compact.")
            return False
        return force or self.needed()

    def _start(self, model: str) -> None:
        """Synchronize the selected model and report the start of compaction."""
        self._model_selection.synchronize_session()
        self._report_start(model)

    async def _start_async(self, model: str) -> None:
        """Synchronize model metadata asynchronously and report compaction start."""
        await self._model_selection.synchronize_session_async()
        self._report_start(model)

    def _report_start(self, model: str) -> None:
        """Report one compaction start after model metadata is synchronized."""
        self._interaction.info("Compacting session context...")
        telemetry_activity(
            "compaction.started",
            severity="info",
            component="compaction",
            model=model,
        )

    def _unsupported(self) -> bool:
        """Report that the backend does not implement compaction."""
        telemetry_activity(
            "compaction.unsupported",
            severity="warning",
            component="compaction",
        )
        self._interaction.warning("The selected backend does not support context compaction.")
        return False

    def _finish(
        self,
        result: CompactionResult | None,
        snapshot: PreparedInstructions,
        model: str,
    ) -> bool:
        """Persist a complete compacted context and report its observed token change."""
        if result is None or not result.items:
            telemetry_error(
                "compaction.failed",
                error_type="compaction.empty_result",
                component="compaction",
            )
            self._interaction.warning("The selected backend did not produce compacted context.")
            return False

        working_directory = str(snapshot.working_directory or self._working_directory)
        previous_tokens = self._session_manager.tokens
        self._session_manager.add_compaction(
            result,
            model=model,
            instructions=snapshot.content,
            working_directory=working_directory,
            active_skills=list(snapshot.active_skills),
            references=tuple(
                section.reference for section in snapshot.sections if section.reference is not None
            ),
        )
        self._report_completion(previous_tokens)
        return True

    async def _finish_async(
        self,
        result: CompactionResult | None,
        snapshot: PreparedInstructions,
        model: str,
    ) -> bool:
        """Persist compacted context asynchronously and report its observed token change."""
        if result is None or not result.items:
            return self._empty_result()

        previous_tokens = self._session_manager.tokens
        await self._session_manager.add_compaction_async(
            result,
            model=model,
            instructions=snapshot.content,
            working_directory=str(snapshot.working_directory or self._working_directory),
            active_skills=list(snapshot.active_skills),
            references=tuple(
                section.reference for section in snapshot.sections if section.reference is not None
            ),
        )
        self._report_completion(previous_tokens)
        return True

    def _report_completion(self, previous_tokens: int) -> None:
        """Report one persisted compaction and its token change."""
        current_tokens = self._session_manager.tokens
        telemetry_activity(
            "compaction.completed",
            severity="info",
            component="compaction",
            input_tokens_before=previous_tokens,
            input_tokens_after=current_tokens,
        )
        if current_tokens != previous_tokens:
            self._interaction.info(
                f"Compacted session context from {previous_tokens:,} to {current_tokens:,} tokens."
            )
        else:
            self._interaction.info("Compacted session context.")

    def _empty_result(self) -> bool:
        """Report a compaction result that has no replacement context."""
        telemetry_error(
            "compaction.failed",
            error_type="compaction.empty_result",
            component="compaction",
        )
        self._interaction.warning("The selected backend did not produce compacted context.")
        return False
