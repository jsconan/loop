"""Coordinate context compaction for one active conversation session."""

from collections.abc import Callable
from pathlib import Path

from .. import constants
from ..backend import Backend
from ..instructions import InstructionsManager
from ..interaction import Interaction
from ..model_selection import ModelSelection
from ..session import SessionManager
from ..telemetry import telemetry_activity, telemetry_error


class ContextCompaction:
    """Compact active model context according to a configurable utilization policy.

    Args:
        backend (Backend): Backend used to produce replacement context.
        session_manager (SessionManager): Active session context and checkpoint persistence.
        model_selection (ModelSelection): Active model and context-window selection.
        instructions_manager (InstructionsManager): Current instructions and active skills.
        interaction (Interaction): Service used to report compaction progress and outcomes.
        working_directory (Callable[[], Path | str]): Provider for the current fallback working
            directory when the instructions manager has no observed directory.
        threshold (float): Context-window utilization that triggers automatic compaction.
            Defaults to ``0.8``.

    Raises:
        ValueError: If ``threshold`` is not strictly between zero and one.
    """

    _backend: Backend
    _session_manager: SessionManager
    _model_selection: ModelSelection
    _instructions_manager: InstructionsManager
    _interaction: Interaction
    _working_directory: Callable[[], Path | str]
    _threshold: float

    def __init__(
        self,
        backend: Backend,
        session_manager: SessionManager,
        model_selection: ModelSelection,
        instructions_manager: InstructionsManager,
        interaction: Interaction,
        working_directory: Callable[[], Path | str],
        *,
        threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Compaction threshold must be between zero and one.")
        self._backend = backend
        self._session_manager = session_manager
        self._model_selection = model_selection
        self._instructions_manager = instructions_manager
        self._interaction = interaction
        self._working_directory = working_directory
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Return the automatic compaction utilization threshold.

        Returns:
            float: Utilization ratio that triggers compaction.
        """
        return self._threshold

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

    def compact_if_needed(self) -> bool:
        """Compact the active context when its utilization policy is satisfied.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            ValueError: If compaction is required but no effective model is configured.
        """
        return self.compact() if self.needed() else False

    def compact(self, *, force: bool = True) -> bool:
        """Compact the active context and persist its replacement checkpoint.

        Args:
            force (bool): Whether to compact regardless of the utilization threshold. Defaults to
                ``True`` for explicit calls.

        Returns:
            bool: ``True`` when a new checkpoint was persisted, otherwise ``False``.

        Raises:
            ValueError: If no effective model is configured.
        """
        if not self.can_compact():
            if force:
                self._interaction.warning("There is no new session context to compact.")
            return False
        if not force and not self.needed():
            return False

        self._instructions_manager.prepare()
        model = self._model_selection.effective
        self._model_selection.synchronize_session()
        self._interaction.info("Compacting session context...")
        telemetry_activity(
            "compaction.started",
            severity="info",
            component="compaction",
            model=model,
        )
        try:
            result = self._backend.compact(
                self._session_manager.model_context,
                instructions=self._instructions_manager.instructions,
                model=model,
            )
        except NotImplementedError:
            telemetry_activity(
                "compaction.unsupported",
                severity="warning",
                component="compaction",
            )
            self._interaction.warning("The selected backend does not support context compaction.")
            return False
        if result is None or not result.items:
            telemetry_error(
                "compaction.failed",
                error_type="compaction.empty_result",
                component="compaction",
            )
            self._interaction.warning("The selected backend did not produce compacted context.")
            return False

        working_directory = str(
            self._instructions_manager.working_directory or self._working_directory()
        )
        previous_tokens = self._session_manager.tokens
        self._session_manager.add_compaction(
            result,
            model=model,
            instructions=self._instructions_manager.instructions,
            working_directory=working_directory,
            active_skills=self._instructions_manager.active_skill_identities,
        )
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
        return True
