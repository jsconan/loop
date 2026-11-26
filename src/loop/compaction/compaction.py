"""Coordinate context compaction for one active conversation session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .. import constants
from ..backend import Backend, estimate_context
from ..interaction import Interaction
from ..models import (
    CompactionContextItem,
    CompactionItem,
    CompactionResult,
    ConversationItem,
    ModelContextItem,
)
from ..session import (
    Compaction,
    InstructionSnapshot,
    ModelWindow,
    ModelWindowConflictError,
    SessionManager,
)
from ..telemetry import telemetry_activity, telemetry_error
from ..utils import PathHolder, PathReference, new_uuid, utc_now
from .planning import CompactionLimitError, RetainedContext, portable_batches

if TYPE_CHECKING:
    from ..agent import AgentContext

_UNDECLARED_CAPACITY = 32_768
_SUMMARY_RESERVE = 1024


@dataclass(frozen=True)
class _Attempt:
    """Capture a rebuild candidate and the session boundaries it may replace."""

    session_id: str
    previous_window: str | None
    previous_checkpoint: str | None
    backend: Backend
    window: ModelWindow
    retained: RetainedContext
    opening: list[ModelContextItem]
    tail: list[ModelContextItem]
    native_tail: list[ModelContextItem] | None
    middle: list[ConversationItem]
    previous_items: tuple[CompactionItem, ...]
    previous_context_tokens: int | None
    uncovered: list[ConversationItem]
    available: int
    summary_budget: int
    native: bool

    @property
    def has_prefix(self) -> bool:
        """Check whether rebuilt input puts canonical items before the checkpoint."""
        return bool(self.opening) or self.window.layout != "leading"


class ContextCompaction:
    """Compact active model context according to a configurable utilization policy.

    Args:
        session_manager (SessionManager): Active session context and checkpoint persistence.
        context (AgentContext): Owner of the candidate model window and its backend.
        interaction (Interaction): Service used to report compaction progress and outcomes.
        working_directory (str | Path | PathReference): Current working-directory reference.
        threshold (float): Context-window utilization that triggers automatic compaction.
            Defaults to ``0.8``.

    Raises:
        ValueError: If ``threshold`` is outside the supported range of (0, 1].
    """

    _session_manager: SessionManager
    _context: AgentContext
    _interaction: Interaction
    _working_directory: PathReference
    _threshold: float

    def __init__(
        self,
        session_manager: SessionManager,
        context: AgentContext,
        interaction: Interaction,
        working_directory: str | Path | PathReference,
        *,
        threshold: float = constants.DEFAULT_COMPACTION_THRESHOLD,
    ) -> None:
        self._session_manager = session_manager
        self._context = context
        self._interaction = interaction
        self._working_directory = PathHolder.from_value(working_directory)
        self.threshold = threshold

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
            threshold (float): Utilization ratio greater than zero and at most one.

        Raises:
            ValueError: If the threshold is outside the supported range.
        """
        if not 0 < threshold <= 1:
            raise ValueError("Compaction threshold must be greater than zero and at most one.")
        self._threshold = threshold

    def can_compact(self, window: ModelWindow | None = None) -> bool:
        """Check whether canonical history has new context beyond the last checkpoint.

        Args:
            window (ModelWindow | None): Staged window, or the active window by default.

        Returns:
            bool: Whether an automatic rebuild could advance the captured source history.
        """
        session = self._session_manager.session
        checkpoint = session.checkpoint(window)
        return len(session.messages) > (checkpoint.boundary if checkpoint is not None else 0)

    def needed(self, occupied: int | None = None, *, window: ModelWindow | None = None) -> bool:
        """Check the compaction threshold against the estimated occupied context.

        Args:
            occupied (int | None): Estimated occupied tokens, or the last observed usage.
            window (ModelWindow | None): Staged window, or the active window by default.

        Returns:
            bool: Whether automatic compaction could make room for the next request.
        """
        limit = self._context.compaction_capacity(window)
        if limit is None or not self.can_compact(window):
            return False
        tokens = self._session_manager.tokens if occupied is None else occupied
        return tokens >= limit * self._threshold

    def compact_if_needed(
        self,
        occupied: int | None = None,
        *,
        window: ModelWindow | None = None,
    ) -> bool:
        """Rebuild the session if observed or projected context crosses the threshold.

        Args:
            occupied (int | None): Estimated occupied tokens, or the last observed usage.
            window (ModelWindow | None): Window whose projected input supplied the estimate.

        Returns:
            bool: Whether a checkpoint was committed.
        """
        return self.needed(occupied, window=window) and self.compact()

    async def compact_if_needed_async(
        self,
        occupied: int | None = None,
        *,
        window: ModelWindow | None = None,
    ) -> bool:
        """Rebuild the session asynchronously after crossing the utilization threshold.

        Args:
            occupied (int | None): Estimated occupied tokens, or the last observed usage.
            window (ModelWindow | None): Window whose projected input supplied the estimate.

        Returns:
            bool: Whether a checkpoint was committed.
        """
        return self.needed(occupied, window=window) and await self.compact_async()

    def compact(self) -> bool:
        """Capture, build, and atomically commit a fresh model context.

        Returns:
            bool: Whether an eligible rebuilt context was committed.
        """
        window = self._context.capture()
        try:
            attempt = self._start(window)
            if attempt is None:
                return False
            with attempt.backend.file_projection(window.settings.file_input_mode):
                result = self._summarize(attempt)
                if result is None or (attempt.middle and not result.items):
                    return self._empty_result()
                tokens = self._estimate(attempt, result)
                if not self._fits(attempt, tokens) and self._native_result(result):
                    result = self._portable(attempt)
                    tokens = self._estimate(attempt, result) if result is not None else None
        except NotImplementedError:
            return self._unsupported()
        except CompactionLimitError as error:
            self._interaction.warning(str(error))
            return False
        if not self._fits(attempt, tokens):
            return self._unfit()
        if not self._context.check_rebuild(window) or self._context.backend is not attempt.backend:
            return self._stale()
        checkpoint = self._checkpoint(attempt, result, tokens)
        try:
            self._session_manager.commit_model_window(
                attempt.session_id,
                attempt.previous_window,
                attempt.previous_checkpoint,
                attempt.window,
                checkpoint=checkpoint,
            )
        except ModelWindowConflictError:
            return self._stale()
        self._context.apply_window(attempt.window)
        self._report_completion(checkpoint)
        return True

    async def compact_async(self) -> bool:
        """Capture and commit a rebuilt context using asynchronous provider operations.

        Returns:
            bool: Whether an eligible rebuilt context was committed.

        Raises:
            asyncio.CancelledError: Rebuild or checkpoint persistence was cancelled.
        """
        window = await self._context.capture_async()
        try:
            attempt = self._start(window)
            if attempt is None:
                return False
            with attempt.backend.file_projection(window.settings.file_input_mode):
                result = await self._summarize_async(attempt)
                if result is None or (attempt.middle and not result.items):
                    return self._empty_result()
                tokens = await self._estimate_async(attempt, result)
                if not self._fits(attempt, tokens) and self._native_result(result):
                    result = await self._portable_async(attempt)
                    tokens = (
                        await self._estimate_async(attempt, result) if result is not None else None
                    )
        except ModelWindowConflictError:
            return self._stale()
        except NotImplementedError:
            return self._unsupported()
        except CompactionLimitError as error:
            self._interaction.warning(str(error))
            return False
        if not self._fits(attempt, tokens):
            return self._unfit()
        checkpoint = self._checkpoint(attempt, result, tokens)
        try:
            await self._session_manager.commit_model_window_async(
                attempt.session_id,
                attempt.previous_window,
                attempt.previous_checkpoint,
                attempt.window,
                checkpoint=checkpoint,
            )
        except ModelWindowConflictError:
            return self._stale()
        self._context.apply_window(attempt.window)
        self._report_completion(checkpoint)
        return True

    def _start(self, window: ModelWindow) -> _Attempt | None:
        """Freeze the source frontier, projected suffix, and eligible earlier checkpoint."""
        manager = self._session_manager
        backend = self._context.backend
        session = manager.session
        previous = session.checkpoint()
        current = session.window
        if (current is None and not session.messages) or (
            not self.can_compact()
            and current is not None
            and window.checkpoint_id == current.checkpoint_id
            and (window.route, window.model, window.settings, window.baseline.digest, window.tools)
            == (
                current.route,
                current.model,
                current.settings,
                current.baseline.digest,
                current.tools,
            )
        ):
            self._interaction.warning("There is no new session context to compact.")
            return None
        checkpoint_id = window.checkpoint_id
        window = window.model_copy(update={"checkpoint_id": str(new_uuid())})
        limit = self._context.compaction_capacity(window)
        capacity = limit or _UNDECLARED_CAPACITY
        available = capacity - backend.generation_reserve(capacity)
        baseline = [window.baseline_item(rebuilt=True)] if window.layout != "leading" else []
        overhead = estimate_context(
            baseline, instructions=window.leading_instructions(rebuilt=True), tools=window.tools
        )
        overhead += self._context.rebuild_guidance_tokens(window)
        retain_budget = available - overhead
        if retain_budget < 0:
            raise CompactionLimitError("Instructions and tools exceed the available context limit.")
        retained = RetainedContext.select(
            session, retain_budget - min(_SUMMARY_RESERVE, retain_budget // 4)
        )
        middle = list(retained.middle)
        previous_items = ()
        previous_context_tokens = None
        uncovered = middle
        opening = [session.messages[index] for index in retained.coverage.opening]
        prefix = bool(opening) or window.layout != "leading"
        if previous is not None and previous.id == checkpoint_id and previous.coverage is not None:
            earlier = RetainedContext(previous.boundary, previous.coverage).middle
            if tuple(middle[: len(earlier)]) == earlier and backend.can_replay_checkpoint(
                previous.context, window.model, prefix=prefix
            ):
                previous_items = previous.context
                previous_context_tokens = previous.input_tokens_after
                uncovered = middle[len(earlier) :]
        self._interaction.info("Compacting session context...")
        telemetry_activity("compaction.started", component="compaction", model=window.model)
        return _Attempt(
            session_id=session.id,
            previous_window=current.id if current is not None else None,
            previous_checkpoint=previous.id if previous is not None else None,
            backend=backend,
            window=window,
            retained=retained,
            opening=opening,
            tail=session.rebuilt_items(retained.coverage.tail),
            native_tail=(
                session.rebuilt_items(
                    retained.coverage.tail, provider_state_start=window.provider_state_start
                )
                if backend.preserves_compacted_reasoning
                else None
            ),
            middle=session.rebuilt_items(middle),
            previous_items=previous_items,
            previous_context_tokens=previous_context_tokens,
            uncovered=session.rebuilt_items(
                uncovered,
                provider_state_start=window.provider_state_start
                if window.layout != "leading"
                else None,
            ),
            available=available,
            summary_budget=max(1, available - min(_SUMMARY_RESERVE, available // 4)),
            native=limit is not None,
        )

    def _summarize(self, attempt: _Attempt) -> CompactionResult | None:
        """Summarize only the middle and retain an eligible prior compacted range."""
        if not attempt.middle or (attempt.previous_items and not attempt.uncovered):
            return CompactionResult(items=attempt.previous_items)
        middle = self._summary_input(attempt)
        if middle is None:
            return self._portable(attempt)
        hydrated = self._session_manager.hydrate_context(middle)
        result = attempt.backend.compact(
            hydrated, model=attempt.window.model, prefix=attempt.has_prefix
        )
        if result is None and attempt.previous_items:
            return self._portable(attempt)
        return result

    async def _summarize_async(self, attempt: _Attempt) -> CompactionResult | None:
        """Summarize the captured middle via async hydration and provider APIs."""
        if not attempt.middle or (attempt.previous_items and not attempt.uncovered):
            return CompactionResult(items=attempt.previous_items)
        middle = self._summary_input(attempt)
        if middle is None:
            return await self._portable_async(attempt)
        hydrated = await self._session_manager.hydrate_context_async(middle)
        result = await attempt.backend.compact_async(
            hydrated, model=attempt.window.model, prefix=attempt.has_prefix
        )
        if result is None and attempt.previous_items:
            return await self._portable_async(attempt)
        return result

    @staticmethod
    def _summary_input(attempt: _Attempt) -> list[ModelContextItem] | None:
        """Budget a reusable native block by the previous rebuilt request's measured cost."""
        if not attempt.native:
            return None
        middle = [*attempt.previous_items, *attempt.uncovered]
        if any(isinstance(item, CompactionContextItem) for item in attempt.previous_items):
            prior = attempt.previous_context_tokens or attempt.summary_budget
            tokens = prior + estimate_context(attempt.uncovered)
        else:
            tokens = estimate_context(middle)
        return middle if tokens <= attempt.summary_budget else None

    def _portable(self, attempt: _Attempt) -> CompactionResult | None:
        """Build bounded summaries from canonical history without native state."""
        result = CompactionResult(items=())
        for batch in portable_batches(attempt.middle, attempt.summary_budget // 2):
            context = self._portable_input(result.items, batch, attempt.summary_budget)
            model_input = self._session_manager.hydrate_context(context)
            result = attempt.backend.compact(model_input, model=attempt.window.model, native=False)
            if result is None or not result.items:
                return None
        return result

    async def _portable_async(self, attempt: _Attempt) -> CompactionResult | None:
        """Build bounded portable summaries with asynchronous I/O."""
        result = CompactionResult(items=())
        for batch in portable_batches(attempt.middle, attempt.summary_budget // 2):
            context = self._portable_input(result.items, batch, attempt.summary_budget)
            model_input = await self._session_manager.hydrate_context_async(context)
            result = await attempt.backend.compact_async(
                model_input, model=attempt.window.model, native=False
            )
            if result is None or not result.items:
                return None
        return result

    @staticmethod
    def _portable_input(
        summary: tuple[CompactionItem, ...],
        batch: list[ConversationItem],
        budget: int,
    ) -> list[ModelContextItem]:
        """Enforce the request budget for the prior summary and its next batch together."""
        items = [*summary, *batch]
        if estimate_context(items) > budget:
            raise CompactionLimitError("The portable checkpoint exceeds the summary limit.")
        return items

    def _estimate(self, attempt: _Attempt, result: CompactionResult) -> int | None:
        """Measure the entire rebuilt generation context."""
        if not attempt.backend.can_replay_checkpoint(
            result.items,
            attempt.window.model,
            prefix=attempt.has_prefix,
        ):
            return None
        context = self._session_manager.hydrate_context(self._rebuilt_items(attempt, result))
        tokens = attempt.backend.estimate_generation(
            context,
            instructions=attempt.window.leading_instructions(rebuilt=True),
            tools=attempt.window.tools,
            model=attempt.window.model,
        )
        if tokens is None:
            return None
        return tokens + self._context.rebuild_guidance_tokens(attempt.window)

    async def _estimate_async(self, attempt: _Attempt, result: CompactionResult) -> int | None:
        """Measure replay, revalidate the candidate, then budget live guidance."""
        if not attempt.backend.can_replay_checkpoint(
            result.items,
            attempt.window.model,
            prefix=attempt.has_prefix,
        ):
            return None
        context = await self._session_manager.hydrate_context_async(
            self._rebuilt_items(attempt, result)
        )
        tokens = await attempt.backend.estimate_generation_async(
            context,
            instructions=attempt.window.leading_instructions(rebuilt=True),
            tools=attempt.window.tools,
            model=attempt.window.model,
        )
        if tokens is None:
            return None
        if (
            not await self._context.check_rebuild_async(attempt.window)
            or self._context.backend is not attempt.backend
        ):
            raise ModelWindowConflictError("The captured model window changed during compaction.")
        return tokens + self._context.rebuild_guidance_tokens(attempt.window)

    def _rebuilt_items(self, attempt: _Attempt, result: CompactionResult) -> list[ModelContextItem]:
        """Build the complete context order used to establish generation headroom."""
        window = attempt.window
        baseline = [window.baseline_item(rebuilt=True)] if window.layout != "leading" else []
        tail = (
            attempt.native_tail
            if attempt.native_tail is not None and self._native_result(result)
            else attempt.tail
        )
        _, update = self._context.reasoning_configuration(window)
        reasoning = [update] if update is not None else []
        return [*baseline, *attempt.opening, *result.items, *reasoning, *tail]

    @staticmethod
    def _fits(attempt: _Attempt, tokens: int | None) -> bool:
        """Accept only a measured or bounded request with room for generation."""
        return tokens is not None and tokens <= attempt.available

    @staticmethod
    def _native_result(result: CompactionResult) -> bool:
        """Check whether the candidate may require canonical portable fallback."""
        return any(isinstance(item, CompactionContextItem) for item in result.items)

    def _checkpoint(self, attempt: _Attempt, result: CompactionResult, tokens: int) -> Compaction:
        """Record captured coverage and a refreshed baseline in one window transition."""
        snapshot = attempt.window.baseline
        return Compaction(
            id=attempt.window.checkpoint_id,
            boundary=attempt.retained.end,
            created_at=utc_now(),
            model=attempt.window.model,
            context=result.items,
            coverage=attempt.retained.coverage.model_copy(
                update={
                    "preserve_reasoning": bool(
                        attempt.native_tail is not None and self._native_result(result)
                    )
                }
            ),
            instructions=InstructionSnapshot(
                working_directory=str(snapshot.working_directory or self._working_directory),
                content=snapshot.content,
                digest=snapshot.digest,
                active_skills=snapshot.active_skills,
                references=tuple(
                    section.reference
                    for section in snapshot.sections
                    if section.reference is not None
                ),
            ),
            input_tokens_before=self._session_manager.tokens,
            input_tokens_after=tokens,
        )

    def _report_completion(self, checkpoint: Compaction) -> None:
        """Report generation-context size separately from backend inference usage."""
        telemetry_activity(
            "compaction.completed",
            component="compaction",
            input_tokens_before=checkpoint.input_tokens_before,
            input_tokens_after=checkpoint.input_tokens_after,
        )
        self._interaction.info(
            f"Compacted session context to an estimated {checkpoint.input_tokens_after:,} tokens."
        )

    def _empty_result(self) -> bool:
        """Report an empty result without replacing the previous checkpoint."""
        telemetry_error(
            "compaction.failed", error_type="compaction.empty_result", component="compaction"
        )
        self._interaction.warning("The selected backend did not produce compacted context.")
        return False

    def _unsupported(self) -> bool:
        """Report a backend that provides neither native nor portable compaction."""
        telemetry_activity("compaction.unsupported", severity="warning", component="compaction")
        self._interaction.warning("The selected backend does not support context compaction.")
        return False

    def _unfit(self) -> bool:
        """Keep the current window when the rebuilt request cannot fit."""
        self._interaction.warning("The rebuilt request would exceed the available context limit.")
        return False

    def _stale(self) -> bool:
        """Keep the current window when the rebuild lost its captured state."""
        self._interaction.warning(
            "The session or its instruction sources changed during compaction."
        )
        return False
