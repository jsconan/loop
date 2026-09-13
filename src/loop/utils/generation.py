"""Detect bounded repetitive text in normalized model response events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from ..models import AnswerDelta, ReasoningDelta, ResponseEvent


@dataclass(frozen=True, slots=True)
class GenerationStop:
    """Describe a safe reason to stop a streamed generation.

    Args:
        reason (str): ``"repetition"``, ``"character_limit"``, or ``"duration"``.
        channel (str | None): Text channel that triggered the stop, when applicable.
        elapsed_seconds (float): Elapsed generation time.
        generated_characters (int): Total answer and reasoning characters observed.
        pattern_size (int | None): Repeated exact suffix size, when applicable.
        repetition_count (int | None): Repeated exact suffix count, when applicable.
    """

    reason: Literal["repetition", "character_limit", "duration"]
    channel: Literal["answer", "reasoning"] | None
    elapsed_seconds: float
    generated_characters: int
    pattern_size: int | None = None
    repetition_count: int | None = None


class GenerationWatchdog:
    """Observe normalized response events without retaining complete generated output.

    Args:
        max_seconds (float | None): Optional explicit total-generation deadline.
        max_chars (int | None): Optional explicit total answer and reasoning character budget.
        min_pattern_size (int): Minimum exact repeated suffix size. Zero disables repetition.
        max_pattern_size (int): Maximum exact repeated suffix size. Zero disables repetition.
        min_count (int): Number of adjacent repeated suffixes required to stop.
    """

    _max_seconds: float | None
    _max_chars: int | None
    _min_pattern_size: int
    _max_pattern_size: int
    _min_count: int
    _started_at: float
    _characters: int
    _suffixes: dict[str, str]
    _repetition_observed: bool

    def __init__(
        self,
        max_seconds: float | None,
        max_chars: int | None,
        min_pattern_size: int,
        max_pattern_size: int,
        min_count: int,
    ) -> None:
        self._max_seconds = max_seconds
        self._max_chars = max_chars
        self._min_pattern_size = min_pattern_size
        self._max_pattern_size = max_pattern_size
        self._min_count = min_count
        self._started_at = perf_counter()
        self._characters = 0
        self._suffixes = {"answer": "", "reasoning": ""}
        self._repetition_observed = False

    def observe(self, events: Iterable[ResponseEvent]) -> GenerationStop | None:
        """Return a safe stop diagnostic when newly emitted events cross a limit.

        Args:
            events (Iterable[ResponseEvent]): Normalized events to inspect.

        Returns:
            GenerationStop | None: Stop diagnostic, or ``None`` when limits remain satisfied.
        """
        elapsed = perf_counter() - self._started_at
        if self._max_seconds is not None and elapsed > self._max_seconds:
            return self._stop("duration", None)
        for event in events:
            if isinstance(event, AnswerDelta):
                stopped = self._observe_text("answer", event.text)
            elif isinstance(event, ReasoningDelta):
                stopped = self._observe_text("reasoning", event.text)
            else:
                continue
            if stopped is not None:
                return stopped
        return None

    def _observe_text(
        self,
        channel: Literal["answer", "reasoning"],
        text: str,
    ) -> GenerationStop | None:
        """Observe newly emitted text for repetition and character limits."""
        self._characters += len(text)
        if self._max_chars is not None and self._characters > self._max_chars:
            return self._stop("character_limit", channel)
        if not self._min_pattern_size or not self._max_pattern_size:
            return None
        if not text:
            return None
        maximum_suffix = self._max_pattern_size * self._min_count
        suffix = (self._suffixes[channel] + text)[-maximum_suffix:]
        self._suffixes[channel] = suffix
        if self._repetition_observed:
            return None
        pattern_size, count = self._repeated_suffix(suffix)
        if pattern_size is None:
            return None
        self._repetition_observed = True
        return self._stop("repetition", channel, pattern_size, count)

    def _repeated_suffix(self, suffix: str) -> tuple[int | None, int | None]:
        """Find the smallest repeated exact suffix within one bounded window."""
        if self._min_count < 1:
            return None, None
        maximum = min(self._max_pattern_size, len(suffix) // self._min_count)
        for size in range(self._min_pattern_size, maximum + 1):
            pattern = suffix[-size:]
            count = 1
            while suffix.endswith(pattern * (count + 1)):
                count += 1
            if count >= self._min_count:
                return size, count
        return None, None

    def _stop(
        self,
        reason: Literal["repetition", "character_limit", "duration"],
        channel: Literal["answer", "reasoning"] | None,
        pattern_size: int | None = None,
        repetition_count: int | None = None,
    ) -> GenerationStop:
        """Build a content-free diagnostic for one stopped stream."""
        return GenerationStop(
            reason=reason,
            channel=channel,
            elapsed_seconds=round(perf_counter() - self._started_at, 3),
            generated_characters=self._characters,
            pattern_size=pattern_size,
            repetition_count=repetition_count,
        )
