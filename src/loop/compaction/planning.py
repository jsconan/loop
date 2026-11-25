"""Select canonical opening, middle, and recent context before compaction."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..backend import estimate_context
from ..models import ConversationItem, Message, ToolCall, ToolResult
from ..session import CompactionCoverage, Session

_OPENING_TOKENS = 1024
_RECENT_COMPLETE_TURNS = 4


class CompactionLimitError(ValueError):
    """Report history that cannot be compacted safely within the available context."""


@dataclass(frozen=True)
class RetainedContext:
    """Capture a compactable range and canonical references to retained history.

    Args:
        end (int): Exclusive history frontier at capture.
        coverage (CompactionCoverage): Cut and ordered retained indices.
    """

    end: int
    coverage: CompactionCoverage

    @property
    def middle(self) -> tuple[int, ...]:
        """Return the eligible canonical indices before the cut.

        Returns:
            tuple[int, ...]: Older indices not retained in the opening or tail.
        """
        retained = set(self.coverage.opening) | set(self.coverage.tail)
        return tuple(index for index in range(self.coverage.cut) if index not in retained)

    @classmethod
    def select(cls, session: Session, budget: int) -> "RetainedContext":
        """Select recent complete turns and preserve the current unfinished turn first.

        Args:
            session (Session): Canonical history and run state to capture.
            budget (int): Portion of the generation context available for retained items.

        Returns:
            RetainedContext: Captured frontier and a valid retention layout.

        Raises:
            CompactionLimitError: The unfinished turn cannot fit even at a tool-round boundary.
        """
        messages = session.messages
        end = len(messages)
        users = [
            index
            for index, item in enumerate(messages)
            if _user_message(item) and (index == 0 or not _user_message(messages[index - 1]))
        ]
        costs = [estimate_context((item,)) for item in messages]
        legal = _complete_boundaries(messages)
        active_start = _active_start(session, users)
        cut = end
        pinned = ()
        remaining = budget
        if active_start is not None:
            cut, pinned, remaining = _retain_active(messages, costs, legal, active_start, budget)

        opening = []
        opening_budget = min(_OPENING_TOKENS, max(0, remaining))
        if users:
            for index in range(users[0], cut):
                if not _user_message(messages[index]):
                    break
                if index in pinned:
                    continue
                if costs[index] > opening_budget:
                    break
                opening.append(index)
                opening_budget -= costs[index]
                remaining -= costs[index]

        recent = [index for index in users if index < cut and index != active_start]
        retained_opening = set(opening)
        for start in reversed(recent[-_RECENT_COMPLETE_TURNS:]):
            if start not in legal:
                continue
            if active_start is not None and cut > active_start:
                break
            cost = sum(costs[index] for index in range(start, cut) if index not in retained_opening)
            if cost > remaining:
                break
            remaining -= cost
            cut = start
        return cls(
            end=end,
            coverage=CompactionCoverage(
                cut=cut,
                opening=tuple(index for index in opening if index < cut),
                tail=(*pinned, *range(cut, end)),
            ),
        )


def portable_batches(
    items: Iterable[ConversationItem],
    budget: int,
) -> Iterable[list[ConversationItem]]:
    """Batch semantic history without dividing a completed tool round.

    Args:
        items (Iterable[ConversationItem]): Canonical middle history.
        budget (int): Allowed estimated context per portable summary request.

    Yields:
        list[ConversationItem]: One bounded batch.

    Raises:
        CompactionLimitError: An indivisible item or unresolved round prevents summarization.
    """
    messages = list(items)
    boundaries = _complete_boundaries(messages)
    batch = []
    cost = 0
    start = 0
    for end in sorted(boundaries - {0}):
        group = messages[start:end]
        group_cost = estimate_context(group)
        if group_cost > budget:
            raise CompactionLimitError(
                "A historical tool round or message exceeds the summary limit."
            )
        if batch and cost + group_cost > budget:
            yield batch
            batch = []
            cost = 0
        batch.extend(group)
        cost += group_cost
        start = end
    if start < len(messages):
        raise CompactionLimitError("An unresolved tool round cannot be summarized.")
    if batch:
        yield batch


def _user_message(item: ConversationItem) -> bool:
    """Identify new user text, including consecutive additions to an opening."""
    return isinstance(item, Message) and item.role == "user"


def _active_start(session: Session, users: list[int]) -> int | None:
    """Find the latest actual request from the session's current execution state."""
    if not users:
        return None
    recovery = session.recovery_state()
    if session.active_run is not None or (
        recovery is not None and recovery.action != "finalize_run"
    ):
        return users[-1]
    return None


def _complete_boundaries(messages: list[ConversationItem]) -> set[int]:
    """Find boundaries that cannot separate a call occurrence from its result."""
    pending: Counter[str] = Counter()
    boundaries = {0}
    for index, item in enumerate(messages):
        if isinstance(item, ToolCall):
            pending[item.call_id] += 1
        elif isinstance(item, ToolResult) and pending[item.call_id]:
            pending[item.call_id] -= 1
            if not pending[item.call_id]:
                del pending[item.call_id]
        if not pending:
            boundaries.add(index + 1)
    return boundaries


def _retain_active(
    messages: list[ConversationItem],
    costs: list[int],
    boundaries: set[int],
    start: int,
    budget: int,
) -> tuple[int, tuple[int, ...], int]:
    """Retain an unfinished turn or split it at a completed parallel tool round."""
    cut = max(boundary for boundary in boundaries if boundary <= start)
    total = sum(costs[cut:])
    if total <= budget:
        return cut, (), budget - total
    user_end = start + 1
    while user_end < len(messages) and _user_message(messages[user_end]):
        user_end += 1
    pin = tuple(range(start, user_end))
    pin_cost = sum(costs[start:user_end])
    for index in range(user_end, len(messages)):
        if index not in boundaries or not isinstance(messages[index - 1], ToolResult):
            continue
        required = pin_cost + sum(costs[index:])
        if required <= budget:
            return index, pin, budget - required
    raise CompactionLimitError("The active request and its tool round exceed the context limit.")
