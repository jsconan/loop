"""Tests for completion adapter aggregation and ranking."""

from collections.abc import Iterable
from unittest.mock import Mock

from prompt_toolkit.document import Document

from loop import CompletionAdapter, CompletionManager, CompletionMatch, CompletionValue


class StaticAdapter(CompletionAdapter):
    """Expose a fixed match and candidate sequence for manager tests.

    Args:
        match (CompletionMatch | None): Match returned for every document.
        values (Iterable[CompletionValue]): Candidates returned for an active match.
        error (Exception | None): Optional error raised while matching.
    """

    def __init__(
        self,
        match: CompletionMatch | None,
        values: Iterable[CompletionValue] = (),
        error: Exception | None = None,
    ) -> None:
        self._match = match
        self._values = values
        self._error = error

    def match(self, document: Document) -> CompletionMatch | None:
        """Return the configured activation or raise the configured error.

        Args:
            document (Document): Current editable input.

        Returns:
            CompletionMatch | None: Configured activation.

        Raises:
            Exception: Configured adapter failure, when present.
        """
        del document
        if self._error is not None:
            raise self._error
        return self._match

    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return the configured candidates.

        Args:
            match (CompletionMatch): Active completion match.

        Returns:
            Iterable[CompletionValue]: Configured candidate sequence.
        """
        del match
        return self._values


def complete(manager: CompletionManager, text: str = ""):
    """Return all completions produced for a document."""
    return list(manager.get_completions(Document(text), Mock()))


def test_manager_aggregates_ranks_deduplicates_and_bounds_registered_adapters():
    """Registered adapters contribute one globally ranked and bounded result sequence."""
    match = CompletionMatch("eta", "eta", "$")
    manager = CompletionManager(
        (
            StaticAdapter(
                match,
                (
                    CompletionValue("dir-eta/path"),
                    CompletionValue("x/beta"),
                    CompletionValue("eta/path"),
                    CompletionValue("x/eta-long"),
                    CompletionValue("eta", "first"),
                ),
            ),
        ),
        max_results=2,
    )
    manager.register(StaticAdapter(match, (CompletionValue("eta", "duplicate"),)))

    results = complete(manager)

    assert [result.text for result in results] == ["$eta", "$x/eta-long"]
    assert results[0].display_meta_text == "first"


def test_manager_isolates_inactive_and_failed_adapters_and_filters_nonmatches():
    """One unavailable capability cannot suppress valid results from another capability."""
    match = CompletionMatch("hit", "hit")
    manager = CompletionManager(
        (
            StaticAdapter(None),
            StaticAdapter(None, error=RuntimeError("unavailable")),
            StaticAdapter(match, (CompletionValue("miss"), CompletionValue("hit"))),
        )
    )

    assert [result.text for result in complete(manager)] == ["hit"]
