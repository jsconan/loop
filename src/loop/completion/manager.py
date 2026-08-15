"""Aggregate registered interactive completion adapters."""

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from .adapters import CompletionAdapter
from .models import CompletionMatch, CompletionValue


class CompletionManager(Completer):
    """Aggregate, rank, and bound independently registered completion adapters.

    Args:
        adapters (Iterable[CompletionAdapter]): Adapters registered in tie-breaking order.
        max_results (int): Maximum candidates yielded for one request. Defaults to 100.
    """

    _adapters: list[CompletionAdapter]
    _max_results: int

    def __init__(
        self,
        adapters: Iterable[CompletionAdapter] = (),
        max_results: int = 100,
    ) -> None:
        self._adapters = list(adapters)
        self._max_results = max_results

    def register(self, adapter: CompletionAdapter) -> CompletionAdapter:
        """Register an adapter after existing adapters.

        Args:
            adapter (CompletionAdapter): Completion capability to register.

        Returns:
            CompletionAdapter: The registered adapter.
        """
        self._adapters.append(adapter)
        return adapter

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Yield aggregated candidates appropriate for the input before the cursor.

        Args:
            document (Document): Current editable input and cursor position.
            complete_event (CompleteEvent): Event that requested completion.

        Yields:
            Completion: Ranked, deduplicated insertion candidates from active adapters.
        """
        del complete_event
        collected = []
        for adapter_index, adapter in enumerate(self._adapters):
            try:
                match = adapter.match(document)
                if match is not None:
                    collected.extend(
                        (value, match, adapter_index) for value in adapter.complete(match)
                    )
            except Exception:  # pylint: disable=broad-exception-caught
                # Completion capabilities are optional, best-effort UI integrations.
                continue

        ranked = sorted(collected, key=self._rank)
        seen = set()
        yielded = 0
        for value, match, _ in ranked:
            searchable = f"{value.display or ''} {value.value}".casefold()
            if match.fragment.casefold() not in searchable:
                continue
            text = f"{match.prefix}{value.value}"
            key = (text, -len(match.replaced))
            if key in seen:
                continue
            seen.add(key)
            yield Completion(
                text,
                start_position=key[1],
                display=f"{match.prefix}{value.display or value.value}",
                display_meta=value.description,
            )
            yielded += 1
            if yielded == self._max_results:
                return

    @staticmethod
    def _rank(
        item: tuple[CompletionValue, CompletionMatch, int],
    ) -> tuple[int, int | str, str | int, int]:
        """Return provider-controlled or relevance-based completion order."""
        value, match, adapter_index = item
        if value.sort_order is not None:
            return 0, value.sort_order, "", adapter_index
        rank, text = CompletionManager._score(value, match)
        return 1, rank, text, adapter_index

    @staticmethod
    def _score(value: CompletionValue, match: CompletionMatch) -> tuple[int, str]:
        """Return a stable relevance score for one candidate."""
        needle = match.fragment.casefold()
        text = (value.display or value.value).casefold()
        basename = text.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if text == needle:
            rank = 0
        elif basename.startswith(needle):
            rank = 1
        elif text.startswith(needle):
            rank = 2
        elif needle in basename:
            rank = 3
        else:
            rank = 4
        return rank, text
