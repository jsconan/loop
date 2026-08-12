"""Coordinate independently injected mention handlers."""

from collections.abc import Iterable

from ..completion import CompletionAdapter
from ..models import ContextReference
from .handlers import MentionHandler
from .parser import parse_mentions


class MentionManager:
    """Parse and dispatch mentions through an injected handler registry.

    Args:
        handlers (Iterable[MentionHandler]): Independently owned mention capabilities. Markers
            must be unique.

    Raises:
        ValueError: If more than one handler declares the same marker.
    """

    _handlers: tuple[MentionHandler, ...]
    _handlers_by_marker: dict[str, MentionHandler]

    def __init__(self, handlers: Iterable[MentionHandler] = ()) -> None:
        self._handlers = tuple(handlers)
        self._handlers_by_marker = {handler.marker: handler for handler in self._handlers}
        if len(self._handlers_by_marker) != len(self._handlers):
            raise ValueError("Mention handler markers must be unique.")

    @property
    def completion_adapters(self) -> tuple[CompletionAdapter, ...]:
        """Return completion adapters declared by registered handlers.

        Returns:
            tuple[CompletionAdapter, ...]: Injected mention completion capabilities.
        """
        return tuple(handler.completion_adapter for handler in self._handlers)

    def resolve(self, content: str) -> tuple[ContextReference, ...]:
        """Resolve one user message through every matching handler.

        Args:
            content (str): Submitted user text.

        Returns:
            tuple[ContextReference, ...]: Context produced by all mentioned capabilities.

        Raises:
            OSError: A handler cannot read its referenced resource.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A handler rejects a mention or cannot complete its operation.
        """
        mentions = parse_mentions(
            content,
            {handler.marker: handler.candidates() for handler in self._handlers},
        )
        references = []
        for handler in self._handlers:
            values = tuple(
                mention.value for mention in mentions if mention.marker == handler.marker
            )
            if values:
                references.extend(handler.resolve(values))
        return tuple(references)
