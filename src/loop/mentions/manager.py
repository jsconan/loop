"""Coordinate independently injected mention handlers."""

from collections.abc import Iterable

from ..completion import CompletionAdapter
from ..models import ContextReference
from .handlers import MentionHandler
from .parser import parse_mentions


class MentionManager:
    """Parse and dispatch unique mention targets through an injected handler registry.

    Args:
        handlers (Iterable[MentionHandler]): Independently owned mention capabilities. Markers
            must be unique, and at most one handler may accept ordinary Markdown links.

    Raises:
        ValueError: If handlers declare duplicate markers or multiple Markdown-link owners.
    """

    _handlers: tuple[MentionHandler, ...]
    _handlers_by_marker: dict[str, MentionHandler]
    _link_handler: MentionHandler | None

    def __init__(self, handlers: Iterable[MentionHandler] = ()) -> None:
        self._handlers = tuple(handlers)
        self._handlers_by_marker = {handler.marker: handler for handler in self._handlers}
        if len(self._handlers_by_marker) != len(self._handlers):
            raise ValueError("Mention handler markers must be unique.")
        link_handlers = tuple(
            handler for handler in self._handlers if handler.accepts_markdown_links
        )
        if len(link_handlers) > 1:
            raise ValueError("Only one mention handler may accept ordinary Markdown links.")
        self._link_handler = link_handlers[0] if link_handlers else None

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
            tuple[ContextReference, ...]: Context produced by all uniquely mentioned capabilities.

        Raises:
            OSError: A handler cannot read its referenced resource.
            UnicodeError: Referenced content is not valid UTF-8.
            ValueError: A handler rejects a mention or cannot complete its operation.
        """
        mentions = parse_mentions(
            content,
            {handler.marker: handler.candidates() for handler in self._handlers},
            optional_link_marker=self._link_handler.marker
            if self._link_handler is not None
            else None,
        )
        references = []
        for handler in self._handlers:
            values = tuple(
                dict.fromkeys(
                    mention.value
                    for mention in mentions
                    if mention.marker == handler.marker and mention.required
                )
            )
            if values:
                references.extend(handler.resolve(values))
            optional_values = tuple(
                dict.fromkeys(
                    mention.value
                    for mention in mentions
                    if (
                        mention.marker == handler.marker
                        and not mention.required
                        and mention.value not in values
                    )
                )
            )
            if optional_values:
                references.extend(handler.resolve_optional(optional_values))
        return tuple(references)
