"""Complete marker-prefixed interactive references."""

from collections.abc import Iterable

from prompt_toolkit.document import Document

from ..models import CompletionMatch, CompletionProvider, CompletionValue
from .adapter import CompletionAdapter


class MarkerCompletionAdapter(CompletionAdapter):
    """Complete bare, quoted, or bracket-bounded fragments after a configurable marker.

    Args:
        marker (str): Single non-alphanumeric symbol that activates completion.
        provider (CompletionProvider): Lazy source of completion candidates.

    Raises:
        ValueError: If ``marker`` is invalid or a named completion source is invalid or duplicated.
    """

    _marker: str
    _provider: CompletionProvider

    def __init__(self, marker: str, provider: CompletionProvider) -> None:
        if len(marker) != 1 or marker.isalnum() or marker.isspace():
            raise ValueError("A completion marker must be one non-alphanumeric character.")
        self._marker = marker
        self._provider = provider

    @property
    def front_markers(self) -> tuple[str, ...]:
        """Return the configured activation marker.

        Returns:
            tuple[str, ...]: Single configured marker.
        """
        return (self._marker,)

    def match(self, document: Document) -> CompletionMatch | None:
        """Match a bounded marker token before the cursor.

        Args:
            document (Document): Current editable input and cursor position.

        Returns:
            CompletionMatch | None: Active bare or unclosed delimited marker fragment, or ``None``
                outside a marker token.
        """
        before = document.text_before_cursor
        bounded, delimiter = max(
            ((before.rfind(f"{self._marker}{item}"), item) for item in "[\"'"),
            key=lambda item: item[0],
        )
        if bounded >= 0 and (not bounded or not before[bounded - 1].isalnum()):
            fragment = self._unclosed_mention_fragment(before, bounded + 2, delimiter)
            if fragment is not None:
                return CompletionMatch(
                    self._decode_mention_fragment(fragment, delimiter),
                    before[bounded:],
                    self._marker,
                )
        for index in range(len(before) - 1, -1, -1):
            character = before[index]
            if character.isspace():
                return None
            if character == self._marker:
                if index and before[index - 1].isalnum():
                    return None
                fragment = before[index + 1 :]
                return CompletionMatch(fragment, f"{self._marker}{fragment}", self._marker)
        return None

    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return values from the bound provider.

        Args:
            match (CompletionMatch): Active marker match.

        Returns:
            Iterable[CompletionValue]: Provider values encoded for unambiguous insertion.
        """
        delimiter = match.replaced[1:2]
        if delimiter not in "[\"'":
            delimiter = ""
        return tuple(self._mention_completion(value, delimiter) for value in self._provider())

    @staticmethod
    def _unclosed_mention_fragment(text: str, start: int, delimiter: str) -> str | None:
        """Return text after an opener unless its unescaped closing delimiter is present."""
        closing = "]" if delimiter == "[" else delimiter
        index = start
        while index < len(text):
            if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in ("\\", closing):
                index += 2
                continue
            if text[index] == closing:
                return None
            index += 1
        return text[start:]

    @staticmethod
    def _decode_mention_fragment(fragment: str, delimiter: str) -> str:
        """Decode supported escapes in an incomplete bounded mention."""
        closing = "]" if delimiter == "[" else delimiter
        value = []
        index = 0
        while index < len(fragment):
            character = fragment[index]
            if (
                character == "\\"
                and index + 1 < len(fragment)
                and fragment[index + 1] in ("\\", closing)
            ):
                index += 1
                character = fragment[index]
            value.append(character)
            index += 1
        return "".join(value)

    @staticmethod
    def _mention_completion(value: CompletionValue, delimiter: str) -> CompletionValue:
        """Return a safely bounded insertion value when its text requires one."""
        if not delimiter and not any(character.isspace() for character in value.value):
            return value
        delimiter = delimiter or '"'
        closing = "]" if delimiter == "[" else delimiter
        escaped = value.value.replace("\\", "\\\\").replace(closing, f"\\{closing}")
        return CompletionValue(
            f"{delimiter}{escaped}{closing}",
            value.description,
            value.display or value.value,
            value.sort_order,
        )
