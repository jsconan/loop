"""Parse exact mentions from submitted user text."""

from collections.abc import Mapping, Sequence

from .models import Mention


def parse_mentions(
    text: str,
    candidates: Mapping[str, Sequence[str]],
) -> tuple[Mention, ...]:
    """Return exact known bare, quoted, or bracket-bounded mentions in source order.

    Args:
        text (str): Submitted user text.
        candidates (Mapping[str, Sequence[str]]): Exact values accepted for each marker. Quoted
            and bracket-bounded values may escape their closing delimiter and backslashes.

    Returns:
        tuple[Mention, ...]: Non-overlapping exact mentions.
    """
    mentions = []
    index = 0
    while index < len(text):
        marker = text[index]
        if marker not in candidates or (index and text[index - 1].isalnum()):
            index += 1
            continue
        bounded = _bounded_value(text, index + 1)
        if bounded is not None:
            value, end = bounded
            if value in candidates[marker]:
                mentions.append(Mention(marker, value, index, end))
                index = end
                continue
        match = next(
            (
                value
                for value in sorted(candidates[marker], key=len, reverse=True)
                if text.startswith(value, index + 1)
                and _is_end_boundary(text, index + 1 + len(value))
            ),
            None,
        )
        if match is None:
            index += 1
            continue
        end = index + 1 + len(match)
        mentions.append(Mention(marker, match, index, end))
        index = end
    return tuple(mentions)


def _bounded_value(text: str, start: int) -> tuple[str, int] | None:
    """Decode one delimited value and its exclusive end offset."""
    if start >= len(text) or text[start] not in "[\"'":
        return None
    closing = "]" if text[start] == "[" else text[start]
    value = []
    index = start + 1
    while index < len(text):
        character = text[index]
        if character == closing:
            return "".join(value), index + 1
        if character == "\\" and index + 1 < len(text) and text[index + 1] in ("\\", closing):
            index += 1
            character = text[index]
        value.append(character)
        index += 1
    return None


def _is_end_boundary(text: str, end: int) -> bool:
    """Return whether an exact candidate ends at a mention boundary."""
    return end == len(text) or text[end].isspace() or text[end] in ",.;:!?)]}"
