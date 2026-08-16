"""Parse exact mentions from submitted user text."""

from collections.abc import Mapping, Sequence

from .models import Mention


def parse_mentions(
    text: str,
    candidates: Mapping[str, Sequence[str]],
    optional_link_marker: str | None = None,
) -> tuple[Mention, ...]:
    """Return bare, explicit, and optional Markdown-link mentions in source order.

    Args:
        text (str): Submitted user text.
        candidates (Mapping[str, Sequence[str]]): Exact values accepted for each marker. Bounded
            values may escape their closing delimiter and backslashes. Only legacy bare mentions
            require a known candidate.
        optional_link_marker (str | None): Marker namespace that should gracefully resolve ordinary
            Markdown link destinations. Defaults to ignoring ordinary links.

    Returns:
        tuple[Mention, ...]: Non-overlapping exact mentions.
    """
    mentions = []
    index = 0
    while index < len(text):
        marker = text[index]
        if (
            marker == "["
            and optional_link_marker is not None
            and (not index or text[index - 1] != "!")
        ):
            linked = _linked_value(text, index)
            if linked is not None:
                value, end = linked
                mentions.append(Mention(optional_link_marker, value, index, end, required=False))
                index = end
                continue
        if marker not in candidates or (index and text[index - 1].isalnum()):
            index += 1
            continue
        bounded = _bounded_value(text, index + 1)
        if bounded is not None:
            value, end = bounded
            if text[index + 1] == "[" and end < len(text) and text[end] == "(":
                linked = _markdown_link_destination(text, end)
                if linked is None:
                    index += 1
                    continue
                value, end = linked
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


def _markdown_link_destination(text: str, start: int) -> tuple[str, int] | None:
    """Decode a Markdown link destination and its exclusive end offset."""
    value = []
    depth = 0
    index = start + 1
    while index < len(text):
        character = text[index]
        if character.isspace() or character in "<>\x00":
            return None
        if character == "\\" and index + 1 < len(text):
            index += 1
            character = text[index]
        elif character == "(":
            depth += 1
        elif character == ")":
            if depth == 0:
                return "".join(value), index + 1
            depth -= 1
        value.append(character)
        index += 1
    return None


def _linked_value(text: str, start: int) -> tuple[str, int] | None:
    """Decode one ordinary Markdown link destination and its exclusive end offset."""
    bounded = _bounded_value(text, start)
    if bounded is None:
        return None
    _, end = bounded
    if end >= len(text) or text[end] != "(":
        return None
    return _markdown_link_destination(text, end)
