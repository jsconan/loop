"""Parse exact mentions from submitted user text."""

from collections.abc import Mapping, Sequence

from .models import Mention


def parse_mentions(
    text: str,
    candidates: Mapping[str, Sequence[str]],
) -> tuple[Mention, ...]:
    """Return exact known mentions in source order.

    Args:
        text (str): Submitted user text.
        candidates (Mapping[str, Sequence[str]]): Exact values accepted for each marker.

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


def _is_end_boundary(text: str, end: int) -> bool:
    """Return whether an exact candidate ends at a mention boundary."""
    return end == len(text) or text[end].isspace() or text[end] in ",.;:!?)]}"
