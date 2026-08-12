"""Define internal mention-domain models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mention:
    """Describe one exact mention in submitted text.

    Args:
        marker (str): Marker introducing the mention.
        value (str): Exact referenced value without the marker.
        start (int): Inclusive source-text offset.
        end (int): Exclusive source-text offset.
    """

    marker: str
    value: str
    start: int
    end: int
