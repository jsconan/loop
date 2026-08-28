"""Provide general scalar and key normalization utilities."""

import re

from .models import Scalar


def normalized_key(key: str) -> str:
    """Return a lowercase snake-case key containing only ASCII letters and digits.

    Args:
        key (str): Key to normalize.

    Returns:
        str: Normalized key with punctuation collapsed into underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def safe_scalar(value: object, *, max_string_length: int = 2_000) -> Scalar:
    """Return a bounded, single-line scalar safe for structured metadata.

    Args:
        value (object): Value to normalize without invoking arbitrary string conversion.
        max_string_length (int): Maximum retained string length. Defaults to 2,000 characters.

    Returns:
        Scalar: Preserved scalar, sanitized string, or the input's qualified type name.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value.replace("\r", "\\r").replace("\n", "\\n")[:max_string_length]
    return type(value).__qualname__
