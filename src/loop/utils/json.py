"""Decode JSON without silently choosing duplicate authority fields."""

import json
from typing import IO


def json_encode(content: object, stream: None | IO[str] = None) -> str:
    """Encode a Python object as canonical JSON string and optionally write it to a stream.

    Args:
        content (object): The Python object to encode.
        stream (None | IO[str]): Optional stream to write the JSON string to.
            If None, the JSON string is only returned.

    Returns:
        str: Canonical JSON representation of the object.
    """
    result = json.dumps(content, sort_keys=True, separators=(",", ":"))
    if stream is not None:
        stream.write(result)
    return result


def load_unique_json(data: str | bytes, *, duplicate_message: str) -> object:
    """Decode JSON while rejecting repeated keys at every object depth.

    Args:
        data (str | bytes): JSON text or encoded bytes accepted by the standard decoder.
        duplicate_message (str): Boundary-specific diagnostic for ambiguous fields.

    Returns:
        object: Decoded JSON value with no duplicate object keys.

    Raises:
        ValueError: Input is malformed or contains duplicate keys.
    """

    def unique(pairs):
        """Reject ambiguity before constructing any authority object."""
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(duplicate_message)
            result[name] = value
        return result

    return json.loads(data, object_pairs_hook=unique)
