"""Provide general hashing utilities."""

from hashlib import sha256


def sha256_digest(content: str | bytes) -> str:
    """Return the SHA-256 hexadecimal digest of text or bytes.

    Args:
        content (str | bytes): Text to encode as UTF-8 or raw bytes to hash directly.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    return sha256(encoded).hexdigest()
