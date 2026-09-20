"""Provide general hashing utilities."""

import json
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


def matches_digest(content: bytes, digest: str) -> bool:
    """Check if the SHA-256 digest of the content matches the given digest.

    Args:
        content (bytes): Raw bytes to hash.
        digest (str): Lowercase hexadecimal SHA-256 digest to compare against.

    Returns:
        bool: True if the digest matches, False otherwise.
    """
    return sha256_digest(content) == digest


def payload_digest(payload: object) -> str:
    """Return the digest of one canonical JSON payload.

    Args:
        payload (object): JSON-compatible value to serialize canonically.

    Returns:
        str: Lowercase SHA-256 hexadecimal digest.
    """
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256_digest(canonical)
