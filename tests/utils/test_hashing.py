"""Tests for general hashing utilities."""

from hashlib import sha256

import pytest

from loop.utils import payload_digest, sha256_digest


@pytest.mark.parametrize("content", ["instructions €", "instructions €".encode()])
def test_sha256_digest_hashes_text_and_bytes(content):
    """Digest generation hashes UTF-8 text and equivalent raw bytes identically."""
    expected = sha256("instructions €".encode()).hexdigest()

    assert sha256_digest(content) == expected


def test_payload_digest_is_canonical():
    """Equivalent JSON mappings produce the same canonical evidence digest."""
    assert payload_digest({"b": 2, "a": 1}) == payload_digest({"a": 1, "b": 2})
