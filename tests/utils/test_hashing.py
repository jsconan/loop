"""Tests for general hashing utilities."""

from hashlib import sha256

import pytest

from loop.utils import sha256_digest


@pytest.mark.parametrize("content", ["instructions €", "instructions €".encode()])
def test_sha256_digest_hashes_text_and_bytes(content):
    """Digest generation hashes UTF-8 text and equivalent raw bytes identically."""
    expected = sha256("instructions €".encode()).hexdigest()

    assert sha256_digest(content) == expected
