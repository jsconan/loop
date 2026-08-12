"""Tests for exact mention parsing."""

from loop.mentions.models import Mention
from loop.mentions.parser import parse_mentions


def test_parser_finds_longest_exact_candidates_with_safe_boundaries():
    """Parsing prefers exact longest values and ignores email-like and unknown marker text."""
    text = "Email a@b.test, use $code, inspect (@my file.py), ignore $unknown."

    assert parse_mentions(
        text,
        {"@": ("my", "my file.py"), "$": ("code",)},
    ) == (
        Mention("$", "code", 20, 25),
        Mention("@", "my file.py", 36, 47),
    )


def test_parser_accepts_start_whitespace_and_punctuation_boundaries():
    """Mentions work at input start and after whitespace while requiring a valid end boundary."""
    assert parse_mentions("@file.py next $skill!", {"@": ("file.py",), "$": ("skill",)}) == (
        Mention("@", "file.py", 0, 8),
        Mention("$", "skill", 14, 20),
    )
    assert parse_mentions("prefix@file.py @file.pyx", {"@": ("file.py",)}) == ()
