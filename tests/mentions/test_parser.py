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


def test_parser_decodes_exact_bounded_mentions_and_preserves_legacy_matching():
    """Delimited syntax disambiguates spaces and escaped delimiters without breaking bare names."""
    text = r"Use $[code] review, $[code review], and @[docs/a\]b\\c.md]."

    assert parse_mentions(
        text,
        {"$": ("code", "code review"), "@": (r"docs/a]b\c.md",)},
    ) == (
        Mention("$", "code", 4, 11),
        Mention("$", "code review", 20, 34),
        Mention("@", r"docs/a]b\c.md", 40, 58),
    )
    assert parse_mentions("$[unknown] $[unclosed", {"$": ("unknown skill",)}) == ()

    assert parse_mentions(
        r'''$"double \" quote" $'single \' quote' @"back\\slash.md"''',
        {"$": ('double " quote', "single ' quote"), "@": (r"back\slash.md",)},
    ) == (
        Mention("$", 'double " quote', 0, 18),
        Mention("$", "single ' quote", 19, 37),
        Mention("@", r"back\slash.md", 38, 55),
    )
