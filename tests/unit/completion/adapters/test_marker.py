"""Tests for marker-prefixed completion."""

from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import CompletionManager, CompletionValue, MarkerCompletionAdapter, Skill


def complete(completer: CompletionManager, text: str):
    """Return all completions produced for text with its cursor at the end."""
    return list(completer.get_completions(Document(text), Mock()))


def test_skill_mentions_work_in_prose_and_require_a_token_boundary(tmp_path):
    """Skill mentions replace only an active bounded token and expose descriptions."""
    skill = Skill("coding", "Implement Python code.", tmp_path / "SKILL.md")
    adapter = MarkerCompletionAdapter(
        "$", lambda: (CompletionValue(skill.name, skill.description),)
    )
    completer = CompletionManager((adapter,))

    assert adapter.front_markers == ("$",)
    results = complete(completer, "Please use ($din")

    assert [(item.text, item.start_position) for item in results] == [("$coding", -4)]
    assert results[0].display_meta_text == "Implement Python code."
    assert complete(completer, "price$din") == []
    assert complete(completer, "plain text") == []


def test_marker_completion_quotes_spaces_and_continues_inside_delimiters():
    """Multi-word targets insert quotes and remain completable inside supported delimiters."""
    completer = CompletionManager(
        (MarkerCompletionAdapter("$", lambda: (CompletionValue("code review"),)),)
    )

    bare = complete(completer, "Use $code")
    quoted = complete(completer, 'Use $"code rev')
    single_quoted = complete(completer, "Use $'code rev")
    bracketed = complete(completer, "Use $[code rev")

    assert [(item.text, item.start_position) for item in bare] == [('$"code review"', -5)]
    assert [(item.text, item.start_position) for item in quoted] == [('$"code review"', -10)]
    assert [(item.text, item.start_position) for item in single_quoted] == [("$'code review'", -10)]
    assert [(item.text, item.start_position) for item in bracketed] == [("$[code review]", -10)]
    assert complete(completer, 'Use $"code review"') == []
    assert complete(completer, "Use $[code review]") == []

    escaped = CompletionManager(
        (MarkerCompletionAdapter("$", lambda: (CompletionValue('say "hi" \\ now'),)),)
    )
    assert [item.text for item in complete(escaped, r'Use $"say \"hi\" \\ n')] == [
        '$"say \\"hi\\" \\\\ now"'
    ]


@pytest.mark.parametrize("marker", ["ab", "a", " "])
def test_marker_adapters_reject_invalid_markers(marker):
    """Marker capabilities reject ambiguous alphanumeric activators."""
    with pytest.raises(ValueError, match="one non-alphanumeric"):
        MarkerCompletionAdapter(marker, lambda: ())
