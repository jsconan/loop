"""Tests for built-in mention capabilities."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from prompt_toolkit.document import Document

from loop import (
    CompletionManager,
    ContextReference,
    InstructionsManager,
    Problem,
    ProjectPathMentionHandler,
    Skill,
    SkillManager,
    SkillMentionHandler,
    constants,
)
from loop.utils import cached_path, decode_content_cursor


def complete(handler, text):
    """Return completion text produced by one mention handler."""
    manager = CompletionManager((handler.completion_adapter,))
    return [item.text for item in manager.get_completions(Document(text), Mock())]


def test_project_paths_complete_after_cache_expiry_and_resolve_unique_snapshots(
    monkeypatch, tmp_path
):
    """Path completion refreshes after its TTL and resolution deduplicates references."""
    now = [10.0]
    monkeypatch.setattr("loop.completion.adapters.time.monotonic", lambda: now[0])
    current = [tmp_path]
    handler = ProjectPathMentionHandler(lambda: current[0])
    assert complete(handler, "@") == []
    (tmp_path / "code.py").write_text("print('ok')\n", encoding="utf-8")
    now[0] += 5.0

    assert complete(handler, "@code") == ["@code.py"]
    assert handler.candidates() == ("code.py",)
    assert handler.resolve(("code.py", "code.py")) == (
        ContextReference(
            kind="file",
            path="code.py",
            content="print('ok')\n",
            size_bytes=12,
            included_bytes=12,
            truncated=False,
        ),
    )

    current[0] = tmp_path / "missing"
    assert complete(handler, "@") == []


def test_directory_paths_attach_only_a_visible_bounded_listing(tmp_path):
    """Directory context lists immediate visible children without recursive expansion."""
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "main.py").write_text("pass", encoding="utf-8")
    (folder / "nested").mkdir()
    (folder / "nested" / "deep.py").write_text("pass", encoding="utf-8")

    context = ProjectPathMentionHandler(lambda: tmp_path).resolve(("src/",))

    assert set(context[0].content.splitlines()) == {"main.py", "nested/"}


def test_project_paths_deduplicate_aliases_by_resolved_resource(tmp_path):
    """Different visible paths to the same resource contribute only one snapshot."""
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(source)

    context = ProjectPathMentionHandler(lambda: tmp_path).resolve(("alias.txt", "source.txt"))

    assert len(context) == 1
    assert context[0].path == "alias.txt"


def test_project_paths_reject_binary_changed_escaping_and_special_files(tmp_path, monkeypatch):
    """Unsafe, unavailable, binary, and unsupported paths cannot become attachments."""
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"binary\0data")
    handler = ProjectPathMentionHandler(lambda: tmp_path)
    with pytest.raises(ValueError, match="binary"):
        handler.resolve(("data.bin",))

    binary.write_text("text", encoding="utf-8")
    original_exists = Path.exists
    monkeypatch.setattr(
        Path,
        "exists",
        lambda candidate: False if candidate == binary else original_exists(candidate),
    )
    with pytest.raises(ValueError, match="unavailable"):
        handler.resolve(("data.bin",))
    monkeypatch.undo()

    outside = tmp_path.parent / "outside-mention.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "escape").symlink_to(outside)
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="escapes the project"):
        handler.resolve(("escape",))
    with pytest.raises(ValueError, match="not a file or directory"):
        handler.resolve(("pipe",))


def test_project_paths_allocate_fair_resumable_attachment_previews(tmp_path):
    """Large mentions share a dedicated budget and retain immutable continuations."""
    size = constants.MAX_ATTACHMENT_CONTENT_BYTES
    (tmp_path / "first.txt").write_text("a" * size, encoding="utf-8")
    (tmp_path / "second.txt").write_text("b" * size, encoding="utf-8")
    handler = ProjectPathMentionHandler(lambda: tmp_path)

    first, second = handler.resolve(("first.txt", "second.txt"))

    assert first.included_bytes == second.included_bytes == size // 2
    assert first.truncated is second.truncated is True
    assert first.handle and first.next_cursor
    assert decode_content_cursor(first.next_cursor, first.handle) == first.included_bytes
    assert cached_path(first.handle)[0].read_text(encoding="utf-8") == "a" * size
    assert first.snapshot_content == "a" * size


def test_project_paths_reclaim_unused_attachment_shares(tmp_path):
    """Small files remain complete while large files receive the unused preview budget."""
    size = constants.MAX_ATTACHMENT_CONTENT_BYTES
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "large.txt").write_text("x" * size, encoding="utf-8")

    small, large = ProjectPathMentionHandler(lambda: tmp_path).resolve(("small.txt", "large.txt"))

    assert small.content == "small"
    assert small.handle is None
    assert large.included_bytes == size - len("small")
    assert large.truncated is True


def test_project_paths_reject_snapshots_above_the_hard_source_limit(monkeypatch, tmp_path):
    """Mentions reject files and listings too large for immutable local snapshots."""
    path = tmp_path / "huge.txt"
    with path.open("wb") as stream:
        stream.truncate(constants.MAX_FETCH_BYTES + 1)
    folder = tmp_path / "folder"
    folder.mkdir()

    with pytest.raises(ValueError, match="snapshot limit"):
        ProjectPathMentionHandler(lambda: tmp_path).resolve(("huge.txt",))
    child = Mock()
    child.relative_to.return_value.as_posix.return_value = "x" * (constants.MAX_FETCH_BYTES + 1)
    child.is_dir.return_value = False
    monkeypatch.setattr(
        "loop.mentions.handlers.iter_visible_paths",
        lambda _: (child,),
    )
    with pytest.raises(ValueError, match="snapshot limit"):
        ProjectPathMentionHandler(lambda: tmp_path).resolve(("folder",))


def test_project_paths_gracefully_resolve_valid_markdown_link_destinations(tmp_path):
    """Optional links attach safe project files and ignore invalid or unsupported destinations."""
    (tmp_path / "guide.md").write_text("Guide", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"binary\0data")
    handler = ProjectPathMentionHandler(lambda: tmp_path)

    context = handler.resolve_optional(
        ("https://example.com", "missing.md", "binary.bin", "guide.md")
    )

    assert context == (
        ContextReference(
            kind="file",
            path="guide.md",
            content="Guide",
            size_bytes=5,
            included_bytes=5,
            truncated=False,
        ),
    )


def test_skill_handler_exposes_live_candidates_and_rolls_back_failed_activation(tmp_path):
    """Skill completion is live and a multi-skill activation is atomic."""
    skills = [
        Skill("first", "First.", tmp_path / "first" / "SKILL.md"),
        Skill("second", "Second.", tmp_path / "second" / "SKILL.md"),
    ]
    instructions = Mock(spec=InstructionsManager)
    instructions.skill_manager = SkillManager(skills)
    instructions.active_skill_identities = []
    instructions.activate_skill.side_effect = [
        {"name": "first", "status": "activated", "instructions_updated": True},
        Problem(
            code="skill.instruction_budget_exceeded",
            title="Instruction budget exceeded",
            detail="Too much skill context.",
        ),
    ]
    handler = SkillMentionHandler(instructions)

    assert handler.candidates() == ("first", "second")
    assert complete(handler, "$fir") == ["$first"]
    with pytest.raises(ValueError, match="Too much skill context"):
        handler.resolve(("first", "second"))
    instructions.deactivate_skill.assert_called_once_with("first")


def test_skill_handler_preserves_already_active_skills(tmp_path):
    """Mentioning an active skill remains idempotent and contributes no attachment."""
    location = tmp_path / "SKILL.md"
    location.write_text(
        "---\nname: review\ndescription: Review.\n---\nReview carefully.\n", encoding="utf-8"
    )
    instructions = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review.", location)])
    )
    instructions.activate_skill("review")

    handler = SkillMentionHandler(instructions)

    assert not handler.resolve(("review", "review"))
    assert not handler.resolve_optional(("review",))
    assert instructions.active_skill_identities == [("review", str(location))]
