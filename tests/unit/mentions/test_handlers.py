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
from loop.mentions.handlers import MentionHandler
from loop.utils import (
    PathHolder,
    cached_path,
    content_identity,
    decode_content_cursor,
)


def complete(handler, text):
    """Return completion text produced by one mention handler."""
    manager = CompletionManager((handler.completion_adapter,))
    return [item.text for item in manager.get_completions(Document(text), Mock())]


def test_base_handler_rejects_markdown_links_by_default():
    """Handlers inherit disabled Markdown-link resolution unless they opt in."""

    class DefaultMentionHandler(MentionHandler):
        """Provide the minimum public behavior needed to test base defaults."""

        @property
        def marker(self):
            return "?"

        @property
        def completion_adapter(self):
            return Mock()

        def candidates(self):
            return ()

        def resolve(self, values):
            del values
            return ()

        def resolve_optional(self, values):
            del values
            return ()

    assert DefaultMentionHandler().accepts_markdown_links is False


def test_project_paths_complete_after_cache_expiry_and_resolve_unique_snapshots(
    monkeypatch, tmp_path
):
    """Path completion refreshes after its TTL and resolution deduplicates references."""
    now = [10.0]
    monkeypatch.setattr("loop.completion.adapters.project_path.time.monotonic", lambda: now[0])
    current = [tmp_path]
    directory = PathHolder(current[0])
    handler = ProjectPathMentionHandler(directory)
    assert handler.marker == "@"
    assert handler.accepts_markdown_links is True
    assert complete(handler, "@") == []
    (tmp_path / "code.py").write_text("print('ok')\n", encoding="utf-8")
    now[0] += 5.0

    assert complete(handler, "@code") == ["@code.py"]
    assert handler.candidates() == ("code.py",)
    resolved = handler.resolve(("code.py", "code.py"))
    assert len(resolved) == 1
    assert resolved[0].model_dump(exclude={"handle"}) == ContextReference(
        kind="file",
        path="code.py",
        content="print('ok')\n",
        size_bytes=12,
        included_bytes=12,
        truncated=False,
        version=content_identity("print('ok')\n")[1],
        media_type="text/x-python",
    ).model_dump(exclude={"handle"})
    assert resolved[0].handle and resolved[0].handle != content_identity("print('ok')\n")[0]

    directory.set(tmp_path / "missing")
    assert complete(handler, "@") == []


def test_directory_paths_attach_only_a_visible_bounded_listing(tmp_path):
    """Directory context lists immediate visible children without recursive expansion."""
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "main.py").write_text("pass", encoding="utf-8")
    (folder / "nested").mkdir()
    (folder / "nested" / "deep.py").write_text("pass", encoding="utf-8")

    context = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(("src/",))

    assert set(context[0].content.splitlines()) == {"main.py", "nested/"}


def test_project_paths_deduplicate_aliases_by_resolved_resource(tmp_path):
    """Different visible paths to the same resource contribute only one snapshot."""
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(source)

    context = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(("alias.txt", "source.txt"))

    assert len(context) == 1
    assert context[0].path == "alias.txt"


def test_project_paths_preserve_binary_changed_escaping_and_special_files(tmp_path, monkeypatch):
    """Binary files retain exact bytes while unsafe and special paths remain unavailable."""
    binary = tmp_path / "data.bin"
    binary.write_bytes(b"binary\0data")
    handler = ProjectPathMentionHandler(PathHolder(tmp_path))
    binary_reference = handler.resolve(("data.bin",))[0]
    assert binary_reference.content == b"binary\0data"
    assert binary_reference.media_type == "application/octet-stream"

    invalid_text = tmp_path / "invalid.txt"
    invalid_text.write_bytes(b"\xff")
    invalid_text_reference = handler.resolve(("invalid.txt",))[0]
    assert invalid_text_reference.content == b"\xff"
    assert invalid_text_reference.media_type == "application/octet-stream"
    invalid_utf8 = tmp_path / "invalid-utf8.bin"
    invalid_utf8.write_bytes(b"\xff")
    invalid_reference = handler.resolve(("invalid-utf8.bin",))[0]
    assert invalid_reference.content == b"\xff"
    assert invalid_reference.included_bytes == 1
    assert invalid_reference.truncated is False

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


def test_project_paths_keep_large_snapshots_local_until_read(tmp_path):
    """Large mentions expose capabilities without inserting content into model context."""
    size = constants.MAX_REFERENCE_PREVIEW_BYTES + 1
    (tmp_path / "first.txt").write_text("a" * size, encoding="utf-8")
    (tmp_path / "second.txt").write_text("b" * size, encoding="utf-8")
    handler = ProjectPathMentionHandler(PathHolder(tmp_path))

    first, second = handler.resolve(("first.txt", "second.txt"))

    assert first.included_bytes == second.included_bytes == 0
    assert first.truncated is second.truncated is True
    assert first.handle and first.next_cursor
    assert decode_content_cursor(first.next_cursor, first.handle) == 0
    assert cached_path(first.handle) is None
    assert first.content == "a" * size
    assert first.version


def test_project_paths_reject_binary_snapshots_that_cannot_be_eagerly_attached(tmp_path):
    """Binary mentions fail instead of advertising an unusable text continuation."""
    path = tmp_path / "large.bin"
    path.write_bytes(b"\0" * (constants.MAX_REFERENCE_PREVIEW_BYTES + 1))

    with pytest.raises(ValueError, match="binary paths must fit the eager attachment budget"):
        ProjectPathMentionHandler(PathHolder(tmp_path)).resolve((path.name,))


def test_optional_project_paths_skip_binary_snapshots_that_cannot_be_eagerly_attached(tmp_path):
    """Optional links ignore binary snapshots that would require a text-only continuation."""
    (tmp_path / "large.bin").write_bytes(b"\0" * (constants.MAX_REFERENCE_PREVIEW_BYTES + 1))
    (tmp_path / "guide.txt").write_text("guide", encoding="utf-8")

    references = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve_optional(
        ("large.bin", "guide.txt")
    )

    assert [reference.path for reference in references] == ["guide.txt"]


def test_project_paths_inline_only_complete_small_files(tmp_path):
    """Small files remain eager while a large neighbor stays entirely local."""
    size = constants.MAX_REFERENCE_PREVIEW_BYTES + 1
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "large.txt").write_text("x" * size, encoding="utf-8")

    small, large = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(
        ("small.txt", "large.txt")
    )

    assert small.content == "small"
    assert small.handle and small.handle != content_identity("small")[0]
    assert large.included_bytes == 0
    assert large.truncated is True


def test_project_paths_keep_many_small_files_lazy(tmp_path):
    """Mentioning many small files does not multiply eager context payloads."""
    for name in ("one.txt", "two.txt", "three.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    references = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(
        ("one.txt", "two.txt", "three.txt")
    )

    assert all(reference.included_bytes == 0 for reference in references)
    assert all(reference.handle and reference.next_cursor for reference in references)


def test_project_paths_bound_aggregate_directory_overviews(monkeypatch, tmp_path):
    """Directory previews share one aggregate budget while retaining lazy snapshots."""
    monkeypatch.setattr(constants, "MAX_REFERENCE_TOTAL_PREVIEW_BYTES", 8)
    values = []
    for index in range(3):
        folder = tmp_path / f"folder-{index}"
        folder.mkdir()
        (folder / "child.txt").write_text("content", encoding="utf-8")
        values.append(folder.name)

    references = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(values)

    assert [reference.included_bytes for reference in references] == [8, 0, 0]
    assert [reference.truncated for reference in references] == [True, True, True]
    assert all(reference.media_type == "text/plain" for reference in references)


def test_project_paths_reclaim_unused_directory_preview_capacity(monkeypatch, tmp_path):
    """Short directory listings leave the aggregate preview budget available to later mentions."""
    monkeypatch.setattr(constants, "MAX_REFERENCE_TOTAL_PREVIEW_BYTES", 8)
    first = tmp_path / "first"
    first.mkdir()
    (first / "a").write_text("", encoding="utf-8")
    second = tmp_path / "second"
    second.mkdir()
    (second / "abcdefghi").write_text("", encoding="utf-8")

    first_reference, second_reference = ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(
        ("first", "second")
    )

    assert first_reference.included_bytes == 1
    assert second_reference.included_bytes == 7


def test_project_paths_reject_snapshots_above_the_hard_source_limit(monkeypatch, tmp_path):
    """Mentions reject files and listings too large for immutable local snapshots."""
    path = tmp_path / "huge.txt"
    with path.open("wb") as stream:
        stream.truncate(constants.MAX_FETCH_BYTES + 1)
    folder = tmp_path / "folder"
    folder.mkdir()

    with pytest.raises(ValueError, match="snapshot limit"):
        ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(("huge.txt",))
    child = Mock()
    child.relative_to.return_value.as_posix.return_value = "x" * (constants.MAX_FETCH_BYTES + 1)
    child.is_dir.return_value = False
    monkeypatch.setattr(
        "loop.mentions.handlers.iter_visible_paths",
        lambda _: (child,),
    )
    with pytest.raises(ValueError, match="snapshot limit"):
        ProjectPathMentionHandler(PathHolder(tmp_path)).resolve(("folder",))


def test_project_paths_gracefully_resolve_valid_markdown_link_destinations(tmp_path):
    """Optional links attach safe project files and ignore invalid or unsupported destinations."""
    (tmp_path / "guide.md").write_text("Guide", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"binary\0data")
    handler = ProjectPathMentionHandler(PathHolder(tmp_path))

    context = handler.resolve_optional(
        ("https://my-host.local", "missing.md", "binary.bin", "guide.md")
    )

    assert [reference.path for reference in context] == ["binary.bin", "guide.md"]
    assert [reference.version for reference in context] == [
        content_identity(b"binary\0data")[1],
        content_identity("Guide")[1],
    ]
    assert all(reference.handle for reference in context)


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

    assert handler.marker == "$"
    assert handler.accepts_markdown_links is False
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
