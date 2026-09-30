"""Tests for instruction discovery and parsing."""

from pathlib import Path

import pytest

from loop.instructions import (
    InstructionBudgetExceededError,
    build_instructions,
    get_agents_files,
    get_skill_directories,
    load_agents_instructions,
    read_instruction_body,
    read_instruction_frontmatter,
)


def test_get_skill_directories_orders_scopes_by_precedence(tmp_path, monkeypatch):
    """Default skill directories start with the closest project scope and end with user scope."""
    project = tmp_path / "project"
    working_directory = project / "packages" / "app"
    home = tmp_path / "home"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert get_skill_directories(working_directory) == [
        working_directory / ".agents/skills",
        project / "packages/.agents/skills",
        project / ".agents/skills",
        home / ".agents/skills",
    ]


def test_get_skill_directories_uses_local_and_user_scopes_outside_project(tmp_path, monkeypatch):
    """A directory outside a project contributes only its local and user scopes."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    assert get_skill_directories(tmp_path, Path("skills")) == [
        tmp_path / "skills",
        home / "skills",
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("No frontmatter", "must start with YAML frontmatter"),
        ("---\nname: broken\n", "frontmatter is not terminated"),
        ("---\n- item\n---\n", "frontmatter must be a mapping"),
    ],
)
def test_read_instruction_frontmatter_validates_structure(tmp_path, content, message):
    """Frontmatter parsing rejects missing, unterminated, and non-mapping metadata."""
    location = tmp_path / "SKILL.md"
    location.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_instruction_frontmatter(location)


def test_read_instruction_frontmatter_returns_yaml_mapping(tmp_path):
    """Frontmatter parsing returns the decoded YAML mapping without reading the body."""
    location = tmp_path / "SKILL.md"
    location.write_text("---\nname: review\n---\nBody", encoding="utf-8")

    assert read_instruction_frontmatter(location) == {"name": "review"}


@pytest.mark.parametrize("field", ["description:", "description: 1", "description: '  '"])
def test_read_instruction_frontmatter_validates_required_fields(tmp_path, field):
    """Required fields must be non-empty strings and valid values are normalized."""
    location = tmp_path / "SKILL.md"
    location.write_text(f"---\nname: review\n{field}\n---\nBody", encoding="utf-8")

    with pytest.raises(ValueError, match="requires a non-empty description"):
        read_instruction_frontmatter(location, required_fields=("name", "description"))

    location.write_text(
        "---\nname: ' review '\ndescription: ' Review work. '\n---\nBody", encoding="utf-8"
    )
    assert read_instruction_frontmatter(location, required_fields=("name", "description")) == {
        "name": "review",
        "description": "Review work.",
    }


def test_read_instruction_body_returns_trimmed_markdown():
    """Body parsing removes frontmatter and surrounding whitespace."""
    assert read_instruction_body("---\nname: review\n---\n\nDo work.\n", "SKILL.md") == ("Do work.")


def test_read_instruction_body_preserves_legacy_plain_markdown():
    """Instruction files without frontmatter retain their complete trimmed body."""
    assert read_instruction_body("\nDo work.\n", "AGENTS.md") == "Do work."


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("---\nname: broken\n", "frontmatter is not terminated"),
        ("---\n- item\n---\n", "frontmatter must be a mapping"),
    ],
)
def test_read_instruction_body_validates_frontmatter(content, message):
    """Body parsing rejects missing and unterminated frontmatter."""
    with pytest.raises(ValueError, match=message):
        read_instruction_body(content, "SKILL.md", require_frontmatter=True)


def test_get_agents_files_returns_existing_files_in_scope_order(tmp_path):
    """Agent file discovery returns canonical root-to-leaf paths in project scope."""
    project = tmp_path / "project"
    working_directory = project / "src" / "feature"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "AGENTS.md").touch()
    (working_directory / "AGENTS.md").touch()

    assert get_agents_files(working_directory) == [
        (project / "AGENTS.md").resolve(),
        (working_directory / "AGENTS.md").resolve(),
    ]


def test_get_agents_files_respects_nested_ignore_rules(tmp_path):
    """Repository discovery excludes instruction files hidden by agent ignore rules."""
    project = tmp_path / "project"
    nested = project / "src"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    (nested / ".agentignore").write_text("AGENTS.md\n", encoding="utf-8")
    (project / "AGENTS.md").touch()
    (nested / "AGENTS.md").touch()

    assert get_agents_files(nested) == [(project / "AGENTS.md").resolve()]


def test_load_agents_instructions_accumulates_only_agents_files_in_scope(tmp_path):
    """Instructions accumulate in scope order and ignore other agent filenames."""
    project = tmp_path / "project"
    working_directory = project / "src" / "feature"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "AGENTS.md").write_text("project rules\n", encoding="utf-8")
    (project / "AGENTS.override.md").write_text("ignored override", encoding="utf-8")
    (project / "src" / "AGENTS.md").write_text("source rules", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("feature rules", encoding="utf-8")

    loaded = load_agents_instructions(working_directory)

    assert loaded.content == "project rules\n\nsource rules\n\nfeature rules"


def test_load_agents_instructions_checks_only_working_directory_without_project(tmp_path):
    """A directory outside a Git project receives only its own instructions."""
    parent = tmp_path / "parent"
    working_directory = parent / "child"
    working_directory.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("parent rules", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("local rules", encoding="utf-8")

    assert load_agents_instructions(working_directory).content == "local rules"


def test_load_agents_instructions_accepts_a_string_path(tmp_path):
    """String working directories are accepted during instruction discovery."""
    (tmp_path / "AGENTS.md").write_text("local rules", encoding="utf-8")

    assert load_agents_instructions(str(tmp_path)).content == "local rules"


def test_load_agents_instructions_skips_empty_files(tmp_path):
    """Empty instruction files do not contribute to the result."""
    project = tmp_path / "project"
    working_directory = project / "src"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "AGENTS.md").write_text("  \n", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("source rules", encoding="utf-8")

    assert load_agents_instructions(working_directory).content == "source rules"


def test_load_agents_instructions_returns_none_without_guidance(tmp_path):
    """Missing and empty instruction chains return no guidance."""
    assert load_agents_instructions(tmp_path).content is None


def test_load_agents_instructions_rejects_default_budget_overflow(tmp_path):
    """Required instructions fail atomically instead of losing a UTF-8 suffix."""
    (tmp_path / "AGENTS.md").write_text("a" * 32767 + "€", encoding="utf-8")
    with pytest.raises(InstructionBudgetExceededError, match="no instructions were truncated"):
        load_agents_instructions(tmp_path)


def test_load_agents_instructions_accepts_exact_custom_budget(tmp_path):
    """Complete instruction provenance survives an exact byte budget."""
    (tmp_path / "AGENTS.md").write_text("abc€", encoding="utf-8")
    loaded = load_agents_instructions(tmp_path, max_bytes=6)
    assert loaded.content == "abc€"
    assert loaded.sources[0].included_bytes == loaded.sources[0].size_bytes == 6
    with pytest.raises(InstructionBudgetExceededError, match="exceed"):
        load_agents_instructions(tmp_path, max_bytes=4)


def test_load_agents_instructions_never_omits_child_rules(tmp_path):
    """A parent cannot silently consume the budget reserved for the complete chain."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "src"
    nested.mkdir()
    (tmp_path / "AGENTS.md").write_text("abcd")
    (nested / "AGENTS.md").write_text("€")
    with pytest.raises(InstructionBudgetExceededError, match="exceed"):
        load_agents_instructions(nested, max_bytes=4)
    loaded = load_agents_instructions(nested, max_bytes=9)
    assert loaded.content == "abcd\n\n€"
    assert [source.included_bytes for source in loaded.sources] == [4, 3]


def test_load_agents_instructions_accepts_custom_filenames(tmp_path):
    """Callers can provide an ordered collection of instruction filenames."""
    (tmp_path / "CUSTOM.md").write_text("custom rules", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path, ("CUSTOM.md",)).content

    assert instructions == "custom rules"


def test_load_agents_instructions_supports_ordered_fallbacks_and_frontmatter(tmp_path):
    """Fallbacks are deduplicated and use only the first available filename per scope."""
    (tmp_path / "AGENTS.md").write_text("---\ntitle: root\n---\nroot", encoding="utf-8")
    (tmp_path / "CUSTOM.md").write_text("---\ntitle: custom\n---\ncustom", encoding="utf-8")

    loaded = load_agents_instructions(tmp_path, ("AGENTS.md", "CUSTOM.md", "AGENTS.md"))

    assert loaded.content == "root"
    assert [source.path.name for source in loaded.sources] == ["AGENTS.md"]


def test_load_agents_instructions_uses_a_fallback_when_the_primary_is_absent(tmp_path):
    """A custom filename applies when its higher-precedence sibling is unavailable."""
    (tmp_path / "CUSTOM.md").write_text("custom", encoding="utf-8")

    loaded = load_agents_instructions(tmp_path, ("AGENTS.md", "CUSTOM.md"))

    assert loaded.content == "custom"
    assert [source.path.name for source in loaded.sources] == ["CUSTOM.md"]


@pytest.mark.parametrize("frontmatter", ["---\n- item\n---\nbody", "---\na: [\n"])
def test_load_agents_instructions_reports_invalid_frontmatter(tmp_path, frontmatter):
    """Invalid AGENTS metadata is skipped with a source-specific diagnostic."""
    location = tmp_path / "AGENTS.md"
    location.write_text(frontmatter, encoding="utf-8")

    loaded = load_agents_instructions(tmp_path)

    assert loaded.content is None
    assert str(location) in loaded.diagnostics[0]


def test_load_agents_instructions_requires_utf8(tmp_path):
    """Instruction files must contain valid UTF-8 text."""
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        load_agents_instructions(tmp_path)


def test_build_instructions_combines_non_empty_sections():
    """Non-empty sections are preserved and separated by a blank line."""
    assert build_instructions("project rules", None, "", "skill catalog") == (
        "project rules\n\nskill catalog"
    )


def test_build_instructions_returns_none_without_content():
    """Missing or empty sections do not produce an instruction string."""
    assert build_instructions(None, "") is None
