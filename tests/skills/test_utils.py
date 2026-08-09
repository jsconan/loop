"""Tests for instruction discovery and parsing."""

from pathlib import Path

import pytest

from loop.skills import (
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

    assert get_skill_directories(tmp_path, Path("instructions")) == [
        tmp_path / "instructions",
        home / "instructions",
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


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("No frontmatter", "must start with YAML frontmatter"),
        ("---\nname: broken\n", "frontmatter is not terminated"),
    ],
)
def test_read_instruction_body_validates_frontmatter(content, message):
    """Body parsing rejects missing and unterminated frontmatter."""
    with pytest.raises(ValueError, match=message):
        read_instruction_body(content, "SKILL.md")


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


def test_load_agents_instructions_uses_a_32_kibibyte_default_limit(tmp_path):
    """The default limit truncates oversized instructions without invalid UTF-8."""
    (tmp_path / "AGENTS.md").write_text("a" * 32767 + "€", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path).content

    assert instructions is not None
    assert instructions.startswith("a")
    assert instructions.endswith("[AGENTS.md truncated: instruction byte limit reached.]")
    assert len(instructions.encode("utf-8")) <= 32 * 1024


def test_load_agents_instructions_accepts_a_custom_byte_limit(tmp_path):
    """Callers can override the maximum encoded instruction size."""
    (tmp_path / "AGENTS.md").write_text("abc€", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path, max_bytes=4).content

    assert instructions == "abc"


def test_load_agents_instructions_exposes_source_provenance_and_truncation(tmp_path):
    """Structured loading reports exact source sizes and omitted content."""
    (tmp_path / "AGENTS.md").write_text("abc€", encoding="utf-8")

    loaded = load_agents_instructions(tmp_path, max_bytes=4)

    assert loaded.truncated is True
    assert loaded.max_bytes == 4
    assert loaded.sources[0].path == (tmp_path / "AGENTS.md").resolve()
    assert loaded.sources[0].size_bytes == 6
    assert loaded.sources[0].included_bytes == 3
    assert loaded.sources[0].truncated is True


def test_load_agents_instructions_records_fully_omitted_sources_and_content(tmp_path):
    """A consumed budget records later files and unrepresentable characters as omitted."""
    project = tmp_path / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "AGENTS.md").write_text("abcd", encoding="utf-8")
    (nested / "AGENTS.md").write_text("€", encoding="utf-8")

    loaded = load_agents_instructions(nested, max_bytes=4)

    assert loaded.content == "abcd"
    assert [source.included_bytes for source in loaded.sources] == [4, 0]
    assert loaded.truncated is True

    isolated = tmp_path / "isolated"
    isolated.mkdir()
    (isolated / "AGENTS.md").write_text("€", encoding="utf-8")
    tiny = load_agents_instructions(isolated, max_bytes=1)
    assert tiny.content is None
    assert tiny.sources[0].included_bytes == 0


def test_load_agents_instructions_accepts_a_custom_filename(tmp_path):
    """Callers can override the instruction filename."""
    (tmp_path / "CUSTOM.md").write_text("custom rules", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path, agents_filename="CUSTOM.md").content

    assert instructions == "custom rules"


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
