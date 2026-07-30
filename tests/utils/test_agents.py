"""Tests for AGENTS.md instruction discovery."""

import pytest

from loop.utils.agents import load_agents_instructions


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

    instructions = load_agents_instructions(working_directory)

    assert instructions == "project rules\n\nsource rules\n\nfeature rules"


def test_load_agents_instructions_checks_only_working_directory_without_project(tmp_path):
    """A directory outside a Git project receives only its own instructions."""
    parent = tmp_path / "parent"
    working_directory = parent / "child"
    working_directory.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("parent rules", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("local rules", encoding="utf-8")

    assert load_agents_instructions(working_directory) == "local rules"


def test_load_agents_instructions_accepts_a_string_path(tmp_path):
    """String working directories are accepted during instruction discovery."""
    (tmp_path / "AGENTS.md").write_text("local rules", encoding="utf-8")

    assert load_agents_instructions(str(tmp_path)) == "local rules"


def test_load_agents_instructions_skips_empty_files(tmp_path):
    """Empty instruction files do not contribute to the result."""
    project = tmp_path / "project"
    working_directory = project / "src"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "AGENTS.md").write_text("  \n", encoding="utf-8")
    (working_directory / "AGENTS.md").write_text("source rules", encoding="utf-8")

    assert load_agents_instructions(working_directory) == "source rules"


def test_load_agents_instructions_returns_none_without_guidance(tmp_path):
    """Missing and empty instruction chains return no guidance."""
    assert load_agents_instructions(tmp_path) is None


def test_load_agents_instructions_uses_a_32_kibibyte_default_limit(tmp_path):
    """The default limit truncates oversized instructions without invalid UTF-8."""
    (tmp_path / "AGENTS.md").write_text("a" * 32767 + "€", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path)

    assert instructions == "a" * 32767
    assert len(instructions.encode("utf-8")) <= 32 * 1024


def test_load_agents_instructions_accepts_a_custom_byte_limit(tmp_path):
    """Callers can override the maximum encoded instruction size."""
    (tmp_path / "AGENTS.md").write_text("abc€", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path, max_bytes=4)

    assert instructions == "abc"


def test_load_agents_instructions_accepts_a_custom_filename(tmp_path):
    """Callers can override the instruction filename."""
    (tmp_path / "CUSTOM.md").write_text("custom rules", encoding="utf-8")

    instructions = load_agents_instructions(tmp_path, agents_filename="CUSTOM.md")

    assert instructions == "custom rules"


def test_load_agents_instructions_requires_utf8(tmp_path):
    """Instruction files must contain valid UTF-8 text."""
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        load_agents_instructions(tmp_path)
