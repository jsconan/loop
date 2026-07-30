"""Tests for progressive Agent Skill discovery and activation."""

from pathlib import Path

import pytest

from loop.skills import SkillManager


def write_skill(directory: Path, name: str, description: str, body: str = "Instructions") -> Path:
    """Write a minimal skill fixture and return its SKILL.md path."""
    directory.mkdir(parents=True)
    location = directory / "SKILL.md"
    location.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return location


def test_discovery_reads_metadata_and_activation_lazily_loads_and_caches_body(
    tmp_path, monkeypatch
):
    """Discovery omits the body while activation reads complete instructions only once."""
    location = write_skill(
        tmp_path / "skills" / "review",
        "review",
        "Review changes & report issues.",
        "Follow the review workflow.",
    )
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])
    original_read_text = Path.read_text
    reads = []

    def tracked_read_text(path, *args, **kwargs):
        reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read_text)

    assert manager.skills[0].description == "Review changes & report issues."
    assert "Follow the review workflow." not in manager.catalog()
    first = manager.activate("review")
    second = manager.activate("review")

    assert first["instructions"] == "Follow the review workflow."
    assert first["skill_root"] == str(location.parent.resolve())
    assert second["status"] == "activated"
    assert reads == [location.resolve()]


def test_discovery_skips_invalid_skills_and_reports_diagnostics(tmp_path):
    """Malformed metadata does not prevent valid skills from loading."""
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "valid", "valid", "A valid skill.")
    invalid = skills_directory / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("No frontmatter", encoding="utf-8")

    manager = SkillManager.discover(tmp_path, [skills_directory])

    assert [skill.name for skill in manager.skills] == ["valid"]
    assert "must start with YAML frontmatter" in manager.diagnostics[0]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("---\nname: broken\n", "frontmatter is not terminated"),
        ("---\n- item\n---\n", "frontmatter must be a mapping"),
        ("---\nname: 1\ndescription: Valid\n---\n", "non-empty name"),
        ("---\nname: '  '\ndescription: Valid\n---\n", "non-empty name"),
        ("---\nname: valid\ndescription: 1\n---\n", "non-empty description"),
        ("---\nname: valid\ndescription: '  '\n---\n", "non-empty description"),
    ],
)
def test_discovery_reports_each_invalid_frontmatter_shape(tmp_path, content, message):
    """Every invalid metadata shape is skipped with a useful diagnostic."""
    location = tmp_path / "skills" / "broken" / "SKILL.md"
    location.parent.mkdir(parents=True)
    location.write_text(content, encoding="utf-8")

    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])

    assert manager.skills == ()
    assert message in manager.diagnostics[0]


def test_default_discovery_orders_project_scopes_before_user_skills(tmp_path, monkeypatch):
    """Default discovery searches every project scope from root to working directory, then home."""
    project = tmp_path / "project"
    working_directory = project / "packages" / "app"
    home = tmp_path / "home"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    write_skill(project / ".agents" / "skills" / "root", "root", "Root skill.")
    write_skill(
        working_directory / ".agents" / "skills" / "local", "local", "Local skill."
    )
    write_skill(home / ".agents" / "skills" / "user", "user", "User skill.")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    manager = SkillManager.discover(working_directory)

    assert [skill.name for skill in manager.skills] == ["root", "local", "user"]


def test_list_activation_errors_and_duplicate_names_are_structured(tmp_path):
    """The manager lists metadata and reports unknown or ambiguous activation requests."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_skill(first / "same", "same", "First definition.")
    write_skill(second / "same", "same", "Second definition.")
    manager = SkillManager.discover(tmp_path, [first, second])

    listing = manager.list()

    assert len(listing["skills"]) == 2
    assert not listing["skills"][0]["activated"]
    assert manager.activate("missing")["error"] == "unknown_skill"
    assert manager.activate("same")["error"] == "ambiguous_skill"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("Instructions without frontmatter", "must start with YAML frontmatter"),
        ("---\nname: changed", "frontmatter is not terminated"),
    ],
)
def test_activation_reports_skill_files_that_become_malformed(
    tmp_path, replacement, message
):
    """Activation validates the complete file even when valid metadata was discovered earlier."""
    location = write_skill(tmp_path / "skills" / "changing", "changing", "Changes later.")
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])
    location.write_text(replacement, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        manager.activate("changing")


def test_catalog_is_metadata_only_escaped_and_bounded(tmp_path):
    """Catalog output is safe structured text that honors its character ceiling."""
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "one", "one", "Use for <special> work.")
    write_skill(skills_directory / "two", "two", "Use for other work.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    catalog = manager.catalog(max_chars=600)

    assert len(catalog) <= 600
    assert "&lt;special&gt;" in catalog
    assert "Instructions" not in catalog


def test_catalog_returns_none_without_skills_and_warns_when_entries_are_omitted(tmp_path):
    """An empty catalog is absent and a constrained catalog reports omitted entries."""
    assert SkillManager().catalog() is None
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "one", "one", "First skill.")
    write_skill(skills_directory / "two", "two", "Second skill.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    catalog = manager.catalog(max_chars=180)

    assert len(catalog) <= 180
    assert "2 skill(s) omitted by catalog limit" in catalog
