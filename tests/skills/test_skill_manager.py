"""Tests for progressive Agent Skill discovery and activation."""

from pathlib import Path

import pytest

from loop import Skill, SkillManager


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

    assert first["skill_root"] == str(location.parent.resolve())
    assert "instructions" not in first
    assert manager.activated_instructions == ((manager.skills[0], "Follow the review workflow."),)
    assert second["status"] == "activated"
    assert reads == [location.resolve()]


def test_activated_skills_returns_immutable_discovery_ordered_snapshot(tmp_path):
    """Activated skills are returned as an immutable snapshot in discovery order."""
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "first", "first", "First skill.")
    write_skill(skills_directory / "second", "second", "Second skill.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    assert manager.activated_skills == ()

    manager.activate("second")
    manager.activate("first")

    assert tuple(skill.name for skill in manager.activated_skills) == ("first", "second")
    assert tuple(skill.name for skill, _ in manager.activated_instructions) == ("first", "second")
    assert tuple(body for _, body in manager.activated_instructions) == (
        "Instructions",
        "Instructions",
    )
    assert manager.active_identities == (
        ("first", str((skills_directory / "first" / "SKILL.md").resolve())),
        ("second", str((skills_directory / "second" / "SKILL.md").resolve())),
    )


def test_counts_and_deactivate_manage_activated_skill_lifecycle(tmp_path):
    """Counts reflect discovery and deactivation releases cached instructions."""
    skills_directory = tmp_path / "skills"
    location = write_skill(skills_directory / "first", "first", "First skill.")
    write_skill(skills_directory / "second", "second", "Second skill.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    assert manager.count == 2
    assert manager.activated == 0
    assert manager.is_active("first") is False
    assert manager.is_active("missing") is False

    assert manager.activate("first")["instructions_updated"] is True
    assert manager.activate("first")["instructions_updated"] is False
    assert manager.is_active("first") is True
    assert manager.activated == 1
    assert manager.deactivate("first")["instructions_updated"] is True
    assert manager.deactivate("first")["instructions_updated"] is False
    assert manager.activated == 0
    assert not manager.list()["skills"][0]["activated"]

    location.write_text(
        "---\nname: first\ndescription: First skill.\n---\n\nNew instructions.\n",
        encoding="utf-8",
    )
    result = manager.activate("first")
    assert "instructions" not in result
    assert manager.activated_instructions == ((manager.skills[0], "New instructions."),)


def test_deactivate_reports_unknown_names_and_deactivate_all_clears_every_skill(tmp_path):
    """Deactivation rejects unknown names and bulk deactivation clears cached instructions."""
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "unique", "unique", "Unique definition.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    manager.activate("unique")

    assert manager.deactivate("missing")["error"] == "unknown_skill"
    assert manager.deactivate("unique")["status"] == "deactivated"
    assert manager.deactivate("unique")["status"] == "deactivated"

    manager.activate("unique")
    assert manager.deactivate_all() == 1
    assert manager.deactivate_all() == 0

    assert manager.activated == 0
    assert manager.activated_skills == ()


def test_restore_owns_identity_matching_and_lifecycle_diagnostics(tmp_path):
    """Restoration accepts exact identities and diagnoses missing or shadowed definitions."""
    skills_directory = tmp_path / "skills"
    location = write_skill(skills_directory / "review", "review", "Review changes.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    results = manager.restore(
        [
            ("review", str(tmp_path / "stale" / "SKILL.md")),
            ("missing", str(tmp_path / "missing" / "SKILL.md")),
            ("review", str(location.resolve())),
        ]
    )

    assert [result["name"] for result in results] == ["review"]
    assert manager.is_active("review") is True
    assert manager.lifecycle_diagnostics == (
        "Could not restore 'review': its definition was removed or shadowed.",
        "Could not restore 'missing': its definition was removed or shadowed.",
    )


def test_rediscovery_preserves_only_exact_valid_active_definitions(tmp_path, monkeypatch):
    """Rediscovery reloads exact identities and contains refresh failures as diagnostics."""
    (tmp_path / ".git").mkdir()
    location = write_skill(
        tmp_path / ".agents" / "skills" / "review",
        "review",
        "Review changes.",
    )
    identity = ("review", str(location.resolve()))

    refreshed = SkillManager.rediscover(tmp_path, [identity, ("missing", "/missing/SKILL.md")])

    assert refreshed.active_identities == (identity,)
    assert refreshed.lifecycle_diagnostics == (
        "Deactivated 'missing' during refresh: its definition was removed or shadowed.",
    )

    def fail_activation(name):
        raise ValueError("changed skill is invalid")

    invalid = SkillManager.discover(tmp_path)
    monkeypatch.setattr(invalid, "activate", fail_activation)
    invalid.restore([identity], refresh=True)

    assert invalid.active_identities == ()
    assert "changed skill is invalid" in invalid.lifecycle_diagnostics[0]

    restoring = SkillManager.discover(tmp_path)
    monkeypatch.setattr(restoring, "activate", fail_activation)
    with pytest.raises(ValueError, match="changed skill is invalid"):
        restoring.restore([identity])


def test_resources_are_listed_and_loaded_only_for_active_skills(tmp_path):
    """References, scripts, and assets load on demand without entering skill instructions."""
    skill_root = tmp_path / "skills" / "resourceful"
    write_skill(skill_root, "resourceful", "Uses resources.")
    reference = skill_root / "references" / "guide.md"
    script = skill_root / "scripts" / "run.py"
    asset = skill_root / "assets" / "image.bin"
    reference.parent.mkdir()
    script.parent.mkdir()
    asset.parent.mkdir()
    reference.write_text("Detailed guide.", encoding="utf-8")
    script.write_text("print('ok')", encoding="utf-8")
    asset.write_bytes(b"\xff\x00")
    outside = tmp_path / "outside.txt"
    outside.write_text("Outside.", encoding="utf-8")
    (reference.parent / "outside.txt").symlink_to(outside)
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])

    assert manager.list_resources("missing")["error"] == "unknown_skill"
    assert manager.list_resources("resourceful")["error"] == "skill_not_active"
    manager.activate("resourceful")

    listed = manager.list_resources("resourceful")
    assert [resource["path"] for resource in listed["resources"]] == [
        "references/guide.md",
        "scripts/run.py",
        "assets/image.bin",
    ]
    text = manager.read_resource("resourceful", "references/guide.md")
    binary = manager.read_resource("resourceful", "assets/image.bin")
    assert text["content"] == "Detailed guide."
    assert text["encoding"] == "utf-8"
    assert binary["content"] == "/wA="
    assert binary["encoding"] == "base64"
    assert "Detailed guide." not in manager.activated_instructions[0][1]


def test_resource_loading_rejects_escapes_and_bounds_oversized_content(tmp_path):
    """Resource loading confines paths and returns resumable oversized content."""
    skill_root = tmp_path / "skills" / "safe"
    write_skill(skill_root, "safe", "Safe resources.")
    assets = skill_root / "assets"
    assets.mkdir()
    (assets / "large.bin").write_bytes(b"x" * (64 * 1024 + 1))
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])
    manager.activate("safe")

    assert manager.read_resource("missing", "assets/a")["error"] == "unknown_skill"
    assert manager.read_resource("safe", "../SKILL.md")["error"] == "invalid_skill_resource"
    assert manager.read_resource("safe", "assets/missing")["error"] == "invalid_skill_resource"
    oversized = manager.read_resource("safe", "assets/large.bin")
    assert oversized["size_bytes"] == 64 * 1024 + 1
    assert oversized["included_bytes"] == 16 * 1024
    assert oversized["truncated"] is True
    assert oversized["next_start_byte"] == 16 * 1024


def test_binary_resource_loading_is_bounded_and_validates_ranges(tmp_path):
    """Binary resources use bounded base64 pages and reject incompatible line access."""
    skill_root = tmp_path / "skills" / "binary"
    write_skill(skill_root, "binary", "Binary resources.")
    assets = skill_root / "assets"
    assets.mkdir()
    payload = b"\0" + b"x" * (16 * 1024)
    (assets / "payload.bin").write_bytes(payload)
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])
    manager.activate("binary")

    first = manager.read_resource("binary", "assets/payload.bin")
    second = manager.read_resource(
        "binary",
        "assets/payload.bin",
        start_byte=first["next_start_byte"],
        start_line=None,
    )

    assert first["encoding"] == "base64"
    assert first["truncated"] is True
    assert first["next_start_byte"] == 12 * 1024
    assert second["included_bytes"] == 4 * 1024 + 1
    with pytest.raises(ValueError, match="binary"):
        manager.read_resource("binary", "assets/payload.bin", start_line=2)
    with pytest.raises(ValueError, match="start_byte"):
        manager.read_resource(
            "binary", "assets/payload.bin", start_byte=-1, start_line=None
        )


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


def test_default_discovery_prefers_the_closest_project_scope_then_user_skills(
    tmp_path, monkeypatch
):
    """Default discovery gives closer project scopes precedence over broader and user scopes."""
    project = tmp_path / "project"
    working_directory = project / "packages" / "app"
    home = tmp_path / "home"
    working_directory.mkdir(parents=True)
    (project / ".git").mkdir()
    root = write_skill(project / ".agents" / "skills" / "root", "root", "Root skill.")
    local = write_skill(working_directory / ".agents" / "skills" / "local", "local", "Local skill.")
    write_skill(home / ".agents" / "skills" / "user", "user", "User skill.")
    write_skill(project / ".agents" / "skills" / "shared", "shared", "Root definition.")
    winner = write_skill(
        working_directory / ".agents" / "skills" / "shared", "shared", "Local definition."
    )
    shadowed = write_skill(home / ".agents" / "skills" / "shared", "shared", "User definition.")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    manager = SkillManager.discover(working_directory)

    assert [skill.name for skill in manager.skills] == ["local", "shared", "root", "user"]
    assert manager.activate("shared")["location"] == str(winner.resolve())
    assert any(str(shadowed.resolve()) in diagnostic for diagnostic in manager.diagnostics)
    assert all(str(local.resolve()) not in diagnostic for diagnostic in manager.diagnostics)
    assert all(str(root.resolve()) not in diagnostic for diagnostic in manager.diagnostics)


def test_explicit_directory_order_resolves_duplicate_names_and_reports_shadowing(tmp_path):
    """The first explicit directory wins while shadowed definitions remain diagnosable."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    winner = write_skill(first / "same", "same", "First definition.", "First instructions.")
    shadowed = write_skill(second / "same", "same", "Second definition.")
    manager = SkillManager.discover(tmp_path, [first, second])

    listing = manager.list()

    assert len(listing["skills"]) == 1
    assert not listing["skills"][0]["activated"]
    assert manager.activate("missing")["error"] == "unknown_skill"
    result = manager.activate("same")
    assert "instructions" not in result
    assert manager.activated_instructions == ((manager.skills[0], "First instructions."),)
    assert manager.deactivate("same")["location"] == str(winner.resolve())
    assert str(shadowed.resolve()) in listing["diagnostics"][0]
    assert listing["diagnostics"][0].endswith(f"overridden by '{winner.resolve()}'.")
    assert manager.catalog().count("<name>same</name>") == 1


def test_constructor_enforces_unique_names_in_input_order(tmp_path):
    """Directly injected skills use the same first-definition-wins invariant."""
    first = Skill("same", "First definition.", tmp_path / "first" / "SKILL.md")
    second = Skill("same", "Second definition.", tmp_path / "second" / "SKILL.md")

    manager = SkillManager([first, second], ["Existing diagnostic."])

    assert manager.skills == (first,)
    assert manager.count == 1
    assert manager.diagnostics[0] == "Existing diagnostic."
    assert str(second.location) in manager.diagnostics[1]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("Instructions without frontmatter", "must start with YAML frontmatter"),
        ("---\nname: changed", "frontmatter is not terminated"),
    ],
)
def test_activation_reports_skill_files_that_become_malformed(tmp_path, replacement, message):
    """Activation validates the complete file even when valid metadata was discovered earlier."""
    location = write_skill(tmp_path / "skills" / "changing", "changing", "Changes later.")
    manager = SkillManager.discover(tmp_path, [tmp_path / "skills"])
    location.write_text(replacement, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        manager.activate("changing")


def test_catalog_is_metadata_only_escaped_and_bounded(tmp_path):
    """Catalog output contains only bounded escaped discovery metadata."""
    skills_directory = tmp_path / "skills"
    write_skill(skills_directory / "one", "one", "Use for <special> work.")
    write_skill(skills_directory / "two", "two", "Use for other work.")
    manager = SkillManager.discover(tmp_path, [skills_directory])

    catalog = manager.catalog(max_chars=600)

    assert len(catalog) <= 600
    assert "&lt;special&gt;" in catalog
    assert "Instructions" not in catalog
    assert str(skills_directory) not in catalog
    assert "<location>" not in catalog


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
