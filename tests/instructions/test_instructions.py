"""Tests for composing dynamic backend instructions."""

import os
from pathlib import Path

import pytest

from loop import (
    Agent,
    AgentIdentity,
    AgentInstructions,
    InstructionsManager,
    RuntimeEnvironment,
    Skill,
    SkillManager,
)

TEST_POLICY = AgentInstructions("Test policy.", "test", "tests")
TEST_AGENT = Agent(AgentIdentity("Test Agent"), TEST_POLICY)


def configured_manager(**kwargs) -> InstructionsManager:
    """Build contextual instruction management for the test agent definition."""
    return InstructionsManager(**kwargs)


def configured_discovered(directory: Path, **kwargs) -> InstructionsManager:
    """Discover contextual sources for the test agent definition."""
    return InstructionsManager.discover(directory, **kwargs)


def prepared(manager: InstructionsManager) -> str:
    """Prepare model-visible instructions for the explicit test agent."""
    return manager.prepare(TEST_AGENT).content


def test_manager_requires_an_explicit_agent_for_each_snapshot():
    """Preparing distinct agents never binds either agent to contextual state."""
    manager = InstructionsManager()
    other = Agent("Other", TEST_POLICY)

    first = manager.prepare(TEST_AGENT)
    second = manager.prepare(other)
    snapshot = manager.snapshot(TEST_AGENT)

    assert 'name="Test Agent"' in first.content
    assert 'name="Other"' in second.content
    assert 'name="Test Agent"' not in second.content
    assert snapshot.content == first.content
    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        InstructionsManager(max_bytes=2).snapshot(TEST_AGENT)


def write_skill(directory: Path, name: str, body: str = "Follow the workflow.") -> Skill:
    """Write and describe a minimal skill fixture."""
    directory.mkdir(parents=True)
    location = directory / "SKILL.md"
    location.write_text(
        f"---\nname: {name}\ndescription: Use the {name} workflow.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return Skill(name, f"Use the {name} workflow.", location)


def test_manager_combines_project_catalog_and_active_skills_in_stable_order(tmp_path):
    """Dynamic instructions preserve the static prefix and discovery-order skill bodies."""
    first = write_skill(tmp_path / "first", "first", "First instructions.")
    second = write_skill(tmp_path / "second", "second", "Second instructions.")
    manager = configured_manager(
        project_instructions="Project rules.",
        skill_manager=SkillManager([first, second]),
    )
    initial = prepared(manager)

    second_result = manager.activate_skill("second")
    first_result = manager.activate_skill("first")

    assert isinstance(initial, str)
    assert initial.startswith('<agent_identity name="Test Agent">')
    assert initial.index("</agent_instructions>") < initial.index("Project rules.")
    assert initial.index("Project rules.") < initial.index("<available_skills>")
    assert manager.max_bytes == 64 * 1024
    assert "First instructions." not in initial
    assert second_result["instructions_updated"] is True
    assert "instructions" not in second_result
    assert first_result["instructions_updated"] is True
    current = prepared(manager)
    assert current.startswith(initial)
    assert current.index("First instructions.") < current.index("Second instructions.")
    context = manager.list_skills()["instruction_context"]
    assert [section["kind"] for section in context["sections"]] == [
        "agents",
        "skill_catalog",
        "active_skill",
        "active_skill",
    ]
    assert len(context["digest"]) == 64


def test_manager_prioritizes_referenced_skill_metadata_without_activation(tmp_path):
    """A request-scoped catalog priority exposes metadata but leaves skills inactive."""
    first = write_skill(tmp_path / "first", "first", "First instructions.")
    second = write_skill(tmp_path / "second", "second", "Second instructions.")
    manager = configured_manager(skill_manager=SkillManager([first, second]))

    manager.set_skill_catalog_priorities(("second", "missing", "second"))
    snapshot = manager.prepare(TEST_AGENT)
    instructions = snapshot.content
    catalog = next(
        section.content for section in snapshot.sections if section.kind == "skill_catalog"
    )

    assert instructions.index("<name>second</name>") < instructions.index("<name>first</name>")
    assert catalog.index("<name>second</name>") < catalog.index("<name>first</name>")
    assert "Second instructions." not in instructions
    assert manager.active_skill_identities == []


def test_manager_rejects_catalog_priorities_that_exceed_the_instruction_budget(tmp_path):
    """Priority changes leave the fitting complete instruction document intact on UTF-8 overflow."""
    skills = [
        Skill("ascii", "a" * 6500, tmp_path / "ascii" / "SKILL.md"),
        Skill("unicode", "€" * 6500, tmp_path / "unicode" / "SKILL.md"),
    ]
    baseline = configured_manager(skill_manager=SkillManager(skills))
    manager = configured_manager(
        skill_manager=SkillManager(skills),
        max_bytes=len(prepared(baseline).encode("utf-8")),
    )
    generation = manager.generation

    accepted = manager.set_skill_catalog_priorities(("unicode",), TEST_AGENT)

    assert accepted is False
    assert "<name>ascii</name>" in manager.instructions
    assert "<name>unicode</name>" not in manager.instructions
    assert manager.generation == generation
    assert prepared(manager) == prepared(baseline)


def test_manager_refreshes_before_validating_catalog_priorities(tmp_path):
    """Priority validation rejects a refreshed document that would exceed its budget."""
    agents = tmp_path / "AGENTS.md"
    skills = [
        Skill("ascii", "a" * 6500, tmp_path / "ascii" / "SKILL.md"),
        Skill("unicode", "€" * 6500, tmp_path / "unicode" / "SKILL.md"),
    ]
    agents.write_text("Old rules.", encoding="utf-8")
    agents.write_text("New rules." * 20, encoding="utf-8")
    baseline = configured_discovered(tmp_path, skill_manager=SkillManager(skills))
    prioritized = configured_discovered(tmp_path, skill_manager=SkillManager(skills))
    assert prioritized.set_skill_catalog_priorities(("unicode",), TEST_AGENT) is True
    limit = len(prepared(prioritized).encode("utf-8")) - 1
    assert len(prepared(baseline).encode("utf-8")) <= limit

    agents.write_text("Old rules.", encoding="utf-8")
    manager = configured_discovered(
        tmp_path,
        skill_manager=SkillManager(skills),
        max_bytes=limit,
    )
    agents.write_text("New rules." * 20, encoding="utf-8")

    assert manager.set_skill_catalog_priorities(("unicode",), TEST_AGENT) is False
    assert "New rules." in manager.instructions
    assert "<name>ascii</name>" in manager.instructions
    assert "<name>unicode</name>" not in manager.instructions
    assert prepared(manager)


def test_manager_revalidates_unchanged_catalog_priorities_for_an_agent(tmp_path):
    """An unchanged priority set still rejects an agent document that exceeds its budget."""
    skills = [
        Skill("ascii", "a" * 6500, tmp_path / "ascii" / "SKILL.md"),
        Skill("unicode", "€" * 6500, tmp_path / "unicode" / "SKILL.md"),
    ]
    probe = configured_manager(skill_manager=SkillManager(skills))
    assert probe.set_skill_catalog_priorities(("unicode",)) is True
    manager = configured_manager(
        skill_manager=SkillManager(skills),
        max_bytes=len(probe.instructions.encode("utf-8")),
    )

    assert manager.set_skill_catalog_priorities(("unicode",)) is True
    assert manager.set_skill_catalog_priorities(("unicode",), TEST_AGENT) is False
    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        manager.prepare(TEST_AGENT)


def test_runtime_environment_is_budgeted_and_only_increment_on_change(tmp_path):
    """Runtime context participates in composition, generation, and the hard budget."""
    baseline_size = len(prepared(configured_manager()).encode("utf-8"))
    manager = configured_manager(max_bytes=baseline_size + 512)
    environment = RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    manager.set_runtime_environment(environment)
    generation = manager.generation

    manager.set_runtime_environment(environment)

    assert str(tmp_path) not in prepared(manager)
    assert "working_directory: /workspace" in prepared(manager)
    assert manager.generation == generation
    manager.set_runtime_environment(RuntimeEnvironment(tmp_path, tmp_path / ("x" * 900)))
    assert "temporary_directory: /tmp" in prepared(manager)


def test_virtual_paths_require_a_runtime_environment_for_workspace_relative_paths():
    """A static instruction manager cannot resolve workspace-relative model paths."""
    with pytest.raises(ValueError):
        configured_manager().virtual_paths.resolve("relative.txt")


def test_runtime_environment_exposes_only_virtual_roots(tmp_path):
    """Instructions and typed file paths use virtual roots without exposing host roots."""
    manager = configured_manager(
        runtime_environment=RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    )

    assert str(tmp_path) not in prepared(manager)
    assert "working_directory: /workspace" in prepared(manager)
    assert "temporary_directory: /tmp" in prepared(manager)
    assert manager.virtual_paths.resolve("/workspace/source.py") == str(tmp_path / "source.py")
    assert manager.virtual_paths.resolve("/tmp/output.txt") == str(
        tmp_path / "temporary/output.txt"
    )
    assert manager.virtual_paths.display(str(tmp_path / "source.py")) == "/workspace/source.py"
    assert (
        manager.virtual_paths.display(str(tmp_path / "temporary/output.txt")) == "/tmp/output.txt"
    )
    assert manager.virtual_paths.redact(f"Failed at {tmp_path}/source.py") == (
        "Failed at /workspace/source.py"
    )
    with pytest.raises(ValueError):
        manager.virtual_paths.resolve("scratch:/output.txt")


def test_runtime_environment_is_independent_from_observed_instruction_directory(tmp_path):
    """Path observations change instruction scope without changing the runtime directory."""
    initial = tmp_path / "initial"
    observed = tmp_path / "observed"
    initial.mkdir()
    observed.mkdir()
    manager = configured_manager(
        runtime_environment=RuntimeEnvironment(initial, tmp_path / "temporary"),
        max_bytes=4096,
    )

    manager.observe_path(observed, directory=True)
    manager.prepare(TEST_AGENT)

    assert "working_directory: /workspace" in prepared(manager)
    assert manager.working_directory == observed.resolve()
    assert manager.virtual_paths.resolve(".") == str(initial.resolve())
    sections = manager.list_skills()["instruction_context"]["sections"]
    assert [section["kind"] for section in sections] == ["runtime_environment"]

    manager.set_working_directory(observed)

    assert manager.virtual_paths.resolve(".") == str(observed.resolve())


def test_manager_discovers_project_instructions_and_skills(tmp_path):
    """Discovery loads applicable AGENTS.md content and repository skill metadata."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    write_skill(tmp_path / ".agents" / "skills" / "review", "review")

    manager = configured_discovered(tmp_path)

    document = prepared(manager)
    assert document.index("</agent_instructions>") < document.index("Project rules.")
    assert manager.skill_manager.count == 1


def test_prepared_sections_capture_relocatable_workspace_and_external_provenance(tmp_path):
    """Prepared snapshots attach relocatable references only to file-backed sections."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    internal = write_skill(workspace / ".agents" / "skills" / "local", "local")
    external = write_skill(tmp_path / "shared" / "external", "external")
    manager = configured_discovered(
        workspace,
        workspace_id="workspace-id",
        workspace_root=workspace,
        skill_manager=SkillManager([internal, external]),
    )
    manager.activate_skill("local")
    manager.activate_skill("external")

    sections = manager.prepare(TEST_AGENT).sections
    references = {
        section.source: section.reference for section in sections if section.reference is not None
    }
    project_reference = references[str((workspace / "AGENTS.md").resolve())]
    internal_reference = references[str(internal.location)]
    external_reference = references[str(external.location)]
    moved = tmp_path / "moved"

    assert project_reference.workspace_id == "workspace-id"
    assert project_reference.workspace_relative_path == "AGENTS.md"
    assert internal_reference.workspace_relative_path == ".agents/skills/local/SKILL.md"
    assert project_reference.resolve("workspace-id", moved) == moved / "AGENTS.md"
    assert project_reference.captured_content == "Project rules."
    assert external_reference.workspace_id is None
    assert external_reference.workspace_relative_path is None
    assert external_reference.resolve("workspace-id", moved) == external.location
    assert (
        next(section for section in sections if section.kind == "agent_identity").reference is None
    )


def test_manager_preserves_complete_sources_within_the_configured_budget(tmp_path):
    """The aggregate budget permits complete instructions beyond the old preview limit."""
    (tmp_path / "AGENTS.md").write_text("x" * 33_000, encoding="utf-8")

    manager = configured_discovered(tmp_path, max_bytes=40_000)
    assert "x" * 33_000 in manager.instructions
    with pytest.raises(ValueError, match="no instructions were truncated"):
        configured_discovered(tmp_path, max_bytes=32_000)


def test_discovery_refresh_preserves_an_explicit_skill_manager(tmp_path):
    """An explicitly supplied catalog remains stable while project instructions refresh."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules.", encoding="utf-8")
    skill = write_skill(tmp_path / "custom" / "review", "review")
    manager = configured_discovered(tmp_path, skill_manager=SkillManager([skill]))
    manager.activate_skill("review")

    agents.write_text("Second rules are longer.", encoding="utf-8")

    assert manager.refresh() is True
    assert manager.active_skill_identities == [("review", str(skill.location))]
    assert "Second rules are longer." in manager.instructions
    manager.invalidate()
    assert manager.refresh() is False

    agents.write_text("short", encoding="utf-8")
    baseline_size = len(prepared(configured_discovered(tmp_path)).encode("utf-8"))
    bounded = configured_discovered(
        tmp_path,
        skill_manager=SkillManager(),
        max_bytes=baseline_size + 10,
    )
    agents.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        bounded.prepare(TEST_AGENT)


def test_prepare_refreshes_changed_project_instructions_without_churning_stable_state(tmp_path):
    """Query preparation refreshes changed sources and preserves stable generations."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules.", encoding="utf-8")
    manager = configured_discovered(tmp_path)

    assert manager.refresh() is False
    assert manager.generation == 0

    agents.write_text("Second and longer rules.", encoding="utf-8")

    assert manager.refresh() is True
    assert prepared(manager).endswith("Second and longer rules.")
    assert manager.generation == 1
    assert manager.list_skills()["instruction_context"]["refresh_changes"] == [
        f"Changed instruction source '{agents.resolve()}'."
    ]
    assert manager.refresh() is False
    assert manager.generation == 1


def test_prepare_detects_content_changes_with_unchanged_file_metadata(tmp_path):
    """Content hashes detect rewrites even when size and modification time are preserved."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules", encoding="utf-8")
    manager = configured_discovered(tmp_path)
    original = agents.stat()

    agents.write_text("Other rules", encoding="utf-8")
    os.utime(agents, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert manager.refresh() is True
    assert prepared(manager).endswith("Other rules")


def test_observed_file_changes_scope_and_rediscovers_nested_sources(tmp_path):
    """A loaded file establishes its containing directory as the next instruction scope."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("Root rules.", encoding="utf-8")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("Nested rules.", encoding="utf-8")
    target = nested / "module.py"
    target.touch()
    manager = configured_discovered(tmp_path)

    manager.observe_path(target)
    manager.observe_path(target)

    assert manager.list_skills()["instruction_context"]["dirty"] is True
    assert manager.refresh() is True
    assert manager.working_directory == nested.resolve()
    assert prepared(manager).endswith("Root rules.\n\nNested rules.")
    changes = manager.list_skills()["instruction_context"]["refresh_changes"]
    assert changes[0] == f"Instruction scope changed to '{nested.resolve()}'."
    assert changes[1] == f"Added instruction source '{(nested / 'AGENTS.md').resolve()}'."


def test_refresh_preserves_same_skill_identity_and_reloads_changed_body(tmp_path):
    """Refresh keeps activation for the same canonical skill while loading its new body."""
    (tmp_path / ".git").mkdir()
    location = write_skill(tmp_path / ".agents" / "skills" / "review", "review", "First.")
    manager = configured_discovered(tmp_path)
    manager.activate_skill("review")

    location.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\nSecond body is longer.\n",
        encoding="utf-8",
    )

    assert manager.refresh() is True
    assert "Second body is longer." in manager.instructions
    assert manager.active_skill_identities == [("review", str(location.location.resolve()))]
    assert manager.list_skills()["instruction_context"]["refresh_changes"] == [
        "Skill catalog or activated skill instructions changed."
    ]


def test_refresh_deactivates_a_skill_replaced_by_a_closer_definition(tmp_path):
    """A newly shadowing definition never inherits activation from the old skill."""
    (tmp_path / ".git").mkdir()
    root_skill = write_skill(tmp_path / ".agents" / "skills" / "review", "review")
    nested = tmp_path / "src"
    nested.mkdir()
    manager = configured_discovered(tmp_path)
    manager.activate_skill("review")
    write_skill(nested / ".agents" / "skills" / "review", "review", "Closer body.")

    manager.observe_path(nested, directory=True)

    assert manager.refresh() is True
    assert manager.active_skill_identities == []
    assert "Closer body." not in manager.instructions
    assert str(root_skill.location.resolve()) not in {
        str(skill.location) for skill in manager.skill_manager.activated_skills
    }
    assert "removed or shadowed" in manager.list_skills()["instruction_context"]["diagnostics"][0]


def test_reactivate_skills_restores_only_matching_current_identities(tmp_path):
    """Restoration activates exact current definitions and ignores missing or stale locations."""
    skill = write_skill(tmp_path / "skills" / "review", "review")
    manager = configured_manager(skill_manager=SkillManager([skill]))

    results = manager.reactivate_skills(
        [
            ("review", str(tmp_path / "stale" / "SKILL.md")),
            ("missing", str(tmp_path / "missing" / "SKILL.md")),
            ("review", str(skill.location)),
        ]
    )

    assert [result["status"] for result in results] == ["activated"]
    assert manager.active_skill_identities == [("review", str(skill.location))]
    assert manager.list_skills()["instruction_context"]["diagnostics"] == [
        "Could not restore 'review': its definition was removed or shadowed.",
        "Could not restore 'missing': its definition was removed or shadowed.",
    ]


def test_reactivation_preserves_idempotence_and_aggregate_budget(tmp_path):
    """Session restoration reports unchanged skills and rejects oversized aggregate instructions."""
    skill = write_skill(tmp_path / "skills" / "large", "large", "x" * 100)
    skill_manager = SkillManager([skill])
    baseline = configured_manager(skill_manager=skill_manager).instructions
    manager = configured_manager(skill_manager=skill_manager, max_bytes=len(baseline) + 10)
    identity = [("large", str(skill.location))]

    rejected = manager.reactivate_skills(identity)

    assert rejected[0].code == "skill.instruction_budget_exceeded"
    assert manager.active_skill_identities == []

    roomy = configured_manager(skill_manager=SkillManager([skill]))
    roomy.activate_skill("large")
    repeated = roomy.reactivate_skills(identity)

    assert repeated[0]["instructions_updated"] is False

    externally_activated = SkillManager([skill])
    constrained = configured_manager(
        skill_manager=externally_activated,
        max_bytes=len(baseline) + 10,
    )
    externally_activated.activate("large")

    repeated_rejection = constrained.reactivate_skills(identity)
    direct_rejection = constrained.activate_skill("large")

    assert repeated_rejection[0].code == "skill.instruction_budget_exceeded"
    assert direct_rejection.code == "skill.instruction_budget_exceeded"
    assert externally_activated.is_active("large") is True


def test_resource_facades_prepare_before_delegating(tmp_path):
    """Compatibility resource façades refresh context before low-level skill reads."""
    skill = write_skill(tmp_path / "skills" / "resourceful", "resourceful")
    reference = skill.location.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("Guide.", encoding="utf-8")
    manager = configured_manager(skill_manager=SkillManager([skill]))
    manager.activate_skill("resourceful")

    assert manager.list_skill_resources("resourceful")["resources"] == [
        {"path": "references/guide.md", "size_bytes": 6}
    ]
    assert manager.read_skill_resource("resourceful", "references/guide.md")["content"] == "Guide."


def test_refresh_deactivates_invalid_and_oversized_active_skills(tmp_path, monkeypatch):
    """Refresh safely drops active bodies that fail validation or the instruction budget."""
    (tmp_path / ".git").mkdir()
    skill = write_skill(tmp_path / ".agents" / "skills" / "review", "review", "Short.")
    baseline = configured_discovered(tmp_path).instructions
    manager = configured_discovered(tmp_path, max_bytes=len(baseline.encode()) + 250)
    assert manager.activate_skill("review")["status"] == "activated"

    skill.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\n" + "x" * 1_000,
        encoding="utf-8",
    )

    assert manager.refresh() is True
    assert manager.active_skill_identities == []
    assert (
        "instruction budget exceeded"
        in manager.list_skills()["instruction_context"]["diagnostics"][0]
    )

    skill.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\nShort again.\n",
        encoding="utf-8",
    )
    manager.prepare(TEST_AGENT)
    manager.activate_skill("review")
    original_manager = manager.skill_manager
    original_activate = SkillManager.activate

    def fail_refreshed_activation(self, name):
        if self is original_manager:
            return original_activate(self, name)
        raise ValueError("changed skill is invalid")

    monkeypatch.setattr(SkillManager, "activate", fail_refreshed_activation)
    manager.invalidate()

    assert manager.refresh() is True
    assert (
        "changed skill is invalid" in manager.list_skills()["instruction_context"]["diagnostics"][0]
    )


def test_invalidation_is_selective_and_oversized_base_refresh_is_atomic(tmp_path, monkeypatch):
    """Only instruction sources invalidate, and an invalid candidate preserves prior content."""
    limit = len(prepared(configured_discovered(tmp_path)).encode("utf-8")) + 20
    manager = configured_discovered(
        tmp_path,
        max_bytes=limit,
    )
    manager.invalidate(tmp_path / "ordinary.py")
    assert manager.list_skills()["instruction_context"]["dirty"] is False

    manager.invalidate()
    assert manager.refresh() is True
    assert manager.generation == 1

    previous = manager.instructions
    (tmp_path / "AGENTS.md").write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        manager.prepare(TEST_AGENT)
    assert manager.instructions == previous

    static = configured_manager()
    static.invalidate()
    assert static.refresh() is False

    clean = tmp_path / "clean"
    clean.mkdir()
    racing = configured_discovered(clean)
    missing = clean / "disappeared" / "AGENTS.md"
    monkeypatch.setattr(
        "loop.instructions.instructions.get_agents_files",
        lambda working_directory, _filenames: [missing],
    )
    racing.invalidate()
    assert racing.refresh() is True


def test_manager_rejects_invalid_and_oversized_initial_limits():
    """Construction requires a positive limit containing the initial document."""
    for limit in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            configured_manager(max_bytes=limit)

    with pytest.raises(ValueError, match="Prepared instructions exceed"):
        prepared(configured_manager(project_instructions="too large", max_bytes=2))


def test_activation_is_idempotent_and_unknown_skills_remain_errors(tmp_path):
    """Repeated activation reports no update while missing skills do not change instructions."""
    skill = write_skill(tmp_path / "example", "example")
    manager = configured_manager(skill_manager=SkillManager([skill]))

    first = manager.activate_skill("example")
    instructions = manager.instructions
    second = manager.activate_skill("example")
    missing = manager.activate_skill("missing")

    assert first["instructions_updated"] is True
    assert second["instructions_updated"] is False
    assert manager.instructions == instructions
    assert missing.code == "skill.unknown"
    assert str(tmp_path) not in instructions
    assert '<skill name="example">' in instructions
    assert " root=" not in instructions


def test_activation_rolls_back_when_complete_instructions_exceed_budget(tmp_path):
    """An oversized skill is rejected atomically without partially changing instructions."""
    skill = write_skill(tmp_path / "large", "large", "x" * 100)
    skill_manager = SkillManager([skill])
    baseline = configured_manager(skill_manager=skill_manager).instructions
    manager = configured_manager(skill_manager=skill_manager, max_bytes=len(baseline) + 10)

    result = manager.activate_skill("large")

    assert result.code == "skill.instruction_budget_exceeded"
    assert result.metadata["required_bytes"] > result.metadata["max_bytes"]
    assert manager.instructions == baseline
    assert skill_manager.activated == 0


def test_deactivation_is_idempotent_and_unknown_skills_remain_errors(tmp_path):
    """Deactivation reports changes while missing skills do not alter instructions."""
    skill = write_skill(tmp_path / "example", "example")
    manager = configured_manager(skill_manager=SkillManager([skill]))
    manager.activate_skill("example")

    result = manager.deactivate_skill("example")
    instructions = manager.instructions
    repeated = manager.deactivate_skill("example")
    missing = manager.deactivate_skill("missing")

    assert result["status"] == "deactivated"
    assert result["instructions_updated"] is True
    assert repeated["instructions_updated"] is False
    assert isinstance(instructions, str)
    assert "Follow the workflow." not in instructions
    assert manager.instructions == instructions
    assert missing.code == "skill.unknown"

    empty = manager.deactivate_all_skills()
    assert empty["deactivated"] == 0
    assert empty["instructions_updated"] is False
