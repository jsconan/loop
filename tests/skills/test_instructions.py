"""Tests for composing dynamic backend instructions."""

import os
from pathlib import Path

import pytest

from loop import (
    AgentPolicy,
    InstructionsManager,
    RuntimeEnvironment,
    Skill,
    SkillManager,
)

TEST_POLICY = AgentPolicy("Test policy.", "test", "tests")


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
    manager = InstructionsManager(
        project_instructions="Project rules.",
        skill_manager=SkillManager([first, second]),
    )
    initial = manager.instructions

    second_result = manager.activate_skill("second")
    first_result = manager.activate_skill("first")

    assert isinstance(initial, str)
    assert manager.agent_policy == AgentPolicy.default()
    assert initial.startswith('<agent_policy version="1"')
    assert initial.index("</agent_policy>") < initial.index("Project rules.")
    assert initial.index("Project rules.") < initial.index("<available_skills>")
    assert manager.max_bytes == 64 * 1024
    assert "First instructions." not in initial
    assert second_result["instructions_updated"] is True
    assert "instructions" not in second_result
    assert first_result["instructions_updated"] is True
    assert manager.instructions.startswith(initial)
    assert manager.instructions.index("First instructions.") < manager.instructions.index(
        "Second instructions."
    )
    context = manager.list_skills()["instruction_context"]
    assert [section["kind"] for section in context["sections"]] == [
        "agent_policy",
        "agents",
        "skill_catalog",
        "active_skill",
        "active_skill",
    ]
    assert len(context["digest"]) == 64


def test_runtime_environment_is_budgeted_and_only_increment_on_change(tmp_path):
    """Runtime context participates in composition, generation, and the hard budget."""
    baseline_size = len(InstructionsManager(agent_policy=TEST_POLICY).instructions.encode("utf-8"))
    manager = InstructionsManager(agent_policy=TEST_POLICY, max_bytes=baseline_size + 512)
    environment = RuntimeEnvironment(tmp_path, tmp_path / "temporary")
    manager.set_runtime_environment(environment)
    generation = manager.generation

    manager.set_runtime_environment(environment)

    assert manager.instructions is not None
    assert str(tmp_path) in manager.instructions
    assert manager.generation == generation
    with pytest.raises(ValueError, match="Runtime environment exceeds"):
        manager.set_runtime_environment(RuntimeEnvironment(tmp_path, tmp_path / ("x" * 900)))


def test_runtime_environment_tracks_the_observed_instruction_directory(tmp_path):
    """Runtime workspace guidance stays aligned with the active instruction scope."""
    initial = tmp_path / "initial"
    observed = tmp_path / "observed"
    initial.mkdir()
    observed.mkdir()
    manager = InstructionsManager(
        runtime_environment=RuntimeEnvironment(initial, tmp_path / "temporary"),
        max_bytes=4096,
    )

    manager.observe_path(observed, directory=True)
    manager.prepare()

    assert manager.instructions is not None
    assert f"working_directory: {observed.resolve()}" in manager.instructions
    sections = manager.list_skills()["instruction_context"]["sections"]
    assert [section["kind"] for section in sections] == [
        "agent_policy",
        "runtime_environment",
    ]


def test_manager_discovers_project_instructions_and_skills(tmp_path):
    """Discovery loads applicable AGENTS.md content and repository skill metadata."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    write_skill(tmp_path / ".agents" / "skills" / "review", "review")

    manager = InstructionsManager.discover(tmp_path)

    assert manager.instructions.index("</agent_policy>") < manager.instructions.index(
        "Project rules."
    )
    assert manager.skill_manager.count == 1


def test_manager_reports_truncated_project_instruction_sources(tmp_path):
    """Discovery exposes bounded source sizes and a persistent truncation diagnostic."""
    (tmp_path / "AGENTS.md").write_text("x" * 33_000, encoding="utf-8")

    manager = InstructionsManager.discover(tmp_path, max_bytes=40_000)
    context = manager.list_skills()["instruction_context"]

    assert context["diagnostics"] == [
        "Agent instructions truncated at 32768 bytes; 288 source byte(s) omitted."
    ]
    assert context["sources"][0]["size_bytes"] == 33_000


def test_discovery_refresh_preserves_an_explicit_skill_manager(tmp_path):
    """An explicitly supplied catalog remains stable while project instructions refresh."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules.", encoding="utf-8")
    skill = write_skill(tmp_path / "custom" / "review", "review")
    manager = InstructionsManager.discover(tmp_path, skill_manager=SkillManager([skill]))
    manager.activate_skill("review")

    agents.write_text("Second rules are longer.", encoding="utf-8")

    assert manager.prepare() is True
    assert manager.active_skill_identities == [("review", str(skill.location))]
    assert "Second rules are longer." in manager.instructions
    manager.invalidate()
    assert manager.prepare() is False

    agents.write_text("short", encoding="utf-8")
    bounded = InstructionsManager.discover(
        tmp_path,
        agent_policy=TEST_POLICY,
        skill_manager=SkillManager(),
        max_bytes=len(TEST_POLICY.render().encode("utf-8")) + 10,
    )
    agents.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="Refreshed base instructions exceed"):
        bounded.prepare()


def test_prepare_refreshes_changed_project_instructions_without_churning_stable_state(tmp_path):
    """Query preparation refreshes changed sources and preserves stable generations."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules.", encoding="utf-8")
    manager = InstructionsManager.discover(tmp_path)

    assert manager.prepare() is False
    assert manager.generation == 0

    agents.write_text("Second and longer rules.", encoding="utf-8")

    assert manager.prepare() is True
    assert manager.instructions.endswith("Second and longer rules.")
    assert manager.generation == 1
    assert manager.list_skills()["instruction_context"]["refresh_changes"] == [
        f"Changed instruction source '{agents.resolve()}'."
    ]
    assert manager.prepare() is False
    assert manager.generation == 1


def test_prepare_detects_content_changes_with_unchanged_file_metadata(tmp_path):
    """Content hashes detect rewrites even when size and modification time are preserved."""
    agents = tmp_path / "AGENTS.md"
    agents.write_text("First rules", encoding="utf-8")
    manager = InstructionsManager.discover(tmp_path)
    original = agents.stat()

    agents.write_text("Other rules", encoding="utf-8")
    os.utime(agents, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert manager.prepare() is True
    assert manager.instructions.endswith("Other rules")


def test_observed_file_changes_scope_and_rediscovers_nested_sources(tmp_path):
    """A loaded file establishes its containing directory as the next instruction scope."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("Root rules.", encoding="utf-8")
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("Nested rules.", encoding="utf-8")
    target = nested / "module.py"
    target.touch()
    manager = InstructionsManager.discover(tmp_path)

    manager.observe_path(target)
    manager.observe_path(target)

    assert manager.list_skills()["instruction_context"]["dirty"] is True
    assert manager.prepare() is True
    assert manager.working_directory == nested.resolve()
    assert manager.instructions.endswith("Root rules.\n\nNested rules.")
    changes = manager.list_skills()["instruction_context"]["refresh_changes"]
    assert changes[0] == f"Instruction scope changed to '{nested.resolve()}'."
    assert changes[1] == f"Added instruction source '{(nested / 'AGENTS.md').resolve()}'."


def test_refresh_preserves_same_skill_identity_and_reloads_changed_body(tmp_path):
    """Refresh keeps activation for the same canonical skill while loading its new body."""
    (tmp_path / ".git").mkdir()
    location = write_skill(tmp_path / ".agents" / "skills" / "review", "review", "First.")
    manager = InstructionsManager.discover(tmp_path)
    manager.activate_skill("review")

    location.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\nSecond body is longer.\n",
        encoding="utf-8",
    )

    assert manager.prepare() is True
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
    manager = InstructionsManager.discover(tmp_path)
    manager.activate_skill("review")
    write_skill(nested / ".agents" / "skills" / "review", "review", "Closer body.")

    manager.observe_path(nested, directory=True)

    assert manager.prepare() is True
    assert manager.active_skill_identities == []
    assert "Closer body." not in manager.instructions
    assert str(root_skill.location.resolve()) not in {
        str(skill.location) for skill in manager.skill_manager.activated_skills
    }
    assert "removed or shadowed" in manager.list_skills()["instruction_context"]["diagnostics"][0]


def test_reactivate_skills_restores_only_matching_current_identities(tmp_path):
    """Restoration activates exact current definitions and ignores missing or stale locations."""
    skill = write_skill(tmp_path / "skills" / "review", "review")
    manager = InstructionsManager(skill_manager=SkillManager([skill]))

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
    baseline = InstructionsManager(skill_manager=skill_manager).instructions
    manager = InstructionsManager(skill_manager=skill_manager, max_bytes=len(baseline) + 10)
    identity = [("large", str(skill.location))]

    rejected = manager.reactivate_skills(identity)

    assert rejected[0].code == "skill.instruction_budget_exceeded"
    assert manager.active_skill_identities == []

    roomy = InstructionsManager(skill_manager=SkillManager([skill]))
    roomy.activate_skill("large")
    repeated = roomy.reactivate_skills(identity)

    assert repeated[0]["instructions_updated"] is False

    externally_activated = SkillManager([skill])
    constrained = InstructionsManager(
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
    manager = InstructionsManager(skill_manager=SkillManager([skill]))
    manager.activate_skill("resourceful")

    assert manager.list_skill_resources("resourceful")["resources"] == [
        {"path": "references/guide.md", "size_bytes": 6}
    ]
    assert manager.read_skill_resource("resourceful", "references/guide.md")["content"] == "Guide."


def test_refresh_deactivates_invalid_and_oversized_active_skills(tmp_path, monkeypatch):
    """Refresh safely drops active bodies that fail validation or the instruction budget."""
    (tmp_path / ".git").mkdir()
    skill = write_skill(tmp_path / ".agents" / "skills" / "review", "review", "Short.")
    baseline = InstructionsManager.discover(tmp_path).instructions
    manager = InstructionsManager.discover(tmp_path, max_bytes=len(baseline.encode()) + 250)
    assert manager.activate_skill("review")["status"] == "activated"

    skill.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\n" + "x" * 1_000,
        encoding="utf-8",
    )

    assert manager.prepare() is True
    assert manager.active_skill_identities == []
    assert (
        "instruction budget exceeded"
        in manager.list_skills()["instruction_context"]["diagnostics"][0]
    )

    skill.location.write_text(
        "---\nname: review\ndescription: Use review.\n---\n\nShort again.\n",
        encoding="utf-8",
    )
    manager.prepare()
    manager.activate_skill("review")
    original_manager = manager.skill_manager
    original_activate = SkillManager.activate

    def fail_refreshed_activation(self, name):
        if self is original_manager:
            return original_activate(self, name)
        raise ValueError("changed skill is invalid")

    monkeypatch.setattr(SkillManager, "activate", fail_refreshed_activation)
    manager.invalidate()

    assert manager.prepare() is True
    assert (
        "changed skill is invalid" in manager.list_skills()["instruction_context"]["diagnostics"][0]
    )


def test_invalidation_is_selective_and_oversized_base_refresh_is_atomic(tmp_path, monkeypatch):
    """Only instruction sources invalidate, and an invalid candidate preserves prior content."""
    limit = len(TEST_POLICY.render().encode("utf-8")) + 20
    manager = InstructionsManager.discover(
        tmp_path,
        agent_policy=TEST_POLICY,
        max_bytes=limit,
    )
    manager.invalidate(tmp_path / "ordinary.py")
    assert manager.list_skills()["instruction_context"]["dirty"] is False

    manager.invalidate()
    assert manager.prepare() is False
    assert manager.generation == 0

    (tmp_path / "AGENTS.md").write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="Refreshed base instructions exceed"):
        manager.prepare()
    assert manager.instructions == TEST_POLICY.render()

    static = InstructionsManager()
    static.invalidate()
    assert static.prepare() is False

    clean = tmp_path / "clean"
    clean.mkdir()
    racing = InstructionsManager.discover(clean)
    missing = clean / "disappeared" / "AGENTS.md"
    monkeypatch.setattr(
        "loop.skills.instructions.get_agents_files",
        lambda working_directory, _filenames: [missing],
    )
    racing.invalidate()
    assert racing.prepare() is False


def test_manager_rejects_invalid_and_oversized_initial_limits():
    """Construction requires a positive limit containing the initial document."""
    for limit in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            InstructionsManager(max_bytes=limit)

    with pytest.raises(ValueError, match="Initial instructions exceed"):
        InstructionsManager(project_instructions="too large", max_bytes=2)


def test_activation_is_idempotent_and_unknown_skills_remain_errors(tmp_path):
    """Repeated activation reports no update while missing skills do not change instructions."""
    skill = write_skill(tmp_path / "example", "example")
    manager = InstructionsManager(skill_manager=SkillManager([skill]))

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
    baseline = InstructionsManager(skill_manager=skill_manager).instructions
    manager = InstructionsManager(skill_manager=skill_manager, max_bytes=len(baseline) + 10)

    result = manager.activate_skill("large")

    assert result.code == "skill.instruction_budget_exceeded"
    assert result.metadata["required_bytes"] > result.metadata["max_bytes"]
    assert manager.instructions == baseline
    assert skill_manager.activated == 0


def test_deactivation_is_idempotent_and_unknown_skills_remain_errors(tmp_path):
    """Deactivation reports changes while missing skills do not alter instructions."""
    skill = write_skill(tmp_path / "example", "example")
    manager = InstructionsManager(skill_manager=SkillManager([skill]))
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
