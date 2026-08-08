"""Tests for composing dynamic backend instructions."""

from pathlib import Path

import pytest

from loop import InstructionsManager, Skill, SkillManager


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
    manager = InstructionsManager("Project rules.", SkillManager([first, second]))
    initial = manager.instructions

    second_result = manager.activate_skill("second")
    first_result = manager.activate_skill("first")

    assert isinstance(initial, str)
    assert initial.startswith("Project rules.\n\n<available_skills>")
    assert manager.max_bytes == 64 * 1024
    assert "First instructions." not in initial
    assert second_result["instructions_updated"] is True
    assert "instructions" not in second_result
    assert first_result["instructions_updated"] is True
    assert manager.instructions.startswith(initial)
    assert manager.instructions.index("First instructions.") < manager.instructions.index(
        "Second instructions."
    )


def test_manager_discovers_project_instructions_and_skills(tmp_path):
    """Discovery loads applicable AGENTS.md content and repository skill metadata."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("Project rules.", encoding="utf-8")
    write_skill(tmp_path / ".agents" / "skills" / "review", "review")

    manager = InstructionsManager.discover(tmp_path)

    assert manager.instructions.startswith("Project rules.")
    assert manager.skill_manager.count == 1


def test_manager_rejects_invalid_and_oversized_initial_limits():
    """Construction requires a positive limit containing the initial document."""
    for limit in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            InstructionsManager(max_bytes=limit)

    with pytest.raises(ValueError, match="Initial instructions exceed"):
        InstructionsManager("too large", max_bytes=2)


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
    assert missing["error"] == "unknown_skill"


def test_activation_rolls_back_when_complete_instructions_exceed_budget(tmp_path):
    """An oversized skill is rejected atomically without partially changing instructions."""
    skill = write_skill(tmp_path / "large", "large", "x" * 100)
    skill_manager = SkillManager([skill])
    baseline = InstructionsManager(skill_manager=skill_manager).instructions
    manager = InstructionsManager(skill_manager=skill_manager, max_bytes=len(baseline) + 10)

    result = manager.activate_skill("large")

    assert result["error"] == "instruction_budget_exceeded"
    assert result["required_bytes"] > result["max_bytes"]
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
    assert missing["error"] == "unknown_skill"
