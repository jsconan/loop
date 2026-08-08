"""Tests for the action-based skill management tool."""

import json
from unittest.mock import Mock

from loop import InstructionsManager, Interaction, Skill, SkillManager, tool_registry


def test_manage_skills_lists_activates_and_deactivates_through_one_tool(tmp_path):
    """One action parameter manages the complete skill lifecycle without returning bodies."""
    location = tmp_path / "example" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: example\ndescription: Example workflow.\n---\nDo the work.",
        encoding="utf-8",
    )
    manager = SkillManager([Skill("example", "Example workflow.", location)])
    instructions_manager = InstructionsManager(skill_manager=manager)
    interaction = Mock(spec=Interaction)

    listed = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"list","name":null}',
            interaction=interaction,
            instructions_manager=instructions_manager,
        )
    )
    activated = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"activate","name":"example"}',
            interaction=interaction,
            instructions_manager=instructions_manager,
        )
    )
    deactivated = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"deactivate","name":"example"}',
            interaction=interaction,
            instructions_manager=instructions_manager,
        )
    )

    assert listed["skills"][0]["name"] == "example"
    assert activated["instructions_updated"] is True
    assert "instructions" not in activated
    assert deactivated["instructions_updated"] is True
    assert "instructions" not in deactivated
    assert "Do the work." not in instructions_manager.instructions


def test_manage_skills_validates_mutations_and_runtime_manager():
    """Mutations need a name while every action needs an active instruction manager."""
    interaction = Mock(spec=Interaction)
    unavailable = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"list","name":null}',
            interaction=interaction,
        )
    )
    missing_name = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"activate","name":null}',
            interaction=interaction,
            instructions_manager=InstructionsManager(),
        )
    )
    missing_deactivation_name = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"deactivate","name":null}',
            interaction=interaction,
            instructions_manager=InstructionsManager(),
        )
    )

    assert unavailable["error"] == "skills_unavailable"
    assert missing_name["error"] == "missing_skill_name"
    assert missing_deactivation_name == {
        "error": "missing_skill_name",
        "message": "The deactivate action requires a skill name.",
    }
