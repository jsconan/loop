"""Tests for the action-based skill management tool."""

import json
from unittest.mock import Mock

from loop import Interaction, Skill, SkillManager, tool_registry


def test_manage_skills_lists_and_activates_through_one_tool(tmp_path):
    """One action parameter selects metadata listing or instruction activation."""
    location = tmp_path / "example" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: example\ndescription: Example workflow.\n---\nDo the work.",
        encoding="utf-8",
    )
    manager = SkillManager([Skill("example", "Example workflow.", location)])
    interaction = Mock(spec=Interaction)

    listed = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"list","name":null}',
            interaction=interaction,
            skill_manager=manager,
        )
    )
    activated = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"activate","name":"example"}',
            interaction=interaction,
            skill_manager=manager,
        )
    )

    assert listed["skills"][0]["name"] == "example"
    assert activated["instructions"] == "Do the work."


def test_manage_skills_validates_activation_and_runtime_manager():
    """Activation needs both a name and an active session manager."""
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
            skill_manager=SkillManager(),
        )
    )

    assert unavailable["error"] == "skills_unavailable"
    assert missing_name["error"] == "missing_skill_name"
