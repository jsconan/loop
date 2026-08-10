"""Tests for the action-based skill management tool."""

import json
from pathlib import Path
from unittest.mock import Mock

from loop import InstructionsManager, Interaction, Skill, SkillManager, tool_registry


def test_manage_skills_lists_activates_and_deactivates_through_one_tool(tmp_path):
    """The skill lifecycle returns only fields required by each model-facing action."""
    location = tmp_path / "example" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: example\ndescription: Example workflow.\n---\nDo the work.",
        encoding="utf-8",
    )
    reference = location.parent / "references" / "guide.md"
    reference.parent.mkdir()
    reference.write_text("Guide content.", encoding="utf-8")
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
    resources = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"list_resources","name":"example","path":null}',
            interaction=interaction,
            instructions_manager=instructions_manager,
        )
    )
    resource = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"read_resource","name":"example","path":"references/guide.md"}',
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
    instructions_manager.activate_skill("example")
    deactivated_all = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"deactivate_all","name":null}',
            interaction=interaction,
            instructions_manager=instructions_manager,
        )
    )

    assert listed == {
        "skills": [{"name": "example", "description": "Example workflow.", "activated": False}]
    }
    assert activated == {
        "name": "example",
        "status": "activated",
        "instructions_updated": True,
    }
    assert resources == {
        "name": "example",
        "resources": [{"path": "references/guide.md", "size_bytes": 14}],
    }
    assert resource == {
        "name": "example",
        "path": "references/guide.md",
        "size_bytes": 14,
        "encoding": "utf-8",
        "content": "Guide content.",
        "start_byte": 0,
        "end_byte": 14,
        "included_bytes": 14,
        "truncated": False,
        "start_line": 1,
        "end_line": 1,
    }
    assert deactivated == {
        "name": "example",
        "status": "deactivated",
        "instructions_updated": True,
    }
    assert deactivated_all == {
        "status": "deactivated_all",
        "deactivated": 1,
        "instructions_updated": True,
    }
    assert str(tmp_path) not in json.dumps(
        [listed, activated, resources, resource, deactivated, deactivated_all]
    )
    assert "skill_root" not in resource
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
    missing_resource_path = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"read_resource","name":"example","path":null}',
            interaction=interaction,
            instructions_manager=InstructionsManager(
                skill_manager=SkillManager([Skill("example", "Example.", Path("SKILL.md"))])
            ),
        )
    )

    assert unavailable["error"] == "skills_unavailable"
    assert missing_name["error"] == "missing_skill_name"
    assert missing_deactivation_name == {
        "error": "missing_skill_name",
        "message": "The deactivate action requires a skill name.",
    }
    assert missing_resource_path["error"] == "missing_resource_path"


def test_manage_skills_sanitizes_activation_failures(tmp_path):
    """Activation failures do not expose exception details or filesystem paths to the model."""
    location = tmp_path / "private" / "SKILL.md"
    manager = InstructionsManager(
        skill_manager=SkillManager([Skill("broken", "Broken skill.", location)])
    )

    result = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"activate","name":"broken"}',
            interaction=Mock(spec=Interaction),
            instructions_manager=manager,
        )
    )

    assert result == {
        "error": "skill_operation_failed",
        "message": "The activate action failed for skill 'broken'.",
    }
    assert str(location) not in json.dumps(result)


def test_manage_skills_returns_only_public_error_details(tmp_path):
    """Manager errors retain useful public details while omitting internal metadata."""
    location = tmp_path / "large" / "SKILL.md"
    location.parent.mkdir()
    location.write_text(
        "---\nname: large\ndescription: Large resource.\n---\nInstructions.",
        encoding="utf-8",
    )
    asset = location.parent / "assets" / "large.bin"
    asset.parent.mkdir()
    asset.write_bytes(b"x" * (64 * 1024 + 1))
    manager = InstructionsManager(
        skill_manager=SkillManager([Skill("large", "Large resource.", location)])
    )
    manager.activate_skill("large")
    interaction = Mock(spec=Interaction)

    unknown = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"activate","name":"missing"}',
            interaction=interaction,
            instructions_manager=manager,
        )
    )
    oversized = json.loads(
        tool_registry.call(
            "manage_skills",
            '{"action":"read_resource","name":"large","path":"assets/large.bin"}',
            interaction=interaction,
            instructions_manager=manager,
        )
    )

    assert unknown == {
        "error": "unknown_skill",
        "message": "Skill 'missing' is not available.",
    }
    assert oversized["size_bytes"] == 64 * 1024 + 1
    assert oversized["included_bytes"] == 16 * 1024
    assert oversized["next_start_byte"] == 16 * 1024
    assert oversized["truncated"] is True
