"""Tests for skill-owned user commands."""

from unittest.mock import Mock

from loop import Agent, CommandManager, InstructionsManager, Interaction, Skill, SkillManager
from loop.instructions import SkillCommands


def test_skill_commands_list_and_change_skill_activation_idempotently(tmp_path):
    """Skill commands display state and report idempotent activation changes."""
    location = tmp_path / "SKILL.md"
    location.write_text("---\nname: review\ndescription: Review code.\n---\nCheck carefully.\n")
    instructions = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review.", location)])
    )
    agent = Agent("Assistant")
    instructions.prepare(agent)
    interaction = Mock(spec=Interaction)
    provider = SkillCommands(instructions)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(provider)

    manager.call("skills")
    manager.call("skills", "activate review")
    manager.call("skills")
    manager.call("use", "review")
    manager.call("skills", "deactivate review")
    manager.call("skills", "deactivate review")

    assert interaction.table.call_args_list[0].args[0][0]["activated"] is False
    assert interaction.table.call_args_list[1].args[0][0]["activated"] is True
    assert interaction.table.call_args_list[0].kwargs == {
        "title": "Discovered skills:",
        "columns": ("name", "description", "activated"),
    }
    assert "Check carefully." not in instructions.instructions
    assert interaction.info.call_args_list[-4].args[0] == "Loaded skill 'review'."
    assert interaction.info.call_args_list[-3].args[0] == "Skill 'review' is already loaded."
    assert interaction.info.call_args_list[-2].args[0] == "Deactivated skill 'review'."
    assert interaction.info.call_args_list[-1].args[0] == "Skill 'review' is already inactive."
    completion = provider.get_completion_providers()[0]
    assert [(value.value, value.description) for value in completion.provider()] == [
        ("review", "Review.")
    ]


def test_skill_commands_deactivate_all_loaded_skills(tmp_path):
    """The bulk command unloads all active skill instructions and is idempotent."""
    first_location = tmp_path / "first" / "SKILL.md"
    second_location = tmp_path / "second" / "SKILL.md"
    first_location.parent.mkdir()
    second_location.parent.mkdir()
    first_location.write_text("---\nname: first\ndescription: First.\n---\nFirst instructions.\n")
    second_location.write_text(
        "---\nname: second\ndescription: Second.\n---\nSecond instructions.\n"
    )
    instructions = InstructionsManager(
        skill_manager=SkillManager(
            [Skill("first", "First.", first_location), Skill("second", "Second.", second_location)]
        )
    )
    instructions.prepare(Agent("Assistant"))
    interaction = Mock(spec=Interaction)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SkillCommands(instructions))

    manager.call("skills", "activate first")
    manager.call("skills", "activate second")
    manager.call("skills", "deactivate-all")
    manager.call("skills", "deactivate-all")

    assert instructions.skill_manager.activated == 0
    assert "First instructions." not in instructions.instructions
    assert "Second instructions." not in instructions.instructions
    assert interaction.info.call_args_list[-2].args[0] == "Deactivated 2 skill(s)."
    assert interaction.info.call_args_list[-1].args[0] == "No skills are currently loaded."


def test_skill_commands_report_invalid_arguments_and_lifecycle_failures():
    """Skill commands validate actions and normalize manager failures."""
    interaction = Mock(spec=Interaction)
    instructions = InstructionsManager()
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SkillCommands(instructions))

    manager.call("skills")
    manager.call("skills", "activate")
    assert "requires a skill name" in interaction.report.call_args.args[0].detail
    manager.call("skills", "deactivate")
    assert "requires a skill name" in interaction.report.call_args.args[0].detail
    manager.call("skills", "deactivate-all review")
    assert "does not accept a skill name" in interaction.report.call_args.args[0].detail
    manager.call("skills", "activate missing")
    assert interaction.info.call_args.args[0] == "No skills discovered."
    assert "Skill 'missing' is not available" in interaction.report.call_args.args[0].detail
    manager.call("skills", "deactivate missing")
    assert "Skill 'missing' is not available" in interaction.report.call_args.args[0].detail

    broken = Mock(spec=InstructionsManager)
    broken.activate_skill.side_effect = ValueError("malformed instructions")
    broken.deactivate_skill.side_effect = ValueError("malformed instructions")
    broken.deactivate_all_skills.side_effect = ValueError("malformed instructions")
    failing = CommandManager(interaction=interaction)
    failing.register_provider(SkillCommands(broken))
    failing.call("skills", "activate broken")
    assert "Could not load skill 'broken'" in interaction.report.call_args.args[0].detail
    failing.call("skills", "deactivate broken")
    assert "Could not unload skill 'broken'" in interaction.report.call_args.args[0].detail
    failing.call("skills", "deactivate-all")
    assert "Could not unload all skills" in interaction.report.call_args.args[0].detail
