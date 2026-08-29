"""Tests for skill-owned user commands."""

from unittest.mock import Mock

from loop import CommandManager, InstructionsManager, Interaction, Skill, SkillManager
from loop.instructions import SkillCommands


def test_skill_commands_list_and_activate_skills_idempotently(tmp_path):
    """Skill commands display discovery and report initial and repeated activation."""
    location = tmp_path / "SKILL.md"
    location.write_text("---\nname: review\ndescription: Review code.\n---\nCheck carefully.\n")
    instructions = InstructionsManager(
        skill_manager=SkillManager([Skill("review", "Review.", location)])
    )
    interaction = Mock(spec=Interaction)
    provider = SkillCommands(instructions)
    manager = CommandManager(interaction=interaction)
    manager.register_provider(provider)

    manager.call("skills")
    manager.call("use", "review")
    manager.call("use", "review")

    interaction.table.assert_called_once_with(
        instructions.skill_manager.skills,
        title="Discovered skills:",
    )
    assert "Check carefully." in instructions.instructions
    assert interaction.info.call_args_list[-2].args[0] == "Loaded skill 'review'."
    assert interaction.info.call_args_list[-1].args[0] == "Skill 'review' is already loaded."
    completion = provider.get_completion_providers()[0]
    assert [(value.value, value.description) for value in completion.provider()] == [
        ("review", "Review.")
    ]


def test_skill_commands_report_empty_unknown_and_loading_failures():
    """Skill commands require a selection, report empty catalogs, and normalize failures."""
    interaction = Mock(spec=Interaction)
    instructions = InstructionsManager()
    manager = CommandManager(interaction=interaction)
    manager.register_provider(SkillCommands(instructions))

    manager.call("skills")
    manager.call("use")
    assert "Field required" in interaction.report.call_args.args[0].detail
    manager.call("use", "missing")
    assert interaction.info.call_args.args[0] == "No skills discovered."
    assert "Skill 'missing' is not available" in interaction.report.call_args.args[0].detail

    broken = Mock(spec=InstructionsManager)
    broken.activate_skill.side_effect = ValueError("malformed instructions")
    failing = CommandManager(interaction=interaction)
    failing.register_provider(SkillCommands(broken))
    failing.call("use", "broken")
    assert "Could not load skill 'broken'" in interaction.report.call_args.args[0].detail
