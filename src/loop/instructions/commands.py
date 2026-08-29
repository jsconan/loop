"""Expose skill discovery and activation as user commands."""

from typing import Annotated

from pydantic import Field

from ..commands import CommandArgumentError, CommandContext, CommandRegistration
from ..completion import CommandCompletion, CompletionProviderRegistration, CompletionValue
from .instructions import InstructionsManager
from .models import SkillOperationError


class SkillCommands:
    """Expose one instructions manager through interactive commands.

    Args:
        instructions_manager (InstructionsManager): Skill lifecycle owner used by the commands.
    """

    def __init__(self, instructions_manager: InstructionsManager) -> None:
        self._instructions_manager = instructions_manager

    def get_commands(self) -> tuple[CommandRegistration, ...]:
        """Return skill command registrations.

        Returns:
            tuple[CommandRegistration, ...]: Skill discovery and activation commands.
        """
        return (
            CommandRegistration(self.skills, name="skills"),
            CommandRegistration(
                self.use,
                name="use",
                completion=CommandCompletion(provider="skills"),
            ),
        )

    def get_completion_providers(self) -> tuple[CompletionProviderRegistration, ...]:
        """Return dynamic skill completion sources.

        Returns:
            tuple[CompletionProviderRegistration, ...]: Named skill completion source.
        """
        return (CompletionProviderRegistration("skills", self._skill_values),)

    def _skill_values(self) -> tuple[CompletionValue, ...]:
        """Return currently discovered skills."""
        return tuple(
            CompletionValue(skill.name, skill.description)
            for skill in self._instructions_manager.skill_manager.skills
        )

    def skills(self, context: CommandContext) -> None:
        """List all discovered skills with their descriptions."""
        skills = self._instructions_manager.skill_manager.skills
        if not skills:
            context.interaction.info("No skills discovered.")
            return
        context.interaction.table(skills, title="Discovered skills:")

    def use(
        self,
        context: CommandContext,
        name: Annotated[str, Field(description="Exact skill name.")],
    ) -> None:
        """Load a skill for subsequent model requests."""
        try:
            result = self._instructions_manager.activate_skill(name)
        except (OSError, UnicodeError, ValueError) as exc:
            raise CommandArgumentError(f"Could not load skill '{name}': {exc}") from exc
        if isinstance(result, SkillOperationError):
            raise CommandArgumentError(result.detail)
        if result["instructions_updated"]:
            context.interaction.info(f"Loaded skill '{name}'.")
        else:
            context.interaction.info(f"Skill '{name}' is already loaded.")
