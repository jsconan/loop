"""Expose skill discovery and activation as user commands."""

from typing import Annotated, Literal

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
            tuple[CommandRegistration, ...]: Skill lifecycle commands.
        """
        return (
            CommandRegistration(
                self.skills,
                name="skills",
                completion=CommandCompletion(
                    values=(
                        CompletionValue("activate", "Load one skill for this session."),
                        CompletionValue("deactivate", "Unload one skill from this session."),
                        CompletionValue(
                            "deactivate-all",
                            "Unload every active skill from this session.",
                        ),
                    ),
                    children={
                        "activate": CommandCompletion(provider="skills"),
                        "deactivate": CommandCompletion(provider="skills"),
                    },
                ),
            ),
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

    def skills(
        self,
        context: CommandContext,
        action: Annotated[
            Literal["activate", "deactivate", "deactivate-all"] | None,
            Field(description="Skill lifecycle action, or omit to list skills."),
        ] = None,
        name: Annotated[
            str | None,
            Field(description="Exact skill name for activation or deactivation."),
        ] = None,
    ) -> None:
        """List skills or change the activation state of one or every skill."""
        if action == "activate":
            if name is None:
                raise CommandArgumentError("The activate action requires a skill name.")
            self._activate(context, name)
            return
        if action == "deactivate":
            if name is None:
                raise CommandArgumentError("The deactivate action requires a skill name.")
            self._deactivate(context, name)
            return
        if action == "deactivate-all":
            if name is not None:
                raise CommandArgumentError(
                    "The deactivate-all action does not accept a skill name."
                )
            self._deactivate_all(context)
            return

        result = self._instructions_manager.list_skills()
        if not result["skills"]:
            context.interaction.info("No skills discovered.")
            return
        context.interaction.table(
            result["skills"],
            title="Discovered skills:",
            columns=("name", "description", "activated"),
        )

    def use(
        self,
        context: CommandContext,
        name: Annotated[str, Field(description="Exact skill name.")],
    ) -> None:
        """Load a skill for subsequent model requests."""
        self._activate(context, name)

    def _activate(self, context: CommandContext, name: str) -> None:
        """Activate one skill and report its state change."""
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

    def _deactivate(self, context: CommandContext, name: str) -> None:
        """Deactivate one skill and report its state change."""
        try:
            result = self._instructions_manager.deactivate_skill(name)
        except (OSError, UnicodeError, ValueError) as exc:
            raise CommandArgumentError(f"Could not unload skill '{name}': {exc}") from exc
        if isinstance(result, SkillOperationError):
            raise CommandArgumentError(result.detail)
        if result["instructions_updated"]:
            context.interaction.info(f"Deactivated skill '{name}'.")
        else:
            context.interaction.info(f"Skill '{name}' is already inactive.")

    def _deactivate_all(self, context: CommandContext) -> None:
        """Deactivate every skill and report how many changed state."""
        try:
            result = self._instructions_manager.deactivate_all_skills()
        except (OSError, UnicodeError, ValueError) as exc:
            raise CommandArgumentError(f"Could not unload all skills: {exc}") from exc
        if result["instructions_updated"]:
            context.interaction.info(f"Deactivated {result['deactivated']} skill(s).")
        else:
            context.interaction.info("No skills are currently loaded.")
