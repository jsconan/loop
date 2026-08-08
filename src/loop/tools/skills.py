"""Expose progressively loaded Agent Skills to the model."""

from typing import Annotated, Literal

from pydantic import Field

from ..context import ToolContext
from ..tooling import tool_registry


@tool_registry.tool
def manage_skills(
    context: ToolContext,
    action: Annotated[
        Literal["list", "activate"],
        Field(description="Whether to list available skills or activate one skill."),
    ],
    name: Annotated[
        str | None,
        Field(description="Exact skill name for activation; null when listing skills."),
    ] = None,
) -> dict:
    """List available skills or activate one skill's instructions on demand."""
    manager = context.instructions_manager
    if manager is None:
        return {"error": "skills_unavailable", "message": "No InstructionsManager is active."}
    if action == "list":
        return manager.list_skills()
    if not name:
        return {
            "error": "missing_skill_name",
            "message": "The activate action requires a skill name.",
        }
    return manager.activate_skill(name)
