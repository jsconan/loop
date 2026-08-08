"""Expose progressively loaded Agent Skills to the model."""

from typing import Annotated, Literal

from pydantic import Field

from ..context import ToolContext
from ..tooling import tool_registry


@tool_registry.tool
def manage_skills(
    context: ToolContext,
    action: Annotated[
        Literal["list", "activate", "deactivate"],
        Field(description="Whether to list, activate, or deactivate skills."),
    ],
    name: Annotated[
        str | None,
        Field(
            description="Exact skill name for activation or deactivation; null when listing."
        ),
    ] = None,
) -> dict:
    """List, activate, or deactivate skill instructions on demand."""
    manager = context.instructions_manager
    if manager is None:
        return {"error": "skills_unavailable", "message": "No InstructionsManager is active."}
    if action == "list":
        return manager.list_skills()
    if not name:
        return {
            "error": "missing_skill_name",
            "message": f"The {action} action requires a skill name.",
        }
    if action == "activate":
        return manager.activate_skill(name)
    return manager.deactivate_skill(name)
