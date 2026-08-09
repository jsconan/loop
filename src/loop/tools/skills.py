"""Expose progressively loaded Agent Skills to the model."""

from typing import Annotated, Any, Literal

from pydantic import Field

from ..context import ToolContext
from ..skills.types import (
    PublicSkillOperationResult,
    SkillOperationError,
    SkillOperationResult,
)
from ..tooling import tool_registry

_FIELDS_BY_NAME = {
    "activate": ("name", "status", "instructions_updated"),
    "deactivate": ("name", "status", "instructions_updated"),
    "deactivate_all": ("status", "deactivated", "instructions_updated"),
    "list_resources": ("name", "resources"),
    "read_resource": ("name", "path", "size_bytes", "encoding", "content"),
}


def _filter_fields(result: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Return only the specified fields from a result dictionary."""
    return {key: result[key] for key in fields if key in result}


def _public_result(action: str, result: SkillOperationResult) -> PublicSkillOperationResult:
    """Return only fields required by the model-facing skill protocol."""
    if "error" in result:
        public = _filter_fields(result, ("error", "message"))
        if action == "read_resource" and "size_bytes" in result:
            public["size_bytes"] = result["size_bytes"]
        return public
    if action == "list":
        return {
            "skills": [
                _filter_fields(skill, ("name", "description", "activated"))
                for skill in result.get("skills", [])
            ]
        }
    return _filter_fields(result, _FIELDS_BY_NAME[action])


@tool_registry.tool
def manage_skills(
    context: ToolContext,
    action: Annotated[
        Literal[
            "list",
            "activate",
            "deactivate",
            "deactivate_all",
            "list_resources",
            "read_resource",
        ],
        Field(description="Whether to list, activate, or deactivate skills."),
    ],
    name: Annotated[
        str | None,
        Field(description="Exact skill name for activation or deactivation; null when listing."),
    ] = None,
    path: Annotated[
        str | None,
        Field(description="Relative skill resource path when reading a resource."),
    ] = None,
) -> PublicSkillOperationResult:
    """List, activate, or deactivate skill instructions on demand."""
    manager = context.instructions_manager
    if manager is None:
        return SkillOperationError(
            error="skills_unavailable",
            message="No InstructionsManager is active.",
        )
    if action not in ("list", "deactivate_all") and not name:
        return SkillOperationError(
            error="missing_skill_name",
            message=f"The {action} action requires a skill name.",
        )
    try:
        if action == "list":
            result = manager.list_skills()
        elif action == "deactivate_all":
            result = manager.deactivate_all_skills()
        elif action == "activate":
            result = manager.activate_skill(name)
        elif action == "list_resources":
            result = manager.list_skill_resources(name)
        elif action == "read_resource":
            if not path:
                return SkillOperationError(
                    error="missing_resource_path",
                    message="The read_resource action requires a relative path.",
                )
            result = manager.read_skill_resource(name, path)
        else:
            result = manager.deactivate_skill(name)
    except OSError, UnicodeError, ValueError:
        target = f" for skill '{name}'" if name else ""
        return SkillOperationError(
            error="skill_operation_failed",
            message=f"The {action} action failed{target}.",
        )
    return _public_result(action, result)
