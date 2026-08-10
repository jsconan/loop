"""Expose progressively loaded Agent Skills to the model."""

from typing import Annotated, Any, Literal

from pydantic import Field

from .. import constants
from ..context import ToolContext
from ..permissions import Capability, PermissionRequest
from ..skills.models import (
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
    "read_resource": (
        "name",
        "path",
        "size_bytes",
        "encoding",
        "content",
        "start_byte",
        "end_byte",
        "included_bytes",
        "truncated",
        "truncation_reason",
        "start_line",
        "end_line",
        "next_start_byte",
        "next_start_line",
    ),
}


def _filter_fields(result: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Return only the specified fields from a result dictionary."""
    return {key: result[key] for key in fields if key in result}


def _public_result(action: str, result: SkillOperationResult) -> PublicSkillOperationResult:
    """Return only fields required by the model-facing skill protocol."""
    if "error" in result:
        return _filter_fields(result, ("error", "message"))
    if action == "list":
        return {
            "skills": [
                _filter_fields(skill, ("name", "description", "activated"))
                for skill in result.get("skills", [])
            ]
        }
    return _filter_fields(result, _FIELDS_BY_NAME[action])


def _skill_permission(arguments: dict[str, Any]) -> tuple[PermissionRequest, ...]:
    """Classify read-only and mutating skill operations."""
    action = arguments["action"]
    capability = (
        Capability.SESSION_WRITE
        if action in {"activate", "deactivate", "deactivate_all"}
        else Capability.FILESYSTEM_READ
    )
    resource = str(arguments.get("path") or arguments.get("name") or action)
    return (PermissionRequest(tool_name="manage_skills", capability=capability, resource=resource),)


@tool_registry.tool(permission_resolver=_skill_permission)
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
    start_byte: Annotated[
        int | None,
        Field(
            description="Zero-based resource offset; start_line may remain 1 only at byte zero.",
            ge=0,
        ),
    ] = None,
    start_line: Annotated[
        int | None,
        Field(description="One-based resource line; set to null for byte access.", ge=1),
    ] = 1,
    max_lines: Annotated[
        int | None,
        Field(
            description="Optional line ceiling; the first reached line or byte limit wins.", ge=1
        ),
    ] = None,
    max_bytes: Annotated[
        int,
        Field(
            description="Maximum raw resource bytes returned.",
            ge=1,
            le=constants.MAX_TOOL_CONTENT_BYTES,
        ),
    ] = constants.MAX_TOOL_CONTENT_BYTES,
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
            result = manager.read_skill_resource(
                name,
                path,
                start_byte=start_byte,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )
        else:
            result = manager.deactivate_skill(name)
    except OSError, UnicodeError, ValueError:
        target = f" for skill '{name}'" if name else ""
        return SkillOperationError(
            error="skill_operation_failed",
            message=f"The {action} action failed{target}.",
        )
    return _public_result(action, result)
