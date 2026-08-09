"""Define structured dictionary contracts for skill operations."""

from typing import Literal, NotRequired, TypedDict


class SkillSummary(TypedDict):
    """Describe one discovered skill."""

    name: str
    description: str
    location: str
    activated: bool


class SkillListResult(TypedDict):
    """Describe discovered skills and their diagnostics."""

    skills: list[SkillSummary]
    diagnostics: list[str]


class SkillOperationError(TypedDict):
    """Describe a failed skill operation."""

    error: str
    message: str
    size_bytes: NotRequired[int]
    max_bytes: NotRequired[int]
    required_bytes: NotRequired[int]


class SkillActivationResult(SkillSummary):
    """Describe a successful skill activation."""

    skill_root: str
    status: Literal["activated"]
    instructions_updated: bool


class SkillDeactivationResult(SkillSummary):
    """Describe a successful skill deactivation."""

    status: Literal["deactivated"]
    instructions_updated: bool


class SkillDeactivationAllResult(TypedDict):
    """Describe deactivation of all active skills."""

    status: Literal["deactivated_all"]
    deactivated: int
    instructions_updated: bool


class SkillResource(TypedDict):
    """Describe one resource belonging to a skill."""

    path: str
    size_bytes: int


class SkillResourceListResult(TypedDict):
    """Describe the resources available to an active skill."""

    name: str
    skill_root: str
    resources: list[SkillResource]


class SkillResourceContentResult(TypedDict):
    """Describe loaded content from one skill resource."""

    name: str
    path: str
    size_bytes: int
    encoding: Literal["utf-8", "base64"]
    content: str


type SkillActivationResponse = SkillActivationResult | SkillOperationError
type SkillDeactivationResponse = SkillDeactivationResult | SkillOperationError
type SkillResourceListResponse = SkillResourceListResult | SkillOperationError
type SkillResourceContentResponse = SkillResourceContentResult | SkillOperationError
type SkillOperationResult = (
    SkillListResult
    | SkillActivationResult
    | SkillDeactivationResult
    | SkillDeactivationAllResult
    | SkillResourceListResult
    | SkillResourceContentResult
    | SkillOperationError
)


class InstructionSourceSummary(TypedDict):
    """Describe one project-instruction source."""

    path: str
    size_bytes: int
    included_bytes: int
    truncated: bool


class InstructionSectionSummary(TypedDict):
    """Describe one composed instruction section."""

    kind: str
    source: str | None
    size_bytes: int
    digest: str


class InstructionContext(TypedDict):
    """Describe the current composed instruction state."""

    working_directory: str | None
    generation: int
    dirty: bool
    diagnostics: list[str]
    refresh_changes: list[str]
    sources: list[InstructionSourceSummary]
    size_bytes: int
    max_bytes: int
    digest: str
    sections: list[InstructionSectionSummary]


class ManagedSkillListResult(SkillListResult):
    """Describe skills together with their composed instruction state."""

    instruction_context: InstructionContext


class PublicSkillSummary(TypedDict):
    """Describe one skill through the model-facing tool contract."""

    name: str
    description: str
    activated: bool


class PublicSkillListResult(TypedDict):
    """Describe available skills through the model-facing tool contract."""

    skills: list[PublicSkillSummary]


class PublicSkillStateResult(TypedDict):
    """Describe one activation-state change through the model-facing tool contract."""

    name: str
    status: Literal["activated", "deactivated"]
    instructions_updated: bool


class PublicSkillResourceListResult(TypedDict):
    """Describe skill resources through the model-facing tool contract."""

    name: str
    resources: list[SkillResource]


type PublicSkillOperationResult = (
    PublicSkillListResult
    | PublicSkillStateResult
    | SkillDeactivationAllResult
    | PublicSkillResourceListResult
    | SkillResourceContentResult
    | SkillOperationError
)
