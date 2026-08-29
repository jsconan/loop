"""Define skill-domain models and structured results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from ..errors import Problem
from ..utils.hashing import sha256_digest


@dataclass(frozen=True)
class InstructionSection:
    """Describe one logical section of the composed instruction document.

    Args:
        kind (str): Stable section category.
        content (str): Exact rendered section content.
        source (str | None): Canonical source path or logical producer.
    """

    kind: str
    content: str
    source: str | None = None

    @property
    def size_bytes(self) -> int:
        """Return the section's UTF-8 size.

        Returns:
            int: Encoded section size in bytes.
        """
        return len(self.content.encode("utf-8"))

    @property
    def digest(self) -> str:
        """Return a stable content digest suitable for cache diagnostics.

        Returns:
            str: SHA-256 hexadecimal digest.
        """
        return sha256_digest(self.content)


@dataclass(frozen=True)
class PreparedInstructions:
    """Capture one immutable aggregate instruction document.

    Args:
        content (str): Exact model-facing instruction document.
        generation (int): Manager generation that produced the document.
        working_directory (Path | None): Active instruction discovery directory.
        active_skills (tuple[tuple[str, str], ...]): Active skill names and locations.
        sections (tuple[InstructionSection, ...]): Ordered instruction provenance.
        digest (str): SHA-256 digest of the complete document.
    """

    content: str
    generation: int
    working_directory: Path | None
    active_skills: tuple[tuple[str, str], ...]
    sections: tuple[InstructionSection, ...]
    digest: str


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Describe runtime paths available to the model.

    Args:
        working_directory (Path): Directory used as the current workspace.
        temporary_directory (Path): Explicitly permitted ephemeral directory for scratch files.
    """

    working_directory: Path
    temporary_directory: Path

    def render(self) -> str:
        """Render the model-facing runtime environment section.

        Returns:
            str: XML-like runtime environment guidance with the current paths.
        """
        return (
            "<runtime_environment>\n"
            f"working_directory: {self.working_directory}\n"
            f"temporary_directory: {self.temporary_directory}\n"
            "Use temporary_directory only for scratch files; its contents are ephemeral.\n"
            "</runtime_environment>"
        )


@dataclass(frozen=True)
class Skill:
    """Describe an Agent Skill without eagerly loading its instructions.

    Args:
        name (str): Public name declared by the skill.
        description (str): Summary used by the model to decide when to activate the skill.
        location (Path): Absolute path to the skill's ``SKILL.md`` file.
    """

    name: str
    description: str
    location: Path


@dataclass(frozen=True)
class AgentInstructionsSource:
    """Describe one discovered project instruction source.

    Args:
        path (Path): Canonical instruction file path.
        size_bytes (int): Complete stripped source size in UTF-8 bytes.
        included_bytes (int): Number of source bytes included in the result.
        truncated (bool): Whether content from this source was omitted.
        content (str): Included source content before composition separators.
    """

    path: Path
    size_bytes: int
    included_bytes: int
    truncated: bool
    content: str


@dataclass(frozen=True)
class LoadedAgentInstructions:
    """Describe composed project instructions and their provenance.

    Args:
        content (str | None): Bounded composed content, or ``None`` when no source applies.
        sources (tuple[AgentInstructionsSource, ...]): Sources in root-to-leaf precedence order.
        max_bytes (int): Configured source-content byte limit.
        truncated (bool): Whether any discovered source content was omitted.
        diagnostics (tuple[str, ...]): Diagnostics for skipped invalid sources.
    """

    content: str | None
    sources: tuple[AgentInstructionsSource, ...]
    max_bytes: int
    truncated: bool
    diagnostics: tuple[str, ...] = ()


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


class SkillOperationError(Problem):
    """Describe an error that occurred during a skill operation."""


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
    start_byte: int
    end_byte: int
    included_bytes: int
    truncated: bool
    truncation_reason: NotRequired[Literal["bytes", "lines"]]
    start_line: NotRequired[int]
    end_line: NotRequired[int]
    next_start_byte: NotRequired[int]
    next_start_line: NotRequired[int]


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
