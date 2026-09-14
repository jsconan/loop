"""Define skill-domain models and structured results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from ..errors import Problem
from ..utils import sha256_digest


@dataclass(frozen=True)
class CapturedInstruction:
    """Preserve immutable instruction content with relocatable source provenance.

    Args:
        workspace_id (str | None): Owning workspace identity for an internal reference.
        workspace_relative_path (str | None): POSIX-style path relative to the workspace root.
        captured_absolute_path (Path): Absolute path observed when the reference was captured.
        content_digest (str): SHA-256 digest of the captured content.
        captured_content (str): Exact immutable captured content.
    """

    workspace_id: str | None
    workspace_relative_path: str | None
    captured_absolute_path: Path
    content_digest: str
    captured_content: str

    def __post_init__(self) -> None:
        if sha256_digest(self.captured_content) != self.content_digest:
            raise ValueError("Instruction content does not match its digest.")

    @classmethod
    def capture(
        cls,
        path: Path | str,
        content: str,
        *,
        workspace_id: str | None = None,
        workspace_root: Path | str | None = None,
    ) -> CapturedInstruction:
        """Capture immutable content and its source provenance.

        Args:
            path (Path | str): Source file path.
            content (str): Exact captured source content.
            workspace_id (str | None): Durable workspace identity.
            workspace_root (Path | str | None): Workspace root used to derive a relative path.
        Returns:
            CapturedInstruction: Immutable capture with relocation metadata when applicable.
        """
        absolute = Path(path).expanduser().resolve()
        relative = None
        owner = None
        if workspace_id is not None and workspace_root is not None:
            root = Path(workspace_root).expanduser().resolve()
            if absolute.is_relative_to(root):
                relative = absolute.relative_to(root).as_posix()
                owner = workspace_id
        return cls(owner, relative, absolute, sha256_digest(content), content)

    def resolve(self, workspace_id: str, workspace_root: Path | str) -> Path:
        """Resolve the current source path without rewriting historical provenance.

        Args:
            workspace_id (str): Active workspace identity.
            workspace_root (Path | str): Current canonical root for that identity.

        Returns:
            Path: Relocated internal path or unchanged external absolute path.

        Raises:
            ValueError: If an internal reference is resolved for a different workspace.
        """
        if self.workspace_relative_path is None:
            return self.captured_absolute_path
        if self.workspace_id != workspace_id:
            raise ValueError("Instruction reference belongs to a different workspace.")
        return Path(workspace_root).expanduser().resolve() / self.workspace_relative_path

    def content(self) -> str:
        """Return the exact captured content without consulting the filesystem.

        Returns:
            str: Immutable captured content matching ``content_digest``.
        """
        return self.captured_content


@dataclass(frozen=True)
class LiveInstructionSource:
    """Represent a relocatable filesystem source whose current content may change.

    Args:
        workspace_id (str | None): Owning workspace identity for an internal source.
        workspace_relative_path (str | None): POSIX-style path relative to the workspace root.
        captured_absolute_path (Path): Absolute path observed when the source was created.
    """

    workspace_id: str | None
    workspace_relative_path: str | None
    captured_absolute_path: Path

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        *,
        workspace_id: str | None = None,
        workspace_root: Path | str | None = None,
    ) -> LiveInstructionSource:
        """Create a live source with relocation metadata when applicable.

        Args:
            path (Path | str): Current source file path.
            workspace_id (str | None): Durable workspace identity.
            workspace_root (Path | str | None): Workspace root used to derive a relative path.

        Returns:
            LiveInstructionSource: Source that reads the current filesystem state.
        """
        absolute = Path(path).expanduser().resolve()
        if workspace_id is not None and workspace_root is not None:
            root = Path(workspace_root).expanduser().resolve()
            if absolute.is_relative_to(root):
                return cls(workspace_id, absolute.relative_to(root).as_posix(), absolute)
        return cls(None, None, absolute)

    def resolve(self, workspace_id: str, workspace_root: Path | str) -> Path:
        """Resolve the current source path.

        Args:
            workspace_id (str): Active workspace identity.
            workspace_root (Path | str): Current canonical root for that identity.

        Returns:
            Path: Relocated internal path or unchanged external absolute path.

        Raises:
            ValueError: If an internal source is resolved for a different workspace.
        """
        if self.workspace_relative_path is None:
            return self.captured_absolute_path
        if self.workspace_id != workspace_id:
            raise ValueError("Instruction source belongs to a different workspace.")
        return Path(workspace_root).expanduser().resolve() / self.workspace_relative_path

    def content(self, workspace_id: str, workspace_root: Path | str) -> str:
        """Read the source's current filesystem content.

        Args:
            workspace_id (str): Active workspace identity.
            workspace_root (Path | str): Current canonical workspace root.

        Returns:
            str: Current source content.
        """
        return self.resolve(workspace_id, workspace_root).read_text(encoding="utf-8")

    def capture(self, workspace_id: str, workspace_root: Path | str) -> CapturedInstruction:
        """Capture the source's current content and fresh digest.

        Args:
            workspace_id (str): Active workspace identity.
            workspace_root (Path | str): Current canonical workspace root.

        Returns:
            CapturedInstruction: Immutable capture of the current source state.
        """
        path = self.resolve(workspace_id, workspace_root)
        return CapturedInstruction.capture(
            path,
            path.read_text(encoding="utf-8"),
            workspace_id=self.workspace_id,
            workspace_root=workspace_root if self.workspace_relative_path is not None else None,
        )


@dataclass(frozen=True)
class InstructionSection:
    """Describe one logical section of the composed instruction document.

    Args:
        kind (str): Stable section category.
        content (str): Exact rendered section content.
        source (str | None): Canonical source path or logical producer.
        reference (CapturedInstruction | None): Immutable file provenance when source is a path.
    """

    kind: str
    content: str
    source: str | None = None
    reference: CapturedInstruction | None = None

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
            "working_directory: /workspace\n"
            "temporary_directory: /tmp\n"
            "File tools accept workspace-relative paths and VirtualPaths below /workspace, /tmp, "
            "or /skills.\n"
            "For terminal commands, select cwd '/workspace' or '/tmp' and use relative paths "
            "inside the command. VirtualPaths are not shell paths.\n"
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
        diagnostics (tuple[str, ...]): Diagnostics for skipped invalid sources.
    """

    content: str | None
    sources: tuple[AgentInstructionsSource, ...]
    max_bytes: int
    diagnostics: tuple[str, ...] = ()


class InstructionBudgetExceededError(ValueError):
    """Report that complete instructions cannot fit within the configured byte limit."""


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
