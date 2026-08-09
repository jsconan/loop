"""Compose the developer instructions used for backend requests."""

from collections.abc import Iterable
from html import escape
from pathlib import Path
from threading import RLock
from typing import Self

from .. import constants
from ..utils import sha256_digest
from .models import (
    AgentInstructionsSource,
    InstructionContext,
    InstructionSection,
    InstructionSectionSummary,
    InstructionSourceSummary,
    LoadedAgentInstructions,
    ManagedSkillListResult,
    SkillActivationResponse,
    SkillActivationResult,
    SkillDeactivationAllResult,
    SkillDeactivationResponse,
    SkillOperationError,
    SkillResourceContentResponse,
    SkillResourceListResponse,
)
from .skill_manager import SkillManager
from .utils import (
    build_instructions,
    get_agents_files,
    load_agents_instructions,
)


class InstructionsManager:
    """Maintain project, catalog, and activated-skill instructions.

    Args:
        project_instructions (str | None): Project instructions loaded from applicable AGENTS.md
            files, or ``None`` when none apply.
        skill_manager (SkillManager | None): Skill manager supplying the catalog and lazily loaded
            skill bodies. Defaults to an empty manager.
        max_bytes (int): Maximum encoded size of the complete instruction document.
        working_directory (Path | str | None): Directory whose project instructions can be
            refreshed, or ``None`` for static injected instructions.
        agents_filenames (tuple[str, ...]): Candidate project instruction filenames in precedence
            order. Defaults to ``("AGENTS.md",)``.

    Raises:
        ValueError: If ``max_bytes`` is not a positive integer or the initial instructions exceed
            the configured limit.
    """

    _project_instructions: str | None
    _skill_manager: SkillManager
    _max_bytes: int
    _working_directory: Path | None
    _dirty: bool
    _generation: int
    _signature: tuple
    _instructions: str | None
    _refresh_diagnostics: list[str]
    _project_sources: tuple[AgentInstructionsSource, ...]
    _skill_discovery_enabled: bool
    _lock: RLock
    _refresh_changes: list[str]
    _agents_filenames: tuple[str, ...]

    def __init__(
        self,
        project_instructions: str | None = None,
        skill_manager: SkillManager | None = None,
        *,
        max_bytes: int = constants.MAX_INSTRUCTIONS_BYTES,
        working_directory: Path | str | None = None,
        agents_filenames: tuple[str, ...] = (constants.DEFAULT_AGENTS_FILENAME,),
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("Instruction limit must be a positive integer.")
        self._project_instructions = project_instructions
        self._skill_manager = skill_manager or SkillManager()
        self._max_bytes = max_bytes
        self._working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        self._agents_filenames = tuple(dict.fromkeys(agents_filenames))
        self._dirty = False
        self._generation = 0
        self._signature = self._discovery_signature(self._working_directory)
        self._refresh_diagnostics = []
        self._project_sources = ()
        self._skill_discovery_enabled = False
        self._lock = RLock()
        self._refresh_changes = []
        self._instructions = self._build_instructions()
        if self._encoded_size(self._instructions) > self._max_bytes:
            raise ValueError("Initial instructions exceed the configured instruction limit.")

    @classmethod
    def discover(
        cls,
        working_directory: Path | str,
        *,
        skill_manager: SkillManager | None = None,
        max_bytes: int = constants.MAX_INSTRUCTIONS_BYTES,
        agents_filenames: tuple[str, ...] = (constants.DEFAULT_AGENTS_FILENAME,),
    ) -> Self:
        """Discover project instructions and skills for a working directory.

        Args:
            working_directory (Path | str): Directory whose instruction scopes should apply.
            skill_manager (SkillManager | None): Explicit skill manager to use instead of
                discovering one.
            max_bytes (int): Maximum encoded size of the complete instruction document.
            agents_filenames (tuple[str, ...]): Candidate instruction filenames in precedence
                order, deduplicated with ``AGENTS.md`` normally first.

        Returns:
            InstructionsManager: Manager containing the discovered instruction sources.

        Raises:
            OSError: An applicable instruction file cannot be read.
            UnicodeError: An applicable instruction file is not valid UTF-8.
            ValueError: The configured limit is invalid or initial instructions exceed it.
        """
        directory = Path(working_directory).resolve()
        loaded = load_agents_instructions(directory, agents_filenames)
        manager = cls(
            loaded.content,
            skill_manager or SkillManager.discover(directory),
            max_bytes=max_bytes,
            working_directory=directory,
            agents_filenames=agents_filenames,
        )
        manager._skill_discovery_enabled = skill_manager is None
        manager._project_sources = loaded.sources
        manager._refresh_diagnostics = manager._load_diagnostics(loaded)
        return manager

    @property
    def instructions(self) -> str | None:
        """Return the complete instructions for the next backend request.

        Returns:
            str | None: Project instructions, skill catalog, and active skill bodies in stable
                order, or ``None`` when no instructions are available.
        """
        return self._instructions

    @property
    def working_directory(self) -> Path | None:
        """Return the directory whose instruction scope is active.

        Returns:
            Path | None: Active resolved instruction directory, or ``None`` for a static manager.
        """
        return self._working_directory

    @property
    def generation(self) -> int:
        """Return the number of effective instruction changes.

        Returns:
            int: Monotonically increasing instruction generation.
        """
        return self._generation

    @property
    def active_skill_identities(self) -> list[tuple[str, str]]:
        """Return active skill names and canonical instruction locations.

        Returns:
            list[tuple[str, str]]: Active identities in discovery order.
        """
        return list(self._skill_manager.active_identities)

    @property
    def skill_manager(self) -> SkillManager:
        """Return the skill manager providing catalog and activation state.

        Returns:
            SkillManager: The associated skill manager.
        """
        return self._skill_manager

    @property
    def max_bytes(self) -> int:
        """Return the complete instruction document size limit.

        Returns:
            int: Maximum encoded instruction size in bytes.
        """
        return self._max_bytes

    def list_skills(self) -> ManagedSkillListResult:
        """Return available skills and activation diagnostics.

        Returns:
            ManagedSkillListResult: Model-readable skill summaries and diagnostics.
        """
        result = self._skill_manager.list()
        sections = self._instruction_sections()
        result["instruction_context"] = InstructionContext(
            working_directory=(
                str(self._working_directory) if self._working_directory is not None else None
            ),
            generation=self._generation,
            dirty=self._dirty,
            diagnostics=[
                *self._refresh_diagnostics,
                *self._skill_manager.lifecycle_diagnostics,
            ],
            refresh_changes=list(self._refresh_changes),
            sources=[
                InstructionSourceSummary(
                    path=str(source.path),
                    size_bytes=source.size_bytes,
                    included_bytes=source.included_bytes,
                    truncated=source.truncated,
                )
                for source in self._project_sources
            ],
            size_bytes=self._encoded_size(self._instructions),
            max_bytes=self._max_bytes,
            digest=sha256_digest(self._instructions or ""),
            sections=[
                InstructionSectionSummary(
                    kind=section.kind,
                    source=section.source,
                    size_bytes=section.size_bytes,
                    digest=section.digest,
                )
                for section in sections
            ],
        )

        return result

    def reactivate_skills(
        self, identities: Iterable[tuple[str, str]]
    ) -> list[SkillActivationResult]:
        """Reactivate skills that still match their persisted identities.

        Args:
            identities (Iterable[tuple[str, str]]): Skill names paired with canonical persisted
                ``SKILL.md`` locations.

        Returns:
            list[SkillActivationResult]: Activation results for matching current definitions.
                Missing or newly shadowed identities are omitted.
        """
        with self._lock:
            self.prepare()
            results = []
            for identity in identities:
                restored = self._skill_manager.restore([identity])
                if not restored:
                    continue
                results.append(self._commit_activation(restored[0]))
            return results

    def observe_path(self, path: Path | str, *, directory: bool = False) -> None:
        """Record a successfully accessed path as the next instruction scope.

        Args:
            path (Path | str): Successfully accessed file or directory.
            directory (bool): Whether ``path`` itself is a directory. File observations use their
                parent directory.
        """
        target = Path(path).resolve()
        if not directory:
            target = target.parent
        with self._lock:
            if self._working_directory != target:
                self._working_directory = target
                self._dirty = True

    def invalidate(self, path: Path | str | None = None) -> None:
        """Mark discovered instructions stale when an applicable source may have changed.

        Args:
            path (Path | str | None): Changed source path, or ``None`` for unconditional
                invalidation. Unrelated paths are ignored when their names cannot affect discovery.
        """
        if path is not None and Path(path).name not in {
            *self._agents_filenames,
            constants.DEFAULT_SKILL_FILENAME,
        }:
            return
        with self._lock:
            if self._working_directory is not None:
                self._dirty = True

    def prepare(self) -> bool:
        """Refresh stale discovered instructions before a backend request.

        Returns:
            bool: Whether the effective instructions or target directory changed.

        Raises:
            OSError: An applicable project instruction cannot be read.
            UnicodeError: An applicable project instruction is not valid UTF-8.
        """
        with self._lock:
            if self._working_directory is None:
                return False
            working_directory = self._working_directory
            signature = self._discovery_signature(working_directory)
            if not self._dirty and signature == self._signature:
                return False
            return self._refresh(working_directory, signature)

    def activate_skill(self, name: str) -> SkillActivationResponse:
        """Activate a skill for subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            SkillActivationResponse: Compact activation acknowledgement or structured error. The
                skill body is retained in the instruction document rather than returned here.

        Raises:
            OSError: The skill instruction file cannot be read.
            UnicodeError: The skill instruction file is not valid UTF-8.
            ValueError: The skill instruction file is malformed.
        """
        with self._lock:
            self.prepare()
            result = self._skill_manager.activate(name)
            if "error" in result:
                return result
            return self._commit_activation(result)

    def deactivate_skill(self, name: str) -> SkillDeactivationResponse:
        """Deactivate a skill and remove it from subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            SkillDeactivationResponse: Compact deactivation acknowledgement or structured error.
        """
        with self._lock:
            self.prepare()
            result = self._skill_manager.deactivate(name)
            if "error" not in result:
                if result["instructions_updated"]:
                    self._instructions = self._build_instructions()
                    self._generation += 1
            return result

    def deactivate_all_skills(self) -> SkillDeactivationAllResult:
        """Deactivate every active skill and rebuild subsequent instructions.

        Returns:
            SkillDeactivationAllResult: Acknowledgement with the number of deactivated skills.
        """
        with self._lock:
            self.prepare()
            count = self._skill_manager.deactivate_all()
            if count:
                self._instructions = self._build_instructions()
                self._generation += 1
            return SkillDeactivationAllResult(
                status="deactivated_all",
                deactivated=count,
                instructions_updated=bool(count),
            )

    def list_skill_resources(self, name: str) -> SkillResourceListResponse:
        """List resources available to one active skill.

        Args:
            name (str): Exact active skill name.

        Returns:
            SkillResourceListResponse: Bounded resource metadata or a structured error.
        """
        with self._lock:
            self.prepare()
            return self._skill_manager.list_resources(name)

    def read_skill_resource(self, name: str, resource_path: str) -> SkillResourceContentResponse:
        """Read one active skill resource on demand.

        Args:
            name (str): Exact active skill name.
            resource_path (str): Relative resource path beneath the skill root.

        Returns:
            SkillResourceContentResponse: Resource content or a structured error.
        """
        with self._lock:
            self.prepare()
            return self._skill_manager.read_resource(name, resource_path)

    def _refresh(self, working_directory: Path, signature: tuple) -> bool:
        """Build and atomically install a refreshed instruction snapshot."""
        previous_target = self._signature[0] if self._signature else None
        previous_instructions = self._instructions
        active = self._skill_manager.active_identities
        loaded = load_agents_instructions(working_directory, self._agents_filenames)
        project_instructions = loaded.content
        refresh_changes = self._describe_refresh(working_directory, loaded)
        if not self._skill_discovery_enabled:
            return self._refresh_static(
                working_directory,
                signature,
                loaded,
                refresh_changes,
                previous_target,
                previous_instructions,
            )

        skill_manager = SkillManager.rediscover(working_directory, active)
        diagnostics = self._load_diagnostics(loaded)

        for skill in skill_manager.activated_skills:
            instructions = self._compose(project_instructions, skill_manager)
            if self._encoded_size(instructions) > self._max_bytes:
                skill_manager.deactivate(skill.name)
                diagnostics.append(
                    f"Deactivated '{skill.name}' during refresh: instruction budget exceeded."
                )

        instructions = self._compose(project_instructions, skill_manager)
        if self._encoded_size(instructions) > self._max_bytes:
            raise ValueError("Refreshed base instructions exceed the configured instruction limit.")

        changed = previous_target != str(working_directory) or previous_instructions != instructions
        self._project_instructions = project_instructions
        self._project_sources = loaded.sources
        self._skill_manager = skill_manager
        self._instructions = instructions
        self._signature = signature
        self._refresh_diagnostics = diagnostics
        self._refresh_changes = refresh_changes
        self._dirty = False
        if changed:
            self._generation += 1
        return changed

    def _refresh_static(
        self,
        working_directory: Path,
        signature: tuple,
        loaded: LoadedAgentInstructions,
        refresh_changes: list[str],
        previous_target: str | None,
        previous_instructions: str | None,
    ) -> bool:
        """Install refreshed project content while preserving an injected skill manager."""
        instructions = self._compose(loaded.content, self._skill_manager)
        if self._encoded_size(instructions) > self._max_bytes:
            raise ValueError("Refreshed base instructions exceed the configured instruction limit.")
        changed = previous_target != str(working_directory) or previous_instructions != instructions
        self._project_instructions = loaded.content
        self._project_sources = loaded.sources
        self._instructions = instructions
        self._signature = signature
        self._refresh_diagnostics = self._load_diagnostics(loaded)
        self._refresh_changes = refresh_changes
        self._dirty = False
        if changed:
            self._generation += 1
        return changed

    def _build_instructions(self) -> str | None:
        """Render the current instruction sources."""
        return self._compose(self._project_instructions, self._skill_manager)

    def _commit_activation(self, result: SkillActivationResult) -> SkillActivationResponse:
        """Commit an activated skill to the aggregate instruction snapshot or roll it back."""
        instructions = self._build_instructions()
        size = self._encoded_size(instructions)
        if size > self._max_bytes:
            if result["instructions_updated"]:
                self._skill_manager.deactivate(result["name"])
            return SkillOperationError(
                error="instruction_budget_exceeded",
                message=f"Activating skill '{result['name']}' exceeds the instruction budget.",
                max_bytes=self._max_bytes,
                required_bytes=size,
            )
        if result["instructions_updated"]:
            self._instructions = instructions
            self._generation += 1
        return result

    def _instruction_sections(self) -> tuple[InstructionSection, ...]:
        """Return typed provenance for every currently composed section."""
        sections = [
            InstructionSection("agents", source.content, str(source.path))
            for source in self._project_sources
            if source.content
        ]
        if self._project_instructions and not sections:
            sections.append(InstructionSection("agents", self._project_instructions, "injected"))
        catalog = self._skill_manager.catalog()
        if catalog:
            sections.append(InstructionSection("skill_catalog", catalog, "skill_discovery"))
        sections.extend(
            InstructionSection("active_skill", instructions, str(skill.location))
            for skill, instructions in self._skill_manager.activated_instructions
        )
        return tuple(sections)

    @classmethod
    def _compose(cls, project_instructions: str | None, skill_manager: SkillManager) -> str | None:
        """Render an instruction document from candidate sources."""
        entries = []
        for skill, instructions in skill_manager.activated_instructions:
            entries.append(f'<skill name="{escape(skill.name)}">\n{instructions}\n</skill>')
        active = (
            "<active_skills>\n" + "\n".join(entries) + "\n</active_skills>" if entries else None
        )
        return build_instructions(project_instructions, skill_manager.catalog(), active)

    def _discovery_signature(self, working_directory: Path | None) -> tuple:
        """Return cheap metadata identifying discoverable instruction sources."""
        if working_directory is None:
            return ()
        paths = get_agents_files(working_directory, self._agents_filenames)
        fingerprints = []
        for path in paths:
            try:
                stat = path.stat()
                digest = sha256_digest(path.read_bytes()) if path.is_file() else None
                fingerprints.append(
                    (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size, digest)
                )
            except FileNotFoundError:
                fingerprints.append((str(path.resolve()), None))
        return (
            str(working_directory),
            *fingerprints,
            SkillManager.discovery_signature(working_directory),
        )

    @staticmethod
    def _encoded_size(instructions: str | None) -> int:
        """Return the UTF-8 size of an optional instruction document."""
        return len((instructions or "").encode("utf-8"))

    @staticmethod
    def _load_diagnostics(loaded: LoadedAgentInstructions) -> list[str]:
        """Return diagnostics produced while loading project instructions."""
        diagnostics = list(loaded.diagnostics)
        if not loaded.truncated:
            return diagnostics
        omitted = sum(source.size_bytes - source.included_bytes for source in loaded.sources)
        diagnostics.append(
            f"Agent instructions truncated at {loaded.max_bytes} bytes; "
            f"{omitted} source byte(s) omitted."
        )
        return diagnostics

    def _describe_refresh(
        self, working_directory: Path, loaded: LoadedAgentInstructions
    ) -> list[str]:
        """Describe effective project-source changes for observability."""
        changes = []
        previous_target = self._signature[0] if self._signature else None
        if previous_target != str(working_directory):
            changes.append(f"Instruction scope changed to '{working_directory}'.")
        previous = {str(source.path): source for source in self._project_sources}
        current = {str(source.path): source for source in loaded.sources}
        for path in sorted(previous.keys() - current.keys()):
            changes.append(f"Removed instruction source '{path}'.")
        for path in sorted(current.keys() - previous.keys()):
            changes.append(f"Added instruction source '{path}'.")
        for path in sorted(previous.keys() & current.keys()):
            before = previous[path]
            after = current[path]
            if before.content != after.content or before.truncated != after.truncated:
                changes.append(f"Changed instruction source '{path}'.")
        if self._signature != self._discovery_signature(working_directory) and not changes:
            changes.append("Skill catalog or activated skill instructions changed.")
        return changes
