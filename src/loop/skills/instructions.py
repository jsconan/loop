"""Compose the developer instructions used for backend requests."""

from collections.abc import Iterable
from html import escape
from pathlib import Path
from threading import RLock
from typing import Any, Self

from ..utils import find_project_root
from .skill_manager import SkillManager
from .utils import (
    DEFAULT_AGENTS_FILENAME,
    build_instructions,
    default_skill_directories,
    load_agents_instructions,
)

MAX_INSTRUCTIONS_BYTES = 64 * 1024


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
    _skill_discovery_enabled: bool
    _lock: RLock

    def __init__(
        self,
        project_instructions: str | None = None,
        skill_manager: SkillManager | None = None,
        *,
        max_bytes: int = MAX_INSTRUCTIONS_BYTES,
        working_directory: Path | str | None = None,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("Instruction limit must be a positive integer.")
        self._project_instructions = project_instructions
        self._skill_manager = skill_manager or SkillManager()
        self._max_bytes = max_bytes
        self._working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        self._dirty = False
        self._generation = 0
        self._signature = self._discovery_signature(self._working_directory)
        self._refresh_diagnostics = []
        self._skill_discovery_enabled = False
        self._lock = RLock()
        self._instructions = self._build_instructions()
        if self._encoded_size(self._instructions) > self._max_bytes:
            raise ValueError("Initial instructions exceed the configured instruction limit.")

    @classmethod
    def discover(
        cls,
        working_directory: Path | str,
        *,
        skill_manager: SkillManager | None = None,
        max_bytes: int = MAX_INSTRUCTIONS_BYTES,
    ) -> Self:
        """Discover project instructions and skills for a working directory.

        Args:
            working_directory (Path | str): Directory whose instruction scopes should apply.
            skill_manager (SkillManager | None): Explicit skill manager to use instead of
                discovering one.
            max_bytes (int): Maximum encoded size of the complete instruction document.

        Returns:
            InstructionsManager: Manager containing the discovered instruction sources.

        Raises:
            OSError: An applicable instruction file cannot be read.
            UnicodeError: An applicable instruction file is not valid UTF-8.
            ValueError: The configured limit is invalid or initial instructions exceed it.
        """
        directory = Path(working_directory).resolve()
        manager = cls(
            load_agents_instructions(directory),
            skill_manager or SkillManager.discover(directory),
            max_bytes=max_bytes,
            working_directory=directory,
        )
        manager._skill_discovery_enabled = skill_manager is None
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
        return [(skill.name, str(skill.location)) for skill in self._skill_manager.activated_skills]

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

    def list_skills(self) -> dict[str, Any]:
        """Return available skills and activation diagnostics.

        Returns:
            dict[str, Any]: Model-readable skill summaries and diagnostics.
        """
        result = self._skill_manager.list()
        result["instruction_context"] = {
            "working_directory": (
                str(self._working_directory) if self._working_directory is not None else None
            ),
            "generation": self._generation,
            "dirty": self._dirty,
            "diagnostics": list(self._refresh_diagnostics),
        }
        return result

    def reactivate_skills(self, identities: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
        """Reactivate skills that still match their persisted identities.

        Args:
            identities (Iterable[tuple[str, str]]): Skill names paired with canonical persisted
                ``SKILL.md`` locations.

        Returns:
            list[dict[str, Any]]: Activation results for matching current definitions. Missing or
                newly shadowed identities are omitted.
        """
        results = []
        for name, location in identities:
            skill = next(
                (
                    candidate
                    for candidate in self._skill_manager.skills
                    if candidate.name == name and str(candidate.location) == location
                ),
                None,
            )
            if skill is not None:
                results.append(self.activate_skill(name))
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
        if path is not None and Path(path).name not in {DEFAULT_AGENTS_FILENAME, "SKILL.md"}:
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

    def activate_skill(self, name: str) -> dict[str, Any]:
        """Activate a skill for subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            dict[str, Any]: Compact activation acknowledgement or structured error. The skill body
                is retained in the instruction document rather than returned in this result.

        Raises:
            OSError: The skill instruction file cannot be read.
            UnicodeError: The skill instruction file is not valid UTF-8.
            ValueError: The skill instruction file is malformed.
        """
        with self._lock:
            self.prepare()
            previously_active = any(
                skill.name == name for skill in self._skill_manager.activated_skills
            )
            result = self._skill_manager.activate(name)
            if "error" in result:
                return result
            instructions = self._build_instructions()
            size = self._encoded_size(instructions)
            if size > self._max_bytes:
                self._skill_manager.deactivate(name)
                return {
                    "error": "instruction_budget_exceeded",
                    "message": f"Activating skill '{name}' exceeds the instruction limit.",
                    "max_bytes": self._max_bytes,
                    "required_bytes": size,
                }
            updated = not previously_active
            if updated:
                self._instructions = instructions
                self._generation += 1
            result["instructions_updated"] = updated
            return result

    def deactivate_skill(self, name: str) -> dict[str, Any]:
        """Deactivate a skill and remove it from subsequent backend requests.

        Args:
            name (str): Exact available skill name.

        Returns:
            dict[str, Any]: Compact deactivation acknowledgement or structured error.
        """
        with self._lock:
            self.prepare()
            previously_active = any(
                skill.name == name for skill in self._skill_manager.activated_skills
            )
            result = self._skill_manager.deactivate(name)
            if "error" not in result:
                result["instructions_updated"] = previously_active
                if previously_active:
                    self._instructions = self._build_instructions()
                    self._generation += 1
            return result

    def _refresh(self, working_directory: Path, signature: tuple) -> bool:
        """Build and atomically install a refreshed instruction snapshot."""
        previous_target = self._signature[0] if self._signature else None
        previous_instructions = self._instructions
        active = {
            (skill.name, skill.location) for skill, _ in self._skill_manager.activated_instructions
        }
        project_instructions = load_agents_instructions(working_directory)
        if not self._skill_discovery_enabled:
            instructions = self._compose(project_instructions, self._skill_manager)
            if self._encoded_size(instructions) > self._max_bytes:
                raise ValueError(
                    "Refreshed base instructions exceed the configured instruction limit."
                )
            changed = (
                previous_target != str(working_directory) or previous_instructions != instructions
            )
            self._project_instructions = project_instructions
            self._instructions = instructions
            self._signature = signature
            self._refresh_diagnostics = []
            self._dirty = False
            if changed:
                self._generation += 1
            return changed

        skill_manager = SkillManager.discover(working_directory)
        diagnostics = []

        for skill in skill_manager.skills:
            identity = (skill.name, skill.location)
            if identity not in active:
                continue
            try:
                skill_manager.activate(skill.name)
            except (OSError, UnicodeError, ValueError) as exc:
                diagnostics.append(f"Deactivated '{skill.name}' during refresh: {exc}")
                continue
            instructions = self._compose(project_instructions, skill_manager)
            if self._encoded_size(instructions) > self._max_bytes:
                skill_manager.deactivate(skill.name)
                diagnostics.append(
                    f"Deactivated '{skill.name}' during refresh: instruction budget exceeded."
                )

        instructions = self._compose(project_instructions, skill_manager)
        if self._encoded_size(instructions) > self._max_bytes:
            raise ValueError("Refreshed base instructions exceed the configured instruction limit.")

        preserved = {(skill.name, skill.location) for skill in skill_manager.activated_skills}
        for name, location in active:
            if (name, location) not in preserved and not any(
                diagnostic.startswith(f"Deactivated '{name}'") for diagnostic in diagnostics
            ):
                diagnostics.append(
                    f"Deactivated '{name}' during refresh: its definition was removed or shadowed."
                )

        changed = previous_target != str(working_directory) or previous_instructions != instructions
        self._project_instructions = project_instructions
        self._skill_manager = skill_manager
        self._instructions = instructions
        self._signature = signature
        self._refresh_diagnostics = diagnostics
        self._dirty = False
        if changed:
            self._generation += 1
        return changed

    def _build_instructions(self) -> str | None:
        """Render the current instruction sources."""
        return self._compose(self._project_instructions, self._skill_manager)

    @classmethod
    def _compose(cls, project_instructions: str | None, skill_manager: SkillManager) -> str | None:
        """Render an instruction document from candidate sources."""
        entries = []
        for skill, instructions in skill_manager.activated_instructions:
            entries.append(
                f'<skill name="{escape(skill.name)}" root="{escape(str(skill.location.parent))}">\n'
                f"{instructions}\n"
                "</skill>"
            )
        active = (
            "<active_skills>\n" + "\n".join(entries) + "\n</active_skills>" if entries else None
        )
        return build_instructions(project_instructions, skill_manager.catalog(), active)

    @staticmethod
    def _discovery_signature(working_directory: Path | None) -> tuple:
        """Return cheap metadata identifying discoverable instruction sources."""
        if working_directory is None:
            return ()
        root = find_project_root(working_directory)
        if root is None:
            directories = [working_directory]
        else:
            directories = [
                directory
                for directory in reversed((working_directory, *working_directory.parents))
                if directory == root or root in directory.parents
            ]
        paths = [directory / DEFAULT_AGENTS_FILENAME for directory in directories]
        for skill_directory in default_skill_directories(working_directory):
            paths.append(skill_directory)
            if skill_directory.is_dir():
                paths.extend(sorted(skill_directory.glob("*/SKILL.md")))
        fingerprints = []
        for path in paths:
            try:
                stat = path.stat()
                fingerprints.append(
                    (str(path.resolve()), stat.st_ino, stat.st_mtime_ns, stat.st_size)
                )
            except FileNotFoundError:
                fingerprints.append((str(path.resolve()), None))
        return (str(working_directory), *fingerprints)

    @staticmethod
    def _encoded_size(instructions: str | None) -> int:
        """Return the UTF-8 size of an optional instruction document."""
        return len((instructions or "").encode("utf-8"))
